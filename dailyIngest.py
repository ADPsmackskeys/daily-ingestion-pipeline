"""
NSE Daily Incremental Ingest
==============================
Fetches only NEW trading data since the last load date and appends
it to BRONZE.RAW_OHLCV and BRONZE.RAW_EVENTS.

Designed to run daily via GitHub Actions after NSE market close (3:30pm IST).
Cron schedule: 30 12 * * 1-5  (12:30 UTC = 6:00pm IST, Mon-Fri)

Unlike the full historical load (nse_bronze_ingest.py), this script:
  - Queries Snowflake for the latest date already loaded
  - Fetches only the gap (last_loaded_date + 1 → today)
  - Appends (overwrite=False) rather than replacing
  - Calls the Snowflake stored procedure to refresh Silver/Gold
  - Exits cleanly with code 0 on success, 1 on failure (for CI visibility)
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime, date, timedelta, timezone

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from snowflake.snowpark import Session

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
SNOWFLAKE_CONFIG = {
    "account"  : os.environ["SNOWFLAKE_ACCOUNT"],
    "user"     : os.environ["SNOWFLAKE_USER"],
    "password" : os.environ["SNOWFLAKE_PASSWORD"],
    "role"     : os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database" : os.getenv("SNOWFLAKE_DATABASE", "NSE_DATABASE"),
    "schema"   : "BRONZE",
}

# Nifty 50 tickers — keep in sync with nse_bronze_ingest.py
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TECHM", "POWERGRID", "NTPC", "ONGC",
    "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "BRITANNIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HEROMOTOCO", "HINDALCO", "INDUSINDBK",
    "CIPLA", "DIVISLAB", "APOLLOHOSP", "BPCL", "TATACONSUM",
    "SBILIFE", "HDFCLIFE", "UPL", "LTIM", "BAJAJ-AUTO",
]

YF_TICKERS = [f"{s}.NS" for s in NIFTY_50_SYMBOLS]


# ── Helpers ───────────────────────────────────────────────────────

def get_session() -> Session:
    log.info("Connecting to Snowflake ...")
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
    log.info("Connected as %s", session.get_current_user())
    return session


def get_last_loaded_date(session: Session) -> date:
    """
    Returns the latest TRADE_DATE already in RAW_OHLCV.
    Falls back to 5 trading days ago if the table is somehow empty.
    """
    result = session.sql(
        "SELECT MAX(TRADE_DATE) AS last_date FROM BRONZE.RAW_OHLCV"
    ).collect()

    last = result[0]["LAST_DATE"]
    if last is None:
        fallback = date.today() - timedelta(days=7)
        log.warning("RAW_OHLCV appears empty — falling back to %s", fallback)
        return fallback

    log.info("Last loaded date in RAW_OHLCV: %s", last)
    return last


def is_nse_trading_day(session: Session, check_date: date) -> bool:
    """Check DIM_DATE to see if today is actually a trading day."""
    result = session.sql(f"""
        SELECT IS_TRADING_DAY
        FROM   SILVER.DIM_DATE
        WHERE  FULL_DATE = '{check_date}'
    """).collect()

    if not result:
        log.warning("Date %s not found in DIM_DATE — assuming trading day", check_date)
        return True

    is_trading = result[0]["IS_TRADING_DAY"]
    log.info("Is %s a trading day? %s", check_date, is_trading)
    return bool(is_trading)


# ── Bronze: incremental OHLCV ─────────────────────────────────────

def fetch_incremental_ohlcv(start: date, end: date) -> pd.DataFrame:
    """
    Fetch OHLCV for the date range (start, end] — exclusive of start
    since that date is already loaded.
    """
    fetch_from = (start + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_to   = (end   + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    log.info("Fetching OHLCV: %s → %s (%s tickers)", fetch_from, fetch_to, len(YF_TICKERS))

    raw = yf.download(
        tickers=YF_TICKERS,
        start=fetch_from,
        end=fetch_to,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    if raw.empty:
        log.info("No new OHLCV data returned (likely a non-trading day).")
        return pd.DataFrame()

    INGEST_TS = datetime.now(timezone.utc)
    records   = []

    for ticker in YF_TICKERS:
        symbol = ticker.replace(".NS", "")
        try:
            df = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date": "trade_date"})

            df["symbol"]    = symbol
            df["yf_ticker"] = ticker
            df["exchange"]  = "NSE"
            df["currency"]  = "INR"
            df["ingest_ts"] = INGEST_TS
            df["source"]    = "YFINANCE_INCREMENTAL"

            # OHLC integrity filter
            df = df[
                (df["low"]    <= df["close"]) &
                (df["close"]  <= df["high"])  &
                (df["open"]   >  0)           &
                (df["volume"] >= 0)
            ]

            records.append(df[[
                "trade_date", "symbol", "yf_ticker", "exchange", "currency",
                "open", "high", "low", "close", "volume", "ingest_ts", "source",
            ]])

        except Exception as e:
            log.warning("OHLCV failed for %s: %s", symbol, e)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.date

    # Remove any dates already in Snowflake (safety dedup)
    result = result[result["trade_date"] > start]

    log.info(
        "New OHLCV rows: %s | dates: %s → %s",
        f"{len(result):,}",
        result["trade_date"].min(),
        result["trade_date"].max(),
    )
    return result


# ── Bronze: incremental events ────────────────────────────────────

def fetch_incremental_events(start: date, end: date) -> pd.DataFrame:
    """
    Fetch dividends and splits for the incremental window.
    yfinance returns the full history for events — we filter to new dates only.
    """
    log.info("Fetching corporate events for new window ...")
    INGEST_TS = datetime.now(timezone.utc)
    records   = []

    for ticker_str in YF_TICKERS:
        symbol = ticker_str.replace(".NS", "")
        try:
            obj = yf.Ticker(ticker_str)

            for event_df, event_type, unit in [
                (obj.dividends, "DIVIDEND", "INR_PER_SHARE"),
                (obj.splits,    "SPLIT",    "RATIO"),
            ]:
                if event_df.empty:
                    continue
                df = event_df.reset_index()
                df.columns = ["event_date", "value"]
                df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

                # Keep only new events
                df = df[(df["event_date"] > start) & (df["event_date"] <= end)]
                if df.empty:
                    continue

                df["event_type"] = event_type
                df["unit"]       = unit
                df["symbol"]     = symbol
                df["yf_ticker"]  = ticker_str
                df["exchange"]   = "NSE"
                df["ingest_ts"]  = INGEST_TS
                records.append(df)

            time.sleep(0.3)

        except Exception as e:
            log.warning("Events failed for %s: %s", symbol, e)

    if not records:
        log.info("No new corporate events.")
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    result = result[["event_date", "symbol", "yf_ticker", "exchange",
                     "event_type", "value", "unit", "ingest_ts"]]
    result = result.drop_duplicates(subset=["event_date", "symbol", "event_type"])
    log.info("New event rows: %s", len(result))
    return result


# ── Silver/Gold refresh ───────────────────────────────────────────

def refresh_silver_gold(session: Session) -> None:
    """
    Call the Silver stored procedure to merge new Bronze rows into
    FACT_DAILY_PRICE. Gold views refresh automatically on next query.
    """
    log.info("Refreshing Silver FACT_DAILY_PRICE ...")
    result = session.sql("CALL SILVER.UPDATE_DIM_STOCK()").collect()
    log.info("  → %s", result[0][0] if result else "done")

    # Refresh event fact table
    log.info("Refreshing Gold FACT_STOCK_EVENT ...")
    session.sql("""
        INSERT INTO GOLD.FACT_STOCK_EVENT (STOCK_KEY, DATE_KEY, EVENT_TYPE, EVENT_VALUE, UNIT)
        SELECT
            ds.STOCK_KEY,
            dd.DATE_KEY,
            e.EVENT_TYPE,
            e.VALUE,
            e.UNIT
        FROM BRONZE.RAW_EVENTS e
        JOIN SILVER.DIM_STOCK ds
            ON e.SYMBOL = ds.SYMBOL AND ds.IS_CURRENT = TRUE
        JOIN SILVER.DIM_DATE dd
            ON e.EVENT_DATE = dd.FULL_DATE
        WHERE NOT EXISTS (
            SELECT 1 FROM GOLD.FACT_STOCK_EVENT tgt
            WHERE tgt.STOCK_KEY  = ds.STOCK_KEY
              AND tgt.DATE_KEY   = dd.DATE_KEY
              AND tgt.EVENT_TYPE = e.EVENT_TYPE
        )
    """).collect()
    log.info("FACT_STOCK_EVENT refreshed.")


# ── Summary ───────────────────────────────────────────────────────

def log_summary(session: Session, run_date: date) -> None:
    db = SNOWFLAKE_CONFIG["database"]
    rows = session.sql(f"""
        SELECT COUNT(*) AS n
        FROM   BRONZE.RAW_OHLCV
        WHERE  TRADE_DATE = '{run_date}'
    """).collect()[0]["N"]

    gold_rows = session.sql("""
        SELECT COUNT(*) AS n FROM GOLD.FACT_DAILY_PRICE
    """).collect()[0]["N"]

    log.info("─" * 50)
    log.info("Run date           : %s", run_date)
    log.info("New OHLCV rows     : %s", rows)
    log.info("FACT_DAILY_PRICE   : %s total rows", f"{gold_rows:,}")
    log.info("─" * 50)


# ── Main ──────────────────────────────────────────────────────────

def main() -> int:
    today = date.today()
    log.info("=" * 50)
    log.info("NSE incremental ingest — %s", today)
    log.info("=" * 50)

    try:
        session = get_session()

        # Skip weekends early (before any Snowflake queries)
        if today.weekday() >= 5:
            log.info("Today is %s — weekend. No NSE trading. Exiting.", today.strftime("%A"))
            session.close()
            return 0

        # Check DIM_DATE for NSE holiday
        if not is_nse_trading_day(session, today):
            log.info("Today (%s) is an NSE holiday. Exiting.", today)
            session.close()
            return 0

        # Get last loaded date
        last_date = get_last_loaded_date(session)

        if last_date >= today:
            log.info("Already up to date (last: %s). Exiting.", last_date)
            session.close()
            return 0

        # ── Fetch and load OHLCV ──────────────────────────────────
        ohlcv_df = fetch_incremental_ohlcv(start=last_date, end=today)
        if not ohlcv_df.empty:
            session.write_pandas(
                ohlcv_df,
                table_name="RAW_OHLCV",
                database=SNOWFLAKE_CONFIG["database"],
                schema="BRONZE",
                auto_create_table=False,
                overwrite=False,         # APPEND — do not replace historical data
                quote_identifiers=False,
            )
            log.info("Appended %s rows to BRONZE.RAW_OHLCV", f"{len(ohlcv_df):,}")
        else:
            log.info("No new OHLCV data to load.")

        # ── Fetch and load events ─────────────────────────────────
        events_df = fetch_incremental_events(start=last_date, end=today)
        if not events_df.empty:
            session.write_pandas(
                events_df,
                table_name="RAW_EVENTS",
                database=SNOWFLAKE_CONFIG["database"],
                schema="BRONZE",
                auto_create_table=False,
                overwrite=False,
                quote_identifiers=False,
            )
            log.info("Appended %s rows to BRONZE.RAW_EVENTS", len(events_df))

        # ── Refresh Silver → Gold ─────────────────────────────────
        refresh_silver_gold(session)

        log_summary(session, today)
        session.close()
        return 0

    except Exception as e:
        log.error("Ingest failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())