# app.py
# Streamlit dashboard: rank-by-prediction + live intraday returns from open→now (Alpaca)
# ---------------------------------------------------
# - Robust to MultiIndex (ticker,date) inputs
# - Enforces session alignment and avoids look-ahead bias by default (uses target session's regular open)
# - Fast batched data fetch via alpaca-py
# - Auto-refresh on a timer; no background infra required
#
# Expected predictions CSV (flexible):
#   Required:   ticker (or symbol), prediction (or pred/pred_score/score)
#   Optional:   date (target session, e.g., 2025-10-08), as_of (ISO timestamp for your model snapshot)
#
# If your file is MultiIndex with ('ticker','date'), that works too — we normalize it.
#
# Usage:
#   1) Put Alpaca keys in Streamlit Secrets (TOML) on Community Cloud, or env vars locally.
#   2) streamlit run app.py
#
# Notes:
# - "Open" means *regular trading session open* (09:30 America/New_York) per the market calendar.
# - Feed selection is honored for both historical bars and latest trades.
# - If you only have the free plan, recent historical data may be delayed (15 min). Use IEX (realtime from a single venue)
#   or upgrade for SIP if you need consolidated realtime. The app exposes this in the sidebar.
#
# Safety:
# - We never place orders; this is read-only market data for monitoring predictions/returns.
# ---------------------------------------------------

from __future__ import annotations

import os
import io
from datetime import datetime, date
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pytz
from dateutil import parser as dateparser

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Alpaca
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Live LTR Monitor (Alpaca)", layout="wide")

NY = pytz.timezone("America/New_York")
BATCH_SIZE = 200  # conservative batch size for any API request loops

# -----------------------
# Secrets / env & helpers
# -----------------------
def _get_secret(name, default=None, table=None):
    try:
        return (st.secrets[table][name] if table else st.secrets[name])
    except Exception:
        return os.environ.get(name, default)

API_KEY    = _get_secret("APCA_API_KEY_ID") or _get_secret("APCA_API_KEY_ID", table="alpaca")
API_SECRET = _get_secret("APCA_API_SECRET_KEY") or _get_secret("APCA_API_SECRET_KEY", table="alpaca")
DATA_FEED_ENV = (_get_secret("APCA_DATA_FEED") or _get_secret("APCA_DATA_FEED", table="alpaca") or "IEX").upper()
BASE_URL   = _get_secret("APCA_API_BASE_URL") or _get_secret("APCA_API_BASE_URL", table="alpaca")

# (optional) mirror to env in case libs auto-read environment variables
if API_KEY:    os.environ["APCA_API_KEY_ID"] = API_KEY
if API_SECRET: os.environ["APCA_API_SECRET_KEY"] = API_SECRET
if BASE_URL:   os.environ["APCA_API_BASE_URL"] = BASE_URL

# Common default CSV locations (Cloud upload mounts sometimes use /mnt/data)
DEFAULT_PREDICTIONS_PATHS = [
    os.environ.get("PREDICTIONS_CSV", "").strip() or "",
    "live_predictions.csv",
    "/mnt/data/live_predictions.csv",
]

FEED_MAP = {"IEX": DataFeed.IEX, "SIP": DataFeed.SIP, "DELAYED_SIP": DataFeed.DELAYED_SIP}
DEFAULT_FEED = FEED_MAP.get(DATA_FEED_ENV, DataFeed.IEX)

def _ensure_api_keys() -> None:
    missing = []
    if not API_KEY:
        missing.append("APCA_API_KEY_ID")
    if not API_SECRET:
        missing.append("APCA_API_SECRET_KEY")
    if missing:
        st.error(
            "Missing Alpaca credentials: "
            + ", ".join(missing)
            + ". Add them in **Streamlit Secrets** (TOML) or environment variables, then rerun."
        )
        st.stop()

# -----------------------
# Debug panel (optional)
# -----------------------
with st.sidebar.expander("🔍 Secrets debug"):
    try:
        st.write("Top-level keys:", list(st.secrets.keys()))
        st.write("Has APCA_API_KEY_ID?", "APCA_API_KEY_ID" in st.secrets)
        st.write("Has [alpaca] table?", "alpaca" in st.secrets)
        if "alpaca" in st.secrets:
            st.write("Keys in [alpaca]:", list(st.secrets["alpaca"].keys()))
    except Exception as e:
        st.write("st.secrets unavailable:", repr(e))
        st.write("env APCA_API_KEY_ID present?", bool(os.environ.get("APCA_API_KEY_ID")))

# -----------------------
# Clients as cached resources (best practice)
# -----------------------
@st.cache_resource(show_spinner=False)
def _init_clients(api_key: str, api_secret: str, base_url: str | None) -> Tuple[StockHistoricalDataClient, TradingClient]:
    """Initialize Alpaca clients once per session/process.

    cache_resource is appropriate for SDK clients (unhashable, stateful).
    """
    data_client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    is_paper = bool(base_url) and ("paper-api" in base_url)
    trading_client = TradingClient(api_key, api_secret, paper=is_paper)
    return data_client, trading_client

# Instantiate resources early so downstream cached functions can reference them
_ensure_api_keys()
data_client, trading_client = _init_clients(API_KEY, API_SECRET, BASE_URL)

# -----------------------
# Sidebar UI
# -----------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.caption(f"🔐 Alpaca key loaded: {'yes' if bool(API_KEY) else 'no'} | feed default={DATA_FEED_ENV}")

    st.write("**Alpaca Data Feed**")
    chosen_feed = st.selectbox(
        "Feed (IEX=single-venue realtime; SIP=consolidated realtime; DELAYED_SIP=15-min delay)",
        options=["IEX", "SIP", "DELAYED_SIP"],
        index=["IEX", "SIP", "DELAYED_SIP"].index(DATA_FEED_ENV if DATA_FEED_ENV in ["IEX","SIP","DELAYED_SIP"] else "IEX"),
        help="Pick SIP if your subscription includes it; DELAYED_SIP is 15-min delayed consolidated tape."
    )
    DATA_FEED = FEED_MAP[chosen_feed]

    st.write("**Auto refresh**")
    refresh_sec = st.slider("Refresh interval (seconds)", min_value=5, max_value=120, value=10, step=5)
    _ = st_autorefresh(interval=refresh_sec * 1000, key="autorefresh")

    st.write("**Predictions file**")
    file_src = st.radio("Load from…", ["Local path", "Upload"])
    if file_src == "Local path":
        # choose first existing default as convenience
        default_guess = next((p for p in DEFAULT_PREDICTIONS_PATHS if p and os.path.exists(p)), DEFAULT_PREDICTIONS_PATHS[0] or "live_predictions.csv")
        preds_path = st.text_input("Path to CSV", value=default_guess, help="e.g., ./live_predictions.csv or /mnt/data/live_predictions.csv")
        upload_file = None
    else:
        upload_file = st.file_uploader("Upload CSV", type=["csv"])
        preds_path = None

    st.write("**Session date**")
    today_ny = datetime.now(tz=NY).date()
    session_date = st.date_input("Target trading session (ET)", value=today_ny)

    st.write("**Ranking**")
    top_n = st.number_input("Show top N by prediction", min_value=1, max_value=5000, value=200, step=1)
    require_asof_guard = st.checkbox(
        "Enforce as_of ≤ session open (avoid look-ahead)", value=True,
        help="If an 'as_of' timestamp is present and is after the session open, that row will be dropped to avoid mixing intraday predictions with open→now returns."
    )
    allow_na_returns = st.checkbox("Include symbols with missing data (NA returns)", value=False)

# -----------------------
# Data loaders / normalizers (cache_data with hashable args only)
# -----------------------
def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    # Reset MultiIndex if needed
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Rename flexible column names
    cols = {c.lower().strip(): c for c in df.columns}
    ticker_col = next((cols[k] for k in cols if k in ["ticker", "symbol"]), None)
    pred_col   = next((cols[k] for k in cols if k in ["prediction", "pred", "pred_score", "score", "yhat"]), None)
    date_col   = next((cols[k] for k in cols if k in ["date", "session_date", "target_date"]), None)
    asof_col   = next((cols[k] for k in cols if k in ["as_of", "asof", "timestamp", "prediction_time"]), None)

    required_missing = []
    if ticker_col is None:
        required_missing.append("ticker")
    if pred_col is None:
        required_missing.append("prediction")
    if required_missing:
        st.error(
            "Your file is missing required column(s): " + ", ".join(required_missing) +
            ". Expected at least: ticker/symbol and prediction/pred."
        )
        st.stop()

    out = df.rename(columns={ticker_col: "ticker", pred_col: "prediction"})
    if date_col:
        out = out.rename(columns={date_col: "date"})
    if asof_col:
        out = out.rename(columns={asof_col: "as_of"})

    # Clean types
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date

    if "as_of" in out.columns:
        # Store 'as_of' as ET tz-aware
        def _parse_asof(x):
            if pd.isna(x):
                return pd.NaT
            try:
                dt = dateparser.parse(str(x))
                if dt.tzinfo is None:
                    dt = NY.localize(dt)
                else:
                    dt = dt.astimezone(NY)
                return dt
            except Exception:
                return pd.NaT
        out["as_of"] = out["as_of"].apply(_parse_asof)

    # De-duplicate: keep latest as_of per ticker if multiple rows
    if "as_of" in out.columns:
        out = out.sort_values(["ticker", "as_of"]).groupby("ticker", as_index=False).tail(1)

    st.caption(f"📐 Predictions shape after normalization: {out.shape}")
    return out

@st.cache_data(show_spinner=False)
def _load_predictions_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_predictions(df)

def _load_predictions_from_upload(upload: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(upload)
    return _normalize_predictions(df)

def _align_and_filter_for_session(df: pd.DataFrame, session: date) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        before_shape = out.shape
        out = out[out["date"] == session]
        st.caption(f"📐 After filtering to session={session} (by 'date'): {before_shape} → {out.shape}")
    return out

def _post_asof_guard(df: pd.DataFrame, session_open: datetime, require_asof_guard: bool) -> pd.DataFrame:
    out = df.copy()
    if require_asof_guard and "as_of" in out.columns:
        mask_ok = df["as_of"].isna() | (df["as_of"] <= session_open)
        before = out.shape
        out = out[mask_ok]
        st.caption(f"📐 After enforcing as_of ≤ session_open (no look-ahead): {before} → {out.shape}")
    return out

# -----------------------
# Market calendar (cache_data keyed only by primitives)
# -----------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _get_session_open_close(session: date) -> tuple[datetime | None, datetime | None]:
    """Fetch the regular session open/close for the given date (ET)."""
    req = GetCalendarRequest(start=session, end=session)
    cal = trading_client.get_calendar(req)
    if not cal:
        return None, None
    c = cal[0]
    # Calendar.open/close are datetimes; ensure ET
    session_open = (c.open if c.open.tzinfo else NY.localize(c.open)).astimezone(NY)
    session_close = (c.close if c.close.tzinfo else NY.localize(c.close)).astimezone(NY)
    return session_open, session_close

# -----------------------
# Data fetchers (cache_data with TTL; never pass client objects)
# -----------------------
@st.cache_data(show_spinner=False, ttl=300)
def _fetch_open_prices(
    symbols: List[str],
    session_open: datetime,
    session_close: datetime,
    feed_name: str
) -> pd.Series:
    """Return a Series indexed by symbol with the first minute bar 'open' at/after session_open."""
    if not symbols:
        return pd.Series(dtype=float)
    feed = FEED_MAP[feed_name]

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=session_open,  # tz-aware ET; SDK treats tz-naive as UTC
        end=min(datetime.now(tz=NY), session_close),
        feed=feed,
        adjustment=Adjustment.SPLIT,
    )

    try:
        bars = data_client.get_stock_bars(req)
        df = bars.df  # MultiIndex: (symbol, timestamp)
    except Exception as e:
        st.error(f"Failed to fetch open prices: {e}")
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    first_open = (
        df.sort_index(level=["symbol", "timestamp"])
          .groupby("symbol")["open"]
          .first()
    )
    first_open.name = "open_today"
    return first_open

@st.cache_data(show_spinner=False, ttl=5)
def _fetch_latest_prices(
    symbols: List[str],
    feed_name: str
) -> pd.Series:
    """Return a Series indexed by symbol with the latest trade price."""
    if not symbols:
        return pd.Series(dtype=float)
    feed = FEED_MAP[feed_name]

    prices: Dict[str, float] = {}
    try:
        for i in range(0, len(symbols), BATCH_SIZE):
            chunk = symbols[i : i + BATCH_SIZE]
            req = StockLatestTradeRequest(symbol_or_symbols=chunk, feed=feed)
            latest = data_client.get_stock_latest_trade(req)  # dict: symbol -> Trade
            for sym, trade in (latest or {}).items():
                try:
                    # Trade model exposes a 'price' attribute
                    prices[sym] = float(trade.price)
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Failed to fetch latest prices: {e}")
        return pd.Series(dtype=float)

    s = pd.Series(prices, dtype=float)
    s.name = "last_price"
    return s

# -----------------------
# Load predictions
# -----------------------
if 'upload_file' in locals() and upload_file is not None:
    preds = _load_predictions_from_upload(upload_file)
else:
    # Try a sensible default path if user didn't upload
    chosen_default = preds_path
    if not chosen_default:
        chosen_default = next((p for p in DEFAULT_PREDICTIONS_PATHS if p and os.path.exists(p)), "live_predictions.csv")
    if chosen_default and os.path.exists(chosen_default):
        preds = _load_predictions_from_path(chosen_default)
    else:
        preds = pd.DataFrame(columns=["ticker", "prediction"])

if preds.empty:
    st.info("Upload or point to your predictions CSV to get started. Expect columns like: "
            "`ticker`, `prediction`, optional `date`, optional `as_of`.")
    st.stop()

# Filter to target session
preds = _align_and_filter_for_session(preds, session_date)

# Rank by prediction (descending)
preds_ranked = preds.sort_values("prediction", ascending=False).reset_index(drop=True)
preds_ranked["rank"] = np.arange(1, len(preds_ranked) + 1)

# Limit to top N requested by the user
preds_ranked = preds_ranked.head(int(top_n))
st.caption(f"📐 Ranked predictions shape (after top-N): {preds_ranked.shape}")

# -----------------------
# Market times
# -----------------------
session_open, session_close = _get_session_open_close(session_date)
if session_open is None:
    st.warning(f"Selected date {session_date} is **not** a US trading session. Pick a valid session.")
    st.stop()

# Enforce as_of guard now that we know session_open
preds_ranked = _post_asof_guard(preds_ranked, session_open, require_asof_guard=require_asof_guard)

if preds_ranked.empty:
    st.warning("No rows remain after enforcing the as_of guard. "
               "Uncheck it in the sidebar if your predictions were created intraday and you still want open→now returns.")
    st.stop()

symbols = preds_ranked["ticker"].dropna().astype(str).str.upper().unique().tolist()

# -----------------------
# Data fetch: open price and latest price
# -----------------------
open_px = _fetch_open_prices(symbols, session_open, session_close, feed_name=chosen_feed)
last_px = _fetch_latest_prices(symbols, feed_name=chosen_feed)

# Merge with predictions
df_view = preds_ranked.merge(open_px.rename("open_today"), left_on="ticker", right_index=True, how="left")
df_view = df_view.merge(last_px.rename("last_price"), left_on="ticker", right_index=True, how="left")

# Compute intraday return from session open → now
df_view["return_open_to_now"] = (df_view["last_price"] / df_view["open_today"] - 1.0).replace([np.inf, -np.inf], np.nan)

# Optionally drop rows with NA returns
if not allow_na_returns:
    before = df_view.shape
    df_view = df_view.dropna(subset=["open_today", "last_price", "return_open_to_now"])
    st.caption(f"📐 After dropping NA returns: {before} → {df_view.shape}")

# Final sorting by your prediction score
df_view = df_view.sort_values("prediction", ascending=False).reset_index(drop=True)

# Pretty formatting
def _fmt_pct(x):
    return "" if pd.isna(x) else f"{x*100:.2f}%"
def _fmt_price(x):
    return "" if pd.isna(x) else f"{x:.2f}"

display_cols = ["rank", "ticker", "prediction", "open_today", "last_price", "return_open_to_now"]
show = df_view[display_cols].copy()
show["open_today"] = show["open_today"].apply(_fmt_price)
show["last_price"] = show["last_price"].apply(_fmt_price)
show["return_open_to_now"] = show["return_open_to_now"].apply(_fmt_pct)

# -----------------------
# Main layout
# -----------------------
st.title("📈 Live LTR Monitor — Predictions vs Open→Now Returns")
st.caption(
    "Sorted by your prediction score. Returns are computed from the **regular session open (09:30 ET)** for the selected session to the latest trade. "
    "Use the sidebar to pick the session date, feed, and refresh interval."
)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Top predictions (live returns)")
    st.dataframe(show, use_container_width=True, height=650)

with right:
    st.subheader("Summary")
    st.metric("Session date (ET)", str(session_date))
    st.metric("Symbols shown", f"{show.shape[0]:,}")
    if df_view["return_open_to_now"].notna().any():
        realized_top_mean = df_view["return_open_to_now"].head(min(20, len(df_view))).mean()
        st.metric("Mean return of Top-20", f"{realized_top_mean*100:.2f}%")
    st.write(" ")
    st.write("**Dimensions audit**")
    st.code(
        f"predictions_raw: {preds.shape}\n"
        f"ranked_topN:   {preds_ranked.shape}\n"
        f"open_prices:   {open_px.shape}\n"
        f"latest_prices: {last_px.shape}\n"
        f"joined_view:   {df_view.shape}",
        language="text",
    )

# Download current table as CSV
csv_buf = io.StringIO()
df_view.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Download full table (CSV)",
    data=csv_buf.getvalue(),
    file_name=f"live_ltr_{session_date}.csv",
    mime="text/csv",
)

st.caption(
    "Tip: If your predictions are for **tomorrow’s** session, set the session date accordingly. "
    "The app will fetch that day’s open once it happens and compute returns thereafter."
)
