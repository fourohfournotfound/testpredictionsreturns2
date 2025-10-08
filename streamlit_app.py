# app.py
# Streamlit dashboard: rank-by-prediction + (delayed) intraday return from open→now using EODHD
# ---------------------------------------------------------------------------------------------
# - Robust to MultiIndex (ticker,date) inputs
# - Enforces session alignment and avoids look-ahead bias by default (uses target session's regular open)
# - Fast batched data fetch via EODHD Live v2 (us-quote-delayed) + optional real-time WebSockets override
# - Auto-refresh on a timer; no background infra required
#
# Expected predictions CSV (flexible):
#   Required:   ticker (or symbol), prediction (or pred/pred_score/score)
#   Optional:   date (target session, e.g., 2025-10-08), as_of (ISO timestamp for your model snapshot)
#
# If your file is MultiIndex with ('ticker','date'), that works too — we normalize it.
#
# Usage:
#   1) Put EODHD API token in Streamlit Secrets or env as EODHD_API_TOKEN.
#   2) streamlit run app.py
#
# Notes:
# - "Open" means *regular trading session open* (09:30 America/New_York); we use EODHD Live v2 'open' field.
# - Default feed is 15–20 min delayed per EODHD docs (Live v2: us-quote-delayed).
# - Optional real-time (WebSocket) override for latest prices (plan-dependent). Falls back gracefully.
#
# Safety:
# - We never place orders; this is read-only market data for monitoring predictions/returns.
# ---------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import io
import json
import math
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
from dateutil import parser as dateparser

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Optional: real-time WS via eodhdc (community client). If missing or plan not entitled -> we fallback.
try:
    from eodhdc import EODHDWebSockets  # type: ignore
    _HAS_EODHDC = True
except Exception:
    _HAS_EODHDC = False

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Live LTR Monitor (EODHD)", layout="wide")

NY = pytz.timezone("America/New_York")

# API & batching params tuned for Live v2 delayed endpoint.
BATCH_SIZE_HTTP = 100   # Live v2 supports 's=' list; docs show pagination up to 100/page.
WS_COLLECT_SECONDS = 2  # short collection window to grab last trade updates when WS enabled
OPEN_LOOKAHEAD_MIN = 5  # only relevant if we ever fall back to intraday bars (not used by default)

# -----------------------
# Secrets / env & helpers
# -----------------------
def _get_secret(name, default=None, table=None):
    try:
        return (st.secrets[table][name] if table else st.secrets[name])
    except Exception:
        return os.environ.get(name, default)

API_TOKEN = _get_secret("EODHD_API_TOKEN") or _get_secret("API_TOKEN", table="eodhd")
BASE_HOST = "https://eodhd.com"
LIVE_V2_URL = f"{BASE_HOST}/api/us-quote-delayed"   # Live v2 (US delayed, extended quotes)
LIVE_V1_URL = f"{BASE_HOST}/api/real-time"          # Live v1 (OHLCV snapshot, multi-asset) [fallback capable]

# Common default CSV locations
DEFAULT_PREDICTIONS_PATHS = [
    os.environ.get("PREDICTIONS_CSV", "").strip() or "",
    "live_predictions.csv",
    "/mnt/data/live_predictions.csv",
]

def _ensure_api_token() -> None:
    if not API_TOKEN:
        st.error(
            "Missing EODHD credentials: **EODHD_API_TOKEN**. "
            "Add it in **Streamlit Secrets** or environment variables, then rerun."
        )
        st.stop()

# -----------------------
# Requests session as cached resource
# -----------------------
@st.cache_resource(show_spinner=False)
def _init_http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "ltr-monitor/1.0"})
    return s

# Instantiate resources early so downstream cached functions can reference them
_ensure_api_token()
_http = _init_http_session()

# -----------------------
# Sidebar UI
# -----------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.caption(f"🔐 EODHD token loaded: {'yes' if bool(API_TOKEN) else 'no'}")

    st.write("**Feed mode**")
    ws_enabled = st.checkbox(
        "Use real‑time WebSocket for latest price (experimental)",
        value=False,
        help=(
            "Requires an EODHD plan with WebSockets entitlement and the 'eodhdc' package. "
            "If unavailable, the app automatically falls back to delayed HTTP (15–20 min)."
        ),
    )

    st.write("**Auto refresh**")
    refresh_sec = st.slider("Refresh interval (seconds)", min_value=5, max_value=120, value=10, step=5)
    _ = st_autorefresh(interval=refresh_sec * 1000, key="autorefresh")

    st.write("**Predictions file**")
    file_src = st.radio("Load from…", ["Local path", "Upload"])
    if file_src == "Local path":
        default_guess = next((p for p in DEFAULT_PREDICTIONS_PATHS if p and os.path.exists(p)), DEFAULT_PREDICTIONS_PATHS[0] or "live_predictions.csv")
        preds_path = st.text_input("Path to CSV", value=default_guess, help="e.g., ./live_predictions.csv or /mnt/data/live_predictions.csv")
        upload_file = None
    else:
        upload_file = st.file_uploader("Upload CSV", type=["csv"])
        preds_path = None

    st.write("**Session date**")
    today_ny = datetime.now(tz=NY).date()
    session_date = st.date_input("Target trading session (ET)", value=toady_ny if (toady_ny:=today_ny) else today_ny)  # defensive alias

    st.write("**Symbol format**")
    exchange_suffix = st.text_input(
        "Default exchange suffix (appends if missing)",
        value="US",
        help="EODHD uses exchange-suffixed symbols (e.g., AAPL.US). "
             "If your CSV already includes suffixes, we keep them as-is.",
    )

    st.write("**Ranking**")
    top_n = st.number_input("Show top N by prediction", min_value=1, max_value=5000, value=200, step=1)
    require_asof_guard = st.checkbox(
        "Enforce as_of ≤ session open (avoid look-ahead)", value=True,
        help="If an 'as_of' timestamp is present and is after the session open, that row will be dropped to avoid mixing intraday predictions with open→now returns."
    )
    allow_na_returns = st.checkbox("Include symbols with missing data (NA returns)", value=False)

# -----------------------
# Data loaders / normalizers
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
# Calendar utilities
# -----------------------
def _regular_session_bounds(session: date) -> Tuple[datetime, datetime]:
    """Default: 09:30–16:00 ET for the given date."""
    return (
        NY.localize(datetime.combine(session, time(9, 30))),
        NY.localize(datetime.combine(session, time(16, 0))),
    )

@st.cache_data(show_spinner=False, ttl=3600)
def _get_session_open_close(session: date) -> Tuple[Optional[datetime], Optional[datetime], str]:
    """
    For robustness and to avoid external entitlements, we use the standard 09:30–16:00 ET window,
    and treat exchange holidays (if we detect them) as closed. If the exchange holiday check fails,
    we still return 09:30–16:00 so the UI remains usable.
    """
    try:
        # Optional: call EODHD exchange details to detect holidays. If it errors, keep regular hours.
        url = f"{BASE_HOST}/api/exchange-details/US?api_token={API_TOKEN}&fmt=json"
        r = _http.get(url, timeout=10)
        if r.ok:
            data = r.json()
            # Best-effort holiday check (shape varies by doc; avoid hard reliance).
            holidays = data.get("holidays") or data.get("StockMarketHolidays") or []
            holiday_dates = set()
            for h in holidays:
                # try common keys, fall back to parsing strings
                d = h.get("date") or h.get("Date")
                if d:
                    try:
                        holiday_dates.add(pd.to_datetime(d).date())
                    except Exception:
                        pass
            if session in holiday_dates:
                return None, None, "holiday"
    except Exception:
        pass

    open_et, close_et = _regular_session_bounds(session)
    return open_et, close_et, "regular"

# -----------------------
# Symbol normalization
# -----------------------
def _to_eod_symbol(ticker: str, suffix: str) -> str:
    ticker = (ticker or "").strip().upper()
    if "." in ticker:
        return ticker  # already suffixed
    return f"{ticker}.{suffix.strip().upper()}"

def _strip_suffix(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    return s.split(".")[0]

# -----------------------
# EODHD Live v2 (delayed) HTTP fetchers
# -----------------------
def _chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(show_spinner=False, ttl=5)
def _fetch_live_v2_quotes(symbols_eod: List[str]) -> pd.DataFrame:
    """
    Calls Live v2 (us-quote-delayed) for up to 100 symbols per request and returns a DataFrame
    indexed by EODHD symbol with columns: ['open', 'lastTradePrice', 'timestamp'].
    """
    if not symbols_eod:
        return pd.DataFrame(columns=["symbol", "open", "lastTradePrice", "timestamp"]).set_index("symbol")

    frames = []
    for chunk_syms in _chunk(symbols_eod, BATCH_SIZE_HTTP):
        s_param = ",".join(chunk_syms)
        url = f"{LIVE_V2_URL}?s={s_param}&api_token={API_TOKEN}&fmt=json"
        try:
            r = _http.get(url, timeout=10)
            if not r.ok:
                continue
            payload = r.json()
            data = payload.get("data", {})
            rows = []
            for sym, d in (data or {}).items():
                if not isinstance(d, dict):
                    continue
                rows.append({
                    "symbol": sym,
                    "open": d.get("open", np.nan),
                    "lastTradePrice": d.get("lastTradePrice", np.nan),
                    "timestamp": d.get("lastTradeTime") or d.get("timestamp")
                })
            if rows:
                frames.append(pd.DataFrame(rows))
        except Exception:
            # continue to next chunk to be robust
            continue

    if not frames:
        return pd.DataFrame(columns=["symbol", "open", "lastTradePrice", "timestamp"]).set_index("symbol")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["symbol"]).set_index("symbol")
    return out

# -----------------------
# Optional: EODHD WebSocket latest prices
# -----------------------
@st.cache_data(show_spinner=False, ttl=2)  # very short TTL; WS provides fresh prints when called
def _fetch_latest_prices_ws(symbols_eod: List[str]) -> pd.Series:
    """
    Uses eodhdc WebSockets 'us' endpoint to collect latest trade prices for given tickers.
    Symbols are subscribed WITHOUT the '.US' suffix per eodhdc examples. If anything fails,
    returns empty Series, and caller should fall back to HTTP delayed.
    """
    if not _HAS_EODHDC or not symbols_eod:
        return pd.Series(dtype=float)

    # WS expects raw tickers (e.g., AAPL, TSLA)
    tickers = sorted({_strip_suffix(s) for s in symbols_eod})

    try:
        import asyncio
        prices: Dict[str, float] = {}

        async def _collect():
            ws = EODHDWebSockets(key=API_TOKEN, buffer=1000)
            # 'us' stream for trades; could also try 'us-quote' for quotes
            async with ws.connect("us") as websocket:
                await ws.subscribe(websocket, tickers)
                start = datetime.now()
                async for msg in ws.receive(websocket):
                    # msg can be dict or string; handle both
                    if isinstance(msg, str):
                        try:
                            msg = json.loads(msg)
                        except Exception:
                            continue
                    # Try common fields across EODHD streams:
                    # symbol under 's' or 'code', price under 'p' or 'price'
                    sym = (msg.get("s") or msg.get("code") or "").upper()
                    px = msg.get("p") or msg.get("price") or msg.get("lastTradePrice")
                    if sym and isinstance(px, (int, float)):
                        prices[sym] = float(px)
                    # Exit after WS_COLLECT_SECONDS
                    if (datetime.now() - start).total_seconds() >= WS_COLLECT_SECONDS:
                        ws.deactivate()  # stop receive loop
                # after exit, prices dict holds most recent prints observed
            # Map back to EODHD '.US' symbols where possible
            mapped = {}
            for s in symbols_eod:
                base = _strip_suffix(s)
                if base in prices:
                    mapped[s] = prices[base]
            return pd.Series(mapped, dtype=float)

        return asyncio.run(_collect())
    except Exception:
        return pd.Series(dtype=float)

# -----------------------
# Load predictions
# -----------------------
if 'upload_file' in locals() and upload_file is not None:
    preds = _load_predictions_from_upload(upload_file)
else:
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
session_open, session_close, session_source = _get_session_open_close(session_date)
if session_open is None:
    st.warning(f"Selected date {session_date} is **not** a US trading session (holiday). Pick a valid session.")
    st.stop()
if session_source != "regular":
    st.caption("⚠️ Using standard 09:30–16:00 ET session bounds (holidays excluded).")

# Enforce as_of guard now that we know session_open
preds_ranked = _post_asof_guard(preds_ranked, session_open, require_asof_guard=require_asof_guard)

if preds_ranked.empty:
    st.warning("No rows remain after enforcing the as_of guard. "
               "Uncheck it in the sidebar if your predictions were created intraday and you still want open→now returns.")
    st.stop()

# Normalize symbols for EODHD
symbols_in = preds_ranked["ticker"].dropna().astype(str).str.upper().tolist()
symbols_eod = [_to_eod_symbol(t, exchange_suffix) for t in symbols_in]

# -----------------------
# Data fetch (Live v2 delayed) + optional realtime WS override
# -----------------------
quotes = _fetch_live_v2_quotes(symbols_eod)  # index: EOD symbol; cols: open, lastTradePrice, timestamp

# Optional WebSocket override for last price (graceful fallback)
if ws_enabled:
    if not _HAS_EODHDC:
        st.info("WebSocket client 'eodhdc' not installed; using delayed HTTP. "
                "Add `eodhdc` to your environment to enable WS override.")
    else:
        ws_px = _fetch_latest_prices_ws(symbols_eod)
        if not ws_px.empty:
            # Align indices and overwrite lastTradePrice with real-time where available
            quotes.loc[ws_px.index, "lastTradePrice"] = ws_px

now_et = datetime.now(tz=NY)
if now_et < session_open:
    st.info("Regular session hasn’t opened yet (09:30 ET). Open→now returns will populate after the open.")

# -----------------------
# Merge with predictions & compute returns
# -----------------------
# Build mapping from original ticker -> EOD symbol to join
map_df = pd.DataFrame({"ticker": symbols_in, "eod_symbol": symbols_eod}).drop_duplicates()

df_view = preds_ranked.merge(map_df, on="ticker", how="left")
df_view = df_view.merge(quotes[["open", "lastTradePrice"]], left_on="eod_symbol", right_index=True, how="left")

# Compute intraday return from session open → now (delayed or realtime depending on source)
df_view["return_open_to_now"] = (df_view["lastTradePrice"] / df_view["open"] - 1.0).replace([np.inf, -np.inf], np.nan)

# Optionally drop rows with NA returns
if not allow_na_returns:
    before = df_view.shape
    df_view = df_view.dropna(subset=["open", "lastTradePrice", "return_open_to_now"])
    st.caption(f"📐 After dropping NA returns: {before} → {df_view.shape}")

# Final sorting by your prediction score
df_view = df_view.sort_values("prediction", ascending=False).reset_index(drop=True)

# Pretty formatting
def _fmt_pct(x):
    return "" if pd.isna(x) else f"{x*100:.2f}%"
def _fmt_price(x):
    return "" if pd.isna(x) else f"{x:.2f}"

display_cols = ["rank", "ticker", "prediction", "open", "lastTradePrice", "return_open_to_now"]
show = df_view[display_cols].copy()
show.rename(columns={"open": "open_today", "lastTradePrice": "last_price"}, inplace=True)
show["open_today"] = show["open_today"].apply(_fmt_price)
show["last_price"] = show["last_price"].apply(_fmt_price)
show["return_open_to_now"] = show["return_open_to_now"].apply(_fmt_pct)

# -----------------------
# Main layout
# -----------------------
st.title("📈 Live LTR Monitor — Predictions vs Open→Now Returns (EODHD)")
st.caption(
    "Sorted by your prediction score. Returns are computed from the **regular session open (09:30 ET)** "
    "for the selected session to the latest price. Default data is **15–20 min delayed**; "
    "enable the WebSocket option for real-time last trades if your plan allows."
)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Top predictions (live or delayed returns)")
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
        f"quotes_livev2: {quotes.shape}\n"
        f"joined_view:   {df_view.shape}\n"
        f"feed_mode:     {'WS realtime override' if ws_enabled and _HAS_EODHDC else 'HTTP delayed (Live v2)'}",
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
    "The app will compute returns after the next open."
)
