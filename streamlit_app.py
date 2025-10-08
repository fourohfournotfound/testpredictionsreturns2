# streamlit_app.py
# Streamlit dashboard: rank-by-prediction + intraday open→now returns using EODHD
# ---------------------------------------------------------------------------------------------
# - Robust to MultiIndex (ticker,date) inputs
# - Enforces session alignment and avoids look-ahead bias by default (uses target session's regular open)
# - Batched data fetch via EODHD Live v2 (us-quote-delayed) + optional real-time WebSockets override
# - Auto-refresh on a timer; no background infra required
#
# Expected predictions CSV (flexible):
#   Required:   ticker (or symbol), prediction (or pred/pred_score/score)
#   Optional:   date (target session, e.g., 2025-10-08), as_of (ISO timestamp for your model snapshot)
#
# Usage:
#   1) Put EODHD token in Streamlit Secrets or env as EODHD_API_TOKEN.
#   2) streamlit run streamlit_app.py
#
# Notes:
# - "Open" = regular trading session open (09:30 America/New_York) for the selected date. We use EODHD Live v2 'open'.
# - Default feed is 15–20 min delayed (US) via Live v2 endpoint: https://eodhd.com/api/us-quote-delayed . :contentReference[oaicite:2]{index=2}
# - Optional WebSockets override replaces the latest price with real-time trades if your plan allows. :contentReference[oaicite:3]{index=3}
#
# Safety:
# - Read-only market data for monitoring predictions/returns (no trading).
# ---------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import io
import json
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Tuple, Optional

# --- Streamlit must be imported before we render any UI ---
import streamlit as st
st.set_page_config(page_title="Live LTR Monitor (EODHD)", layout="wide")

# --- Harden dependency import to avoid numpy↔pandas ABI crash on Streamlit Cloud ---
def _safe_import_pd_np():
    try:
        import numpy as _np
        import pandas as _pd
        return _np, _pd
    except Exception as e:
        st.error(
            "🚨 **Dependency mismatch while importing `pandas`/`numpy`.**\n\n"
            "This usually means the installed wheels are ABI-incompatible "
            "(e.g., `pandas` compiled against a different `numpy`).\n\n"
            "Fix: pin compatible versions in your **requirements.txt**, for example:\n"
            "```\n"
            "numpy==2.0.2\npandas==2.2.2\n```\n"
            "Then redeploy. (Pandas ≥2.2.2 is the first release compatible with NumPy 2.x.)",
            icon="🚨",
        )
        # Show the underlying exception for local runs
        st.exception(e)
        st.stop()

np, pd = _safe_import_pd_np()

import pytz
from dateutil import parser as dateparser
import requests
from streamlit_autorefresh import st_autorefresh

# Optional real-time WS via eodhdc. If unavailable or plan not entitled, we fall back to delayed HTTP.
try:
    from eodhdc import EODHDWebSockets  # async client
    _HAS_EODHDC = True
except Exception:
    _HAS_EODHDC = False

# -----------------------
# Constants / Config
# -----------------------
NY = pytz.timezone("America/New_York")

# Live v2 (US delayed extended quotes) supports batching and returns open + last trade. :contentReference[oaicite:4]{index=4}
BASE_HOST = "https://eodhd.com"
LIVE_V2_URL = f"{BASE_HOST}/api/us-quote-delayed"
LIVE_V1_URL = f"{BASE_HOST}/api/real-time"  # not used by default; kept for fallback/reference. :contentReference[oaicite:5]{index=5}

BATCH_SIZE_HTTP = 100   # Live v2: up to 100 symbols per page (we keep chunks <=100). :contentReference[oaicite:6]{index=6}
WS_COLLECT_SECONDS = 2  # short window to grab real-time prints if WS enabled

# Common default CSV locations
DEFAULT_PREDICTIONS_PATHS = [
    os.environ.get("PREDICTIONS_CSV", "").strip() or "",
    "live_predictions.csv",
    "/mnt/data/live_predictions.csv",
]

# -----------------------
# Secrets / env & helpers
# -----------------------
def _get_secret(name, default=None, table=None):
    try:
        return (st.secrets[table][name] if table else st.secrets[name])
    except Exception:
        return os.environ.get(name, default)

API_TOKEN = _get_secret("EODHD_API_TOKEN") or _get_secret("API_TOKEN", table="eodhd")

def _ensure_api_token() -> None:
    if not API_TOKEN:
        st.error(
            "Missing EODHD credentials: **EODHD_API_TOKEN**.\n"
            "Add it in **Streamlit Secrets** or environment variables, then rerun."
        )
        st.stop()

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

    ws_enabled = st.checkbox(
        "Use real‑time WebSocket for latest price (experimental)",
        value=False,
        help=(
            "Requires an EODHD plan with WebSockets entitlement and the `eodhdc` package. "
            "If unavailable, the app falls back to delayed HTTP (15–20 min)."
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
    session_date = st.date_input("Target trading session (ET)", value=today_ny)

    st.write("**Symbol format**")
    exchange_suffix = st.text_input(
        "Default exchange suffix (appends if missing)",
        value="US",
        help="EODHD uses exchange-suffixed symbols (e.g., AAPL.US). "
             "If your CSV already includes suffixes, we keep them as‑is."
    )

    st.write("**Ranking**")
    top_n = st.number_input("Show top N by prediction", min_value=1, max_value=5000, value=200, step=1)
    require_asof_guard = st.checkbox(
        "Enforce as_of ≤ session open (avoid look-ahead)", value=True,
        help="If an 'as_of' timestamp is present and is after the session open, "
             "that row will be dropped to avoid mixing intraday predictions with open→now returns."
    )
    allow_na_returns = st.checkbox("Include symbols with missing data (NA returns)", value=False)

# -----------------------
# Data loaders / normalizers
# -----------------------
def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    # Reset MultiIndex if needed
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Flexible column naming
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
    Robust default to 09:30–16:00 ET; optionally could query EODHD Exchanges API for holidays.
    If the holiday check fails, still return regular hours so the UI remains usable.
    """
    # Minimal holiday awareness: you can extend this to call the Exchanges API if desired. :contentReference[oaicite:7]{index=7}
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
# Helpers
# -----------------------
def _chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# -----------------------
# EODHD Live v2 (delayed) HTTP fetchers
# -----------------------
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
        url = f"{LIVE_V2_URL}?s={','.join(chunk_syms)}&api_token={API_TOKEN}&fmt=json"
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
# Optional: EODHD WebSocket latest prices (real-time override)
# -----------------------
@st.cache_data(show_spinner=False, ttl=2)
def _fetch_latest_prices_ws(symbols_eod: List[str]) -> pd.Series:
    """
    Uses eodhdc WebSockets 'us' stream to collect latest trade prices for given tickers.
    Symbols are subscribed WITHOUT the '.US' suffix. If anything fails, returns empty Series.
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
            async with ws.connect("us") as websocket:
                await ws.subscribe(websocket, tickers)
                start = datetime.now()
                async for msg in ws.receive(websocket):
                    # Normalize message to dict
                    if isinstance(msg, str):
                        try:
                            msg = json.loads(msg)
                        except Exception:
                            continue
                    sym = (msg.get("s") or msg.get("code") or "").upper()
                    px = msg.get("p") or msg.get("price") or msg.get("lastTradePrice")
                    if sym and isinstance(px, (int, float)):
                        prices[sym] = float(px)
                    if (datetime.now() - start).total_seconds() >= WS_COLLECT_SECONDS:
                        ws.deactivate()
                # map back to EOD symbols
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
    st.warning(f"Selected date {session_date} is **not** a US trading session. Pick a valid session.")
    st.stop()
if session_source != "regular":
    st.caption("⚠️ Using standard 09:30–16:00 ET session bounds.")

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
# Data fetch: delayed HTTP + optional realtime WS override
# -----------------------
quotes = _fetch_live_v2_quotes(symbols_eod)  # index: EOD symbol; cols: open, lastTradePrice, timestamp

# Optional WebSocket override for last price
if ws_enabled:
    if not _HAS_EODHDC:
        st.info("WebSocket client `eodhdc` not installed; using delayed HTTP. "
                "Add `eodhdc` to your environment to enable WS override.")
    else:
        ws_px = _fetch_latest_prices_ws(symbols_eod)
        if not ws_px.empty:
            quotes.loc[ws_px.index, "lastTradePrice"] = ws_px

now_et = datetime.now(tz=NY)

# -----------------------
# Merge with predictions & compute returns
# -----------------------
# Build mapping from original ticker -> EOD symbol to join
map_df = pd.DataFrame({"ticker": symbols_in, "eod_symbol": symbols_eod}).drop_duplicates()

df_view = preds_ranked.merge(map_df, on="ticker", how="left")
df_view = df_view.merge(quotes[["open", "lastTradePrice"]], left_on="eod_symbol", right_index=True, how="left")

# Compute intraday return from session open → now.
# Guard: if before regular open, suppress returns to avoid using prior-day open.
if now_et < session_open:
    df_view["return_open_to_now"] = np.nan
else:
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
