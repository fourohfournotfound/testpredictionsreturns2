# streamlit_app.py
# Live LTR Monitor — EODHD (opens via Live v2, rotating real-time WS for last prices)
# -----------------------------------------------------------------------------------
# - MultiIndex-safe loader (('ticker','date') accepted)
# - Shape-first logging at each stage
# - Strict as_of ≤ session_open guard to avoid look-ahead
# - HTTP: one-time batched fetch from Live v2 (us-quote-delayed) to get today's OPEN for all symbols
# - WS: rotate subscriptions in chunks (default 50) to update last prices without burning API calls
# - Auto-refresh UI; HTTP snapshot refresh cadence is configurable (default = never, to save quota)
#
# Docs used:
# - Live v2 (US extended quotes): endpoint /api/us-quote-delayed, s= batching, fields (open, lastTradePrice), 1 call / ticker, page[limit]≤100.  [EODHD "Live v2 for US Stocks: Extended Quotes (2025)"]
# - WebSockets: wss://ws.eodhistoricaldata.com/ws/us?api_token=..., subscribe/unsubscribe JSON; ~50 symbols/connection by default; WS does NOT consume API calls.  [EODHD "Real-Time Data API via WebSockets"]
# - API limits: daily calls counted per symbol, 100k/day default.  [EODHD "API Limits"]
# -----------------------------------------------------------------------------------

from __future__ import annotations

import os
import io
import json
import time
from datetime import datetime, date, time as dt_time, timedelta
from typing import List, Dict, Tuple, Optional

import streamlit as st
st.set_page_config(page_title="Live LTR Monitor (EODHD, rotating WS)", layout="wide")

# Friendly import guard for NumPy/Pandas ABI issues on hosted builds.
def _safe_import_pd_np():
    try:
        import numpy as _np
        import pandas as _pd
        return _np, _pd
    except Exception as e:
        st.error(
            "🚨 Could not import NumPy/Pandas (wheel/ABI mismatch). "
            "Use recent wheels (e.g., numpy>=2,<3 and pandas>=2.2,<3), update requirements, redeploy."
        )
        st.exception(e)
        st.stop()

np, pd = _safe_import_pd_np()

import pytz
from dateutil import parser as dateparser
import requests
import altair as alt

try:
    import websocket  # websocket-client
    _HAS_WS = True
except Exception:
    _HAS_WS = False

from streamlit_autorefresh import st_autorefresh

# -----------------------
# Constants / Config
# -----------------------
NY = pytz.timezone("America/New_York")

BASE_HOST = "https://eodhd.com"
LIVE_V2_URL = f"{BASE_HOST}/api/us-quote-delayed"   # US extended quotes (delayed); includes 'open' and 'lastTradePrice'

HTTP_BATCH = 100           # Live v2: page[limit] max 100
DEFAULT_WS_WINDOW = 50     # per EODHD docs: ~50 concurrent symbols per connection by default
DEFAULT_WS_DWELL = 2       # seconds to sit on each WS rotation slice
CACHE_TTL_LIVEV2 = 60 * 60 * 6   # 6h; 'open' is fixed after regular open, lastTradePrice is WS-overridden

DEFAULT_PREDICTIONS_PATHS = [
    os.environ.get("PREDICTIONS_CSV", "").strip() or "",
    "live_predictions.csv",
    "/mnt/data/live_predictions.csv",
]

# -----------------------
# Secrets / env
# -----------------------
def _get_secret(name, default=None, table=None):
    try:
        return (st.secrets[table][name] if table else st.secrets[name])
    except Exception:
        return os.environ.get(name, default)

API_TOKEN = _get_secret("EODHD_API_TOKEN") or _get_secret("API_TOKEN", table="eodhd")

def _ensure_api_token() -> None:
    if not API_TOKEN:
        st.error("Missing **EODHD_API_TOKEN**. Add it in Streamlit Secrets or env, then rerun.")
        st.stop()

@st.cache_resource(show_spinner=False)
def _http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "ltr-monitor/rotate-ws/1.0"})
    return s

_ensure_api_token()
_http = _http_session()

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.caption(f"🔐 EODHD token loaded: {'yes' if bool(API_TOKEN) else 'no'}")

    # Refresh cadence
    st.write("**Auto refresh**")
    refresh_sec = st.slider("Refresh interval (seconds)", min_value=5, max_value=120, value=10, step=5)
    _ = st_autorefresh(interval=refresh_sec * 1000, key="autorefresh")

    # Predictions source
    st.write("**Predictions file**")
    file_src = st.radio("Load from…", ["Local path", "Upload"])
    if file_src == "Local path":
        default_guess = next((p for p in DEFAULT_PREDICTIONS_PATHS if p and os.path.exists(p)), DEFAULT_PREDICTIONS_PATHS[0] or "live_predictions.csv")
        preds_path = st.text_input("Path to CSV", value=default_guess, help="e.g., ./live_predictions.csv or /mnt/data/live_predictions.csv")
        upload_file = None
    else:
        upload_file = st.file_uploader("Upload CSV", type=["csv"])
        preds_path = None

    # Session date
    st.write("**Session date**")
    today_ny = datetime.now(tz=NY).date()
    session_date = st.date_input("Target trading session (ET)", value=today_ny)

    # Price sourcing
    st.write("**Price data source**")
    price_source = st.selectbox(
        "Where should OHLC prices come from?",
        (
            "Auto (live today, historical API otherwise)",
            "Force live (today only)",
            "Historical via EODHD API",
            "Upload daily OHLCV",
        ),
        help=(
            "Auto = live quotes when the session matches today, otherwise daily bars via the EODHD EOD API. "
            "Upload lets you provide your own daily OHLCV file (MultiIndex CSV supported)."
        ),
    )

    uploaded_ohlcv_file = None
    if price_source == "Upload daily OHLCV":
        uploaded_ohlcv_file = st.file_uploader(
            "Upload daily OHLCV (CSV)",
            type=["csv"],
            help="Expect columns like ticker/date/open/close. MultiIndex CSVs are also supported.",
        )

    # Symbol format
    st.write("**Symbol format**")
    exchange_suffix = st.text_input(
        "Default exchange suffix (appends if missing)",
        value="US",
        help="EODHD uses exchange-suffixed symbols, e.g., AAPL.US."
    )

    # Ranking / guards
    st.write("**Ranking**")
    top_n = st.number_input("Show top N by prediction", min_value=1, max_value=5000, value=200, step=1)
    require_asof_guard = st.checkbox(
        "Enforce as_of ≤ session open (avoid look-ahead)", value=True,
        help="If an 'as_of' timestamp is present and is after the session open, the row is dropped."
    )
    allow_na_returns = st.checkbox("Include symbols with missing data (NA returns)", value=False)

    st.write("**WebSocket (realtime) override**")
    ws_enabled = st.checkbox(
        "Enable rotating WS updates for last price",
        value=True,
        help="Updates a subset of symbols each refresh via WebSocket; doesn't consume API calls."
    )
    ws_window = st.slider("WS chunk size (≤50 per connection)", 5, DEFAULT_WS_WINDOW, DEFAULT_WS_WINDOW, step=5)
    ws_dwell = st.slider("WS dwell seconds per chunk", 1, 5, DEFAULT_WS_DWELL, step=1)
    # Build a default WS URL per docs; allow override
    default_ws_url = f"wss://ws.eodhistoricaldata.com/ws/us?api_token={API_TOKEN}"
    ws_url = st.text_input("WebSocket URL", value=default_ws_url)

    st.write("**HTTP snapshot refresh**")
    http_refresh_mins = st.number_input(
        "Refresh Live v2 HTTP snapshot every N minutes (0 = never, save quota)",
        min_value=0, max_value=120, value=0, step=5,
        help="Live v2 costs 1 API call per ticker (batched). Keep 0 to avoid burning calls."
    )

    # --- NEW: robust evaluation controls (metrics only; no API calls) ---
    st.write("**Evaluation metrics (robust)**")
    metrics_topk = st.slider(
        "Top-K for metrics (used for medians & win rate)",
        min_value=5,
        max_value=int(min(200, max(5, top_n))),
        value=int(min(20, top_n)),
        step=5,
        help="Applies to Top/Bottom median returns and Top-K win rate."
    )
    winsor_tail_pct = st.slider(
        "Winsorize tails of 'return_open_to_now' (%) for metrics",
        min_value=0, max_value=10, value=1, step=1,
        help="Clips bottom/top tails for metrics only. Spearman/Kendall are rank-based (already robust)."
    )

    st.write("**Summary metrics (long vs short)**")
    summary_topk_default = int(min(20, max(1, top_n)))
    summary_topk = st.number_input(
        "Top-K (long) rows",
        min_value=1,
        max_value=int(top_n),
        value=summary_topk_default,
        step=1,
        help="Controls the Top-K slice used for summary metric cards.",
    )
    summary_bottomk = st.number_input(
        "Bottom-K (short) rows",
        min_value=1,
        max_value=int(top_n),
        value=summary_topk_default,
        step=1,
        help="Controls the Bottom-K slice (worst predictions among the visible set).",
    )

# -----------------------
# Loaders / normalizers
# -----------------------
def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    cols = {c.lower().strip(): c for c in df.columns}
    ticker_col = next((cols[k] for k in cols if k in ["ticker", "symbol"]), None)
    pred_col   = next((cols[k] for k in cols if k in ["prediction", "pred", "pred_score", "score", "yhat"]), None)
    date_col   = next((cols[k] for k in cols if k in ["date", "session_date", "target_date"]), None)
    asof_col   = next((cols[k] for k in cols if k in ["as_of", "asof", "timestamp", "prediction_time"]), None)

    missing = []
    if ticker_col is None: missing.append("ticker")
    if pred_col is None:   missing.append("prediction")
    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
        st.stop()

    out = df.rename(columns={ticker_col: "ticker", pred_col: "prediction"})
    if date_col: out = out.rename(columns={date_col: "date"})
    if asof_col: out = out.rename(columns={asof_col: "as_of"})

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date

    if "as_of" in out.columns:
        def _parse_asof(x):
            if pd.isna(x): return pd.NaT
            try:
                dt = dateparser.parse(str(x))
                dt = NY.localize(dt) if dt.tzinfo is None else dt.astimezone(NY)
                return dt
            except Exception:
                return pd.NaT
        out["as_of"] = out["as_of"].apply(_parse_asof)
        out = out.sort_values(["ticker", "as_of"]).groupby("ticker", as_index=False).tail(1)

    st.caption(f"📐 Predictions shape after normalization: {out.shape}")
    return out

@st.cache_data(show_spinner=False)
def _load_predictions_from_path(path: str) -> pd.DataFrame:
    return _normalize_predictions(pd.read_csv(path))

def _load_predictions_from_upload(upload: io.BytesIO) -> pd.DataFrame:
    return _normalize_predictions(pd.read_csv(upload))

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV daily bars (ticker/date/open/close at minimum)."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    cols = {c.lower().strip(): c for c in df.columns}
    ticker_col = next((cols[k] for k in cols if k in ["ticker", "symbol"]), None)
    date_col = next((cols[k] for k in cols if k in ["date", "session_date", "day"]), None)
    open_col = next((cols[k] for k in cols if k in ["open", "open_price", "o"]), None)
    close_col = next((cols[k] for k in cols if k in ["close", "close_price", "c", "last"]), None)

    missing = []
    if ticker_col is None: missing.append("ticker")
    if date_col is None:   missing.append("date")
    if open_col is None:   missing.append("open")
    if close_col is None:  missing.append("close")
    if missing:
        st.error("OHLCV file missing required column(s): " + ", ".join(missing))
        st.stop()

    out = df.rename(columns={
        ticker_col: "ticker",
        date_col: "date",
        open_col: "open",
        close_col: "close",
    })

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date
    numeric_cols = [c for c in ["open", "close", "adj_close"] if c in out.columns]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    keep_cols = [c for c in ["ticker", "date", "open", "close", "adj_close", "high", "low", "volume"] if c in out.columns]
    return out[keep_cols]

def _load_ohlcv_from_upload(upload: io.BytesIO) -> pd.DataFrame:
    return _normalize_ohlcv(pd.read_csv(upload))

def _align_and_filter_for_session(df: pd.DataFrame, session: date) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        before = out.shape
        out = out[out["date"] == session]
        st.caption(f"📐 After filtering to session={session} (by 'date'): {before} → {out.shape}")
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
# Session bounds (simple 09:30–16:00 ET)
# -----------------------
def _regular_session_bounds(session: date) -> Tuple[datetime, datetime]:
    return (
        NY.localize(datetime.combine(session, dt_time(9, 30))),
        NY.localize(datetime.combine(session, dt_time(16, 0))),
    )

session_open, session_close = _regular_session_bounds(session_date)
now_et = datetime.now(tz=NY)

# -----------------------
# Symbol normalization
# -----------------------
def _to_eod_symbol(ticker: str, suffix: str) -> str:
    t = (ticker or "").strip().upper()
    return t if "." in t else f"{t}.{suffix.strip().upper()}"

def _strip_suffix(symbol: str) -> str:
    return (symbol or "").strip().upper().split(".")[0]

# -----------------------
# Live v2 (delayed) HTTP: batch quotes (open + lastTradePrice)
# -----------------------
def _chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _historical_daily_bars(symbols_eod: Tuple[str, ...], session: date) -> pd.DataFrame:
    """Fetch historical daily bars (open/close) for a single session via EODHD EOD API."""
    if not symbols_eod:
        return pd.DataFrame(columns=["symbol", "open", "close"]).set_index("symbol")

    iso = session.strftime("%Y-%m-%d")
    frames = []
    for sym in symbols_eod:
        url = f"{BASE_HOST}/api/eod/{sym}?from={iso}&to={iso}&api_token={API_TOKEN}&fmt=json"
        try:
            r = _http.get(url, timeout=10)
            if not r.ok:
                continue
            payload = r.json()
            if isinstance(payload, list) and payload:
                bar = payload[0]
                frames.append({
                    "symbol": sym,
                    "open": bar.get("open", np.nan),
                    "close": bar.get("close", np.nan),
                })
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["symbol", "open", "close"]).set_index("symbol")

    out = pd.DataFrame(frames).drop_duplicates(subset=["symbol"]).set_index("symbol")
    return out

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_LIVEV2)
def _livev2_quotes_once(symbols_eod: Tuple[str, ...]) -> pd.DataFrame:
    """
    Fetch delayed quotes for all requested symbols (one-time snapshot).
    Returns DataFrame indexed by symbol with at least columns: ['open', 'lastTradePrice'].
    NOTE: Consumes 1 API call per symbol per EODHD docs. Keep TTL high to save quota.
    """
    if not symbols_eod:
        return pd.DataFrame(columns=["symbol", "open", "lastTradePrice"]).set_index("symbol")

    frames = []
    for group in _chunk(list(symbols_eod), HTTP_BATCH):
        url = f"{LIVE_V2_URL}?s={','.join(group)}&api_token={API_TOKEN}&fmt=json"
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
            continue

    if not frames:
        return pd.DataFrame(columns=["symbol", "open", "lastTradePrice"]).set_index("symbol")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["symbol"]).set_index("symbol")
    return out

# Optional periodic HTTP refresh (off by default to save quota)
def _should_http_refresh() -> bool:
    if http_refresh_mins <= 0:
        return False
    last = st.session_state.get("last_http_refresh_at")
    if last is None:
        return True
    return (datetime.now(tz=NY) - last) >= timedelta(minutes=int(http_refresh_mins))

def _mark_http_refreshed():
    st.session_state["last_http_refresh_at"] = datetime.now(tz=NY)

# -----------------------
# Rotating WS updater (no SDK)
# -----------------------
def _ws_update_prices_rotating(ws_url: str, symbols_eod: List[str], window: int, dwell_s: int, rot_idx: int) -> Dict[str, float]:
    """
    Subscribe to a slice of symbols (size=window) and collect trade prices for dwell_s seconds.
    Returns mapping EOD symbol -> last price for the slice.
    """
    if not _HAS_WS or not ws_url or not symbols_eod or window <= 0:
        return {}

    total = len(symbols_eod)
    start = (rot_idx * window) % max(total, 1)
    slice_syms = symbols_eod[start:start+window]
    if len(slice_syms) < window and total > 0:
        slice_syms += symbols_eod[0:max(0, window - len(slice_syms))]

    base_tickers = ",".join(sorted({_strip_suffix(s) for s in slice_syms}))

    try:
        # Direct, synchronous connection with controlled lifespan.
        conn = websocket.create_connection(ws_url, timeout=5)
        conn.settimeout(1.0)
        conn.send(json.dumps({"action": "subscribe", "symbols": base_tickers}))
        end_time = time.time() + max(1, int(dwell_s))
        prices: Dict[str, float] = {}
        while time.time() < end_time:
            try:
                msg = conn.recv()
            except Exception:
                continue
            try:
                d = json.loads(msg)
            except Exception:
                continue
            sym = str(d.get("s") or d.get("code") or "").upper()
            px = d.get("p") or d.get("price") or d.get("lastTradePrice")
            if sym and isinstance(px, (int, float)):
                prices[sym] = float(px)
        try:
            conn.close()
        except Exception:
            pass

        # Map base tickers back to EOD symbols in this slice
        out = {}
        for s in slice_syms:
            base = _strip_suffix(s)
            if base in prices:
                out[s] = prices[base]
        return out
    except Exception:
        return {}

# -----------------------
# Load predictions
# -----------------------
if 'upload_file' in locals() and upload_file is not None:
    preds = _load_predictions_from_upload(upload_file)
else:
    chosen_default = preds_path if 'preds_path' in locals() else None
    if not chosen_default:
        chosen_default = next((p for p in DEFAULT_PREDICTIONS_PATHS if p and os.path.exists(p)), "live_predictions.csv")
    if chosen_default and os.path.exists(chosen_default):
        preds = _load_predictions_from_path(chosen_default)
    else:
        preds = pd.DataFrame(columns=["ticker", "prediction"])

if preds.empty:
    st.info("Upload or point to your predictions CSV to get started. Expect columns like: `ticker`, `prediction`, optional `date`, optional `as_of`.")
    st.stop()

# Filter to session & rank
preds = _align_and_filter_for_session(preds, session_date)
preds_ranked = preds.sort_values("prediction", ascending=False).reset_index(drop=True)
preds_ranked["rank"] = np.arange(1, len(preds_ranked) + 1)
preds_ranked = preds_ranked.head(int(top_n))
preds_ranked["ticker"] = preds_ranked["ticker"].astype(str).str.upper().str.strip()
st.caption(f"📐 Ranked predictions shape (after top-N): {preds_ranked.shape}")

# Session guard
preds_ranked = _post_asof_guard(preds_ranked, session_open, require_asof_guard=require_asof_guard)
if preds_ranked.empty:
    st.warning("No rows remain after enforcing the as_of guard.")
    st.stop()

# Symbols
symbols_in = preds_ranked["ticker"].dropna().astype(str).str.upper().tolist()
symbols_eod = [_to_eod_symbol(t, exchange_suffix) for t in symbols_in]
symbols_eod_tuple = tuple(symbols_eod)  # for cache key stability

is_today_session = session_date == today_ny
is_past_session = session_date < today_ny

if price_source == "Force live (today only)":
    price_mode = "live"
elif price_source == "Historical via EODHD API":
    price_mode = "historical_api"
elif price_source == "Upload daily OHLCV":
    price_mode = "upload"
else:
    price_mode = "live" if is_today_session else ("historical_api" if is_past_session else "future")

if price_mode == "live" and not is_today_session:
    st.warning("Live mode only supports today's session — falling back to historical daily bars.")
    price_mode = "historical_api"

ohlcv_upload_df = None
if price_mode == "upload":
    if uploaded_ohlcv_file is None:
        st.info("Upload a daily OHLCV CSV to evaluate past sessions without API calls.")
        st.stop()
    ohlcv_upload_df = _load_ohlcv_from_upload(uploaded_ohlcv_file)
    st.caption(f"📐 Uploaded OHLCV shape after normalization: {ohlcv_upload_df.shape}")

# -----------------------
# Price data assembly (live vs historical vs upload)
# -----------------------
map_df = pd.DataFrame({"ticker": symbols_in, "eod_symbol": symbols_eod}).drop_duplicates()
map_df["base_ticker"] = map_df["ticker"].str.upper().str.strip()
df_view = preds_ranked.merge(map_df, on="ticker", how="left")
if "base_ticker" in df_view.columns:
    df_view["base_ticker"] = df_view["base_ticker"].fillna(df_view["ticker"].astype(str).str.upper().str.strip())
else:
    df_view["base_ticker"] = df_view["ticker"].astype(str).str.upper().str.strip()

df_view["open_today"] = np.nan
df_view["last_price"] = np.nan

quotes_snapshot = None
hist = None
session_rows_all = None
session_rows = None

if price_mode == "live":
    quotes_snapshot = _livev2_quotes_once(symbols_eod_tuple)  # index: EOD symbol
    if _should_http_refresh():
        _livev2_quotes_once.clear()
        quotes_snapshot = _livev2_quotes_once(symbols_eod_tuple)
        _mark_http_refreshed()

    if "last_price_cache" not in st.session_state:
        st.session_state["last_price_cache"] = {}

    for sym in symbols_eod:
        if sym not in st.session_state["last_price_cache"]:
            try:
                st.session_state["last_price_cache"][sym] = float(quotes_snapshot.at[sym, "lastTradePrice"])
            except Exception:
                pass

    if ws_enabled:
        if not _HAS_WS:
            st.info("websocket-client not installed; staying on delayed HTTP only.")
        else:
            rot = st.session_state.get("ws_rot_idx", 0)
            ws_prices = _ws_update_prices_rotating(ws_url, symbols_eod, min(ws_window, DEFAULT_WS_WINDOW), ws_dwell, rot)
            for sym, px in ws_prices.items():
                st.session_state["last_price_cache"][sym] = px
            st.session_state["ws_rot_idx"] = rot + 1

    open_series = quotes_snapshot["open"] if "open" in quotes_snapshot.columns else pd.Series(dtype=float)
    last_series = pd.Series({s: st.session_state["last_price_cache"].get(s, np.nan) for s in symbols_eod}, dtype=float)
    df_view["open_today"] = df_view["eod_symbol"].map(open_series)
    df_view["last_price"] = df_view["eod_symbol"].map(last_series)

elif price_mode == "historical_api":
    hist = _historical_daily_bars(symbols_eod_tuple, session_date)
    st.caption(f"📐 Historical EOD bars fetched: {hist.shape}")
    if hist.empty:
        st.warning("No historical bars returned for this session from EODHD.")
    df_view["open_today"] = df_view["eod_symbol"].map(hist["open"] if "open" in hist.columns else pd.Series(dtype=float))
    df_view["last_price"] = df_view["eod_symbol"].map(hist["close"] if "close" in hist.columns else pd.Series(dtype=float))

elif price_mode == "upload":
    session_rows_all = ohlcv_upload_df[ohlcv_upload_df["date"] == session_date]
    st.caption(f"📐 Uploaded OHLCV rows for {session_date}: {session_rows_all.shape}")
    if session_rows_all.empty:
        st.warning("Uploaded OHLCV file has no rows for this session date.")
    session_rows = session_rows_all.drop_duplicates(subset=["ticker"]).set_index("ticker")
    df_view["open_today"] = df_view["base_ticker"].map(session_rows["open"] if "open" in session_rows.columns else pd.Series(dtype=float))
    close_col = "close" if "close" in session_rows.columns else None
    if close_col is None and "adj_close" in session_rows.columns:
        close_col = "adj_close"
    if close_col is None:
        st.error("Uploaded OHLCV file needs a 'close' or 'adj_close' column for returns.")
        st.stop()
    df_view["last_price"] = df_view["base_ticker"].map(session_rows[close_col])

else:  # future session
    st.info("Selected session is in the future. Returns will populate once data is available.")

# Compute open→now return (suppress before regular open)
if price_mode == "live" and now_et < session_open:
    df_view["return_open_to_now"] = np.nan
else:
    df_view["return_open_to_now"] = (df_view["last_price"] / df_view["open_today"] - 1.0).replace([np.inf, -np.inf], np.nan)

# Optionally drop NA rows
if not allow_na_returns:
    before = df_view.shape
    df_view = df_view.dropna(subset=["open_today", "last_price", "return_open_to_now"])
    st.caption(f"📐 After dropping NA returns: {before} → {df_view.shape}")

# Final sort by your prediction score
df_view = df_view.sort_values("prediction", ascending=False).reset_index(drop=True)

# Pretty formatting
def _fmt_pct(x):   return "" if pd.isna(x) else f"{x*100:.2f}%"
def _fmt_price(x): return "" if pd.isna(x) else f"{x:.2f}"

display_cols = ["rank", "ticker", "prediction", "open_today", "last_price", "return_open_to_now"]
show = df_view[display_cols].copy()
show["open_today"] = show["open_today"].apply(_fmt_price)
show["last_price"] = show["last_price"].apply(_fmt_price)
show["return_open_to_now"] = show["return_open_to_now"].apply(_fmt_pct)

chart_data = (
    df_view[["rank", "ticker", "prediction", "return_open_to_now"]]
    .dropna(subset=["return_open_to_now"])
    .copy()
)
if not chart_data.empty:
    chart_data["return_direction"] = np.where(
        chart_data["return_open_to_now"] >= 0,
        "Positive",
        "Negative",
    )
    chart_data.sort_values("rank", inplace=True)


if price_mode == "live":
    open_label = "Open (today)"
    last_label = "Last price"
    return_label = "Return (open→now)"
elif price_mode in {"historical_api", "upload"}:
    open_label = "Open"
    last_label = "Close"
    return_label = "Return (open→close)"
else:
    open_label = "Open"
    last_label = "Last price"
    return_label = "Return"

show = show.rename(columns={
    "open_today": open_label,
    "last_price": last_label,
    "return_open_to_now": return_label,
})

# -----------------------
# NEW: Robust rank⇄return metrics (no API calls; in-memory)
# -----------------------
def _winsorize_series(s: pd.Series, p: float) -> pd.Series:
    """Clip s to [p, 1-p] quantiles; p in [0, 0.5)."""
    s = pd.to_numeric(s, errors="coerce")
    if p <= 0 or s.dropna().empty:
        return s
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lower=lo, upper=hi)

def _pearson_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = a.notna() & b.notna()
    if m.sum() < 2:
        return np.nan
    ax = a[m] - a[m].mean()
    bx = b[m] - b[m].mean()
    denom = np.sqrt((ax**2).sum() * (bx**2).sum())
    return float((ax * bx).sum() / denom) if denom > 0 else np.nan

def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman via Pearson of ranks (average-tie method)."""
    ar = a.rank(method="average")
    br = b.rank(method="average")
    return _pearson_corr(ar, br)

def _kendall_tau_b(x: pd.Series, y: pd.Series) -> float:
    """Naive O(n^2) Kendall tau-b; fine for typical top-N≤~1000."""
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()
    n = len(d)
    if n < 2:
        return np.nan
    xv = d["x"].to_numpy()
    yv = d["y"].to_numpy()
    C = 0  # concordant
    D = 0  # discordant
    for i in range(n - 1):
        dx = xv[i+1:] - xv[i]
        dy = yv[i+1:] - yv[i]
        prod = dx * dy
        C += int((prod > 0).sum())
        D += int((prod < 0).sum())
    # tie corrections
    n0 = n * (n - 1) // 2
    counts_x = pd.Series(xv).value_counts().to_numpy()
    counts_y = pd.Series(yv).value_counts().to_numpy()
    n1 = int(((counts_x * (counts_x - 1)) // 2).sum())
    n2 = int(((counts_y * (counts_y - 1)) // 2).sum())
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    return float((C - D) / denom) if denom > 0 else np.nan

def _theil_sen_slope_via_deciles(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """
    Robust Theil–Sen slope of y vs x computed on decile medians to avoid O(n^2).
    Returns slope only. If insufficient points, returns np.nan.
    """
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if d.shape[0] < 3:
        return np.nan
    bins = int(max(3, min(bins, d.shape[0] // 2)))
    # bin by rank to evenly spread observations
    # labels 1..bins with 1 = lowest x; we want x increasing with "better", so we will pass x as (-rank)
    try:
        q = pd.qcut(d["x"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    except Exception:
        q = pd.cut(d["x"], bins=bins, labels=False, duplicates="drop")
    g = d.assign(bin=q).groupby("bin", as_index=False).agg(x_med=("x", "median"), y_med=("y", "median")).dropna()
    xv = g["x_med"].to_numpy()
    yv = g["y_med"].to_numpy()
    m = len(xv)
    if m < 3:
        return np.nan
    slopes = []
    for i in range(m - 1):
        dx = xv[i+1:] - xv[i]
        dy = yv[i+1:] - yv[i]
        valid = dx != 0
        if np.any(valid):
            slopes.extend((dy[valid] / dx[valid]).tolist())
    return float(np.median(slopes)) if slopes else np.nan

def _compute_rank_metrics(df: pd.DataFrame, topk: int, winsor_pct: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute robust rank↔return metrics; returns (metrics_table, deciles_table)."""
    ready = df[["rank", "prediction", "return_open_to_now"]].dropna().copy()
    st.caption(f"📐 Metrics-ready rows (after NA drop): {ready.shape}")

    if ready.empty:
        return pd.DataFrame(columns=["metric", "value"]), pd.DataFrame(columns=["decile", "median_return"])

    # Use negative rank so "higher is better" for correlation orientation
    ready["neg_rank"] = -ready["rank"].astype(float)

    # Winsorize returns for Pearson-style only (Spearman/Kendall are rank-based)
    p = max(0.0, min(0.5, float(winsor_pct) / 100.0))
    ret_w = _winsorize_series(ready["return_open_to_now"], p)

    # Correlations
    spearman = _spearman_corr(ready["neg_rank"], ready["return_open_to_now"])
    kendall  = _kendall_tau_b(ready["neg_rank"], ready["return_open_to_now"])
    pearson_w = _pearson_corr(ready["neg_rank"], ret_w)

    # Theil–Sen slope (robust): y vs x where x = -rank
    ts_slope = _theil_sen_slope_via_deciles(ready["neg_rank"], ready["return_open_to_now"], bins=10)
    bps_per_10rank = (ts_slope * 10.0) * 10000.0 if pd.notna(ts_slope) else np.nan

    # Top/Bottom-K medians + win rate
    k = int(min(topk, ready.shape[0] // 2 if ready.shape[0] >= 2 else topk))
    top = ready.nsmallest(k, "rank")  # rank 1..k
    bot = ready.nlargest(k, "rank")
    top_med = float(top["return_open_to_now"].median()) if not top.empty else np.nan
    bot_med = float(bot["return_open_to_now"].median()) if not bot.empty else np.nan
    l_s_med = top_med - bot_med if pd.notna(top_med) and pd.notna(bot_med) else np.nan
    top_win = float((top["return_open_to_now"] > 0).mean()) if not top.empty else np.nan

    metrics_rows = [
        ("Spearman ρ (−rank vs return)", f"{spearman:.3f}" if pd.notna(spearman) else "—"),
        ("Kendall τ-b (−rank vs return)", f"{kendall:.3f}" if pd.notna(kendall) else "—"),
        (f"Winsorized Pearson r (tails={winsor_pct}%)", f"{pearson_w:.3f}" if pd.notna(pearson_w) else "—"),
        ("Theil–Sen slope (return per rank)", f"{ts_slope:.6f}" if pd.notna(ts_slope) else "—"),
        ("≈ bps per 10-rank improvement", f"{bps_per_10rank:.1f}" if pd.notna(bps_per_10rank) else "—"),
        (f"Top-{k} median return", f"{top_med*100:.2f}%" if pd.notna(top_med) else "—"),
        (f"Bottom-{k} median return", f"{bot_med*100:.2f}%" if pd.notna(bot_med) else "—"),
        (f"Median long–short (Top-{k} − Bottom-{k})", f"{l_s_med*100:.2f}%" if pd.notna(l_s_med) else "—"),
        (f"Top-{k} win rate (>0%)", f"{top_win*100:.1f}%" if pd.notna(top_win) else "—"),
    ]
    metrics_tbl = pd.DataFrame(metrics_rows, columns=["metric", "value"])

    # Deciles by rank: 1 (best) → D (worst)
    D = int(min(10, max(3, ready.shape[0] // 10)))
    ready["decile"] = pd.qcut(ready["rank"], q=D, labels=False, duplicates="drop") if ready["rank"].nunique() > 1 else 0
    # make 0 = best rank bucket
    deciles = (ready
               .groupby("decile", as_index=False)
               .agg(median_return=("return_open_to_now", "median"),
                    count=("return_open_to_now", "size"))
               .sort_values("decile"))
    deciles["bucket"] = deciles["decile"].astype(int) + 1  # 1..D
    deciles = deciles[["bucket", "count", "median_return"]]
    deciles.rename(columns={"bucket": f"rank_decile(1=best, D={deciles.shape[0]})"}, inplace=True)
    return metrics_tbl, deciles

metrics_tbl, deciles_tbl = _compute_rank_metrics(df_view, metrics_topk, winsor_tail_pct)

deciles_chart_data = pd.DataFrame()
if deciles_tbl is not None and not deciles_tbl.empty:
    deciles_chart_data = deciles_tbl.copy()
    decile_label = deciles_chart_data.columns[0]
    deciles_chart_data = deciles_chart_data.rename(columns={decile_label: "rank_decile"})
    deciles_chart_data["rank_decile"] = deciles_chart_data["rank_decile"].astype(str)


# -----------------------
# UI
# -----------------------

price_mode_label_map = {
    "live": "HTTP + rotating WS" if ws_enabled else "HTTP (delayed only)",
    "historical_api": "Historical daily bars (EODHD API)",
    "upload": "Uploaded OHLCV file",
    "future": "Future session (awaiting data)",
}
price_mode_label = price_mode_label_map.get(price_mode, str(price_mode))

price_dim_lines = [
    f"predictions_raw: {preds.shape}",
    f"ranked_topN:     {preds_ranked.shape}",
]

if price_mode == "live":
    snap_shape = quotes_snapshot.shape if isinstance(quotes_snapshot, pd.DataFrame) else (0, 0)
    price_dim_lines.append(f"quotes_snapshot: {snap_shape}")
elif price_mode == "historical_api":
    hist_shape = hist.shape if isinstance(hist, pd.DataFrame) else (0, 0)
    price_dim_lines.append(f"historical_bars: {hist_shape}")
elif price_mode == "upload":
    upload_shape = session_rows_all.shape if isinstance(session_rows_all, pd.DataFrame) else (0, 0)
    price_dim_lines.append(f"uploaded_rows: {upload_shape}")
else:
    price_dim_lines.append("price_data: future (n/a)")

price_dim_lines.extend([
    f"joined_view:     {df_view.shape}",
    f"mode:            {price_mode_label}",
])
st.title("📈 Live LTR Monitor — Predictions vs Open→Now Returns (EODHD, rotating WS)")
st.caption(
    "Opens fetched once via **Live v2 (delayed)**; last prices updated by **rotating WebSocket slices** "
    f"(chunk ≤{DEFAULT_WS_WINDOW}). This keeps HTTP API calls low while still refreshing a large watchlist."
)

if price_mode in {"historical_api", "upload"}:
    st.caption("Historical mode: returns use session open → close (no WebSocket refresh required).")
elif price_mode == "future":
    st.caption("Future session selected — waiting for market data to arrive.")

left, right = st.columns([3, 2], gap="large")
with left:
    if price_mode == "live":
        table_phrase = "returns from open → latest"
    elif price_mode in {"historical_api", "upload"}:
        table_phrase = "returns from open → close"
    else:
        table_phrase = "returns (pending data)"
    st.subheader(f"Top predictions ({table_phrase})")
    if not chart_data.empty:
        scatter = (
            alt.Chart(chart_data)
            .mark_circle(size=70, opacity=0.75)
            .encode(
                x=alt.X("rank:Q", title="Model rank (1 = best)", scale=alt.Scale(zero=False)),
                y=alt.Y(
                    "return_open_to_now:Q",
                    title="Realized return (open→now)",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color(
                    "return_direction:N",
                    title="Return direction",
                    scale=alt.Scale(domain=["Positive", "Negative"], range=["#2ecc71", "#e74c3c"]),
                ),
                tooltip=[
                    alt.Tooltip("ticker:N", title="Ticker"),
                    alt.Tooltip("prediction:Q", title="Prediction", format=".4f"),
                    alt.Tooltip("return_open_to_now:Q", title="Return", format=".2%"),
                    alt.Tooltip("rank:Q", title="Rank"),
                ],
            )
        )

        rolling = (
            alt.Chart(chart_data)
            .transform_window(
                rolling_mean="mean(return_open_to_now)",
                sort=[{"field": "rank"}],
                frame=[-5, 5],
            )
            .mark_line(color="#34495e", strokeWidth=2)
            .encode(
                x=alt.X("rank:Q"),
                y=alt.Y("rolling_mean:Q", title="Rolling mean"),
            )
        )

        st.altair_chart((scatter + rolling).interactive(), use_container_width=True)
    else:
        st.info("Waiting for realized returns to plot.")
    st.dataframe(show, use_container_width=True, height=650)

with right:
    st.subheader("Summary")
    st.metric("Session date (ET)", str(session_date))
    st.metric("Symbols shown", f"{show.shape[0]:,}")
    if df_view["return_open_to_now"].notna().any():
        realized_scope = df_view[df_view["return_open_to_now"].notna()]
        if not realized_scope.empty:
            top_slice = realized_scope.head(min(int(summary_topk), len(realized_scope)))
            bottom_slice = realized_scope.tail(min(int(summary_bottomk), len(realized_scope)))

            top_mean = top_slice["return_open_to_now"].mean()
            bottom_mean = bottom_slice["return_open_to_now"].mean()

            st.metric(
                f"Mean return of Top-{len(top_slice)}",
                f"{top_mean * 100:.2f}%",
            )
            st.metric(
                f"Mean return of Bottom-{len(bottom_slice)}",
                f"{bottom_mean * 100:.2f}%",
            )

    if price_mode == "live":
        total = len(symbols_eod)
        rot = st.session_state.get("ws_rot_idx", 0)
        current_slice_start = (rot * min(ws_window, DEFAULT_WS_WINDOW)) % max(total, 1) if total else 0
        st.write(" ")
        st.write("**Rotation status**")
        st.code(
            f"total_symbols:    {total}\n"
            f"ws_chunk_size:    {min(ws_window, DEFAULT_WS_WINDOW)}\n"
            f"ws_dwell_seconds: {ws_dwell}\n"
            f"rotation_index:   {rot}\n"
            f"slice_start_idx:  {current_slice_start}",
            language="text",
        )

    st.write(" ")
    st.write("**Dimensions audit**")
    st.code("\n".join(price_dim_lines), language="text")

st.write("---")
met_left, met_right = st.columns([2, 3], gap="large")
with met_left:
    metrics_scope = "today" if price_mode == "live" and is_today_session else str(session_date)
    st.subheader(f"🧪 Robust rank⇄return metrics ({metrics_scope})")
    if not metrics_tbl.empty:
        st.dataframe(metrics_tbl, use_container_width=True, height=330)
    else:
        st.info("Metrics unavailable (no non-NA returns yet).")

with met_right:
    st.subheader("Rank deciles → median returns")
    if not deciles_tbl.empty:
        if not deciles_chart_data.empty:
            decile_chart = (
                alt.Chart(deciles_chart_data)
                .mark_bar(color="#3498db", opacity=0.7)
                .encode(
                    x=alt.X("rank_decile:N", title="Rank decile"),
                    y=alt.Y("median_return:Q", title="Median return", axis=alt.Axis(format="%")),
                    tooltip=[
                        alt.Tooltip("rank_decile:N", title="Decile"),
                        alt.Tooltip("median_return:Q", title="Median return", format=".2%"),
                        alt.Tooltip("count:Q", title="Count"),
                    ],
                )
            )
            decile_line = (
                alt.Chart(deciles_chart_data)
                .mark_line(color="#2c3e50", point=alt.OverlayMarkDef(filled=True, color="#2c3e50"))
                .encode(
                    x="rank_decile:N",
                    y=alt.Y("median_return:Q", axis=alt.Axis(format="%")),
                )
            )
            st.altair_chart(decile_chart + decile_line, use_container_width=True)
        # Format and show decile medians to visualize monotonicity
        deciles_fmt = deciles_tbl.copy()
        deciles_fmt["median_return"] = deciles_fmt["median_return"].apply(_fmt_pct)
        st.dataframe(deciles_fmt, use_container_width=True, height=330)
    else:
        st.info("Decile table unavailable (insufficient rows).")

# Download
csv_buf = io.StringIO()
df_view.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Download full table (CSV)",
    data=csv_buf.getvalue(),
    file_name=f"live_ltr_{session_date}.csv",
    mime="text/csv",
)

if price_mode == "live":
    st.caption(
        "Tip: Keep HTTP refresh at **0 minutes** when monitoring large lists to avoid burning daily API calls. "
        "WebSockets don’t consume API calls; default WS limit is ~50 symbols per connection (upgradeable)."
    )
