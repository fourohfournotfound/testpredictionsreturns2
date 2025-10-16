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

DEFAULT_BASELINE_PATHS = [
    os.environ.get("BASELINE_PREDICTIONS_CSV", "").strip() or "",
    "baseline_predictions.csv",
    "/mnt/data/baseline_predictions.csv",
]

DEFAULT_SHARADAR_PATHS = [
    os.environ.get("SHARADAR_DAILY_CSV", "").strip() or "",
    "sharadar_sep.csv",
    "/mnt/data/sharadar_sep.csv",
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

    st.write("**Baseline predictions (optional)**")
    baseline_src = st.radio("Baseline source", ["None", "Local path", "Upload"], index=0)
    baseline_upload = None
    baseline_path = None
    if baseline_src == "Local path":
        baseline_default = next((p for p in DEFAULT_BASELINE_PATHS if p and os.path.exists(p)), DEFAULT_BASELINE_PATHS[0] or "baseline_predictions.csv")
        baseline_path = st.text_input(
            "Baseline CSV path",
            value=baseline_default,
            help="Optional reference model to compare rank shifts.",
            key="baseline_path",
        )
    elif baseline_src == "Upload":
        baseline_upload = st.file_uploader("Upload baseline CSV", type=["csv"], key="baseline_upload")

    today_ny = datetime.now(tz=NY).date()

    # Price sourcing
    st.write("**Price data source**")
    price_source = st.selectbox(
        "Where should OHLC prices come from?",
        (
            "Auto (live today, historical API otherwise)",
            "Force live (today only)",
            "Historical via EODHD API",
            "Sharadar daily (local CSV)",
            "Upload daily OHLCV",
        ),
        help=(
            "Auto = live quotes when the session matches today, otherwise daily bars via the EODHD EOD API. "
            "Upload lets you provide your own daily OHLCV file (MultiIndex CSV supported)."
        ),
    )

    sharadar_path = None
    sharadar_upload = None
    uploaded_ohlcv_file = None
    if price_source == "Upload daily OHLCV":
        uploaded_ohlcv_file = st.file_uploader(
            "Upload daily OHLCV (CSV)",
            type=["csv"],
            help="Expect columns like ticker/date/open/close. MultiIndex CSVs are also supported.",
        )
    elif price_source == "Sharadar daily (local CSV)":
        default_sharadar = next(
            (p for p in DEFAULT_SHARADAR_PATHS if p and os.path.exists(p)),
            DEFAULT_SHARADAR_PATHS[0] or "sharadar_sep.csv",
        )
        sharadar_path = st.text_input(
            "Sharadar CSV path",
            value=default_sharadar,
            help=(
                "Path to a Sharadar SEP-style daily prices CSV (needs ticker/date/open/high/low/close/volume). "
                "Upload a slice instead if the full file is too large."
            ),
        )
        sharadar_upload = st.file_uploader(
            "Or upload Sharadar daily CSV",
            type=["csv"],
            key="sharadar_upload",
        )

    st.write("**Return horizon**")
    horizon_options = {
        "next_open_to_close": "Next session: open → close",
        "next_open_to_open": "Next session: open → following open",
        "same_session": "Same session: open → close",
    }
    return_horizon = st.selectbox(
        "Evaluate realized returns using…",
        list(horizon_options.keys()),
        index=0,
        format_func=lambda k: horizon_options[k],
        help=(
            "Default = trade on the next day at the open and exit at the close."
            " Choose open→open for overnight holds or same-session for intraday."
        ),
    )

    st.write("**Trading session**")
    session_picker = st.empty()
    st.caption(
        "Prediction dates appear here once a CSV is loaded. We pick the latest by default."
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
        help="Applies to Top/Bottom median returns, Top-K win rate, and ranking metrics (Precision@K / MAP@K / NDCG@K)."
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

    if ticker_col is None and date_col is not None:
        wide_value_cols = [c for c in df.columns if c != date_col]
        if wide_value_cols:
            melted = df.melt(
                id_vars=[date_col],
                value_vars=wide_value_cols,
                var_name="ticker",
                value_name="prediction",
            )
            return _normalize_predictions(melted)

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

    out = out.dropna(subset=["prediction"])

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

@st.cache_data(show_spinner=False)
def _load_ohlcv_from_path(path: str) -> pd.DataFrame:
    return _normalize_ohlcv(pd.read_csv(path))

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

def _next_weekday(d: date) -> date:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt

def _shift_weekday(d: date, steps: int) -> date:
    if steps == 0:
        return d
    direction = 1 if steps > 0 else -1
    remaining = abs(steps)
    current = d
    while remaining > 0:
        current += timedelta(days=direction)
        if current.weekday() < 5:
            remaining -= 1
    return current

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

preds_full_history = preds.copy()

if preds.empty:
    st.info("Upload or point to your predictions CSV to get started. Expect columns like: `ticker`, `prediction`, optional `date`, optional `as_of`.")
    st.stop()

prediction_date = None
if "date" in preds.columns and preds["date"].notna().any():
    available_dates = sorted(preds["date"].dropna().unique())
    default_index = len(available_dates) - 1
    with st.sidebar:
        prediction_date = session_picker.selectbox(
            "Prediction date",
            options=available_dates,
            index=default_index,
            key="prediction_date_select",
            format_func=lambda d: d.strftime("%Y-%m-%d"),
        )
else:
    with st.sidebar:
        prediction_date = session_picker.date_input(
            "Prediction date",
            value=today_ny,
            key="prediction_date_manual",
        )

if isinstance(prediction_date, pd.Timestamp):
    prediction_date = prediction_date.date()

if not isinstance(prediction_date, date):
    prediction_date = today_ny


if return_horizon == "same_session":
    session_date_guess = prediction_date
else:
    session_date_guess = _next_weekday(prediction_date)

session_open_guess, session_close_guess = _regular_session_bounds(session_date_guess)
session_open, session_close = session_open_guess, session_close_guess


# Optional baseline predictions
baseline = pd.DataFrame(columns=["ticker", "prediction"])
baseline_ranked = pd.DataFrame()
if baseline_src != "None":
    try:
        if baseline_src == "Local path" and baseline_path:
            if os.path.exists(baseline_path):
                baseline = _load_predictions_from_path(baseline_path)
            else:
                st.warning(f"Baseline path not found: {baseline_path}")
        elif baseline_src == "Upload" and baseline_upload is not None:
            baseline = _load_predictions_from_upload(baseline_upload)
    except Exception as e:
        st.warning(f"Failed to load baseline predictions: {e}")

if not baseline.empty:
    baseline = _align_and_filter_for_session(baseline, prediction_date)
    baseline = _post_asof_guard(baseline, session_open, require_asof_guard=require_asof_guard)
    baseline_ranked = baseline.sort_values("prediction", ascending=False).reset_index(drop=True)
    baseline_ranked["baseline_rank"] = np.arange(1, len(baseline_ranked) + 1)
    baseline_ranked["ticker"] = baseline_ranked["ticker"].astype(str).str.upper().str.strip()
    baseline_ranked = baseline_ranked.rename(columns={"prediction": "baseline_prediction"})
    st.caption(f"📐 Baseline predictions shape (after normalization): {baseline_ranked.shape}")

# Filter to session & rank
preds = _align_and_filter_for_session(preds, prediction_date)
preds_ranked_full = preds.sort_values("prediction", ascending=False).reset_index(drop=True)
preds_ranked_full["full_rank"] = np.arange(1, len(preds_ranked_full) + 1)
preds_ranked_full["ticker"] = preds_ranked_full["ticker"].astype(str).str.upper().str.strip()
preds_ranked = preds_ranked_full.head(int(top_n)).copy()
preds_ranked["rank"] = np.arange(1, len(preds_ranked) + 1)
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

is_today_session = session_date_guess == today_ny
is_past_session = session_date_guess < today_ny

if price_source == "Force live (today only)":
    price_mode = "live"
elif price_source == "Historical via EODHD API":
    price_mode = "historical_api"
elif price_source == "Sharadar daily (local CSV)":
    price_mode = "sharadar"
elif price_source == "Upload daily OHLCV":
    price_mode = "upload"
else:
    price_mode = "live" if is_today_session else ("historical_api" if is_past_session else "future")

if price_mode == "live" and not is_today_session:
    st.warning("Live mode only supports today's session — falling back to historical daily bars.")
    price_mode = "historical_api"

effective_horizon = return_horizon
if price_mode == "live":
    if return_horizon != "same_session":
        st.info("Live mode is limited to same-session open→now returns; overriding the selected horizon.")
    effective_horizon = "same_session"
    session_date = prediction_date
elif price_mode in {"historical_api", "upload", "sharadar"}:
    if return_horizon == "same_session":
        session_date = prediction_date
    else:
        session_date = _next_weekday(prediction_date)
    effective_horizon = return_horizon
else:
    session_date = prediction_date

entry_date = session_date
exit_date = session_date
if effective_horizon == "next_open_to_open":
    exit_date = _next_weekday(entry_date)

session_open, session_close = _regular_session_bounds(session_date)

is_today_session = session_date == today_ny
is_past_session = session_date < today_ny

if session_open != session_open_guess:
    preds_ranked = _post_asof_guard(preds_ranked, session_open, require_asof_guard=require_asof_guard)
    if preds_ranked.empty:
        st.warning("No rows remain after enforcing the as_of guard for the active trading session.")
        st.stop()

ohlcv_upload_df = None
sharadar_df = None
if price_mode == "upload":
    if uploaded_ohlcv_file is None:
        st.info("Upload a daily OHLCV CSV to evaluate past sessions without API calls.")
        st.stop()
    ohlcv_upload_df = _load_ohlcv_from_upload(uploaded_ohlcv_file)
    st.caption(f"📐 Uploaded OHLCV shape after normalization: {ohlcv_upload_df.shape}")
elif price_mode == "sharadar":
    if sharadar_upload is not None:
        sharadar_df = _load_ohlcv_from_upload(sharadar_upload)
    elif sharadar_path and os.path.exists(sharadar_path):
        sharadar_df = _load_ohlcv_from_path(sharadar_path)
    else:
        st.error("Sharadar CSV not found. Provide a valid path or upload a file.")
        st.stop()
    st.caption(f"📐 Sharadar daily rows (after normalization): {sharadar_df.shape}")

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

if not baseline_ranked.empty:
    baseline_cols = baseline_ranked[["ticker", "baseline_prediction", "baseline_rank"]]
    df_view = df_view.merge(baseline_cols, on="ticker", how="left")
else:
    df_view["baseline_prediction"] = np.nan
    df_view["baseline_rank"] = np.nan

df_view["rank_delta_vs_baseline"] = df_view["baseline_rank"] - df_view["rank"]

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
    hist_entry = _historical_daily_bars(symbols_eod_tuple, entry_date)
    hist_exit = hist_entry if exit_date == entry_date else _historical_daily_bars(symbols_eod_tuple, exit_date)
    hist = hist_entry
    st.caption(
        f"📐 Historical EOD bars fetched: entry={hist_entry.shape}, exit={hist_exit.shape if exit_date != entry_date else hist_entry.shape}"
    )
    if hist_entry.empty:
        st.warning("No historical bars returned for the entry session from EODHD.")
    df_view["open_today"] = df_view["eod_symbol"].map(hist_entry["open"] if "open" in hist_entry.columns else pd.Series(dtype=float))
    if effective_horizon == "next_open_to_open" and exit_date != entry_date:
        df_view["last_price"] = df_view["eod_symbol"].map(
            hist_exit["open"] if "open" in hist_exit.columns else pd.Series(dtype=float)
        )
    else:
        close_series = hist_entry["close"] if "close" in hist_entry.columns else pd.Series(dtype=float)
        df_view["last_price"] = df_view["eod_symbol"].map(close_series)

elif price_mode == "sharadar":
    entry_rows_all = sharadar_df[sharadar_df["date"] == entry_date]
    st.caption(f"📐 Sharadar rows for entry {entry_date}: {entry_rows_all.shape}")
    if entry_rows_all.empty:
        st.warning("Sharadar CSV has no rows for the entry session date.")
    entry_rows = entry_rows_all.drop_duplicates(subset=["ticker"]).set_index("ticker")
    df_view["open_today"] = df_view["base_ticker"].map(
        entry_rows["open"] if "open" in entry_rows.columns else pd.Series(dtype=float)
    )

    if effective_horizon == "next_open_to_open" and exit_date != entry_date:
        exit_rows_all = sharadar_df[sharadar_df["date"] == exit_date]
        st.caption(f"📐 Sharadar rows for exit {exit_date}: {exit_rows_all.shape}")
        exit_rows = exit_rows_all.drop_duplicates(subset=["ticker"]).set_index("ticker")
        df_view["last_price"] = df_view["base_ticker"].map(
            exit_rows["open"] if "open" in exit_rows.columns else pd.Series(dtype=float)
        )
    else:
        exit_rows = entry_rows
        close_col = "close" if "close" in exit_rows.columns else None
        if close_col is None and "adj_close" in exit_rows.columns:
            close_col = "adj_close"
        if close_col is None:
            st.error("Sharadar CSV needs a 'close' or 'adj_close' column for returns.")
            st.stop()
        df_view["last_price"] = df_view["base_ticker"].map(exit_rows[close_col])

elif price_mode == "upload":
    session_rows_entry_all = ohlcv_upload_df[ohlcv_upload_df["date"] == entry_date]
    st.caption(f"📐 Uploaded OHLCV rows for entry {entry_date}: {session_rows_entry_all.shape}")
    if session_rows_entry_all.empty:
        st.warning("Uploaded OHLCV file has no rows for the entry session date.")
    session_rows_all = session_rows_entry_all
    session_rows_entry = session_rows_entry_all.drop_duplicates(subset=["ticker"]).set_index("ticker")
    df_view["open_today"] = df_view["base_ticker"].map(
        session_rows_entry["open"] if "open" in session_rows_entry.columns else pd.Series(dtype=float)
    )

    if effective_horizon == "next_open_to_open" and exit_date != entry_date:
        session_rows_exit_all = ohlcv_upload_df[ohlcv_upload_df["date"] == exit_date]
        st.caption(f"📐 Uploaded OHLCV rows for exit {exit_date}: {session_rows_exit_all.shape}")
        session_rows_exit = session_rows_exit_all.drop_duplicates(subset=["ticker"]).set_index("ticker")
        df_view["last_price"] = df_view["base_ticker"].map(
            session_rows_exit["open"] if "open" in session_rows_exit.columns else pd.Series(dtype=float)
        )
    else:
        close_col = "close" if "close" in session_rows_entry.columns else None
        if close_col is None and "adj_close" in session_rows_entry.columns:
            close_col = "adj_close"
        if close_col is None:
            st.error("Uploaded OHLCV file needs a 'close' or 'adj_close' column for returns.")
            st.stop()
        df_view["last_price"] = df_view["base_ticker"].map(session_rows_entry[close_col])

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
def _fmt_rank(x):  return "" if pd.isna(x) else f"{int(x)}"
def _fmt_delta(x):
    if pd.isna(x):
        return ""
    return f"{int(x):+d}"

def _fmt_volume(x):
    if pd.isna(x):
        return ""
    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}K"
    return f"{x:.0f}"

def _fmt_pct_display(x: float) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.2f}%"

def _fmt_delta_points(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.1f} pts"

display_cols = [
    "rank",
    "baseline_rank",
    "rank_delta_vs_baseline",
    "ticker",
    "prediction",
    "baseline_prediction",
    "open_today",
    "last_price",
    "return_open_to_now",
]
show = df_view[display_cols].copy()
show["rank"] = show["rank"].apply(_fmt_rank)
show["baseline_rank"] = show["baseline_rank"].apply(_fmt_rank)
show["rank_delta_vs_baseline"] = show["rank_delta_vs_baseline"].apply(_fmt_delta)
show["open_today"] = show["open_today"].apply(_fmt_price)
show["last_price"] = show["last_price"].apply(_fmt_price)
show["return_open_to_now"] = show["return_open_to_now"].apply(_fmt_pct)
show["baseline_prediction"] = show["baseline_prediction"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")

returns_ready = df_view[df_view["return_open_to_now"].notna()].copy()
returns_ready_sorted = returns_ready.sort_values("rank") if not returns_ready.empty else returns_ready
avg_return_all = float(returns_ready_sorted["return_open_to_now"].mean()) if not returns_ready_sorted.empty else np.nan
benchmark_return = avg_return_all

chart_data = (
    df_view[["rank", "ticker", "prediction", "return_open_to_now"]]
    .dropna(subset=["return_open_to_now"])
    .copy()
)
if not chart_data.empty:
    if pd.notna(benchmark_return):
        outperform_label = "Outperform (≥ benchmark)"
        underperform_label = "Underperform (< benchmark)"
        chart_data["return_direction"] = np.where(
            chart_data["return_open_to_now"] >= benchmark_return,
            outperform_label,
            underperform_label,
        )
    else:
        outperform_label = "Positive return"
        underperform_label = "Negative return"
        chart_data["return_direction"] = np.where(
            chart_data["return_open_to_now"] >= 0,
            outperform_label,
            underperform_label,
        )
    chart_data.sort_values("rank", inplace=True)


top_k_count = int(min(int(summary_topk), len(returns_ready_sorted))) if len(returns_ready_sorted) else 0
bottom_k_count = int(min(int(summary_bottomk), len(returns_ready_sorted))) if len(returns_ready_sorted) else 0

top_slice = returns_ready_sorted.head(top_k_count) if top_k_count else pd.DataFrame(columns=returns_ready.columns)
bottom_slice = returns_ready_sorted.tail(bottom_k_count) if bottom_k_count else pd.DataFrame(columns=returns_ready.columns)

top_mean = float(top_slice["return_open_to_now"].mean()) if not top_slice.empty else np.nan
bottom_short_mean = (
    float((-bottom_slice["return_open_to_now"]).mean()) if not bottom_slice.empty else np.nan
)
if not top_slice.empty and not bottom_slice.empty:
    long_leg_sum = float(top_slice["return_open_to_now"].sum())
    short_leg_sum = float((-bottom_slice["return_open_to_now"]).sum())
    total_positions = top_slice.shape[0] + bottom_slice.shape[0]
    long_short_spread = (long_leg_sum + short_leg_sum) / total_positions if total_positions else np.nan
else:
    long_short_spread = np.nan
def _long_hit_mask(returns: pd.Series, benchmark: float) -> pd.Series:
    if pd.isna(benchmark):
        return returns > 0
    return returns > benchmark

def _short_hit_mask(returns: pd.Series, benchmark: float) -> pd.Series:
    if pd.isna(benchmark):
        return returns < 0
    return returns < benchmark

top_win_rate = (
    float(_long_hit_mask(top_slice["return_open_to_now"], benchmark_return).mean())
    if not top_slice.empty
    else np.nan
)
bottom_win_rate = (
    float(_short_hit_mask(bottom_slice["return_open_to_now"], benchmark_return).mean())
    if not bottom_slice.empty
    else np.nan
)

topk_depth_returns = pd.DataFrame()
topk_depth_hits = pd.DataFrame()
if not returns_ready_sorted.empty:
    long_curve = returns_ready_sorted[["return_open_to_now"]].reset_index(drop=True)
    long_curve["top_k"] = np.arange(1, len(long_curve) + 1)
    long_curve["mean_return"] = long_curve["return_open_to_now"].expanding().mean()
    long_curve["hit_rate"] = (
        _long_hit_mask(long_curve["return_open_to_now"], benchmark_return).expanding().mean()
    )

    short_curve = (
        returns_ready_sorted.sort_values("rank", ascending=False)[
            ["return_open_to_now"]
        ]
        .reset_index(drop=True)
    )
    short_curve["top_k"] = np.arange(1, len(short_curve) + 1)
    short_curve["mean_return"] = short_curve["return_open_to_now"].expanding().mean()
    short_curve["hit_rate"] = (
        _short_hit_mask(short_curve["return_open_to_now"], benchmark_return).expanding().mean()
    )

    short_curve["short_mean_pnl"] = -short_curve["mean_return"]
    long_short_spread_curve = 0.5 * (
        long_curve["mean_return"].values + short_curve["short_mean_pnl"].values
    )

    topk_depth_returns = pd.concat(
        [
            pd.DataFrame(
                {
                    "top_k": long_curve["top_k"],
                    "series": "Long cumulative avg return",
                    "value": long_curve["mean_return"],
                }
            ),
            pd.DataFrame(
                {
                    "top_k": short_curve["top_k"],
                    "series": "Short cumulative avg return (short P&L)",
                    "value": short_curve["short_mean_pnl"],
                }
            ),
            pd.DataFrame(
                {
                    "top_k": long_curve["top_k"],
                    "series": "Equal-weight long–short spread (avg)",
                    "value": long_short_spread_curve,
                }
            ),
        ],
        ignore_index=True,
    )

    topk_depth_hits = pd.concat(
        [
            pd.DataFrame(
                {
                    "top_k": long_curve["top_k"],
                    "series": "Long hit rate (> equal-weight benchmark)",
                    "value": long_curve["hit_rate"],
                }
            ),
            pd.DataFrame(
                {
                    "top_k": short_curve["top_k"],
                    "series": "Short hit rate (< equal-weight benchmark)",
                    "value": short_curve["hit_rate"],
                }
            ),
        ],
        ignore_index=True,
    )


if price_mode == "live":
    open_label = "Open (today)"
    last_label = "Last price"
    return_label = "Return (open→now)"
else:
    if effective_horizon == "same_session":
        open_label = "Open"
        last_label = "Close"
        return_label = "Return (open→close)"
    elif effective_horizon == "next_open_to_close":
        open_label = "Next open"
        last_label = "Next close"
        return_label = "Return (next open→next close)"
    elif effective_horizon == "next_open_to_open":
        open_label = "Next open"
        last_label = "Following open"
        return_label = "Return (next open→following open)"
    else:
        open_label = "Open"
        last_label = "Last price"
        return_label = "Return"

show = show.rename(columns={
    "rank": "Rank",
    "open_today": open_label,
    "last_price": last_label,
    "return_open_to_now": return_label,
    "baseline_rank": "Baseline rank",
    "rank_delta_vs_baseline": "Δ rank (baseline→now)",
    "baseline_prediction": "Baseline prediction",
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

def _compute_rank_metrics(
    df: pd.DataFrame, topk: int, winsor_pct: int, benchmark: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute robust rank↔return metrics; returns (metrics_table, deciles_table)."""
    ready = df[["rank", "prediction", "return_open_to_now"]].dropna().copy()
    st.caption(f"📐 Metrics-ready rows (after NA drop): {ready.shape}")

    if ready.empty:
        return pd.DataFrame(columns=["Focus", "Metric", "Value"]), pd.DataFrame(columns=["decile", "median_return"])

    # Use negative rank so "higher is better" for correlation orientation
    ready["neg_rank"] = -ready["rank"].astype(float)
    returns_numeric = pd.to_numeric(ready["return_open_to_now"], errors="coerce")
    if pd.notna(benchmark):
        ready["relevance"] = (returns_numeric - benchmark).clip(lower=0)
    else:
        ready["relevance"] = returns_numeric.clip(lower=0)

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
    bot_med_raw = float(bot["return_open_to_now"].median()) if not bot.empty else np.nan
    bot_med_short = -bot_med_raw if pd.notna(bot_med_raw) else np.nan
    l_s_med = (
        top_med + bot_med_short
        if pd.notna(top_med) and pd.notna(bot_med_short)
        else np.nan
    )
    top_win = (
        float(_long_hit_mask(top["return_open_to_now"], benchmark).mean())
        if not top.empty
        else np.nan
    )
    bot_win = (
        float(_short_hit_mask(bot["return_open_to_now"], benchmark).mean())
        if not bot.empty
        else np.nan
    )

    # Ranking metrics @K (Top-K by predicted rank)
    ready_sorted = ready.sort_values("rank")
    k_rank = int(max(1, min(topk, ready_sorted.shape[0])))
    rel = ready_sorted["relevance"].to_numpy()[:k_rank]
    if k_rank > 0:
        discounts = 1.0 / np.log2(np.arange(2, k_rank + 2))
        gains = (np.power(2.0, rel) - 1.0) * discounts
        dcg = float(np.sum(gains))
        ideal_rel = np.sort(ready_sorted["relevance"].to_numpy())[::-1][:k_rank]
        ideal_gains = (np.power(2.0, ideal_rel) - 1.0) * discounts
        ideal_dcg = float(np.sum(ideal_gains))
        ndcg_k = float(dcg / ideal_dcg) if ideal_dcg > 0 else np.nan

        hits = _long_hit_mask(
            pd.Series(ready_sorted["return_open_to_now"].to_numpy()[:k_rank]), benchmark
        ).to_numpy()
        precision_k = float(hits.mean()) if k_rank > 0 else np.nan
        cum_hits = np.cumsum(hits)
        precisions = cum_hits / (np.arange(1, k_rank + 1))
        relevant_total = int(hits.sum())
        map_k = float(np.sum(precisions * hits) / relevant_total) if relevant_total > 0 else 0.0
    else:
        ndcg_k = np.nan
        precision_k = np.nan
        map_k = np.nan

    metrics_rows = [
        ("All ranks", "Spearman ρ (ranks ↔ returns)", f"{spearman:.3f}" if pd.notna(spearman) else "—", "↑ toward +1"),
        ("All ranks", "Kendall τ-b (ranks ↔ returns)", f"{kendall:.3f}" if pd.notna(kendall) else "—", "↑ toward +1"),
        ("All ranks", f"Winsorized Pearson r (tails={winsor_pct}%)", f"{pearson_w:.3f}" if pd.notna(pearson_w) else "—", "↑ toward +1"),
        ("All ranks", "Theil–Sen slope (return per rank)", f"{ts_slope:.6f}" if pd.notna(ts_slope) else "—", "↑ more positive"),
        ("All ranks", "≈ bps per 10-rank improvement", f"{bps_per_10rank:.1f}" if pd.notna(bps_per_10rank) else "—", "↑ more positive"),
        ("Long bucket", f"Top-{k} median return", f"{top_med*100:.2f}%" if pd.notna(top_med) else "—", "↑ more positive"),
        (
            "Short bucket",
            f"Bottom-{k} median short P&L",
            f"{bot_med_short*100:.2f}%" if pd.notna(bot_med_short) else "—",
            "↑ more positive",
        ),
        ("Spread", f"Median long–short (Top-{k} + Short-{k})", f"{l_s_med*100:.2f}%" if pd.notna(l_s_med) else "—", "↑ wider"),
        (
            "Long bucket",
            f"Top-{k} win rate (> benchmark)",
            f"{top_win*100:.1f}%" if pd.notna(top_win) else "—",
            "↑ toward 100%",
        ),
        (
            "Short bucket",
            f"Bottom-{k} win rate (< benchmark)",
            f"{bot_win*100:.1f}%" if pd.notna(bot_win) else "—",
            "↑ toward 100%",
        ),
        ("Top-K", f"NDCG@{k_rank} (relevance from returns)", f"{ndcg_k:.3f}" if pd.notna(ndcg_k) else "—", "↑ toward 1"),
        (
            "Top-K",
            f"MAP@{k_rank} (> benchmark returns as relevant)",
            f"{map_k:.3f}" if pd.notna(map_k) else "—",
            "↑ toward 1",
        ),
        (
            "Top-K",
            f"Precision@{k_rank} (> benchmark returns)",
            f"{precision_k*100:.1f}%" if pd.notna(precision_k) else "—",
            "↑ toward 100%",
        ),
    ]
    metrics_tbl = pd.DataFrame(metrics_rows, columns=["Focus", "Metric", "Value", "Better ↗︎"])

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

metrics_tbl, deciles_tbl = _compute_rank_metrics(
    df_view, metrics_topk, winsor_tail_pct, benchmark_return
)

deciles_chart_data = pd.DataFrame()
if deciles_tbl is not None and not deciles_tbl.empty:
    deciles_chart_data = deciles_tbl.copy()
    decile_label = deciles_chart_data.columns[0]
    deciles_chart_data = deciles_chart_data.rename(columns={decile_label: "rank_decile"})
    deciles_chart_data["rank_decile"] = deciles_chart_data["rank_decile"].astype(str)

baseline_overlap_tbl = pd.DataFrame()
baseline_corr_tbl = pd.DataFrame()
baseline_shift_up = pd.DataFrame()
baseline_shift_down = pd.DataFrame()
if not baseline_ranked.empty:
    long_main = preds_ranked["ticker"].tolist()
    long_baseline = baseline_ranked.head(int(top_n))["ticker"].tolist()
    long_overlap = len(set(long_main) & set(long_baseline))
    long_pct = long_overlap / max(1, len(long_main))

    short_cut = int(min(int(top_n), len(preds))) if len(preds) else 0
    short_baseline_cut = int(min(int(top_n), len(baseline_ranked))) if len(baseline_ranked) else 0
    short_main_df = preds.sort_values("prediction", ascending=True).head(short_cut)
    short_baseline_df = baseline_ranked.sort_values("baseline_prediction", ascending=True).head(short_baseline_cut)
    short_main = short_main_df["ticker"].astype(str).str.upper().tolist()
    short_baseline = short_baseline_df["ticker"].tolist()
    short_overlap = len(set(short_main) & set(short_baseline))
    short_pct = short_overlap / max(1, len(short_main))

    baseline_overlap_rows = [
        ("Long", "Main Top-N size", f"{len(long_main):,}"),
        ("Long", "Baseline Top-N size", f"{len(long_baseline):,}"),
        ("Long", "Overlap count", f"{long_overlap:,}"),
        ("Long", "Overlap % of main", f"{long_pct*100:.1f}%"),
        ("Short", "Main Bottom-N size", f"{len(short_main):,}"),
        ("Short", "Baseline Bottom-N size", f"{len(short_baseline):,}"),
        ("Short", "Overlap count", f"{short_overlap:,}"),
        ("Short", "Overlap % of main", f"{short_pct*100:.1f}%"),
    ]
    baseline_overlap_tbl = pd.DataFrame(baseline_overlap_rows, columns=["Focus", "Metric", "Value"])

    merged_full = preds_ranked_full.merge(
        baseline_ranked[["ticker", "baseline_rank", "baseline_prediction"]],
        on="ticker",
        how="inner",
    )
    merged_full = merged_full.rename(columns={"full_rank": "rank"})
    if merged_full.shape[0] >= 2:
        rank_spearman = _spearman_corr(merged_full["rank"], merged_full["baseline_rank"])
        rank_kendall = _kendall_tau_b(merged_full["rank"], merged_full["baseline_rank"])
        pred_pearson = _pearson_corr(merged_full["prediction"], merged_full["baseline_prediction"])
    else:
        rank_spearman = np.nan
        rank_kendall = np.nan
        pred_pearson = np.nan

    baseline_corr_rows = [
        ("Ranks", "Spearman ρ", f"{rank_spearman:.3f}" if pd.notna(rank_spearman) else "—"),
        ("Ranks", "Kendall τ-b", f"{rank_kendall:.3f}" if pd.notna(rank_kendall) else "—"),
        ("Predictions", "Pearson r", f"{pred_pearson:.3f}" if pd.notna(pred_pearson) else "—"),
    ]
    baseline_corr_tbl = pd.DataFrame(baseline_corr_rows, columns=["Focus", "Metric", "Value"])

    shift_scope = df_view[df_view["baseline_rank"].notna()].copy()
    if not shift_scope.empty:
        shift_scope["rank_delta_vs_baseline"] = shift_scope["rank_delta_vs_baseline"].astype(float)
        baseline_shift_up = (
            shift_scope.sort_values("rank_delta_vs_baseline", ascending=False)
            .head(5)
            [["ticker", "rank", "baseline_rank", "rank_delta_vs_baseline", "prediction", "baseline_prediction"]]
        )
        baseline_shift_up = baseline_shift_up[baseline_shift_up["rank_delta_vs_baseline"] > 0]
        baseline_shift_down = (
            shift_scope.sort_values("rank_delta_vs_baseline", ascending=True)
            .head(5)
            [["ticker", "rank", "baseline_rank", "rank_delta_vs_baseline", "prediction", "baseline_prediction"]]
        )
        baseline_shift_down = baseline_shift_down[baseline_shift_down["rank_delta_vs_baseline"] < 0]

    def _format_shift(tbl: pd.DataFrame) -> pd.DataFrame:
        if tbl is None or tbl.empty:
            return pd.DataFrame()
        out = tbl.copy()
        out["rank"] = out["rank"].astype(int)
        out["baseline_rank"] = out["baseline_rank"].astype(int)
        out["rank_delta_vs_baseline"] = out["rank_delta_vs_baseline"].astype(int)
        out["prediction"] = out["prediction"].apply(lambda x: f"{float(x):.6f}")
        out["baseline_prediction"] = out["baseline_prediction"].apply(lambda x: f"{float(x):.6f}")
        return out.rename(
            columns={
                "rank": "Now rank",
                "baseline_rank": "Baseline rank",
                "rank_delta_vs_baseline": "Δ rank",
                "prediction": "Now pred",
                "baseline_prediction": "Baseline pred",
            }
        )

    baseline_shift_up = _format_shift(baseline_shift_up)
    baseline_shift_down = _format_shift(baseline_shift_down)


def _compute_recent_prediction_summaries(
    preds_all: pd.DataFrame,
    prices_daily: Optional[pd.DataFrame],
    horizon: str,
    *,
    max_days: int = 5,
    k_values: Tuple[int, ...] = (1, 2, 3),
    today_cutoff: Optional[date] = None,
) -> pd.DataFrame:
    """Summarize recent prediction sessions with compact long/short metrics."""

    if (
        preds_all is None
        or preds_all.empty
        or "date" not in preds_all.columns
        or prices_daily is None
        or prices_daily.empty
        or "open" not in prices_daily.columns
    ):
        return pd.DataFrame()

    preds_hist = preds_all.dropna(subset=["date", "prediction", "ticker"]).copy()
    if preds_hist.empty:
        return pd.DataFrame()

    preds_hist["date"] = pd.to_datetime(preds_hist["date"]).dt.date
    preds_hist["ticker"] = preds_hist["ticker"].astype(str).str.upper().str.strip()

    price_hist = prices_daily.copy()
    price_hist["date"] = pd.to_datetime(price_hist["date"]).dt.date
    price_hist["ticker"] = price_hist["ticker"].astype(str).str.upper().str.strip()

    unique_dates = sorted(preds_hist["date"].dropna().unique(), reverse=True)
    rows = []

    for pred_date in unique_dates:
        entry_date = pred_date if horizon == "same_session" else _next_weekday(pred_date)
        exit_date = entry_date
        if horizon == "next_open_to_open":
            exit_date = _next_weekday(entry_date)

        if today_cutoff is not None and (
            entry_date > today_cutoff or exit_date > today_cutoff
        ):
            continue

        entry_prices_df = price_hist[price_hist["date"] == entry_date]
        if entry_prices_df.empty:
            continue

        entry_open = entry_prices_df.dropna(subset=["open"]).set_index("ticker")["open"]
        if entry_open.empty:
            continue

        if horizon == "next_open_to_open":
            exit_prices_df = price_hist[price_hist["date"] == exit_date]
            if exit_prices_df.empty or "open" not in exit_prices_df.columns:
                continue
            exit_series = exit_prices_df.dropna(subset=["open"]).set_index("ticker")["open"]
        else:
            if exit_date == entry_date:
                exit_prices_df = entry_prices_df
            else:
                exit_prices_df = price_hist[price_hist["date"] == exit_date]
            if exit_prices_df.empty:
                continue
            close_col = "close" if "close" in exit_prices_df.columns else None
            if close_col is None and "adj_close" in exit_prices_df.columns:
                close_col = "adj_close"
            if close_col is None:
                continue
            exit_series = exit_prices_df.dropna(subset=[close_col]).set_index("ticker")[close_col]

        combined = (
            pd.DataFrame({"open": entry_open})
            .join(exit_series.rename("exit"), how="inner")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if combined.empty:
            continue

        combined["return"] = combined["exit"] / combined["open"] - 1.0

        preds_day = preds_hist[preds_hist["date"] == pred_date].copy()
        preds_day = preds_day.merge(
            combined[["return"]], left_on="ticker", right_index=True, how="left"
        )
        if preds_day.empty:
            continue

        preds_day = preds_day.sort_values("prediction", ascending=False).reset_index(drop=True)
        preds_day["rank"] = np.arange(1, len(preds_day) + 1)

        realized = preds_day[preds_day["return"].notna()].copy()
        if realized.empty:
            continue

        row = {
            "prediction_date": pred_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "realized_count": int(realized.shape[0]),
            "rank_ic": _spearman_corr(preds_day["prediction"], preds_day["return"]),
        }

        bottom_sorted = preds_day.sort_values("prediction", ascending=True).reset_index(drop=True)

        for k in k_values:
            top_slice = preds_day.head(int(k))
            top_slice = top_slice[top_slice["return"].notna()]
            long_mean = float(top_slice["return"].mean()) if not top_slice.empty else np.nan

            bottom_slice = bottom_sorted.head(int(k))
            bottom_slice = bottom_slice[bottom_slice["return"].notna()]
            short_mean = float(bottom_slice["return"].mean()) if not bottom_slice.empty else np.nan

            short_pnl = float(-short_mean) if pd.notna(short_mean) else np.nan
            long_short = (
                float(long_mean - short_mean)
                if pd.notna(long_mean) and pd.notna(short_mean)
                else np.nan
            )

            row[f"top{k}_long"] = long_mean
            row[f"top{k}_short"] = short_pnl
            row[f"top{k}_ls"] = long_short

        rows.append(row)

        if len(rows) >= max_days:
            break

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values("entry_date", ascending=False).reset_index(drop=True)
    return out


# -----------------------
# UI
# -----------------------

price_mode_label_map = {
    "live": "HTTP + rotating WS" if ws_enabled else "HTTP (delayed only)",
    "historical_api": "Historical daily bars (EODHD API)",
    "sharadar": "Sharadar daily CSV",
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
elif price_mode == "sharadar":
    sharadar_shape = sharadar_df.shape if isinstance(sharadar_df, pd.DataFrame) else (0, 0)
    price_dim_lines.append(f"sharadar_rows: {sharadar_shape}")
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

horizon_readable = horizon_options.get(effective_horizon, effective_horizon)
trade_caption = (
    f"Prediction date: **{prediction_date}**, trade session: **{session_date}** ({horizon_readable})."
)
if effective_horizon == "next_open_to_open" and exit_date != entry_date:
    trade_caption += f" Exit session: **{exit_date}**."
st.caption(trade_caption)

if price_mode in {"historical_api", "upload", "sharadar"}:
    horizon_copy = {
        "same_session": "same-session open → close",
        "next_open_to_close": "next-session open → close",
        "next_open_to_open": "next-session open → following open",
    }
    st.caption(
        f"Historical mode: returns use {horizon_copy.get(effective_horizon, 'the selected horizon')} (no WebSocket refresh required)."
    )
elif price_mode == "future":
    st.caption("Future session selected — waiting for market data to arrive.")

if not returns_ready_sorted.empty:
    st.subheader("🚦 Performance snapshot")
    summary_cols = st.columns(4, gap="large")
    with summary_cols[0]:
        st.metric("Equal-weight benchmark return", _fmt_pct_display(benchmark_return))
    with summary_cols[1]:
        st.metric(
            f"Top-{top_k_count or summary_topk} mean",
            _fmt_pct_display(top_mean),
        )
    with summary_cols[2]:
        st.metric(
            f"Bottom-{bottom_k_count or summary_bottomk} mean short P&L",
            _fmt_pct_display(bottom_short_mean),
        )
    with summary_cols[3]:
        st.metric("Equal-weight long–short spread", _fmt_pct_display(long_short_spread))

    rate_cols = st.columns(2, gap="large")
    with rate_cols[0]:
        st.metric(
            f"Top-{top_k_count or summary_topk} hit rate vs benchmark",
            _fmt_pct_display(top_win_rate),
            delta=_fmt_delta_points(top_win_rate - 0.5) if pd.notna(top_win_rate) else None,
        )
    with rate_cols[1]:
        st.metric(
            f"Bottom-{bottom_k_count or summary_bottomk} short hit rate vs benchmark",
            _fmt_pct_display(bottom_win_rate),
            delta=_fmt_delta_points(bottom_win_rate - 0.5) if pd.notna(bottom_win_rate) else None,
        )

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Portfolio build-up by rank")
    st.caption(
        "Shows how cumulative average returns evolve as you add more names from the ranked list. "
        "The short line converts negative returns into short-side P&L."
    )
    if not topk_depth_returns.empty:
        returns_chart = (
            alt.Chart(topk_depth_returns)
            .mark_line(point=alt.OverlayMarkDef(size=60, filled=True))
            .encode(
                x=alt.X("top_k:Q", title="Portfolio size (K)"),
                y=alt.Y(
                    "value:Q",
                    title="Cumulative avg return",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color(
                    "series:N",
                    title="",
                    scale=alt.Scale(
                        domain=[
                            "Long cumulative avg return",
                            "Short cumulative avg return (short P&L)",
                            "Equal-weight long–short spread (avg)",
                        ],
                        range=["#1abc9c", "#e67e22", "#f1c40f"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("series:N", title="Metric"),
                    alt.Tooltip("top_k:Q", title="K size"),
                    alt.Tooltip("value:Q", title="Value", format=".2%"),
                ],
            )
        )
        st.altair_chart(returns_chart.interactive(), use_container_width=True)

        hits_chart = (
            alt.Chart(topk_depth_hits)
            .mark_line(point=alt.OverlayMarkDef(size=50, filled=True), strokeDash=[4, 4])
            .encode(
                x=alt.X("top_k:Q", title="Portfolio size (K)"),
                y=alt.Y(
                    "value:Q",
                    title="Cumulative hit rate",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color(
                    "series:N",
                    title="",
                    scale=alt.Scale(
                        domain=[
                            "Long hit rate (> equal-weight benchmark)",
                            "Short hit rate (< equal-weight benchmark)",
                        ],
                        range=["#2980b9", "#8e44ad"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("series:N", title="Metric"),
                    alt.Tooltip("top_k:Q", title="K size"),
                    alt.Tooltip("value:Q", title="Value", format=".2%"),
                ],
            )
        )
        st.altair_chart(hits_chart.interactive(), use_container_width=True)
    else:
        st.info("Performance lines populate once realized returns are available.")

    if price_mode == "live":
        table_phrase = "returns from open → latest"
    elif price_mode in {"historical_api", "upload"}:
        table_phrase = "returns from open → close"
    else:
        table_phrase = "returns (pending data)"
    st.subheader(f"Rank vs realized returns ({table_phrase})")
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
                    title="Return vs benchmark" if pd.notna(benchmark_return) else "Return direction",
                    scale=alt.Scale(
                        domain=[outperform_label, underperform_label],
                        range=["#2ecc71", "#e74c3c"],
                    ),
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
        layers = scatter + rolling
        if pd.notna(benchmark_return):
            benchmark_line = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "label": ["Equal-weight benchmark"],
                            "benchmark": [benchmark_return],
                        }
                    )
                )
                .mark_rule(color="#f1c40f", strokeDash=[6, 4])
                .encode(
                    y=alt.Y("benchmark:Q"),
                    tooltip=[
                        alt.Tooltip("label:N", title=""),
                        alt.Tooltip("benchmark:Q", title="Benchmark", format=".2%"),
                    ],
                )
            )
            layers = layers + benchmark_line

        st.altair_chart(layers.interactive(), use_container_width=True)
    else:
        st.info("Waiting for realized returns to plot.")
    st.subheader("Ranked predictions table")
    st.dataframe(show, use_container_width=True, height=650)

with right:
    st.subheader("Summary")
    st.metric("Session date (ET)", str(session_date))
    st.metric("Symbols shown", f"{show.shape[0]:,}")
    st.metric("Symbols with realized returns", f"{len(returns_ready_sorted):,}")
    if not returns_ready_sorted.empty:
        st.metric("Equal-weight benchmark return", _fmt_pct_display(benchmark_return))
        st.metric(
            f"Top-{top_k_count or summary_topk} mean return",
            _fmt_pct_display(top_mean),
        )
        st.metric(
            f"Bottom-{bottom_k_count or summary_bottomk} mean short P&L",
            _fmt_pct_display(bottom_short_mean),
        )
        st.metric("Equal-weight long–short spread", _fmt_pct_display(long_short_spread))
        st.metric(
            f"Top-{top_k_count or summary_topk} hit rate vs benchmark",
            _fmt_pct_display(top_win_rate),
        )
        st.metric(
            f"Bottom-{bottom_k_count or summary_bottomk} short hit rate vs benchmark",
            _fmt_pct_display(bottom_win_rate),
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
        st.caption(
            "Focus splits long/short/aggregate views; **Better ↗︎** is a quick tooltip on which way you want each metric to move."
        )
        st.dataframe(
            metrics_tbl,
            use_container_width=True,
            height=330,
            column_config={
                "Better ↗︎": st.column_config.TextColumn(
                    "Better ↗︎",
                    help="Directionally helpful move for that metric (e.g., higher, more negative).",
                    disabled=True,
                )
            },
        )
        with st.expander("How to read these numbers", expanded=False):
            st.markdown(
                """
                **Correlations (Spearman/Kendall/Pearson)** — +1 is perfect rank↔return alignment, 0 is random, −1 is inverted.
                Daily open→now data is noisy, so sustained readings above ~0.20 usually indicate meaningful skill.

                **Slope & bps/10 ranks** — positive values mean returns improve as rank gets better. Values near zero imply
                little payoff for reordering; +10 bps per 10 ranks means a 10-place improvement is worth ~0.10%.

                **Bucket medians & win rates** — long medians/win rates should trend positive and short medians translate
                into positive short P&L (i.e., underlying returns negative). A long–short spread above 0 and win rates
                above 55–60% are typical of a healthy day.

                **Ranking scores (NDCG/MAP/Precision)** — all live in [0, 1]. Higher means more positive-return ideas surfaced
                near the top. 0.5 is “coin flip” behaviour; >0.7 usually reflects strong signal for the current session.
                """
            )
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

if not baseline_ranked.empty:
    st.write("---")
    base_left, base_right = st.columns([2, 3], gap="large")
    with base_left:
        st.subheader("Baseline comparison")
        st.caption("How the current ranking differs from the reference model.")
        if not baseline_overlap_tbl.empty:
            st.markdown("**Overlap & coverage**")
            st.dataframe(baseline_overlap_tbl, use_container_width=True, height=240)
        if not baseline_corr_tbl.empty:
            st.markdown("**Rank / score correlations**")
            st.dataframe(baseline_corr_tbl, use_container_width=True, height=150)
    with base_right:
        st.subheader("Largest rank moves vs baseline")
        shifts_up_col, shifts_down_col = st.columns(2, gap="medium")
        with shifts_up_col:
            st.markdown("⬆️ Improved placement")
            if not baseline_shift_up.empty:
                st.dataframe(baseline_shift_up, use_container_width=True, height=230)
            else:
                st.info("No overlapping symbols yet.")
        with shifts_down_col:
            st.markdown("⬇️ Dropped placement")
            if not baseline_shift_down.empty:
                st.dataframe(baseline_shift_down, use_container_width=True, height=230)
            else:
                st.info("No overlapping symbols yet.")

history_source = None
if price_mode == "sharadar" and isinstance(sharadar_df, pd.DataFrame):
    history_source = sharadar_df.copy()
elif price_mode == "upload" and isinstance(ohlcv_upload_df, pd.DataFrame):
    history_source = ohlcv_upload_df.copy()

financial_history_tbl = pd.DataFrame()
if history_source is not None and not history_source.empty:
    tickers_scope = df_view["base_ticker"].dropna().unique().tolist()
    history_scope = history_source[history_source["ticker"].isin(tickers_scope)].copy()
    if not history_scope.empty and "date" in history_scope.columns:
        history_scope["date"] = pd.to_datetime(history_scope["date"]).dt.date
        past_scope = history_scope[history_scope["date"] <= entry_date]
        unique_dates = sorted(past_scope["date"].unique())
        lookback = unique_dates[-5:] if len(unique_dates) > 5 else unique_dates
        rows = []
        for dt in lookback:
            day = past_scope[past_scope["date"] == dt]
            if day.empty:
                continue
            open_mean = day["open"].mean() if "open" in day.columns else np.nan
            close_col = "close" if "close" in day.columns else ("adj_close" if "adj_close" in day.columns else None)
            close_mean = day[close_col].mean() if close_col else np.nan
            volume_mean = day["volume"].mean() if "volume" in day.columns else np.nan
            if close_col and "open" in day.columns:
                returns = day[close_col] / day["open"] - 1.0
                returns = returns.replace([np.inf, -np.inf], np.nan)
                avg_ret = returns.mean()
            else:
                avg_ret = np.nan
            rows.append({
                "date": dt,
                "avg_open": float(open_mean) if pd.notna(open_mean) else np.nan,
                "avg_close": float(close_mean) if pd.notna(close_mean) else np.nan,
                "avg_volume": float(volume_mean) if pd.notna(volume_mean) else np.nan,
                "avg_open_close_return": float(avg_ret) if pd.notna(avg_ret) else np.nan,
            })
        financial_history_tbl = pd.DataFrame(rows)

recent_prediction_metrics = _compute_recent_prediction_summaries(
    preds_full_history,
    history_source,
    effective_horizon,
    max_days=5,
    today_cutoff=today_ny,
)

if not financial_history_tbl.empty or not recent_prediction_metrics.empty:
    st.write("---")
    with st.expander("📊 Recent daily metrics (Sharadar / uploaded data)", expanded=False):
        st.caption(
            "Five most recent sessions for the tickers in view, plus realized performance for recent prediction dates."
        )
        if not financial_history_tbl.empty:
            history_display = financial_history_tbl.sort_values("date", ascending=False).copy()
            history_display["Date"] = history_display["date"].apply(lambda d: d.strftime("%Y-%m-%d"))
            history_display["Avg open"] = history_display["avg_open"].apply(_fmt_price)
            history_display["Avg close"] = history_display["avg_close"].apply(_fmt_price)
            history_display["Avg volume"] = history_display["avg_volume"].apply(_fmt_volume)
            history_display["Avg open→close"] = history_display["avg_open_close_return"].apply(_fmt_pct)
            cols_order = ["Date", "Avg open", "Avg close", "Avg open→close", "Avg volume"]
            st.markdown("**Price overview**")
            st.dataframe(history_display[cols_order], use_container_width=True, height=220)
        else:
            st.info("Price overview unavailable (no Sharadar/uploaded daily history for recent sessions).")

        if not recent_prediction_metrics.empty:
            recent_display = recent_prediction_metrics.copy()
            recent_display["Prediction"] = recent_display["prediction_date"].apply(
                lambda d: d.strftime("%Y-%m-%d") if isinstance(d, date) else str(d)
            )
            recent_display["Session"] = recent_display["entry_date"].apply(
                lambda d: d.strftime("%Y-%m-%d") if isinstance(d, date) else str(d)
            )
            recent_display["Exit"] = recent_display["exit_date"].apply(
                lambda d: d.strftime("%Y-%m-%d") if isinstance(d, date) else str(d)
            )
            recent_display["Realized #"] = recent_display["realized_count"].apply(lambda x: f"{int(x):,}")
            recent_display["Rank IC (Spearman)"] = recent_display["rank_ic"].apply(
                lambda x: "—" if pd.isna(x) else f"{x:.3f}"
            )
            for k in (1, 2, 3):
                recent_display[f"Top{k} long"] = recent_display[f"top{k}_long"].apply(_fmt_pct_display)
                recent_display[f"Top{k} short"] = recent_display[f"top{k}_short"].apply(_fmt_pct_display)
                recent_display[f"Top{k} L-S"] = recent_display[f"top{k}_ls"].apply(_fmt_pct_display)

            cols = [
                "Prediction",
                "Session",
                "Exit",
                "Realized #",
                "Rank IC (Spearman)",
                "Top1 long",
                "Top1 short",
                "Top1 L-S",
                "Top2 long",
                "Top2 short",
                "Top2 L-S",
                "Top3 long",
                "Top3 short",
                "Top3 L-S",
            ]
            st.markdown("**Prediction performance detail**")
            st.dataframe(recent_display[cols], use_container_width=True, height=260)
        else:
            st.info(
                "Prediction performance detail unavailable. Provide a daily OHLCV history that covers recent prediction dates."
            )

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
