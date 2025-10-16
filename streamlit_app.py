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
# - Live v2 (US extended quotes): endpoint /api/us-quote-delayed, s= batching, fields (open, lastTradePrice), pagination page[limit]≤100, ~1 API call / ticker.  [EODHD "Live v2 for US Stocks: Extended Quotes (2025)"]
# - WebSockets: wss://ws.eodhistoricaldata.com/ws/us?api_token=..., subscribe/unsubscribe JSON; ~50 symbols/connection by default; WS does NOT consume API calls.  [EODHD "Real-Time Data API via WebSockets"]
# - EOD historical: /api/eod/{SYMBOL}?from=&to=&fmt=json, one call covers any range; US EOD publishing often ~2–3 hours after close.  [EODHD "End-of-Day Historical Stock Market Data API"]
# - Bulk last day (not defaulted due to cost): /api/eod-bulk-last-day/US?date=YYYY-MM-DD.  [EODHD "Bulk API for EOD, Splits and Dividends"]
# -----------------------------------------------------------------------------------

from __future__ import annotations

import os
import io
import json
import time
from datetime import datetime, date, time as dt_time, timedelta
from typing import List, Dict, Tuple, Optional, Iterable, Sequence

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
DEFAULT_WS_WINDOW = 50     # EODHD WS: ~50 concurrent symbols per connection by default
DEFAULT_WS_DWELL = 2       # seconds to sit on each WS rotation slice
CACHE_TTL_LIVEV2 = 60 * 60 * 6   # 6h; 'open' is fixed after regular open, lastTradePrice is WS-overridden
CACHE_TTL_EOD_RANGE = 60 * 60 * 6

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

SHARE_CLASS_DOT_TO_HYPHEN = {
    # Common US share-class notations used by models/files vs EODHD
    "BRK.A": "BRK-A", "BRK.B": "BRK-B",
    "BF.A": "BF-A", "BF.B": "BF-B",
    "HEI.A": "HEI-A", "HEI.B": "HEI-B",
}

DEFAULT_DAILY_METRICS_LOOKBACK = 30

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
    s.headers.update({"User-Agent": "ltr-monitor/rotate-ws/1.1"})
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
        help="Live v2 costs ~1 API call per ticker (batched). Keep 0 to avoid burning calls."
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

    # --- NEW: Daily metrics (multi-session) ---
    st.write("**Recent daily metrics**")
    lookback_days = st.number_input(
        "Lookback sessions (0 = All)", min_value=0, max_value=1000, value=DEFAULT_DAILY_METRICS_LOOKBACK, step=5,
        help="Used for daily Top1/2/3 long & long–short and Rank IC (per-day) with EOD open→close returns."
    )
    daily_metrics_topks = (1, 2, 3)  # fixed per request (Top 1,2,3 visibility)

    st.write("**After‑close EOD**")
    if st.button("🔄 Force EOD refresh (latest session)", use_container_width=True):
        # Clear only the EOD caches; leave others intact
        try:
            _livev2_quotes_once.clear()
        except Exception:
            pass
        try:
            _historical_daily_bars.clear()
        except Exception:
            pass
        try:
            _eod_bars_range.clear()
        except Exception:
            pass
        st.session_state["__force_eod_refresh_clicked__"] = True
        st.success("Will re-fetch EOD data for the active session/date this run.")

# -----------------------
# Helpers — time / sessions
# -----------------------
NY = pytz.timezone("America/New_York")
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
def _normalize_base_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    return SHARE_CLASS_DOT_TO_HYPHEN.get(t, t)

def _to_eod_symbol(ticker: str, suffix: str) -> str:
    base = _normalize_base_ticker((ticker or "").strip().upper())
    return base if "." in base else f"{base}.{suffix.strip().upper()}"

def _strip_suffix(symbol: str) -> str:
    return (symbol or "").strip().upper().split(".")[0]

# -----------------------
# Predictions loaders / normalizers
# -----------------------
def _try_parse_date_col(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: return column name that is 'date-like' (case-insensitive) or first column if mostly datey."""
    cols = list(df.columns)
    low = {c.lower().strip(): c for c in cols}
    for k in ["date", "session_date", "target_date", "day"]:
        if k in low:
            return low[k]
    # fallback: if first column parses to dates for ≥80% of non-null rows
    first = cols[0] if cols else None
    if first is not None:
        try:
            parsed = pd.to_datetime(df[first], errors="coerce")
            valid_ratio = parsed.notna().mean()
            if valid_ratio >= 0.80:
                return first
        except Exception:
            pass
    return None

def _normalize_predictions_any(df: pd.DataFrame, assume_date: date) -> pd.DataFrame:
    """
    Accepts:
      - Long: columns like ['ticker','prediction', optional 'date','as_of']
      - 'This style': ['ticker','prediction'] (no 'date') -> we attach assume_date
      - Wide: columns ['date', 'AAPL','ABT', ...] -> melt to long
    Returns columns: ['ticker','prediction','date', optional 'as_of'] with standardized types.
    """
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    cols_map = {c.lower().strip(): c for c in df.columns}
    ticker_col = next((cols_map[k] for k in ["ticker", "symbol"] if k in cols_map), None)
    pred_col   = next((cols_map[k] for k in ["prediction", "pred", "pred_score", "score", "yhat"] if k in cols_map), None)
    date_col   = next((cols_map[k] for k in ["date", "session_date", "target_date", "day"] if k in cols_map), None)
    asof_col   = next((cols_map[k] for k in ["as_of", "asof", "timestamp", "prediction_time"] if k in cols_map), None)

    # Case A: already long (ticker/pred present)
    if ticker_col is not None and pred_col is not None:
        out = df.rename(columns={ticker_col: "ticker", pred_col: "prediction"})
        if date_col: out = out.rename(columns={date_col: "date"})
        if asof_col: out = out.rename(columns={asof_col: "as_of"})
        if "date" not in out.columns:
            out["date"] = pd.to_datetime(assume_date).date()
    else:
        # Case B: probably wide (melt)
        date_col = _try_parse_date_col(df)
        if not date_col:
            st.error(
                "Could not auto-detect predictions format. Provide either long format "
                "(['ticker','prediction', optional 'date']) or wide format (first column = date, remaining = tickers)."
            )
            st.stop()
        tickers = [c for c in df.columns if c != date_col]
        long_df = df.melt(id_vars=[date_col], value_vars=tickers, var_name="ticker", value_name="prediction")
        out = long_df.rename(columns={date_col: "date"})

    # Clean
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip().map(_normalize_base_ticker)
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date

    # as_of (optional) parse as NY TZ
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
        # keep latest as_of per ticker+date if multiple candidates
        out = out.sort_values(["ticker", "date", "as_of"]).groupby(["ticker", "date"], as_index=False).tail(1)

    st.caption(f"📐 Predictions shape after normalization (all days): {out.shape}")
    return out[["ticker", "prediction", "date"] + (["as_of"] if "as_of" in out.columns else [])]

@st.cache_data(show_spinner=False)
def _load_predictions_from_path(path: str, assume_date: date) -> pd.DataFrame:
    return _normalize_predictions_any(pd.read_csv(path), assume_date)

def _load_predictions_from_upload(upload: io.BytesIO, assume_date: date) -> pd.DataFrame:
    return _normalize_predictions_any(pd.read_csv(upload), assume_date)

# -----------------------
# OHLCV normalization
# -----------------------
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

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip().map(_normalize_base_ticker)
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

def _post_asof_guard_df_date(df: pd.DataFrame, session: date, require_asof_guard: bool) -> pd.DataFrame:
    """Enforce as_of ≤ session open, evaluated at 09:30 ET of that session date."""
    if not require_asof_guard or "as_of" not in df.columns:
        return df
    session_open_dt, _ = _regular_session_bounds(session)
    mask_ok = df["as_of"].isna() | (df["as_of"] <= session_open_dt)
    before = df.shape
    out = df[mask_ok]
    st.caption(f"📐 as_of guard for session={session} (≤ 09:30 ET): {before} → {out.shape}")
    return out

def _post_asof_guard(df: pd.DataFrame, session_open: datetime, require_asof_guard: bool) -> pd.DataFrame:
    """Legacy guard for already-selected session."""
    out = df.copy()
    if require_asof_guard and "as_of" in out.columns:
        mask_ok = df["as_of"].isna() | (df["as_of"] <= session_open)
        before = out.shape
        out = out[mask_ok]
        st.caption(f"📐 After enforcing as_of ≤ session_open (no look-ahead): {before} → {out.shape}")
    return out

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
                    "date": bar.get("date"),
                })
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["symbol", "open", "close"]).set_index("symbol")

    out = pd.DataFrame(frames).drop_duplicates(subset=["symbol"]).set_index("symbol")
    return out

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_EOD_RANGE)
def _eod_bars_range(symbols_eod: Tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """
    Fetch EOD bars for a date RANGE for each symbol (one call per symbol, covers the whole range).
    Returns: columns ['symbol','date','open','close'] and derived ['ticker'].
    """
    if not symbols_eod:
        return pd.DataFrame(columns=["symbol", "date", "open", "close", "ticker"])

    start_iso, end_iso = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    frames = []
    for sym in symbols_eod:
        url = f"{BASE_HOST}/api/eod/{sym}?from={start_iso}&to={end_iso}&api_token={API_TOKEN}&fmt=json"
        try:
            r = _http.get(url, timeout=12)
            if not r.ok:
                continue
            payload = r.json()
            if isinstance(payload, list) and payload:
                for bar in payload:
                    frames.append({
                        "symbol": sym,
                        "date": bar.get("date"),
                        "open": bar.get("open", np.nan),
                        "close": bar.get("close", np.nan),
                    })
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "open", "close", "ticker"])

    out = pd.DataFrame(frames)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date
    out["ticker"] = out["symbol"].map(lambda s: _strip_suffix(s))
    # Consistency: map back to our base ticker normalization (e.g., BRK-A)
    out["ticker"] = out["ticker"].astype(str).str.upper().map(_normalize_base_ticker)
    # Remove dupes, keep last per ticker-date (rare)
    out = out.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
    st.caption(f"📐 EOD range bars fetched: {out.shape} (symbols={len(set(out['ticker'])):,}, days={len(set(out['date'])):,})")
    return out[["ticker", "date", "open", "close"]]

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_LIVEV2)
def _livev2_quotes_once(symbols_eod: Tuple[str, ...]) -> pd.DataFrame:
    """
    Fetch delayed quotes for all requested symbols (one-time snapshot).
    Returns DataFrame indexed by symbol with at least columns: ['open', 'lastTradePrice'].
    NOTE: Consumes ~1 API call per ticker per EODHD docs (batched with page[limit]≤100).
    """
    if not symbols_eod:
        return pd.DataFrame(columns=["symbol", "open", "lastTradePrice"]).set_index("symbol")

    frames = []
    for group in _chunk(list(symbols_eod), HTTP_BATCH):
        url = f"{LIVE_V2_URL}?s={','.join(group)}&api_token={API_TOKEN}&fmt=json&page[limit]={min(len(group), HTTP_BATCH)}"
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
# Load predictions (ALL days) and per-session slice
# -----------------------
if 'upload_file' in locals() and upload_file is not None:
    preds_all = _load_predictions_from_upload(upload_file, assume_date=session_date)
else:
    chosen_default = preds_path if 'preds_path' in locals() else None
    if not chosen_default:
        chosen_default = next((p for p in DEFAULT_PREDICTIONS_PATHS if p and os.path.exists(p)), "live_predictions.csv")
    if chosen_default and os.path.exists(chosen_default):
        preds_all = _load_predictions_from_path(chosen_default, assume_date=session_date)
    else:
        preds_all = pd.DataFrame(columns=["ticker", "prediction", "date"])

if preds_all.empty:
    st.info("Upload or point to your predictions CSV to get started. Accepted formats: long ('ticker','prediction'[, 'date']) or wide (first column 'date', remaining columns as tickers).")
    st.stop()

# Optional baseline predictions (only per-session comparison, we keep your previous logic)
baseline = pd.DataFrame(columns=["ticker", "prediction", "date"])
baseline_ranked = pd.DataFrame()
if baseline_src != "None":
    try:
        if baseline_src == "Local path" and baseline_path:
            if os.path.exists(baseline_path):
                baseline = _load_predictions_from_path(baseline_path, assume_date=session_date)
            else:
                st.warning(f"Baseline path not found: {baseline_path}")
        elif baseline_src == "Upload" and baseline_upload is not None:
            baseline = _load_predictions_from_upload(baseline_upload, assume_date=session_date)
    except Exception as e:
        st.warning(f"Failed to load baseline predictions: {e}")

# -------------- per-session slice (MAIN view) --------------
preds = _align_and_filter_for_session(preds_all, session_date)
preds = _post_asof_guard_df_date(preds, session_date, require_asof_guard=require_asof_guard)
preds_ranked_full = preds.sort_values("prediction", ascending=False).reset_index(drop=True)
preds_ranked_full["full_rank"] = np.arange(1, len(preds_ranked_full) + 1)
preds_ranked_full["ticker"] = preds_ranked_full["ticker"].astype(str).str.upper().map(_normalize_base_ticker)

preds_ranked = preds_ranked_full.head(int(top_n)).copy()
preds_ranked["rank"] = np.arange(1, len(preds_ranked) + 1)
preds_ranked["ticker"] = preds_ranked["ticker"].astype(str).str.upper().map(_normalize_base_ticker)
st.caption(f"📐 Ranked predictions shape (after top-N): {preds_ranked.shape}")

# Baseline per-session
if not baseline.empty:
    baseline = _align_and_filter_for_session(baseline, session_date)
    baseline = _post_asof_guard_df_date(baseline, session_date, require_asof_guard=require_asof_guard)
    baseline_ranked = baseline.sort_values("prediction", ascending=False).reset_index(drop=True)
    baseline_ranked["baseline_rank"] = np.arange(1, len(baseline_ranked) + 1)
    baseline_ranked["ticker"] = baseline_ranked["ticker"].astype(str).str.upper().map(_normalize_base_ticker)
    baseline_ranked = baseline_ranked.rename(columns={"prediction": "baseline_prediction"})
    st.caption(f"📐 Baseline predictions shape (after normalization): {baseline_ranked.shape}")

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
map_df["base_ticker"] = map_df["ticker"].astype(str).str.upper().str.strip()
df_view = preds_ranked.merge(map_df, on="ticker", how="left")

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
    if http_refresh_mins > 0:
        last = st.session_state.get("last_http_refresh_at")
        if (last is None) or ((datetime.now(tz=NY) - last) >= timedelta(minutes=int(http_refresh_mins))):
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

# --- NEW: After‑close override with official EOD close (if available or forced) ---
if price_mode == "live" and (now_et >= session_close or st.session_state.get("__force_eod_refresh_clicked__", False)):
    eod_today = _historical_daily_bars(symbols_eod_tuple, session_date)
    if not eod_today.empty:
        df_view["open_today"] = df_view["eod_symbol"].map(eod_today["open"])
        df_view["last_price"] = df_view["eod_symbol"].map(eod_today["close"])
        df_view["return_open_to_now"] = (df_view["last_price"] / df_view["open_today"] - 1.0).replace([np.inf, -np.inf], np.nan)
        st.caption("✅ After‑close: using **official EOD open→close** for today's final returns (EODHD historical API).")
    else:
        st.caption("ℹ️ After‑close: EOD data not yet published per EODHD schedule (~2–3h after close for US). Using lastTradePrice as provisional close.")[  # US EOD delay
            0
        ]  # lint silence

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
bottom_short_mean = float((-bottom_slice["return_open_to_now"]).mean()) if not bottom_slice.empty else np.nan
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

top_win_rate = float(_long_hit_mask(top_slice["return_open_to_now"], benchmark_return).mean()) if not top_slice.empty else np.nan
bottom_win_rate = float(_short_hit_mask(bottom_slice["return_open_to_now"], benchmark_return).mean()) if not bottom_slice.empty else np.nan

# Curves for long/short accumulation
topk_depth_returns = pd.DataFrame()
topk_depth_hits = pd.DataFrame()
if not returns_ready_sorted.empty:
    long_curve = returns_ready_sorted[["return_open_to_now"]].reset_index(drop=True)
    long_curve["top_k"] = np.arange(1, len(long_curve) + 1)
    long_curve["mean_return"] = long_curve["return_open_to_now"].expanding().mean()
    long_curve["hit_rate"] = _long_hit_mask(long_curve["return_open_to_now"], benchmark_return).expanding().mean()

    short_curve = returns_ready_sorted.sort_values("rank", ascending=False)[["return_open_to_now"]].reset_index(drop=True)
    short_curve["top_k"] = np.arange(1, len(short_curve) + 1)
    short_curve["mean_return"] = short_curve["return_open_to_now"].expanding().mean()
    short_curve["hit_rate"] = _short_hit_mask(short_curve["return_open_to_now"], benchmark_return).expanding().mean()

    short_curve["short_mean_pnl"] = -short_curve["mean_return"]
    long_short_spread_curve = 0.5 * (long_curve["mean_return"].values + short_curve["short_mean_pnl"].values)

    topk_depth_returns = pd.concat(
        [
            pd.DataFrame({"top_k": long_curve["top_k"], "series": "Long cumulative avg return", "value": long_curve["mean_return"]}),
            pd.DataFrame({"top_k": short_curve["top_k"], "series": "Short cumulative avg return (short P&L)", "value": short_curve["short_mean_pnl"]}),
            pd.DataFrame({"top_k": long_curve["top_k"], "series": "Equal-weight long–short spread (avg)", "value": long_short_spread_curve}),
        ],
        ignore_index=True,
    )

    topk_depth_hits = pd.concat(
        [
            pd.DataFrame({"top_k": long_curve["top_k"], "series": "Long hit rate (> equal-weight benchmark)", "value": long_curve["hit_rate"]}),
            pd.DataFrame({"top_k": short_curve["top_k"], "series": "Short hit rate (< equal-weight benchmark)", "value": short_curve["hit_rate"]}),
        ],
        ignore_index=True,
    )

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
    "rank": "Rank",
    "open_today": open_label,
    "last_price": last_label,
    "return_open_to_now": return_label,
    "baseline_rank": "Baseline rank",
    "rank_delta_vs_baseline": "Δ rank (baseline→now)",
    "baseline_prediction": "Baseline prediction",
})

# -----------------------
# Robust rank⇄return metrics (single-session)
# -----------------------
def _winsorize_series(s: pd.Series, p: float) -> pd.Series:
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
    ar = a.rank(method="average")
    br = b.rank(method="average")
    return _pearson_corr(ar, br)

def _kendall_tau_b(x: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()
    n = len(d)
    if n < 2:
        return np.nan
    xv = d["x"].to_numpy(); yv = d["y"].to_numpy()
    C = 0; D = 0
    for i in range(n - 1):
        dx = xv[i+1:] - xv[i]; dy = yv[i+1:] - yv[i]
        prod = dx * dy
        C += int((prod > 0).sum()); D += int((prod < 0).sum())
    n0 = n * (n - 1) // 2
    counts_x = pd.Series(xv).value_counts().to_numpy()
    counts_y = pd.Series(yv).value_counts().to_numpy()
    n1 = int(((counts_x * (counts_x - 1)) // 2).sum())
    n2 = int(((counts_y * (counts_y - 1)) // 2).sum())
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    return float((C - D) / denom) if denom > 0 else np.nan

def _theil_sen_slope_via_deciles(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if d.shape[0] < 3:
        return np.nan
    bins = int(max(3, min(bins, d.shape[0] // 2)))
    try:
        q = pd.qcut(d["x"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    except Exception:
        q = pd.cut(d["x"], bins=bins, labels=False, duplicates="drop")
    g = d.assign(bin=q).groupby("bin", as_index=False).agg(x_med=("x", "median"), y_med=("y", "median")).dropna()
    xv = g["x_med"].to_numpy(); yv = g["y_med"].to_numpy()
    m = len(xv)
    if m < 3:
        return np.nan
    slopes = []
    for i in range(m - 1):
        dx = xv[i+1:] - xv[i]; dy = yv[i+1:] - yv[i]
        valid = dx != 0
        if np.any(valid):
            slopes.extend((dy[valid] / dx[valid]).tolist())
    return float(np.median(slopes)) if slopes else np.nan

def _compute_rank_metrics(df: pd.DataFrame, topk: int, winsor_pct: int, benchmark: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ready = df[["rank", "prediction", "return_open_to_now"]].dropna().copy()
    st.caption(f"📐 Metrics-ready rows (after NA drop): {ready.shape}")

    if ready.empty:
        return pd.DataFrame(columns=["Focus", "Metric", "Value"]), pd.DataFrame(columns=["decile", "median_return"])

    ready["neg_rank"] = -ready["rank"].astype(float)
    returns_numeric = pd.to_numeric(ready["return_open_to_now"], errors="coerce")
    if pd.notna(benchmark):
        ready["relevance"] = (returns_numeric - benchmark).clip(lower=0)
    else:
        ready["relevance"] = returns_numeric.clip(lower=0)

    p = max(0.0, min(0.5, float(winsor_pct) / 100.0))
    ret_w = _winsorize_series(ready["return_open_to_now"], p)

    spearman = _spearman_corr(ready["neg_rank"], ready["return_open_to_now"])
    kendall  = _kendall_tau_b(ready["neg_rank"], ready["return_open_to_now"])
    pearson_w = _pearson_corr(ready["neg_rank"], ret_w)

    ts_slope = _theil_sen_slope_via_deciles(ready["neg_rank"], ready["return_open_to_now"], bins=10)
    bps_per_10rank = (ts_slope * 10.0) * 10000.0 if pd.notna(ts_slope) else np.nan

    k = int(min(topk, ready.shape[0] // 2 if ready.shape[0] >= 2 else topk))
    top = ready.nsmallest(k, "rank")
    bot = ready.nlargest(k, "rank")
    top_med = float(top["return_open_to_now"].median()) if not top.empty else np.nan
    bot_med_raw = float(bot["return_open_to_now"].median()) if not bot.empty else np.nan
    bot_med_short = -bot_med_raw if pd.notna(bot_med_raw) else np.nan
    l_s_med = top_med + bot_med_short if (pd.notna(top_med) and pd.notna(bot_med_short)) else np.nan

    top_win = float(_long_hit_mask(top["return_open_to_now"], benchmark).mean()) if not top.empty else np.nan
    bot_win = float(_short_hit_mask(bot["return_open_to_now"], benchmark).mean()) if not bot.empty else np.nan

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

        hits = _long_hit_mask(pd.Series(ready_sorted["return_open_to_now"].to_numpy()[:k_rank]), benchmark).to_numpy()
        precision_k = float(hits.mean()) if k_rank > 0 else np.nan
        cum_hits = np.cumsum(hits)
        precisions = cum_hits / (np.arange(1, k_rank + 1))
        relevant_total = int(hits.sum())
        map_k = float(np.sum(precisions * hits) / relevant_total) if relevant_total > 0 else 0.0
    else:
        ndcg_k = np.nan; precision_k = np.nan; map_k = np.nan

    metrics_rows = [
        ("All ranks", "Spearman ρ (ranks ↔ returns)", f"{spearman:.3f}" if pd.notna(spearman) else "—", "↑ toward +1"),
        ("All ranks", "Kendall τ-b (ranks ↔ returns)", f"{kendall:.3f}" if pd.notna(kendall) else "—", "↑ toward +1"),
        ("All ranks", f"Winsorized Pearson r (tails={winsor_pct}%)", f"{pearson_w:.3f}" if pd.notna(pearson_w) else "—", "↑ toward +1"),
        ("All ranks", "Theil–Sen slope (return per rank)", f"{ts_slope:.6f}" if pd.notna(ts_slope) else "—", "↑ more positive"),
        ("All ranks", "≈ bps per 10-rank improvement", f"{bps_per_10rank:.1f}" if pd.notna(bps_per_10rank) else "—", "↑ more positive"),
        ("Long bucket", f"Top-{k} median return", f"{top_med*100:.2f}%" if pd.notna(top_med) else "—", "↑ more positive"),
        ("Short bucket", f"Bottom-{k} median short P&L", f"{bot_med_short*100:.2f}%" if pd.notna(bot_med_short) else "—", "↑ more positive"),
        ("Spread", f"Median long–short (Top-{k} + Short-{k})", f"{l_s_med*100:.2f}%" if pd.notna(l_s_med) else "—", "↑ wider"),
        ("Long bucket", f"Top-{k} win rate (> benchmark)", f"{top_win*100:.1f}%" if pd.notna(top_win) else "—", "↑ toward 100%"),
        ("Short bucket", f"Bottom-{k} short hit rate", f"{bot_win*100:.1f}%" if pd.notna(bot_win) else "—", "↑ toward 100%"),
        ("Top-K", f"NDCG@{k_rank}", f"{ndcg_k:.3f}" if pd.notna(ndcg_k) else "—", "↑ toward 1"),
        ("Top-K", f"MAP@{k_rank}", f"{map_k:.3f}" if pd.notna(map_k) else "—", "↑ toward 1"),
        ("Top-K", f"Precision@{k_rank}", f"{precision_k*100:.1f}%" if pd.notna(precision_k) else "—", "↑ toward 100%"),
    ]
    metrics_tbl = pd.DataFrame(metrics_rows, columns=["Focus", "Metric", "Value", "Better ↗︎"])

    D = int(min(10, max(3, ready.shape[0] // 10)))
    ready["decile"] = pd.qcut(ready["rank"], q=D, labels=False, duplicates="drop") if ready["rank"].nunique() > 1 else 0
    deciles = (ready
               .groupby("decile", as_index=False)
               .agg(median_return=("return_open_to_now", "median"),
                    count=("return_open_to_now", "size"))
               .sort_values("decile"))
    deciles["bucket"] = deciles["decile"].astype(int) + 1
    deciles = deciles[["bucket", "count", "median_return"]]
    deciles.rename(columns={"bucket": f"rank_decile(1=best, D={deciles.shape[0]})"}, inplace=True)
    return metrics_tbl, deciles

metrics_tbl, deciles_tbl = _compute_rank_metrics(df_view, metrics_topk, winsor_tail_pct, benchmark_return)

deciles_chart_data = pd.DataFrame()
if deciles_tbl is not None and not deciles_tbl.empty:
    deciles_chart_data = deciles_tbl.copy()
    decile_label = deciles_chart_data.columns[0]
    deciles_chart_data = deciles_chart_data.rename(columns={decile_label: "rank_decile"})
    deciles_chart_data["rank_decile"] = deciles_chart_data["rank_decile"].astype(str)

# -----------------------
# Baseline comparison (unchanged)
# -----------------------
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
        rank_spearman = np.nan; rank_kendall = np.nan; pred_pearson = np.nan

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

# -----------------------
# UI — header and dimensions
# -----------------------
price_mode_label_map = {
    "live": "HTTP + rotating WS" if ws_enabled else "HTTP (delayed only)",
    "historical_api": "Historical daily bars (EODHD API)",
    "upload": "Uploaded OHLCV file",
    "future": "Future session (awaiting data)",
}
price_mode_label = price_mode_label_map.get(price_mode, str(price_mode))

price_dim_lines = [
    f"predictions_all: {preds_all.shape}",
    f"predictions_raw_session: {preds.shape}",
    f"ranked_topN: {preds_ranked.shape}",
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
    f"joined_view: {df_view.shape}",
    f"mode: {price_mode_label}",
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

# -----------------------
# Session snapshot cards and charts
# -----------------------
if not returns_ready_sorted.empty:
    st.subheader("🚦 Performance snapshot")
    summary_cols = st.columns(4, gap="large")
    with summary_cols[0]:
        st.metric("Equal-weight benchmark return", _fmt_pct_display(benchmark_return))
    with summary_cols[1]:
        st.metric(f"Top-{top_k_count or summary_topk} mean", _fmt_pct_display(top_mean))
    with summary_cols[2]:
        st.metric(f"Bottom-{bottom_k_count or summary_bottomk} mean short P&L", _fmt_pct_display(bottom_short_mean))
    with summary_cols[3]:
        st.metric("Equal-weight long–short spread", _fmt_pct_display(long_short_spread))

    rate_cols = st.columns(2, gap="large")
    with rate_cols[0]:
        st.metric(f"Top-{top_k_count or summary_topk} hit rate vs benchmark", _fmt_pct_display(top_win_rate),
                  delta=_fmt_delta_points(top_win_rate - 0.5) if pd.notna(top_win_rate) else None)
    with rate_cols[1]:
        st.metric(f"Bottom-{bottom_k_count or summary_bottomk} short hit rate vs benchmark", _fmt_pct_display(bottom_win_rate),
                  delta=_fmt_delta_points(bottom_win_rate - 0.5) if pd.notna(bottom_win_rate) else None)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Portfolio build-up by rank")
    st.caption("Cumulative average returns as you add more names. Short line converts negative returns into short-side P&L.")
    if not topk_depth_returns.empty:
        returns_chart = (
            alt.Chart(topk_depth_returns)
            .mark_line(point=alt.OverlayMarkDef(size=60, filled=True))
            .encode(
                x=alt.X("top_k:Q", title="Portfolio size (K)"),
                y=alt.Y("value:Q", title="Cumulative avg return", axis=alt.Axis(format="%")),
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
                tooltip=[alt.Tooltip("series:N", title="Metric"), alt.Tooltip("top_k:Q", title="K size"), alt.Tooltip("value:Q", title="Value", format=".2%")],
            )
        )
        st.altair_chart(returns_chart.interactive(), use_container_width=True)

        hits_chart = (
            alt.Chart(topk_depth_hits)
            .mark_line(point=alt.OverlayMarkDef(size=50, filled=True), strokeDash=[4, 4])
            .encode(
                x=alt.X("top_k:Q", title="Portfolio size (K)"),
                y=alt.Y("value:Q", title="Cumulative hit rate", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "series:N",
                    title="",
                    scale=alt.Scale(
                        domain=["Long hit rate (> equal-weight benchmark)", "Short hit rate (< equal-weight benchmark)"],
                        range=["#2980b9", "#8e44ad"],
                    ),
                ),
                tooltip=[alt.Tooltip("series:N", title="Metric"), alt.Tooltip("top_k:Q", title="K size"), alt.Tooltip("value:Q", title="Value", format=".2%")],
            )
        )
        st.altair_chart(hits_chart.interactive(), use_container_width=True)
    else:
        st.info("Performance lines populate once realized returns are available.")

    table_phrase = "returns from open → latest" if price_mode == "live" else ("returns from open → close" if price_mode in {"historical_api", "upload"} else "returns (pending data)")
    st.subheader(f"Rank vs realized returns ({table_phrase})")
    if not chart_data.empty:
        scatter = (
            alt.Chart(chart_data)
            .mark_circle(size=70, opacity=0.75)
            .encode(
                x=alt.X("rank:Q", title="Model rank (1 = best)", scale=alt.Scale(zero=False)),
                y=alt.Y("return_open_to_now:Q", title=return_label, axis=alt.Axis(format="%")),
                color=alt.Color(
                    "return_direction:N",
                    title="Return vs benchmark" if pd.notna(benchmark_return) else "Return direction",
                    scale=alt.Scale(domain=["Outperform (≥ benchmark)", "Underperform (< benchmark)"], range=["#2ecc71", "#e74c3c"]),
                ),
                tooltip=[alt.Tooltip("ticker:N", title="Ticker"), alt.Tooltip("prediction:Q", title="Prediction", format=".4f"), alt.Tooltip("return_open_to_now:Q", title="Return", format=".2%"), alt.Tooltip("rank:Q", title="Rank")],
            )
        )

        rolling = (
            alt.Chart(chart_data)
            .transform_window(rolling_mean="mean(return_open_to_now)", sort=[{"field": "rank"}], frame=[-5, 5])
            .mark_line(color="#34495e", strokeWidth=2)
            .encode(x=alt.X("rank:Q"), y=alt.Y("rolling_mean:Q", title="Rolling mean"))
        )
        layers = scatter + rolling
        if pd.notna(benchmark_return):
            benchmark_line = (
                alt.Chart(pd.DataFrame({"label": ["Equal-weight benchmark"], "benchmark": [benchmark_return]}))
                .mark_rule(color="#f1c40f", strokeDash=[6, 4])
                .encode(y=alt.Y("benchmark:Q"), tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("benchmark:Q", title="Benchmark", format=".2%")])
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
        st.metric(f"Top-{top_k_count or summary_topk} mean return", _fmt_pct_display(top_mean))
        st.metric(f"Bottom-{bottom_k_count or summary_bottomk} mean short P&L", _fmt_pct_display(bottom_short_mean))
        st.metric("Equal-weight long–short spread", _fmt_pct_display(long_short_spread))
        st.metric(f"Top-{top_k_count or summary_topk} hit rate vs benchmark", _fmt_pct_display(top_win_rate))
        st.metric(f"Bottom-{bottom_k_count or summary_bottomk} short hit rate vs benchmark", _fmt_pct_display(bottom_win_rate))

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
        st.caption("Focus splits long/short/aggregate views; **Better ↗︎** indicates the helpful direction.")
        st.dataframe(
            metrics_tbl, use_container_width=True, height=330,
            column_config={"Better ↗︎": st.column_config.TextColumn("Better ↗︎", help="Directionally helpful move for that metric.", disabled=True)}
        )
        with st.expander("How to read these numbers", expanded=False):
            st.markdown(
                """
                **Correlations (Spearman/Kendall/Pearson)** — +1 is perfect rank↔return alignment, 0 is random, −1 is inverted.

                **Slope & bps/10 ranks** — positive means better rank → higher return; e.g., +10 bps/10 ranks implies ~0.10% gain per 10 places.

                **Bucket medians & win rates** — long medians should be positive and short medians translate to positive short P&L. A long–short spread > 0 with win rates >55–60% is healthy.

                **Ranking scores (NDCG/MAP/Precision)** — all in [0, 1]. Higher shows more relevant (positive-return) names near the top.
                """
            )
    else:
        st.info("Metrics unavailable (no non-NA returns yet).")

with met_right:
    st.subheader("Rank deciles → median returns")
    if not deciles_tbl.empty:
        deciles_chart_data = deciles_tbl.rename(columns={deciles_tbl.columns[0]: "rank_decile"})
        deciles_chart_data["rank_decile"] = deciles_chart_data["rank_decile"].astype(str)
        decile_chart = (
            alt.Chart(deciles_chart_data)
            .mark_bar(color="#3498db", opacity=0.7)
            .encode(
                x=alt.X("rank_decile:N", title="Rank decile"),
                y=alt.Y("median_return:Q", title="Median return", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("rank_decile:N", title="Decile"), alt.Tooltip("median_return:Q", title="Median return", format=".2%"), alt.Tooltip("count:Q", title="Count")],
            )
        )
        decile_line = (
            alt.Chart(deciles_chart_data)
            .mark_line(color="#2c3e50", point=alt.OverlayMarkDef(filled=True, color="#2c3e50"))
            .encode(x="rank_decile:N", y=alt.Y("median_return:Q", axis=alt.Axis(format="%")))
        )
        st.altair_chart(decile_chart + decile_line, use_container_width=True)
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

# -----------------------
# NEW: Recent Daily Metrics (multi-session: Top1/2/3 long & long–short, Rank IC)
# -----------------------
st.write("---")
st.subheader("📅 Recent Daily Metrics (EOD open→close)")

def _compute_daily_metrics(
    preds_all: pd.DataFrame,
    lookback_days: int,
    session_cutoff: date,
    topks: Sequence[int],
    exchange_suffix: str,
    require_asof_guard: bool,
    max_universe_per_day: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - daily_tbl: rows per date with Top1/2/3 long, Top1/2/3 long–short, Rank IC (Spearman)
      - bars_used: EOD bars frame used for reference (ticker,date,open,close)
    """
    if preds_all.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Only use predictions up to the last available/on or before session_cutoff
    all_dates = sorted(d for d in pd.unique(preds_all["date"]) if d <= session_cutoff)
    if not all_dates:
        return pd.DataFrame(), pd.DataFrame()

    if lookback_days and lookback_days > 0:
        start_date = max(min(all_dates), session_cutoff - timedelta(days=int(lookback_days) - 1))
        target_dates = [d for d in all_dates if start_date <= d <= session_cutoff]
    else:
        start_date = min(all_dates)
        target_dates = [d for d in all_dates if d <= session_cutoff]

    if not target_dates:
        return pd.DataFrame(), pd.DataFrame()

    # Prepare per-day ranks and gather tickers for the whole window
    maxk = int(max(topks)) if topks else 3
    tickers_union: set = set()
    per_day_frames: Dict[date, pd.DataFrame] = {}

    for d in target_dates:
        df_d = preds_all[preds_all["date"] == d].copy()
        # as_of guard per day
        df_d = _post_asof_guard_df_date(df_d, d, require_asof_guard=require_asof_guard)
        df_d = df_d.dropna(subset=["prediction"]).copy()
        if df_d.empty:
            continue
        df_d = df_d.sort_values("prediction", ascending=False).reset_index(drop=True)
        df_d["rank"] = np.arange(1, len(df_d) + 1)
        if max_universe_per_day is not None and len(df_d) > max_universe_per_day:
            df_d = df_d.head(int(max_universe_per_day))  # optional clip for extremely large universes
        per_day_frames[d] = df_d
        tickers_union |= set(df_d["ticker"].astype(str))

    if not per_day_frames:
        return pd.DataFrame(), pd.DataFrame()

    # Fetch EOD bars for the WHOLE window in one pass per symbol
    symbols_all = tuple(_to_eod_symbol(t, exchange_suffix) for t in sorted(tickers_union))
    bars = _eod_bars_range(symbols_all, min(target_dates), max(target_dates))
    if bars.empty:
        return pd.DataFrame(), pd.DataFrame()

    bars["ret"] = (bars["close"] / bars["open"] - 1.0).replace([np.inf, -np.inf], np.nan)

    rows = []
    for d in target_dates:
        df_d = per_day_frames.get(d)
        if df_d is None or df_d.empty:
            continue
        b_d = bars[bars["date"] == d][["ticker", "ret"]].copy()
        m = df_d.merge(b_d, on="ticker", how="left")
        m = m.dropna(subset=["ret"])
        if m.empty:
            continue

        # Rank IC across the day's full available universe
        ic = _spearman_corr(-m["rank"].astype(float), m["ret"].astype(float))

        # K-bucket returns
        metrics = {"date": d, "rank_ic": ic}
        for k in topks:
            topk = m.nsmallest(int(k), "rank")               # rank 1..k
            botk = m.nlargest(int(k), "rank")                # worst k by prediction
            long_k = float(topk["ret"].mean()) if not topk.empty else np.nan
            short_pnl_k = float((-botk["ret"]).mean()) if not botk.empty else np.nan
            ls_k = (long_k + short_pnl_k) if (pd.notna(long_k) and pd.notna(short_pnl_k)) else np.nan
            metrics[f"top{k}_long"] = long_k
            metrics[f"top{k}_long_short"] = ls_k
        rows.append(metrics)

    daily_tbl = pd.DataFrame(rows).sort_values("date")
    return daily_tbl, bars

daily_tbl, bars_used = _compute_daily_metrics(
    preds_all=preds_all,
    lookback_days=int(lookback_days),
    session_cutoff=session_date,
    topks=(1, 2, 3),
    exchange_suffix=exchange_suffix,
    require_asof_guard=require_asof_guard,
    max_universe_per_day=None,  # keep full universe for IC fidelity
)

if not daily_tbl.empty:
    # 30-day (or configured) averages
    avg_cols = ["rank_ic", "top1_long", "top1_long_short", "top2_long", "top2_long_short", "top3_long", "top3_long_short"]
    avgs = {c: (float(daily_tbl[c].mean()) if c in daily_tbl.columns else np.nan) for c in avg_cols}

    cards = st.columns(4, gap="large")
    with cards[0]:
        st.metric("Avg Rank IC", _fmt_pct_display(avgs["rank_ic"]))
    with cards[1]:
        st.metric("Avg Top‑1 Long", _fmt_pct_display(avgs["top1_long"]))
    with cards[2]:
        st.metric("Avg Top‑2 Long‑Short", _fmt_pct_display(avgs["top2_long_short"]))
    with cards[3]:
        st.metric("Avg Top‑3 Long", _fmt_pct_display(avgs["top3_long"]))

    # Charts: Long returns (Top1/2/3), Long-Short (Top1/2/3), Rank IC
    chart_df_long = daily_tbl.melt(id_vars=["date", "rank_ic"], value_vars=["top1_long", "top2_long", "top3_long"], var_name="series", value_name="value")
    chart_df_ls = daily_tbl.melt(id_vars=["date", "rank_ic"], value_vars=["top1_long_short", "top2_long_short", "top3_long_short"], var_name="series", value_name="value")
    chart_df_ic = daily_tbl[["date", "rank_ic"]].rename(columns={"rank_ic": "value"}).assign(series="Rank IC")

    c1 = (
        alt.Chart(chart_df_long).mark_line(point=alt.OverlayMarkDef(size=45, filled=True))
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Top‑K long return (EOD)", axis=alt.Axis(format="%")),
            color=alt.Color("series:N", title="", scale=alt.Scale(domain=["top1_long", "top2_long", "top3_long"], range=["#16a085", "#27ae60", "#2ecc71"])),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("series:N", title="Metric"), alt.Tooltip("value:Q", title="Value", format=".2%")],
        )
    )
    c2 = (
        alt.Chart(chart_df_ls).mark_line(point=alt.OverlayMarkDef(size=45, filled=True))
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Top‑K long–short (EOD)", axis=alt.Axis(format="%")),
            color=alt.Color("series:N", title="", scale=alt.Scale(domain=["top1_long_short", "top2_long_short", "top3_long_short"], range=["#d35400", "#e67e22", "#f39c12"])),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("series:N", title="Metric"), alt.Tooltip("value:Q", title="Value", format=".2%")],
        )
    )
    c3 = (
        alt.Chart(chart_df_ic).mark_line(point=alt.OverlayMarkDef(size=45, filled=True))
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Rank IC (Spearman)", axis=alt.Axis(format="%")),
            color=alt.Color("series:N", title="", scale=alt.Scale(domain=["Rank IC"], range=["#34495e"])),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("value:Q", title="Value", format=".2%")],
        )
    )
    st.altair_chart(c1.resolve_scale(y='independent') & c2.resolve_scale(y='independent') & c3.resolve_scale(y='independent'), use_container_width=True)

    # Table (daily)
    daily_fmt = daily_tbl.copy()
    for c in daily_fmt.columns:
        if c != "date":
            daily_fmt[c] = daily_fmt[c].apply(_fmt_pct)
    st.dataframe(daily_fmt, use_container_width=True, height=420)
else:
    st.info("Daily metrics will populate once your predictions file includes multiple dates or once EOD data is available for those dates.")

# -----------------------
# Download
# -----------------------
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
        "WebSockets don’t consume API calls; default WS limit is ~50 symbols per connection (upgradeable in your dashboard)."
    )
