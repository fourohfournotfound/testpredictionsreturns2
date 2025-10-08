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
# - Live v2 (US extended quotes): endpoint, s= batching, fields (open, lastTradePrice), 1 call / symbol.  (https://eodhd.com/api/us-quote-delayed)  [See EODHD docs page "Live v2 for US Stocks: Extended Quotes (2025)"]
# - WebSockets: wss://ws.eodhistoricaldata.com/ws/us?api_token=..., subscribe/unsubscribe JSON; ~50 symbols/connection by default; WS does NOT consume API calls.
# - API limits: daily calls counted per symbol, 100k/day default.
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
DEFAULT_WS_WINDOW = 50     # per EODHD docs: 50 concurrent symbols per connection by default
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

# -----------------------
# HTTP snapshot (opens + initial last)
# -----------------------
# One-time (cached) snapshot
quotes_snapshot = _livev2_quotes_once(symbols_eod_tuple)  # index: EOD symbol
# Optional periodic refresh to gently correct drift if desired
if _should_http_refresh():
    quotes_snapshot = _livev2_quotes_once.clear() and _livev2_quotes_once(symbols_eod_tuple)  # invalidate + refetch
    _mark_http_refreshed()

# -----------------------
# Last-price cache persisted across refreshes
# -----------------------
if "last_price_cache" not in st.session_state:
    st.session_state["last_price_cache"] = {}

# Seed cache from snapshot for any missing symbols
for sym in symbols_eod:
    if sym not in st.session_state["last_price_cache"]:
        try:
            st.session_state["last_price_cache"][sym] = float(quotes_snapshot.at[sym, "lastTradePrice"])
        except Exception:
            pass

# -----------------------
# Rotating WS update (optional)
# -----------------------
if ws_enabled:
    if not _HAS_WS:
        st.info("websocket-client not installed; staying on delayed HTTP only.")
    else:
        rot = st.session_state.get("ws_rot_idx", 0)
        ws_prices = _ws_update_prices_rotating(ws_url, symbols_eod, min(ws_window, DEFAULT_WS_WINDOW), ws_dwell, rot)
        # Update cache with any fresh prints from this slice
        for sym, px in ws_prices.items():
            st.session_state["last_price_cache"][sym] = px
        st.session_state["ws_rot_idx"] = rot + 1

# -----------------------
# Build view: open from HTTP snapshot; last price from cache (or snapshot fallback)
# -----------------------
open_series = quotes_snapshot["open"] if "open" in quotes_snapshot.columns else pd.Series(dtype=float)
last_series = pd.Series({s: st.session_state["last_price_cache"].get(s, np.nan) for s in symbols_eod}, dtype=float)

map_df = pd.DataFrame({"ticker": symbols_in, "eod_symbol": symbols_eod}).drop_duplicates()
df_view = preds_ranked.merge(map_df, on="ticker", how="left")
df_view = df_view.merge(open_series.rename("open_today"), left_on="eod_symbol", right_index=True, how="left")
df_view = df_view.merge(last_series.rename("last_price"), left_on="eod_symbol", right_index=True, how="left")

# Compute open→now return (suppress before regular open)
if now_et < session_open:
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

# -----------------------
# UI
# -----------------------
st.title("📈 Live LTR Monitor — Predictions vs Open→Now Returns (EODHD, rotating WS)")
st.caption(
    "Opens fetched once via **Live v2 (delayed)**; last prices updated by **rotating WebSocket slices** "
    f"(chunk ≤{DEFAULT_WS_WINDOW}). This keeps HTTP API calls low while still refreshing a large watchlist."
)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Top predictions (returns from open → latest)")
    st.dataframe(show, use_container_width=True, height=650)

with right:
    st.subheader("Summary")
    st.metric("Session date (ET)", str(session_date))
    st.metric("Symbols shown", f"{show.shape[0]:,}")
    if df_view["return_open_to_now"].notna().any():
        realized_top_mean = df_view["return_open_to_now"].head(min(20, len(df_view))).mean()
        st.metric("Mean return of Top-20", f"{realized_top_mean*100:.2f}%")

    # Rotation diagnostics
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
    st.code(
        f"predictions_raw: {preds.shape}\n"
        f"ranked_topN:     {preds_ranked.shape}\n"
        f"quotes_snapshot: {quotes_snapshot.shape}\n"
        f"joined_view:     {df_view.shape}\n"
        f"mode:            {'HTTP + rotating WS' if ws_enabled else 'HTTP (delayed only)'}",
        language="text",
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

st.caption(
    "Tip: Keep HTTP refresh at **0 minutes** when monitoring large lists to avoid burning daily API calls. "
    "WebSockets don’t consume API calls; default WS limit is ~50 symbols per connection (upgradeable)."
)
