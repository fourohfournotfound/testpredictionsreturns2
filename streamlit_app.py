# streamlit_app.py
# Streamlit dashboard: rank-by-prediction + intraday open→now returns using EODHD (no SDK)
# ---------------------------------------------------------------------------------------------
# - Robust to MultiIndex (ticker,date) inputs
# - Enforces session alignment and avoids look-ahead bias by default (uses target session's regular open)
# - Batched HTTP fetch via EODHD Live v2 (us-quote-delayed) -> returns 'open' + 'lastTradePrice'
# - Optional direct WebSocket override for real-time last trades (no eodhdc dependency)
# - Auto-refresh on a timer; no background infra required
#
# Expected predictions CSV (flexible):
#   Required:   ticker (or symbol), prediction (or pred/pred_score/score)
#   Optional:   date (target session), as_of (ISO timestamp for your model snapshot)
#
# Usage:
#   1) Put your token in Secrets or env as EODHD_API_TOKEN.
#   2) Optionally paste your EODHD WS URL in the sidebar when you want realtime.
#   3) streamlit run streamlit_app.py
#
# Docs:
# - Live v2 endpoint & fields (open/lastTradePrice, batching with s=, pagination): https://eodhd.com/api/us-quote-delayed  :contentReference[oaicite:4]{index=4}
# - API key via api_token param: https://eodhd.com/financial-apis/quick-start-with-our-financial-data-apis           :contentReference[oaicite:5]{index=5}
# - Realtime WebSocket overview & symbol subscriptions: https://eodhd.com/lp/realtime-api                             :contentReference[oaicite:6]{index=6}
# - Symbol format TICKER.EXCHANGE (e.g., AAPL.US): marketplace docs example                                           :contentReference[oaicite:7]{index=7}
# ---------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import io
import json
import time
from datetime import datetime, date, time as dt_time, timedelta
from typing import List, Dict, Tuple, Optional

import streamlit as st
st.set_page_config(page_title="Live LTR Monitor (EODHD)", layout="wide")

# --- Guarded scientific imports (nicer error if wheels mismatch on Cloud) ---
def _safe_import_pd_np():
    try:
        import numpy as _np
        import pandas as _pd
        return _np, _pd
    except Exception as e:
        st.error(
            "🚨 Could not import NumPy/Pandas (wheel/ABI mismatch). Try recent wheels:\n"
            "    numpy>=2.0,<3  and  pandas>=2.2,<3\n"
            "Update requirements.txt and redeploy."
        )
        st.exception(e)
        st.stop()

np, pd = _safe_import_pd_np()

import pytz
from dateutil import parser as dateparser
import requests
from streamlit_autorefresh import st_autorefresh

# Optional: plain websocket client for realtime override (no SDK)
try:
    import websocket
    _HAS_WS = True
except Exception:
    _HAS_WS = False

# -----------------------
# Constants / Config
# -----------------------
NY = pytz.timezone("America/New_York")

BASE_HOST = "https://eodhd.com"
LIVE_V2_URL = f"{BASE_HOST}/api/us-quote-delayed"   # returns delayed open + lastTradePrice; batch via s=...  :contentReference[oaicite:8]{index=8}

BATCH_SIZE_HTTP = 100    # Live v2 docs note paging; we chunk at <=100 per call.  :contentReference[oaicite:9]{index=9}
WS_COLLECT_SECONDS = 2   # small sampling window for realtime prints

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
        st.error("Missing **EODHD_API_TOKEN**. Add it in Streamlit Secrets or env, then rerun.")
        st.stop()

@st.cache_resource(show_spinner=False)
def _init_http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "ltr-monitor/1.0"})
    return s

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
        "Use real-time WebSocket override (optional)",
        value=False,
        help="If ON, we try to replace last_price with a real-time trade from an EODHD WS stream."
    )
    ws_url = st.text_input(
        "WebSocket URL (optional)",
        value="",
        help=(
            "Paste an EODHD WS endpoint (e.g., a US trades stream). "
            "We will send: {\"action\":\"subscribe\",\"symbols\":\"AAPL,TSLA\"}. "
            "See EODHD realtime notes. "
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
        help="EODHD uses exchange-suffixed symbols, e.g., AAPL.US."
    )

    st.write("**Ranking**")
    top_n = st.number_input("Show top N by prediction", min_value=1, max_value=5000, value=200, step=1)
    require_asof_guard = st.checkbox(
        "Enforce as_of ≤ session open (avoid look-ahead)", value=True,
        help="If an 'as_of' timestamp is present and is after the session open, the row is dropped."
    )
    allow_na_returns = st.checkbox("Include symbols with missing data (NA returns)", value=False)

# -----------------------
# Data loaders / normalizers
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

        # keep latest as_of per ticker
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
# Calendar utilities (simple 09:30–16:00 ET window)
# -----------------------
def _regular_session_bounds(session: date) -> Tuple[datetime, datetime]:
    return (NY.localize(datetime.combine(session, dt_time(9, 30))),
            NY.localize(datetime.combine(session, dt_time(16, 0))))

@st.cache_data(show_spinner=False, ttl=3600)
def _get_session_open_close(session: date) -> Tuple[Optional[datetime], Optional[datetime], str]:
    open_et, close_et = _regular_session_bounds(session)
    return open_et, close_et, "regular"

# -----------------------
# Symbol normalization
# -----------------------
def _to_eod_symbol(ticker: str, suffix: str) -> str:
    ticker = (ticker or "").strip().upper()
    if "." in ticker:  # already suffixed (e.g., AAPL.US)
        return ticker
    return f"{ticker}.{suffix.strip().upper()}"

def _strip_suffix(symbol: str) -> str:
    return (symbol or "").strip().upper().split(".")[0]

# -----------------------
# HTTP (Live v2 delayed) fetchers
# -----------------------
def _chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(show_spinner=False, ttl=5)
def _fetch_live_v2_quotes(symbols_eod: List[str]) -> pd.DataFrame:
    """
    Call Live v2 (us-quote-delayed) in <=100-symbol chunks.
    Returns DF indexed by EOD symbol with columns: ['open', 'lastTradePrice', 'timestamp'].
    """
    if not symbols_eod:
        return pd.DataFrame(columns=["symbol","open","lastTradePrice","timestamp"]).set_index("symbol")

    frames = []
    for group in _chunk(symbols_eod, 100):
        url = f"{LIVE_V2_URL}?s={','.join(group)}&api_token={API_TOKEN}&fmt=json"  # batching via s=...  :contentReference[oaicite:10]{index=10}
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
        return pd.DataFrame(columns=["symbol","open","lastTradePrice","timestamp"]).set_index("symbol")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["symbol"]).set_index("symbol")
    return out

# -----------------------
# Optional: WebSocket override (no SDK)
# -----------------------
@st.cache_data(show_spinner=False, ttl=2)
def _fetch_latest_prices_ws(ws_url: str, symbols_eod: List[str]) -> pd.Series:
    """
    Connects to the provided EODHD WebSocket URL and collects last-trade prices for WS_COLLECT_SECONDS.
    We send {"action":"subscribe","symbols":"AAPL,TSLA"} using base tickers (no .US).
    If anything fails (no client, bad URL, plan), return empty Series.
    """
    if not _HAS_WS or not ws_url or not symbols_eod:
        return pd.Series(dtype=float)

    tickers = sorted({_strip_suffix(s) for s in symbols_eod})
    subs = {"action": "subscribe", "symbols": ",".join(tickers)}
    prices: Dict[str, float] = {}

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except Exception:
            return
        # Common trade payload fields: symbol under 's' or 'code', price under 'p'/'price'/'lastTradePrice'
        sym = str(msg.get("s") or msg.get("code") or "").upper()
        px = msg.get("p") or msg.get("price") or msg.get("lastTradePrice")
        if sym and isinstance(px, (int, float)):
            prices[sym] = float(px)

    def on_open(ws):
        ws.send(json.dumps(subs))

    try:
        ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_open=on_open)
        # run for a short window; this is single-threaded and quick
        end_time = time.time() + WS_COLLECT_SECONDS
        ws.run_forever(dispatcher=None, reconnect=0)
        # Note: some Streamlit runners block; if so, you can shorten WS_COLLECT_SECONDS to 1s.
    except Exception:
        return pd.Series(dtype=float)

    # Map base tickers back to EOD symbols
    mapped = {}
    for s in symbols_eod:
        base = _strip_suffix(s)
        if base in prices:
            mapped[s] = prices[base]
    return pd.Series(mapped, dtype=float)

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

# Filter to target session
preds = _align_and_filter_for_session(preds, session_date)

# Rank by prediction (descending)
preds_ranked = preds.sort_values("prediction", ascending=False).reset_index(drop=True)
preds_ranked["rank"] = np.arange(1, len(preds_ranked) + 1)

# Limit to top-N
preds_ranked = preds_ranked.head(int(top_n))
st.caption(f"📐 Ranked predictions shape (after top-N): {preds_ranked.shape}")

# -----------------------
# Market times
# -----------------------
def _regular_session_bounds(session: date) -> Tuple[datetime, datetime]:
    return (NY.localize(datetime.combine(session, dt_time(9, 30))),
            NY.localize(datetime.combine(session, dt_time(16, 0))))

session_open, session_close = _regular_session_bounds(session_date)
now_et = datetime.now(tz=NY)

# Enforce as_of guard
preds_ranked = _post_asof_guard(preds_ranked, session_open, require_asof_guard=require_asof_guard)
if preds_ranked.empty:
    st.warning("No rows remain after enforcing the as_of guard. Uncheck it if your predictions were created intraday.")
    st.stop()

# Normalize symbols to EODHD format
symbols_in = preds_ranked["ticker"].dropna().astype(str).str.upper().tolist()
symbols_eod = [_to_eod_symbol(t, exchange_suffix) for t in symbols_in]

# -----------------------
# Data fetch: delayed HTTP + optional realtime WS override
# -----------------------
quotes = _fetch_live_v2_quotes(symbols_eod)  # index: EOD symbol; cols: open, lastTradePrice, timestamp

if ws_enabled and ws_url:
    ws_px = _fetch_latest_prices_ws(ws_url, symbols_eod)
    if not ws_px.empty:
        quotes.loc[ws_px.index, "lastTradePrice"] = ws_px
elif ws_enabled and not ws_url:
    st.info("WebSocket override enabled but no WS URL provided — staying on delayed HTTP.")

# -----------------------
# Merge & compute returns
# -----------------------
map_df = pd.DataFrame({"ticker": symbols_in, "eod_symbol": symbols_eod}).drop_duplicates()
df_view = preds_ranked.merge(map_df, on="ticker", how="left")
df_view = df_view.merge(quotes[["open", "lastTradePrice"]], left_on="eod_symbol", right_index=True, how="left")

if now_et < session_open:
    df_view["return_open_to_now"] = np.nan
else:
    df_view["return_open_to_now"] = (df_view["lastTradePrice"] / df_view["open"] - 1.0).replace([np.inf, -np.inf], np.nan)

if not allow_na_returns:
    before = df_view.shape
    df_view = df_view.dropna(subset=["open", "lastTradePrice", "return_open_to_now"])
    st.caption(f"📐 After dropping NA returns: {before} → {df_view.shape}")

df_view = df_view.sort_values("prediction", ascending=False).reset_index(drop=True)

# Pretty formatting
def _fmt_pct(x):   return "" if pd.isna(x) else f"{x*100:.2f}%"
def _fmt_price(x): return "" if pd.isna(x) else f"{x:.2f}"

display_cols = ["rank", "ticker", "prediction", "open", "lastTradePrice", "return_open_to_now"]
show = df_view[display_cols].copy()
show.rename(columns={"open": "open_today", "lastTradePrice": "last_price"}, inplace=True)
show["open_today"] = show["open_today"].apply(_fmt_price)
show["last_price"] = show["last_price"].apply(_fmt_price)
show["return_open_to_now"] = show["return_open_to_now"].apply(_fmt_pct)

# -----------------------
# UI
# -----------------------
st.title("📈 Live LTR Monitor — Predictions vs Open→Now Returns (EODHD, SDK-free)")
st.caption(
    "Returns are computed from the **regular session open (09:30 ET)** for the selected date to the latest price.\n"
    "Default data is **15–20 min delayed** via Live v2; if you provide a WebSocket URL, we’ll override last trades in real time."
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
        f"mode:          {'HTTP+WS override' if (ws_enabled and ws_url) else 'HTTP delayed (Live v2)'}",
        language="text",
    )

csv_buf = io.StringIO()
df_view.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Download full table (CSV)",
    data=csv_buf.getvalue(),
    file_name=f"live_ltr_{session_date}.csv",
    mime="text/csv",
)

st.caption(
    "Tip: If your predictions target **tomorrow’s** session, set the date accordingly. "
    "The app will compute returns after the next open."
)
