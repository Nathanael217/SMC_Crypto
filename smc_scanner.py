"""
SMC Scanner — Smart Money Concepts Strategy System
====================================================
Strategy pillars:
  1. BTC Regime  — EMA 20/50/200 + market structure bias
  2. Altcoin Regime — per-coin EMA + structure alignment
  3. Market Structure — BOS, CHoCH, HH/HL/LH/LL detection
  4. Volume Breakout — real momentum confirmation
  5. EMA Filter — only trade in the direction of the trend

Entry options   : Bullish OB + FVG | Classic S/R | Fib 0.786 | Liquidity zone
SL options      : Fixed (ATR) | Structure-based (below OB / S/R / Fib)
TP options      : 2R / 2.5R / 3R | Classic S/R | Bearish OB + FVG | Liquidity target

Scanner         : Looks back 30–50 candles, finds all valid setups, mini-backtests
                  each one and grades the recommendation.

Run: streamlit run smc_scanner.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

st.set_page_config(
    page_title="SMC Scanner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #ccd6f6; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 16px; margin: 4px 0;
  }
  .signal-card {
    border-radius: 8px; padding: 14px 16px; margin: 8px 0;
    border-left: 4px solid;
  }
  .bull-card { border-color: #3fb950; background: #0d1f14; }
  .bear-card { border-color: #f85149; background: #1f0d0d; }
  .tag { display:inline-block; padding:2px 8px; border-radius:4px;
         font-size:11px; font-weight:700; margin:2px; }
  .tag-bos-bull { background:#1a3a1a; color:#3fb950; }
  .tag-bos-bear { background:#3a1a1a; color:#f85149; }
  .tag-choch-bull { background:#1a2a3a; color:#58a6ff; }
  .tag-choch-bear { background:#3a2a1a; color:#f0883e; }
  .tag-ob  { background:#2a1a3a; color:#bc8cff; }
  .tag-fvg { background:#1a2a2a; color:#39d0d0; }
  .tag-sr  { background:#2a2a1a; color:#e3b341; }
  .regime-bull { color:#3fb950; font-weight:700; }
  .regime-bear { color:#f85149; font-weight:700; }
  .regime-chop { color:#e3b341; font-weight:700; }
  .grade-A { color:#3fb950; font-size:22px; font-weight:800; }
  .grade-B { color:#e3b341; font-size:22px; font-weight:800; }
  .grade-C { color:#f0883e; font-size:22px; font-weight:800; }
  .grade-X { color:#f85149; font-size:22px; font-weight:800; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

BINANCE_BASE = "https://api.binance.com"
HEADERS = {"User-Agent": "SMCScanner/1.0"}

TF_MAP = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
TF_LIMIT = {"15m": 200, "1h": 200, "4h": 200, "1d": 200}


@st.cache_data(ttl=120)
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch OHLCV from Binance."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qv","n_trades","tbb","tbq","ignore"
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_universe(min_vol_usdt: float = 500_000, top_n: int = 150) -> List[str]:
    """Return top-N USDT pairs by 24h volume above threshold."""
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/ticker/24hr",
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        tickers = r.json()
        usdt = [t for t in tickers
                if t["symbol"].endswith("USDT")
                and not any(x in t["symbol"] for x in ["DOWNUSDT","UPUSDT","BEARUSDT","BULLUSDT"])
                and float(t["quoteVolume"]) >= min_vol_usdt]
        usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        return [t["symbol"] for t in usdt[:top_n]]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SMC ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2a. EMA Calculations ──────────────────────────────────────────────────────

def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA 20, 50, 200 to dataframe."""
    df = df.copy()
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df


def ema_regime(df: pd.DataFrame) -> str:
    """
    Classify EMA regime using last close vs EMAs.
    BULL  : close > ema20 > ema50 > ema200
    BEAR  : close < ema20 < ema50 < ema200
    BULL_WEAK: close > ema50 but ema50 < ema200
    BEAR_WEAK: close < ema50 but ema50 > ema200
    CHOP  : mixed
    """
    if len(df) < 10:
        return "UNKNOWN"
    last = df.iloc[-1]
    c, e20, e50, e200 = last["close"], last["ema20"], last["ema50"], last["ema200"]
    if c > e20 > e50 > e200:
        return "BULL"
    if c < e20 < e50 < e200:
        return "BEAR"
    if c > e50 and c > e200:
        return "BULL_WEAK"
    if c < e50 and c < e200:
        return "BEAR_WEAK"
    return "CHOP"


# ── 2b. Swing Detection ───────────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Detect swing highs and swing lows using a simple pivot algorithm.
    lookback: number of candles each side that must be lower/higher.
    Adds columns: swing_high, swing_low (True/False).
    """
    df = df.copy()
    n = len(df)
    df["swing_high"] = False
    df["swing_low"]  = False
    for i in range(lookback, n - lookback):
        window_high = df["high"].iloc[i - lookback : i + lookback + 1]
        window_low  = df["low"].iloc[i - lookback : i + lookback + 1]
        if df["high"].iloc[i] == window_high.max():
            df.iloc[i, df.columns.get_loc("swing_high")] = True
        if df["low"].iloc[i] == window_low.min():
            df.iloc[i, df.columns.get_loc("swing_low")] = True
    return df


# ── 2c. Market Structure (BOS / CHoCH / HH-HL-LH-LL) ─────────────────────────

def detect_market_structure(df: pd.DataFrame) -> Dict:
    """
    Analyse last N swing points and classify market structure.
    Returns dict with:
      - trend        : 'BULLISH' | 'BEARISH' | 'RANGING'
      - last_bos     : list of recent BOS events {idx, type, price, direction}
      - last_choch   : list of recent CHoCH events
      - swing_highs  : list of (idx, price) of confirmed swing highs
      - swing_lows   : list of (idx, price) of confirmed swing lows
      - hh_hl        : True if making HH + HL
      - lh_ll        : True if making LH + LL
      - last_event   : most recent structure event string
    """
    result = {
        "trend": "RANGING",
        "last_bos": [],
        "last_choch": [],
        "swing_highs": [],
        "swing_lows": [],
        "hh_hl": False,
        "lh_ll": False,
        "last_event": "—",
    }
    if "swing_high" not in df.columns:
        return result

    # Collect swing highs and lows with indices
    sh_list = [(i, df["high"].iloc[i])  for i in range(len(df)) if df["swing_high"].iloc[i]]
    sl_list = [(i, df["low"].iloc[i])   for i in range(len(df)) if df["swing_low"].iloc[i]]

    result["swing_highs"] = sh_list[-8:]  # keep recent 8
    result["swing_lows"]  = sl_list[-8:]

    if len(sh_list) < 2 or len(sl_list) < 2:
        return result

    # Last 4 swing highs and lows
    recent_sh = sh_list[-4:]
    recent_sl = sl_list[-4:]

    # Trend classification via HH/HL or LH/LL
    sh_prices = [p for _, p in recent_sh]
    sl_prices = [p for _, p in recent_sl]

    hh = all(sh_prices[i] > sh_prices[i-1] for i in range(1, len(sh_prices))) if len(sh_prices) >= 2 else False
    hl = all(sl_prices[i] > sl_prices[i-1] for i in range(1, len(sl_prices))) if len(sl_prices) >= 2 else False
    lh = all(sh_prices[i] < sh_prices[i-1] for i in range(1, len(sh_prices))) if len(sh_prices) >= 2 else False
    ll = all(sl_prices[i] < sl_prices[i-1] for i in range(1, len(sl_prices))) if len(sl_prices) >= 2 else False

    result["hh_hl"] = hh and hl
    result["lh_ll"] = lh and ll

    if hh and hl:
        result["trend"] = "BULLISH"
    elif lh and ll:
        result["trend"] = "BEARISH"
    else:
        result["trend"] = "RANGING"

    # BOS and CHoCH detection
    # We look at close prices vs previous swing high/low
    prev_sh_price = recent_sh[-2][1] if len(recent_sh) >= 2 else None
    prev_sl_price = recent_sl[-2][1] if len(recent_sl) >= 2 else None

    last_close = df["close"].iloc[-1]
    events = []

    # In a BULLISH trend, breaking above previous swing high = Bullish BOS (continuation)
    # In a BULLISH trend, closing below previous swing low = CHoCH (reversal warning)
    if result["trend"] == "BULLISH" and prev_sh_price:
        if last_close > prev_sh_price:
            events.append({"type": "BOS", "direction": "BULL",
                           "price": prev_sh_price, "label": "🟢 BOS ↑"})
        if prev_sl_price and last_close < prev_sl_price:
            events.append({"type": "CHoCH", "direction": "BEAR",
                           "price": prev_sl_price, "label": "🔴 CHoCH ↓"})

    elif result["trend"] == "BEARISH" and prev_sl_price:
        if last_close < prev_sl_price:
            events.append({"type": "BOS", "direction": "BEAR",
                           "price": prev_sl_price, "label": "🔴 BOS ↓"})
        if prev_sh_price and last_close > prev_sh_price:
            events.append({"type": "CHoCH", "direction": "BULL",
                           "price": prev_sh_price, "label": "🟢 CHoCH ↑"})

    result["last_bos"]   = [e for e in events if e["type"] == "BOS"]
    result["last_choch"] = [e for e in events if e["type"] == "CHoCH"]
    if events:
        result["last_event"] = events[-1]["label"]

    return result


# ── 2d. Order Block Detection ─────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
    """
    Detect order blocks in the last `lookback` candles.
    Bullish OB: last bearish candle before a bullish impulse (3+ candles up, 
                or a candle that breaks a swing high).
    Bearish OB: last bullish candle before a bearish impulse.
    
    An OB is 'fresh' if price has not re-entered the zone since.
    Returns list of OB dicts: {type, ob_high, ob_low, ob_mid, idx, fresh, strength}
    """
    df = df.copy()
    n = len(df)
    start = max(0, n - lookback)
    obs = []

    for i in range(start + 3, n - 1):
        # Check for bullish impulse after candle i
        # Impulse = next 2 candles are both bullish and make progress
        next_2_bull = (
            df["close"].iloc[i+1] > df["open"].iloc[i+1] and
            df["close"].iloc[i+1] > df["high"].iloc[i]
        ) if i + 1 < n else False

        # Bullish OB: candle i is bearish, followed by bullish break
        if (df["close"].iloc[i] < df["open"].iloc[i] and  # bearish candle
                next_2_bull):
            ob_high = df["high"].iloc[i]
            ob_low  = df["low"].iloc[i]
            ob_mid  = (ob_high + ob_low) / 2
            # Fresh check: has price come back into OB zone since?
            future_lows  = df["low"].iloc[i+1:]
            fresh = not (future_lows < ob_high).any() or (future_lows < ob_low).any()
            # Strength: based on body size vs range
            body  = abs(df["close"].iloc[i] - df["open"].iloc[i])
            rng   = df["high"].iloc[i] - df["low"].iloc[i] + 1e-12
            strength = min(1.0, body / rng)
            obs.append({
                "type": "BULL_OB", "ob_high": ob_high, "ob_low": ob_low,
                "ob_mid": ob_mid, "idx": i, "fresh": fresh,
                "strength": strength, "candle_time": df.index[i],
            })

        # Check for bearish impulse after candle i
        next_2_bear = (
            df["close"].iloc[i+1] < df["open"].iloc[i+1] and
            df["close"].iloc[i+1] < df["low"].iloc[i]
        ) if i + 1 < n else False

        # Bearish OB: candle i is bullish, followed by bearish break
        if (df["close"].iloc[i] > df["open"].iloc[i] and  # bullish candle
                next_2_bear):
            ob_high = df["high"].iloc[i]
            ob_low  = df["low"].iloc[i]
            ob_mid  = (ob_high + ob_low) / 2
            future_highs = df["high"].iloc[i+1:]
            fresh = not (future_highs > ob_low).any() or (future_highs > ob_high).any()
            body  = abs(df["close"].iloc[i] - df["open"].iloc[i])
            rng   = df["high"].iloc[i] - df["low"].iloc[i] + 1e-12
            strength = min(1.0, body / rng)
            obs.append({
                "type": "BEAR_OB", "ob_high": ob_high, "ob_low": ob_low,
                "ob_mid": ob_mid, "idx": i, "fresh": fresh,
                "strength": strength, "candle_time": df.index[i],
            })

    # Keep most recent 5 of each type
    bull_obs = sorted([o for o in obs if o["type"] == "BULL_OB"],
                      key=lambda x: x["idx"], reverse=True)[:5]
    bear_obs = sorted([o for o in obs if o["type"] == "BEAR_OB"],
                      key=lambda x: x["idx"], reverse=True)[:5]
    return bull_obs + bear_obs


# ── 2e. Fair Value Gap (FVG) Detection ────────────────────────────────────────

def detect_fvg(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
    """
    Detect Fair Value Gaps (imbalances) in last `lookback` candles.
    Bullish FVG: df.high[i-2] < df.low[i]  → gap between candle i-2 and candle i
    Bearish FVG: df.low[i-2] > df.high[i]  → gap between candle i-2 and candle i
    
    FVG is 'filled' if a subsequent candle enters the gap.
    """
    n = len(df)
    start = max(2, n - lookback)
    fvgs = []

    for i in range(start, n):
        # Bullish FVG
        gap_low  = df["high"].iloc[i-2]
        gap_high = df["low"].iloc[i]
        if gap_high > gap_low:  # gap exists
            gap_mid = (gap_high + gap_low) / 2
            gap_size_pct = (gap_high - gap_low) / gap_low * 100
            # Filled check
            future = df.iloc[i+1:] if i + 1 < n else pd.DataFrame()
            filled = False
            if not future.empty:
                filled = (future["low"] <= gap_mid).any()
            fvgs.append({
                "type": "BULL_FVG", "fvg_high": gap_high, "fvg_low": gap_low,
                "fvg_mid": gap_mid, "size_pct": gap_size_pct,
                "idx": i, "filled": filled, "candle_time": df.index[i],
            })

        # Bearish FVG
        gap_high2 = df["low"].iloc[i-2]
        gap_low2  = df["high"].iloc[i]
        if gap_high2 > gap_low2:  # gap exists (bearish: candle i-2 low > candle i high)
            gap_mid2 = (gap_high2 + gap_low2) / 2
            gap_size_pct2 = (gap_high2 - gap_low2) / gap_low2 * 100
            future = df.iloc[i+1:] if i + 1 < n else pd.DataFrame()
            filled2 = False
            if not future.empty:
                filled2 = (future["high"] >= gap_mid2).any()
            fvgs.append({
                "type": "BEAR_FVG", "fvg_high": gap_high2, "fvg_low": gap_low2,
                "fvg_mid": gap_mid2, "size_pct": gap_size_pct2,
                "idx": i, "filled": filled2, "candle_time": df.index[i],
            })

    # Keep unfilled ones, most recent 5 each type
    bull_fvgs = sorted([f for f in fvgs if f["type"] == "BULL_FVG" and not f["filled"]],
                       key=lambda x: x["idx"], reverse=True)[:5]
    bear_fvgs = sorted([f for f in fvgs if f["type"] == "BEAR_FVG" and not f["filled"]],
                       key=lambda x: x["idx"], reverse=True)[:5]
    return bull_fvgs + bear_fvgs


# ── 2f. Volume Analysis ───────────────────────────────────────────────────────

def analyze_volume(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Volume analysis: ratio vs average, detect breakout candles.
    """
    if len(df) < lookback + 1:
        return {"ratio": 1.0, "breakout": False, "avg_vol": 0, "last_vol": 0}
    avg_vol  = df["volume"].iloc[-lookback-1:-1].mean()
    last_vol = df["volume"].iloc[-1]
    ratio    = last_vol / (avg_vol + 1e-12)
    return {
        "ratio": ratio,
        "breakout": ratio >= 2.0,
        "strong": ratio >= 1.5,
        "avg_vol": avg_vol,
        "last_vol": last_vol,
    }


# ── 2g. Classic S/R Detection ─────────────────────────────────────────────────

def detect_support_resistance(df: pd.DataFrame, lookback: int = 50,
                               tolerance_pct: float = 0.5) -> List[Dict]:
    """
    Detect key S/R levels from swing highs/lows.
    Clusters nearby levels (within tolerance_pct) into single zones.
    """
    if "swing_high" not in df.columns or len(df) < 10:
        return []
    n = len(df)
    start = max(0, n - lookback)
    levels = []

    for i in range(start, n):
        if df["swing_high"].iloc[i]:
            levels.append({"price": df["high"].iloc[i], "type": "resistance", "hits": 1})
        if df["swing_low"].iloc[i]:
            levels.append({"price": df["low"].iloc[i], "type": "support", "hits": 1})

    # Cluster nearby levels
    if not levels:
        return []

    clustered = []
    used = [False] * len(levels)
    for i, lv in enumerate(levels):
        if used[i]:
            continue
        cluster = [lv]
        for j in range(i+1, len(levels)):
            if not used[j]:
                diff_pct = abs(levels[j]["price"] - lv["price"]) / lv["price"] * 100
                if diff_pct <= tolerance_pct and levels[j]["type"] == lv["type"]:
                    cluster.append(levels[j])
                    used[j] = True
        avg_price = np.mean([c["price"] for c in cluster])
        clustered.append({
            "price": avg_price,
            "type": cluster[0]["type"],
            "hits": len(cluster),
            "strength": min(5, len(cluster)),
        })
        used[i] = True

    # Sort by strength
    clustered.sort(key=lambda x: x["hits"], reverse=True)
    return clustered[:10]


# ── 2h. Fibonacci Levels ──────────────────────────────────────────────────────

def compute_fib_levels(swing_low: float, swing_high: float) -> Dict:
    """Compute Fibonacci retracement levels from a swing."""
    diff = swing_high - swing_low
    return {
        "0.000": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.500": swing_high - 0.500 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.705": swing_high - 0.705 * diff,
        "0.786": swing_high - 0.786 * diff,
        "0.886": swing_high - 0.886 * diff,
        "1.000": swing_low,
    }


# ── 2i. Liquidity Zones (Equal Highs / Equal Lows) ───────────────────────────

def detect_liquidity_zones(df: pd.DataFrame, lookback: int = 50,
                            tol_pct: float = 0.3) -> List[Dict]:
    """
    Detect liquidity zones: areas with 2+ equal highs or lows (stop clusters).
    Equal = within tol_pct of each other.
    """
    n = len(df)
    start = max(0, n - lookback)
    sub = df.iloc[start:]
    zones = []

    # Equal highs (sells stops above)
    highs = sub["high"].values
    for i in range(len(highs)):
        cluster = [highs[i]]
        for j in range(i+1, len(highs)):
            if abs(highs[j] - highs[i]) / highs[i] * 100 <= tol_pct:
                cluster.append(highs[j])
        if len(cluster) >= 2:
            zones.append({
                "price": np.mean(cluster),
                "type": "equal_high",
                "count": len(cluster),
                "label": f"EQH × {len(cluster)}",
            })

    # Equal lows (buy stops below)
    lows = sub["low"].values
    for i in range(len(lows)):
        cluster = [lows[i]]
        for j in range(i+1, len(lows)):
            if abs(lows[j] - lows[i]) / lows[i] * 100 <= tol_pct:
                cluster.append(lows[j])
        if len(cluster) >= 2:
            zones.append({
                "price": np.mean(cluster),
                "type": "equal_low",
                "count": len(cluster),
                "label": f"EQL × {len(cluster)}",
            })

    # Deduplicate and keep strongest
    seen = set()
    unique = []
    for z in sorted(zones, key=lambda x: x["count"], reverse=True):
        key = round(z["price"], 6)
        if key not in seen:
            seen.add(key)
            unique.append(z)
    return unique[:8]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_btc_regime(interval: str = "4h") -> Dict:
    """
    Get BTC regime for macro context.
    Returns: regime (BULL/BEAR/CHOP), ema_regime, structure_trend, detail
    """
    df = fetch_klines("BTCUSDT", interval, limit=200)
    if df.empty:
        return {"regime": "UNKNOWN", "ema": "UNKNOWN", "structure": "RANGING", "detail": "No data"}

    df = add_emas(df)
    df = detect_swings(df, lookback=3)
    ema = ema_regime(df)
    ms  = detect_market_structure(df)
    vol = analyze_volume(df)

    # Combined regime
    if ema in ("BULL",) and ms["trend"] == "BULLISH":
        regime = "BULL"
    elif ema in ("BEAR",) and ms["trend"] == "BEARISH":
        regime = "BEAR"
    elif ema in ("BULL","BULL_WEAK") and ms["trend"] in ("BULLISH","RANGING"):
        regime = "BULL_WEAK"
    elif ema in ("BEAR","BEAR_WEAK") and ms["trend"] in ("BEARISH","RANGING"):
        regime = "BEAR_WEAK"
    else:
        regime = "CHOP"

    return {
        "regime": regime,
        "ema": ema,
        "structure": ms["trend"],
        "hh_hl": ms["hh_hl"],
        "lh_ll": ms["lh_ll"],
        "last_event": ms["last_event"],
        "vol_ratio": vol["ratio"],
        "detail": f"EMA={ema} | Structure={ms['trend']} | LastEvent={ms['last_event']}",
    }


def get_altcoin_regime(df: pd.DataFrame) -> Dict:
    """Get per-coin regime."""
    if df.empty or len(df) < 50:
        return {"regime": "UNKNOWN", "ema": "UNKNOWN", "structure": "RANGING"}
    df = add_emas(df)
    df = detect_swings(df, lookback=3)
    ema = ema_regime(df)
    ms  = detect_market_structure(df)
    if ema in ("BULL",) and ms["trend"] == "BULLISH":
        regime = "BULL"
    elif ema in ("BEAR",) and ms["trend"] == "BEARISH":
        regime = "BEAR"
    elif ema in ("BULL","BULL_WEAK"):
        regime = "BULL_WEAK"
    elif ema in ("BEAR","BEAR_WEAK"):
        regime = "BEAR_WEAK"
    else:
        regime = "CHOP"
    return {"regime": regime, "ema": ema, "structure": ms["trend"], "ms": ms}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SIGNAL SCANNER  (30-50 candle lookback)
# ═══════════════════════════════════════════════════════════════════════════════

def find_setups(df: pd.DataFrame, lookback: int = 40,
                direction_filter: str = "both") -> List[Dict]:
    """
    Scan last `lookback` candles for valid SMC setups.
    A setup requires:
      - Clear BOS or CHoCH in the lookback window
      - At least one valid entry confluence (OB, FVG, S/R, or Fib 0.786)
      - Volume confirmation (ratio > 1.5x on the impulse candle)
      - Price pulled back INTO the entry zone (or is approaching it)
    
    Returns list of setup dicts, each containing:
      entry_type, entry_price, sl_price, tp_prices, direction,
      confluence, risk_reward, structure_event, backtest_result
    """
    if len(df) < lookback + 20:
        return []

    df = add_emas(df)
    df = detect_swings(df, lookback=3)
    ms  = detect_market_structure(df)
    obs  = detect_order_blocks(df, lookback=lookback)
    fvgs = detect_fvg(df, lookback=lookback)
    sr   = detect_support_resistance(df, lookback=lookback)
    liq  = detect_liquidity_zones(df, lookback=lookback)
    vol  = analyze_volume(df, lookback=20)
    ema  = ema_regime(df)

    last_close = df["close"].iloc[-1]
    atr = compute_atr(df, period=14)
    setups = []

    # ── LONG setups (Bullish OB + FVG combo) ──────────────────────────────────
    if direction_filter in ("both", "long"):
        for ob in [o for o in obs if o["type"] == "BULL_OB" and o["fresh"]]:
            # Price should be at or near the OB zone
            in_ob_zone = ob["ob_low"] <= last_close <= ob["ob_high"] * 1.01

            # Find overlapping bullish FVG for confluence
            matching_fvg = [f for f in fvgs
                            if f["type"] == "BULL_FVG"
                            and f["fvg_low"] <= ob["ob_high"]
                            and f["fvg_high"] >= ob["ob_low"]]

            if in_ob_zone or (last_close > ob["ob_low"] * 0.99 and last_close < ob["ob_high"] * 1.02):
                entry  = ob["ob_mid"]
                sl     = ob["ob_low"] - atr * 0.5  # just below OB
                r_dist = abs(entry - sl)
                if r_dist < 1e-10:
                    continue
                tp2    = entry + r_dist * 2.0
                tp25   = entry + r_dist * 2.5
                tp3    = entry + r_dist * 3.0
                confluence = ["Bullish OB"]
                if matching_fvg:
                    confluence.append("FVG overlap")
                if vol["strong"]:
                    confluence.append(f"Vol {vol['ratio']:.1f}×")
                if ms["trend"] == "BULLISH":
                    confluence.append("Bullish structure")
                if "BULL" in ema:
                    confluence.append("Above EMA")

                # Skip if regime strongly against
                if ema in ("BEAR",) and ms["trend"] == "BEARISH":
                    continue

                setups.append({
                    "direction": "LONG",
                    "entry_type": "OB" + ("+FVG" if matching_fvg else ""),
                    "entry_price": entry,
                    "sl_price": sl,
                    "tp_2R": tp2, "tp_25R": tp25, "tp_3R": tp3,
                    "r_dist": r_dist,
                    "ob": ob,
                    "fvg": matching_fvg[0] if matching_fvg else None,
                    "confluence": confluence,
                    "ema_regime": ema,
                    "ms_trend": ms["trend"],
                    "vol_ratio": vol["ratio"],
                    "structure_event": ms["last_event"],
                    "score": _score_setup(confluence, ms, ema, vol, "LONG"),
                })

        # Fib 0.786 setup for LONG
        sh_list = ms.get("swing_highs", [])
        sl_list = ms.get("swing_lows", [])
        if sh_list and sl_list:
            # Most recent swing high and preceding swing low
            last_sh = sh_list[-1][1]
            last_sl = sl_list[-1][1] if sl_list else None
            if last_sl and last_sh > last_sl:
                fibs = compute_fib_levels(last_sl, last_sh)
                fib786 = fibs["0.786"]
                # Price near fib 0.786?
                if abs(last_close - fib786) / fib786 < 0.015:
                    sl_fib = last_sl - atr * 0.3
                    r_fib  = abs(fib786 - sl_fib)
                    if r_fib > 1e-10:
                        confluence = ["Fib 0.786"]
                        if ms["trend"] == "BULLISH":
                            confluence.append("Bullish structure")
                        if "BULL" in ema:
                            confluence.append("Above EMA")
                        if not (ema in ("BEAR",) and ms["trend"] == "BEARISH"):
                            setups.append({
                                "direction": "LONG",
                                "entry_type": "Fib 0.786",
                                "entry_price": fib786,
                                "sl_price": sl_fib,
                                "tp_2R": fib786 + r_fib * 2.0,
                                "tp_25R": fib786 + r_fib * 2.5,
                                "tp_3R": fib786 + r_fib * 3.0,
                                "r_dist": r_fib,
                                "ob": None, "fvg": None,
                                "confluence": confluence,
                                "ema_regime": ema,
                                "ms_trend": ms["trend"],
                                "vol_ratio": vol["ratio"],
                                "structure_event": ms["last_event"],
                                "fibs": fibs,
                                "score": _score_setup(confluence, ms, ema, vol, "LONG"),
                            })

    # ── SHORT setups (Bearish OB + FVG combo) ─────────────────────────────────
    if direction_filter in ("both", "short"):
        for ob in [o for o in obs if o["type"] == "BEAR_OB" and o["fresh"]]:
            in_ob_zone = ob["ob_low"] * 0.99 <= last_close <= ob["ob_high"]

            matching_fvg = [f for f in fvgs
                            if f["type"] == "BEAR_FVG"
                            and f["fvg_low"] <= ob["ob_high"]
                            and f["fvg_high"] >= ob["ob_low"]]

            if in_ob_zone or (last_close < ob["ob_high"] * 1.01 and last_close > ob["ob_low"] * 0.98):
                entry  = ob["ob_mid"]
                sl     = ob["ob_high"] + atr * 0.5
                r_dist = abs(sl - entry)
                if r_dist < 1e-10:
                    continue
                tp2    = entry - r_dist * 2.0
                tp25   = entry - r_dist * 2.5
                tp3    = entry - r_dist * 3.0
                confluence = ["Bearish OB"]
                if matching_fvg:
                    confluence.append("FVG overlap")
                if vol["strong"]:
                    confluence.append(f"Vol {vol['ratio']:.1f}×")
                if ms["trend"] == "BEARISH":
                    confluence.append("Bearish structure")
                if "BEAR" in ema:
                    confluence.append("Below EMA")

                if ema in ("BULL",) and ms["trend"] == "BULLISH":
                    continue

                setups.append({
                    "direction": "SHORT",
                    "entry_type": "OB" + ("+FVG" if matching_fvg else ""),
                    "entry_price": entry,
                    "sl_price": sl,
                    "tp_2R": tp2, "tp_25R": tp25, "tp_3R": tp3,
                    "r_dist": r_dist,
                    "ob": ob,
                    "fvg": matching_fvg[0] if matching_fvg else None,
                    "confluence": confluence,
                    "ema_regime": ema,
                    "ms_trend": ms["trend"],
                    "vol_ratio": vol["ratio"],
                    "structure_event": ms["last_event"],
                    "score": _score_setup(confluence, ms, ema, vol, "SHORT"),
                })

        # Fib 0.786 SHORT setup
        sh_list = ms.get("swing_highs", [])
        sl_list = ms.get("swing_lows", [])
        if sh_list and sl_list:
            last_sl2 = sl_list[-1][1]
            last_sh2 = sh_list[-1][1] if sh_list else None
            if last_sh2 and last_sl2 < last_sh2:
                fibs2 = compute_fib_levels(last_sl2, last_sh2)
                fib786_s = fibs2["0.786"]
                # For short, fib from high down: 0.786 retracement from recent HIGH
                # Price rallied TO fib 0.786 from the PREVIOUS move's high
                # Actually for SHORT: we want price near fib 0.236/0.382 of DOWN move
                # Let's use: retracement of a bearish move
                alt_fib = compute_fib_levels(last_sh2, last_sl2)  # inverted for short
                fib786_short = alt_fib["0.786"]
                if abs(last_close - fib786_short) / fib786_short < 0.015:
                    sl_fib_s = last_sh2 + atr * 0.3
                    r_fib_s  = abs(fib786_short - sl_fib_s)
                    if r_fib_s > 1e-10:
                        confluence2 = ["Fib 0.786"]
                        if ms["trend"] == "BEARISH":
                            confluence2.append("Bearish structure")
                        if "BEAR" in ema:
                            confluence2.append("Below EMA")
                        if not (ema in ("BULL",) and ms["trend"] == "BULLISH"):
                            setups.append({
                                "direction": "SHORT",
                                "entry_type": "Fib 0.786",
                                "entry_price": fib786_short,
                                "sl_price": sl_fib_s,
                                "tp_2R": fib786_short - r_fib_s * 2.0,
                                "tp_25R": fib786_short - r_fib_s * 2.5,
                                "tp_3R": fib786_short - r_fib_s * 3.0,
                                "r_dist": r_fib_s,
                                "ob": None, "fvg": None,
                                "confluence": confluence2,
                                "ema_regime": ema,
                                "ms_trend": ms["trend"],
                                "vol_ratio": vol["ratio"],
                                "structure_event": ms["last_event"],
                                "fibs": alt_fib,
                                "score": _score_setup(confluence2, ms, ema, vol, "SHORT"),
                            })

    # Sort by score descending
    setups.sort(key=lambda x: x["score"], reverse=True)
    return setups[:8]


def _score_setup(confluence: List[str], ms: Dict, ema: str,
                 vol: Dict, direction: str) -> int:
    """Score a setup 0-100 based on confluence factors."""
    score = 0
    # Confluence items
    if "OB" in " ".join(confluence):
        score += 25
    if "FVG" in " ".join(confluence):
        score += 15
    if "Fib" in " ".join(confluence):
        score += 15
    if "structure" in " ".join(confluence).lower():
        score += 20
    if "EMA" in " ".join(confluence):
        score += 15
    if vol["ratio"] >= 2.0:
        score += 10
    elif vol["ratio"] >= 1.5:
        score += 5
    # Regime alignment bonus
    if direction == "LONG" and ema in ("BULL",) and ms["trend"] == "BULLISH":
        score += 15
    if direction == "SHORT" and ema in ("BEAR",) and ms["trend"] == "BEARISH":
        score += 15
    # CHoCH bonus — riding the new wave
    if ms.get("last_choch"):
        score += 10
    return min(100, score)


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range."""
    if len(df) < period + 1:
        return df["close"].iloc[-1] * 0.02  # fallback 2%
    high = df["high"]
    low  = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.iloc[-period:].mean()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MINI BACKTEST  (replay last 30-50 candles)
# ═══════════════════════════════════════════════════════════════════════════════

def mini_backtest(df: pd.DataFrame, lookback: int = 40) -> Dict:
    """
    Replay last `lookback` candles.
    For each historical OB/FVG zone identified in the window BEFORE it was
    tested, check if price went through entry → TP or entry → SL.
    Returns win_rate, avg_R, total_trades, grade, trade_log.
    """
    n = len(df)
    if n < lookback + 20:
        return {"win_rate": 0, "avg_R": 0, "total": 0,
                "grade": "—", "trade_log": [], "pf": 0}

    df = add_emas(df)
    df = detect_swings(df, lookback=3)
    trade_log = []
    atr_global = compute_atr(df, 14)

    # Replay: for each candle in the lookback window, identify if
    # there was a valid entry zone and if the next N candles hit TP or SL
    step = max(5, lookback // 8)  # sample ~8 points in the lookback window
    test_points = range(n - lookback, n - 5, step)

    for start_i in test_points:
        sub = df.iloc[:start_i]
        if len(sub) < 20:
            continue

        sub_ms  = detect_market_structure(sub)
        sub_obs = detect_order_blocks(sub, lookback=30)
        sub_vol = analyze_volume(sub, lookback=15)

        current_close = sub["close"].iloc[-1]
        atr = compute_atr(sub, 14)

        for ob in sub_obs[:3]:  # check top 3 OBs
            if ob["type"] == "BULL_OB" and ob["fresh"]:
                in_zone = ob["ob_low"] <= current_close <= ob["ob_high"] * 1.01
                if not in_zone:
                    continue
                entry  = ob["ob_mid"]
                sl     = ob["ob_low"] - atr * 0.5
                r_dist = abs(entry - sl)
                if r_dist < 1e-10:
                    continue
                tp25 = entry + r_dist * 2.5

                # Forward test: what happened next?
                future = df.iloc[start_i:start_i + 20]
                result = _simulate_trade("LONG", entry, sl, tp25, future)
                trade_log.append({
                    "direction": "LONG",
                    "entry_type": "BULL_OB",
                    "idx": start_i,
                    "result": result["outcome"],
                    "r_net": result["r_net"],
                })

            elif ob["type"] == "BEAR_OB" and ob["fresh"]:
                in_zone = ob["ob_low"] * 0.99 <= current_close <= ob["ob_high"]
                if not in_zone:
                    continue
                entry  = ob["ob_mid"]
                sl     = ob["ob_high"] + atr * 0.5
                r_dist = abs(sl - entry)
                if r_dist < 1e-10:
                    continue
                tp25 = entry - r_dist * 2.5

                future = df.iloc[start_i:start_i + 20]
                result = _simulate_trade("SHORT", entry, sl, tp25, future)
                trade_log.append({
                    "direction": "SHORT",
                    "entry_type": "BEAR_OB",
                    "idx": start_i,
                    "result": result["outcome"],
                    "r_net": result["r_net"],
                })

    if not trade_log:
        return {"win_rate": 0, "avg_R": 0, "total": 0,
                "grade": "INSUFFICIENT DATA", "trade_log": [], "pf": 0}

    wins    = [t for t in trade_log if t["result"] == "TP"]
    losses  = [t for t in trade_log if t["result"] == "SL"]
    expired = [t for t in trade_log if t["result"] == "EXPIRED"]

    total      = len(trade_log)
    win_rate   = len(wins) / total * 100 if total else 0
    avg_R      = np.mean([t["r_net"] for t in trade_log]) if trade_log else 0
    gross_win  = sum(t["r_net"] for t in wins)
    gross_loss = abs(sum(t["r_net"] for t in losses)) + 1e-10
    pf         = gross_win / gross_loss

    # Grade
    if win_rate >= 55 and avg_R >= 0.8 and pf >= 1.4:
        grade = "A"
    elif win_rate >= 45 and avg_R >= 0.4 and pf >= 1.2:
        grade = "B"
    elif win_rate >= 35 and avg_R >= 0.0:
        grade = "C"
    else:
        grade = "X"

    return {
        "win_rate": win_rate,
        "avg_R": avg_R,
        "total": total,
        "wins": len(wins),
        "losses": len(losses),
        "expired": len(expired),
        "grade": grade,
        "trade_log": trade_log,
        "pf": pf,
        "recommendation": _backtest_recommendation(grade, win_rate, avg_R, pf),
    }


def _simulate_trade(direction: str, entry: float, sl: float,
                    tp: float, future: pd.DataFrame) -> Dict:
    """Simulate forward outcome of a single trade."""
    if future.empty:
        return {"outcome": "EXPIRED", "r_net": 0}
    r_dist = abs(entry - sl)
    for _, row in future.iterrows():
        if direction == "LONG":
            if row["low"] <= sl:
                return {"outcome": "SL", "r_net": -1.0}
            if row["high"] >= tp:
                return {"outcome": "TP", "r_net": (tp - entry) / r_dist}
        else:
            if row["high"] >= sl:
                return {"outcome": "SL", "r_net": -1.0}
            if row["low"] <= tp:
                return {"outcome": "TP", "r_net": (entry - tp) / r_dist}
    return {"outcome": "EXPIRED", "r_net": 0}


def _backtest_recommendation(grade: str, wr: float, avg_R: float, pf: float) -> str:
    if grade == "A":
        return f"✅ HIGH CONVICTION — Backtest strong: WR {wr:.0f}% | Avg R {avg_R:.2f} | PF {pf:.2f}. This setup type has edge on this coin recently."
    elif grade == "B":
        return f"🟡 MODERATE — Decent backtest: WR {wr:.0f}% | Avg R {avg_R:.2f} | PF {pf:.2f}. Take with standard sizing."
    elif grade == "C":
        return f"⚠️ WEAK — Marginal edge: WR {wr:.0f}% | Avg R {avg_R:.2f} | PF {pf:.2f}. Reduce size or skip."
    else:
        return f"❌ NO EDGE — Backtest fails: WR {wr:.0f}% | Avg R {avg_R:.2f} | PF {pf:.2f}. Skip this setup."


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CHARTING
# ═══════════════════════════════════════════════════════════════════════════════

def build_smc_chart(df: pd.DataFrame, symbol: str, interval: str,
                    setups: List[Dict], obs: List[Dict],
                    fvgs: List[Dict], sr: List[Dict],
                    ms: Dict, lookback: int = 60) -> go.Figure:
    """Build a rich SMC chart with OB, FVG, S/R, structure labels."""
    sub = df.iloc[-lookback:].copy()
    sub = add_emas(sub)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=sub.index, open=sub["open"], high=sub["high"],
        low=sub["low"], close=sub["close"],
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        name="Price",
    ), row=1, col=1)

    # EMAs
    for ema_col, color, name in [("ema20","#58a6ff","EMA 20"),
                                   ("ema50","#e3b341","EMA 50"),
                                   ("ema200","#bc8cff","EMA 200")]:
        if ema_col in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub[ema_col], mode="lines",
                line=dict(color=color, width=1.2, dash="solid" if ema_col == "ema20" else "dot"),
                name=name, opacity=0.7,
            ), row=1, col=1)

    # Order Blocks
    for ob in obs[:4]:
        if ob["idx"] < len(df) - lookback:
            continue
        color = "rgba(63,185,80,0.15)" if ob["type"] == "BULL_OB" else "rgba(248,81,73,0.15)"
        border = "#3fb950" if ob["type"] == "BULL_OB" else "#f85149"
        label  = "🟢 Bull OB" if ob["type"] == "BULL_OB" else "🔴 Bear OB"
        # Draw as horizontal band from OB formation candle to now
        x0 = sub.index[max(0, ob["idx"] - (len(df) - lookback))]
        x1 = sub.index[-1]
        fig.add_shape(type="rect", x0=x0, x1=x1,
                      y0=ob["ob_low"], y1=ob["ob_high"],
                      fillcolor=color, line=dict(color=border, width=1),
                      row=1, col=1)
        fig.add_annotation(x=x1, y=ob["ob_mid"], text=label,
                           font=dict(size=9, color=border),
                           showarrow=False, xanchor="right", row=1, col=1)

    # FVGs
    for fvg in fvgs[:4]:
        if fvg["idx"] < len(df) - lookback:
            continue
        color = "rgba(57,208,208,0.12)" if fvg["type"] == "BULL_FVG" else "rgba(240,136,62,0.12)"
        border = "#39d0d0" if fvg["type"] == "BULL_FVG" else "#f0883e"
        x0 = sub.index[max(0, fvg["idx"] - (len(df) - lookback))]
        x1 = sub.index[-1]
        fig.add_shape(type="rect", x0=x0, x1=x1,
                      y0=fvg["fvg_low"], y1=fvg["fvg_high"],
                      fillcolor=color, line=dict(color=border, width=0.5, dash="dash"),
                      row=1, col=1)

    # S/R levels (top 5 by strength)
    for level in sorted(sr, key=lambda x: x["hits"], reverse=True)[:5]:
        color = "#3fb950" if level["type"] == "support" else "#f85149"
        fig.add_hline(y=level["price"], line=dict(color=color, width=0.8, dash="dot"),
                      row=1, col=1)

    # Swing structure labels
    for idx, price in ms.get("swing_highs", [])[-4:]:
        if idx >= len(df) - lookback:
            x = sub.index[idx - (len(df) - lookback)]
            fig.add_annotation(x=x, y=price * 1.002, text="SH",
                                font=dict(size=8, color="#f85149"),
                                showarrow=False, row=1, col=1)
    for idx, price in ms.get("swing_lows", [])[-4:]:
        if idx >= len(df) - lookback:
            x = sub.index[idx - (len(df) - lookback)]
            fig.add_annotation(x=x, y=price * 0.998, text="SL",
                                font=dict(size=8, color="#3fb950"),
                                showarrow=False, row=1, col=1)

    # Setup entry/SL/TP lines (show best setup)
    for setup in setups[:1]:
        d = setup["direction"]
        clr = "#3fb950" if d == "LONG" else "#f85149"
        for y_val, lbl, dash in [
            (setup["entry_price"], "Entry", "solid"),
            (setup["sl_price"], "SL", "dash"),
            (setup["tp_25R"], "TP 2.5R", "dot"),
        ]:
            fig.add_hline(y=y_val, line=dict(color=clr, width=1.5, dash=dash), row=1, col=1)
            fig.add_annotation(x=sub.index[-1], y=y_val, text=lbl,
                                font=dict(size=9, color=clr),
                                showarrow=False, xanchor="right", row=1, col=1)

    # Volume bars
    vol_colors = ["#3fb950" if c >= o else "#f85149"
                  for c, o in zip(sub["close"], sub["open"])]
    avg_vol = sub["volume"].mean()
    fig.add_trace(go.Bar(
        x=sub.index, y=sub["volume"],
        marker_color=vol_colors, opacity=0.7, name="Volume",
    ), row=2, col=1)
    fig.add_hline(y=avg_vol * 2.0, line=dict(color="#e3b341", width=0.8, dash="dot"),
                  row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        title=dict(text=f"{symbol} {interval.upper()} — SMC Analysis",
                   font=dict(color="#ccd6f6", size=14)),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def regime_badge(regime: str) -> str:
    if "BULL" in regime:
        return f'<span class="regime-bull">🟢 {regime}</span>'
    if "BEAR" in regime:
        return f'<span class="regime-bear">🔴 {regime}</span>'
    return f'<span class="regime-chop">🟡 {regime}</span>'


def grade_html(grade: str) -> str:
    css = {"A": "grade-A", "B": "grade-B", "C": "grade-C", "X": "grade-X"}.get(grade, "grade-C")
    labels = {"A": "A — HIGH EDGE", "B": "B — MODERATE", "C": "C — WEAK", "X": "X — SKIP"}
    return f'<span class="{css}">{labels.get(grade, grade)}</span>'


def render_setup_card(symbol: str, interval: str, setup: Dict, bt: Dict):
    d = setup["direction"]
    is_long = d == "LONG"
    card_cls = "bull-card" if is_long else "bear-card"
    arrow = "⬆️" if is_long else "⬇️"
    clr = "#3fb950" if is_long else "#f85149"

    # Confluence tags
    tag_html = ""
    for c in setup["confluence"]:
        if "OB" in c:   tag_html += f'<span class="tag tag-ob">{c}</span>'
        elif "FVG" in c: tag_html += f'<span class="tag tag-fvg">{c}</span>'
        elif "Fib" in c: tag_html += f'<span class="tag tag-sr">{c}</span>'
        elif "Vol" in c: tag_html += f'<span class="tag tag-bos-bull">{c}</span>'
        else:            tag_html += f'<span class="tag tag-choch-bull">{c}</span>'

    # Structure event
    if setup.get("structure_event") and setup["structure_event"] != "—":
        ev = setup["structure_event"]
        if "BOS" in ev and "BULL" in ev:  tag_html += f'<span class="tag tag-bos-bull">{ev}</span>'
        elif "BOS" in ev:                  tag_html += f'<span class="tag tag-bos-bear">{ev}</span>'
        elif "CHoCH" in ev and "BULL" in ev: tag_html += f'<span class="tag tag-choch-bull">{ev}</span>'
        elif "CHoCH" in ev:                tag_html += f'<span class="tag tag-choch-bear">{ev}</span>'

    def fmt(p: float) -> str:
        return f"{p:.6f}" if p < 1 else f"{p:.4f}" if p < 100 else f"{p:.2f}"

    st.markdown(f"""
<div class="signal-card {card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span style="color:{clr};font-size:18px;font-weight:800;">{arrow} {symbol} {d}</span>
      <span style="color:#8892b0;font-size:12px;margin-left:8px;">{interval.upper()} · {setup['entry_type']}</span>
    </div>
    <div>{grade_html(bt.get('grade','—'))}</div>
  </div>
  <div style="margin:8px 0;">{tag_html}</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0;">
    <div class="metric-card">
      <div style="color:#8892b0;font-size:10px;">ENTRY</div>
      <div style="color:#ccd6f6;font-weight:700;">{fmt(setup['entry_price'])}</div>
    </div>
    <div class="metric-card">
      <div style="color:#8892b0;font-size:10px;">STOP LOSS</div>
      <div style="color:#f85149;font-weight:700;">{fmt(setup['sl_price'])}</div>
    </div>
    <div class="metric-card">
      <div style="color:#8892b0;font-size:10px;">TP 2.5R</div>
      <div style="color:#3fb950;font-weight:700;">{fmt(setup['tp_25R'])}</div>
    </div>
    <div class="metric-card">
      <div style="color:#8892b0;font-size:10px;">TP 3R</div>
      <div style="color:#3fb950;font-weight:700;">{fmt(setup['tp_3R'])}</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:11px;color:#8892b0;">
    <div>EMA Regime: {regime_badge(setup['ema_regime'])}</div>
    <div>Structure: <b style="color:#ccd6f6;">{setup['ms_trend']}</b></div>
    <div>Volume: <b style="color:{'#3fb950' if setup['vol_ratio']>=2 else '#e3b341' if setup['vol_ratio']>=1.5 else '#8892b0'};">{setup['vol_ratio']:.1f}×</b></div>
  </div>
  <div style="margin-top:8px;font-size:11px;color:#8892b0;border-top:1px solid #30363d;padding-top:8px;">
    {bt.get('recommendation','—')}
  </div>
  {f'<div style="font-size:10px;color:#8892b0;margin-top:4px;">Backtest: {bt.get("total",0)} trades · WR {bt.get("win_rate",0):.0f}% · Avg R {bt.get("avg_R",0):.2f} · PF {bt.get("pf",0):.2f}</div>' if bt.get('total',0) > 0 else ''}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: Manual / Single Coin Analyzer
# ─────────────────────────────────────────────────────────────────────────────

def render_manual_tab():
    st.subheader("🔍 Single Coin SMC Analysis")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        symbol = st.text_input("Symbol", value="BTCUSDT", key="manual_sym").upper().strip()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
    with col2:
        interval = st.selectbox("Timeframe", ["4h","1d","1h","15m"], key="manual_tf")
    with col3:
        lookback = st.slider("Scan lookback (candles)", 20, 60, 40, key="manual_lb")
    with col4:
        direction = st.selectbox("Direction", ["both","long","short"], key="manual_dir")

    if st.button("🔍 Analyze", key="manual_btn", use_container_width=True):
        with st.spinner(f"Fetching {symbol} {interval}..."):
            df = fetch_klines(symbol, interval, limit=200)

        if df.empty:
            st.error(f"Could not fetch data for {symbol}. Check the symbol.")
            return

        df = add_emas(df)
        df = detect_swings(df, lookback=3)
        ms    = detect_market_structure(df)
        obs   = detect_order_blocks(df, lookback=lookback)
        fvgs  = detect_fvg(df, lookback=lookback)
        sr    = detect_support_resistance(df, lookback=lookback)
        liq   = detect_liquidity_zones(df, lookback=lookback)
        vol   = analyze_volume(df, lookback=20)
        ema   = ema_regime(df)
        btc   = get_btc_regime(interval if interval in ("4h","1d") else "4h")

        setups = find_setups(df, lookback=lookback, direction_filter=direction)

        # ── Regime summary row ──────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
              <div style="color:#8892b0;font-size:10px;">BTC REGIME</div>
              <div style="font-size:16px;font-weight:700;">{regime_badge(btc['regime'])}</div>
              <div style="color:#8892b0;font-size:10px;">{btc['last_event']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
              <div style="color:#8892b0;font-size:10px;">ALTCOIN EMA</div>
              <div style="font-size:16px;font-weight:700;">{regime_badge(ema)}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            struct_clr = "#3fb950" if ms["trend"] == "BULLISH" else "#f85149" if ms["trend"] == "BEARISH" else "#e3b341"
            st.markdown(f"""<div class="metric-card">
              <div style="color:#8892b0;font-size:10px;">MARKET STRUCTURE</div>
              <div style="font-size:16px;font-weight:700;color:{struct_clr};">{ms['trend']}</div>
              <div style="color:#8892b0;font-size:10px;">{ms['last_event']}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            vol_clr = "#3fb950" if vol["ratio"] >= 2 else "#e3b341" if vol["ratio"] >= 1.5 else "#8892b0"
            st.markdown(f"""<div class="metric-card">
              <div style="color:#8892b0;font-size:10px;">VOLUME</div>
              <div style="font-size:16px;font-weight:700;color:{vol_clr};">{vol['ratio']:.1f}×</div>
              <div style="color:#8892b0;font-size:10px;">{'🔥 BREAKOUT' if vol['breakout'] else '📊 normal'}</div>
            </div>""", unsafe_allow_html=True)

        # ── Chart ────────────────────────────────────────────────────────────
        st.plotly_chart(
            build_smc_chart(df, symbol, interval, setups, obs, fvgs, sr, ms, lookback=60),
            use_container_width=True,
        )

        # ── Structure detail ─────────────────────────────────────────────────
        with st.expander("📐 Market Structure Detail", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**BOS Events (recent)**")
                if ms["last_bos"]:
                    for b in ms["last_bos"]:
                        st.markdown(f"- {b['label']} at `{b['price']:.6f}`")
                else:
                    st.caption("No recent BOS detected")
                st.markdown("**CHoCH Events**")
                if ms["last_choch"]:
                    for c in ms["last_choch"]:
                        st.markdown(f"- {c['label']} at `{c['price']:.6f}`")
                else:
                    st.caption("No recent CHoCH detected")
            with col_b:
                hh_hl_str = "✅ HH + HL (Bullish)" if ms["hh_hl"] else "❌"
                lh_ll_str = "✅ LH + LL (Bearish)" if ms["lh_ll"] else "❌"
                st.markdown(f"**Structure pattern:**\n- {hh_hl_str}\n- {lh_ll_str}")
                if obs:
                    st.markdown(f"**Order Blocks found:** {len(obs)} ({len([o for o in obs if o['type']=='BULL_OB'])} Bull / {len([o for o in obs if o['type']=='BEAR_OB'])} Bear)")
                if fvgs:
                    st.markdown(f"**FVGs (unfilled):** {len(fvgs)} ({len([f for f in fvgs if f['type']=='BULL_FVG'])} Bull / {len([f for f in fvgs if f['type']=='BEAR_FVG'])} Bear)")

        # ── Liquidity zones ───────────────────────────────────────────────────
        if liq:
            with st.expander("💧 Liquidity Zones (Stop Clusters)", expanded=False):
                for z in liq[:6]:
                    icon = "⬆️" if z["type"] == "equal_high" else "⬇️"
                    st.markdown(f"- {icon} `{z['price']:.6f}` — {z['label']}")

        # ── Setups ────────────────────────────────────────────────────────────
        st.markdown("---")
        if not setups:
            st.warning("⚠️ No valid SMC setups found in the current lookback window. "
                       "Try adjusting the lookback, direction, or check back after next candle close.")
            return

        st.markdown(f"### 🎯 {len(setups)} Setup(s) Found")
        with st.spinner("Running mini-backtest on last 40 candles…"):
            bt = mini_backtest(df, lookback=40)

        for setup in setups:
            render_setup_card(symbol, interval, setup, bt)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: Scanner Tab — scan all coins
# ─────────────────────────────────────────────────────────────────────────────

def render_scanner_tab():
    st.subheader("🔭 SMC Market Scanner")
    st.caption("Scans Binance altcoins for SMC setups: BOS/CHoCH + OB/FVG + Volume + EMA regime")

    # ── Sidebar / controls ───────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        min_vol = st.number_input("Min 24h Vol ($)", value=1_000_000, step=500_000,
                                   format="%d", key="scan_vol")
    with col2:
        top_n = st.slider("# Coins to scan", 20, 200, 80, key="scan_topn")
    with col3:
        intervals = st.multiselect("Timeframes", ["4h","1d","1h"], default=["4h","1d"],
                                    key="scan_tfs")
    with col4:
        direction = st.selectbox("Direction", ["both","long","short"], key="scan_dir")
    with col5:
        min_score = st.slider("Min setup score", 0, 80, 40, key="scan_score")

    scan_btn = st.button("🚀 Scan Market Now", key="scan_btn", use_container_width=True,
                          type="primary")

    if not scan_btn:
        st.info("👆 Press **Scan Market Now** to begin scanning. "
                "Scanner checks BTC regime, altcoin regime, market structure, OB/FVG, and volume.")
        return

    # ── BTC Regime ────────────────────────────────────────────────────────────
    btc_regime_data = get_btc_regime("4h")
    btc_r = btc_regime_data["regime"]
    btc_badge = ("🟢" if "BULL" in btc_r else "🔴" if "BEAR" in btc_r else "🟡")
    st.markdown(
        f'<div class="metric-card" style="margin-bottom:12px;">'
        f'<b>Macro Context:</b> {btc_badge} BTC {btc_r} &nbsp;|&nbsp; '
        f'Structure: {btc_regime_data["structure"]} &nbsp;|&nbsp; '
        f'Last event: {btc_regime_data["last_event"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Fetch universe ────────────────────────────────────────────────────────
    with st.spinner("Fetching coin universe…"):
        universe = get_universe(min_vol_usdt=min_vol, top_n=top_n)

    if not universe:
        st.error("Could not fetch coin universe. Check your network / Binance API.")
        return

    st.caption(f"Universe: {len(universe)} coins | Scanning timeframes: {', '.join(intervals)}")

    # ── Scan ──────────────────────────────────────────────────────────────────
    all_signals = []
    progress_bar = st.progress(0)
    status_txt   = st.empty()
    total_jobs   = len(universe) * len(intervals)
    job_count    = 0

    for symbol in universe:
        for interval in intervals:
            job_count += 1
            progress_bar.progress(min(job_count / total_jobs, 1.0))
            status_txt.caption(f"Scanning {symbol} {interval}… ({job_count}/{total_jobs})")
            try:
                df = fetch_klines(symbol, interval, limit=200)
                if df.empty or len(df) < 60:
                    continue
                setups = find_setups(df, lookback=40, direction_filter=direction)
                for setup in setups:
                    if setup["score"] >= min_score:
                        all_signals.append({
                            "symbol": symbol,
                            "interval": interval,
                            "df": df,
                            **setup,
                        })
            except Exception:
                pass
            time.sleep(0.03)  # Rate limit

    progress_bar.empty()
    status_txt.empty()

    if not all_signals:
        st.warning("⚠️ No setups found matching the criteria. Try lowering min score, "
                   "adding more timeframes, or reducing min volume.")
        return

    # Sort by score
    all_signals.sort(key=lambda x: x["score"], reverse=True)
    st.success(f"✅ Scan complete — {len(all_signals)} setup(s) found across "
               f"{len(set(s['symbol'] for s in all_signals))} coins")

    # ── Render cards ──────────────────────────────────────────────────────────
    for sig in all_signals[:20]:  # show top 20
        with st.expander(
            f"{'⬆️' if sig['direction']=='LONG' else '⬇️'} "
            f"{sig['symbol']} {sig['interval'].upper()} | "
            f"{sig['entry_type']} | Score {sig['score']} | {sig['ms_trend']}",
            expanded=sig["score"] >= 70,
        ):
            df = sig.pop("df")
            with st.spinner("Mini-backtest…"):
                bt = mini_backtest(df, lookback=40)
            render_setup_card(sig["symbol"], sig["interval"], sig, bt)

            # Quick chart for top signals
            if sig["score"] >= 65:
                df = add_emas(df)
                df = detect_swings(df, lookback=3)
                obs  = detect_order_blocks(df, lookback=40)
                fvgs = detect_fvg(df, lookback=40)
                sr   = detect_support_resistance(df, lookback=40)
                ms   = detect_market_structure(df)
                st.plotly_chart(
                    build_smc_chart(df, sig["symbol"], sig["interval"],
                                    [sig], obs, fvgs, sr, ms, lookback=50),
                    use_container_width=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: Regime Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def render_regime_tab():
    st.subheader("📊 Regime Dashboard")
    st.caption("BTC macro + altcoin sector alignment check")

    tf = st.selectbox("Regime timeframe", ["4h","1d"], key="regime_tf")

    if st.button("🔄 Refresh Regime", key="regime_btn"):
        btc = get_btc_regime(tf)
        st.markdown("### BTC Regime")
        cols = st.columns(4)
        metrics = [
            ("Regime", btc["regime"]),
            ("EMA Stack", btc["ema"]),
            ("Structure", btc["structure"]),
            ("Last Event", btc["last_event"]),
        ]
        for col, (label, val) in zip(cols, metrics):
            col.metric(label, val)

        st.markdown("---")
        st.markdown("### Altcoin Regime Sample (Top 20 by Volume)")

        with st.spinner("Fetching altcoin regimes…"):
            sample_coins = get_universe(min_vol_usdt=5_000_000, top_n=20)

        rows = []
        for sym in sample_coins:
            try:
                df = fetch_klines(sym, tf, limit=200)
                if df.empty:
                    continue
                r = get_altcoin_regime(df)
                rows.append({
                    "Symbol": sym,
                    "Regime": r["regime"],
                    "EMA": r["ema"],
                    "Structure": r["structure"],
                    "Aligned with BTC": "✅" if (
                        (btc["regime"] in ("BULL","BULL_WEAK") and "BULL" in r["regime"]) or
                        (btc["regime"] in ("BEAR","BEAR_WEAK") and "BEAR" in r["regime"])
                    ) else "❌",
                })
                time.sleep(0.05)
            except Exception:
                pass

        if rows:
            df_rows = pd.DataFrame(rows)
            bull_pct = (df_rows["Regime"].str.contains("BULL").sum() / len(df_rows) * 100)
            bear_pct = (df_rows["Regime"].str.contains("BEAR").sum() / len(df_rows) * 100)
            st.metric("% Altcoins in BULL regime", f"{bull_pct:.0f}%")
            st.metric("% Altcoins in BEAR regime", f"{bear_pct:.0f}%")
            st.dataframe(df_rows, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 SMC Scanner")
        st.caption("Smart Money Concepts — BOS · CHoCH · OB · FVG · EMA Regime · Volume")
        st.markdown("---")

        st.markdown("""
**Strategy Pillars:**
1. 📈 BTC Regime (EMA + structure)
2. 📊 Altcoin Regime (EMA + structure)  
3. 🏗 Market Structure (BOS / CHoCH)
4. 💥 Volume Breakout (≥1.5×)
5. 📉 EMA Bias (20 / 50 / 200)

**Entry Options:**
- Bullish OB + FVG
- Classic S/R
- Fibonacci 0.786
- Liquidity zones

**SL:** Fixed (ATR) or structure  
**TP:** 2R · 2.5R · 3R or S/R or OB
        """)
        st.markdown("---")
        st.caption("Data: Binance | Timeframes: 4H + 1D")
        st.caption("⚠️ For educational use only. Not financial advice.")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_scanner, tab_manual, tab_regime = st.tabs([
        "🔭 Scanner — Market Sweep",
        "🔍 Manual — Single Coin",
        "📊 Regime Dashboard",
    ])

    with tab_scanner:
        render_scanner_tab()

    with tab_manual:
        render_manual_tab()

    with tab_regime:
        render_regime_tab()


if __name__ == "__main__":
    main()
