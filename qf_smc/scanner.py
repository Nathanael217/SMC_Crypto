"""
qf_smc/scanner.py — Multi-TF SMC scanner orchestrator
======================================================
Coordinates structure detection, zone detection, LTF confirmation, backtest,
and trade plan building per QUANTFLOW SMC Long spec v1.1 §5.5.

Public API:
  - MODE_CONFIG: dict
  - run_scan(mode, symbols, btc_regime, progress_callback)
  - scan_one_symbol(symbol, mode, btc_regime)
  - helpers: check_ltf_confirmation, build_trade_plan, classify_ema_tier, pick_primary_zone
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable

from qf_shared import (
    _scanner_fetch_candles,
    calculate_ema,
    _clean_df,
)
from qf_smc.structure import detect_swings, classify_structure, is_uptrend_confirmed
from qf_smc.zones import (
    detect_smart_obs, detect_fvgs, detect_fibo_levels, detect_sr_levels,
    classify_current_price_in_zones,
)
from qf_smc.backtest import (
    backtest_per_coin, backtest_universe_baseline,
    bayesian_blend, compute_recent_check,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "SWING": {"htf": "1d",  "ltf": "4h",  "lookback": 50, "htf_label": "1D", "ltf_label": "4H"},
    "DAY":   {"htf": "4h",  "ltf": "15m", "lookback": 50, "htf_label": "4H", "ltf_label": "15m"},
    "SCALP": {"htf": "1h",  "ltf": "5m",  "lookback": 50, "htf_label": "1H", "ltf_label": "5m"},
}

# Map entry zone types to backtest method combos (sl_method, tp_method)
_ZONE_BACKTEST_PARAMS: Dict[str, Dict[str, str]] = {
    "smart_ob":  {"sl_method": "structural", "tp_method": "fixed_R"},
    "fvg":       {"sl_method": "structural", "tp_method": "fixed_R"},
    "fibo_786":  {"sl_method": "fixed",      "tp_method": "fixed_R"},
    "sr":        {"sl_method": "structural", "tp_method": "fixed_R"},
}

# Default TP R-multiple for backtest
_DEFAULT_TP_R = 2.0


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_scan(
    mode: str,
    symbols: List[str],
    btc_regime: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Run a full SMC scan over the provided symbol list in the chosen mode.

    Args:
        mode: "SWING" | "DAY" | "SCALP"
        symbols: list of symbol strings to scan (from Session 04 screener output)
        btc_regime: "BULL" | "CHOP" | "BEAR" | "UNKNOWN"
                    (used for BADGE + AI context only — NOT a skip gate.
                    Scanner runs symbols regardless of BTC regime.)
        progress_callback: optional callback(current_idx, total, symbol_name)
                          fired once per symbol for UI progress display

    Returns:
        List of result dicts (one per symbol that produced a setup).
        Symbols that fail HTF structure check or have no zone retracement
        return None and are excluded from the result list.
    """
    results: List[Dict[str, Any]] = []

    for idx, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(idx + 1, len(symbols), symbol)
        try:
            result = scan_one_symbol(symbol, mode, btc_regime)
            if result is not None:
                results.append(result)
        except Exception as e:
            # Log error to console but continue scanning
            print(f"[scanner] error scanning {symbol}: {e}")
            continue

    return results


# ============================================================================
# SINGLE-SYMBOL PIPELINE
# ============================================================================

def scan_one_symbol(
    symbol: str,
    mode: str,
    btc_regime: str,
) -> Optional[Dict[str, Any]]:
    """
    Full pipeline for ONE symbol.

    Steps:
        1.  Fetch HTF candles (mode-specific TF, lookback bars)
        2.  Fetch LTF candles (mode-specific TF, lookback bars)
        3.  If either fetch fails or returns < 30 bars → return None
        4.  Run structure.detect_swings + classify_structure on HTF
        5.  If state not in {"BOS", "UPTREND"} → return None
        6.  Run zones detectors on HTF
        7.  Check EMA 50 & EMA 200 confidence tier
        8.  classify_current_price_in_zones(current_price, ...)
        9.  If classification['any'] is False → return None
       10.  Run LTF confirmation check
       11.  Pick PRIMARY zone
       12.  Run backtest for the primary zone's setup
       13.  Compute fights_macro = btc_regime in {"BEAR"}
       14.  Build trade_plan
       15.  Return result dict

    Returns:
        Full result dict, or None if filtered out.
    """
    cfg = MODE_CONFIG.get(mode)
    if cfg is None:
        print(f"[scanner] Unknown mode {mode!r}; valid: {list(MODE_CONFIG.keys())}")
        return None

    htf_interval = cfg["htf"]
    ltf_interval = cfg["ltf"]
    lookback     = cfg["lookback"]

    # ── Step 1-2: Fetch candles ──────────────────────────────────────────────
    # Use deeper fetch for HTF to give backtest meaningful history (≥200 bars)
    df_htf = _scanner_fetch_candles(symbol, htf_interval, limit=500)
    df_ltf = _scanner_fetch_candles(symbol, ltf_interval, limit=lookback + 50)

    # ── Step 3: Validate data ────────────────────────────────────────────────
    if df_htf is None or df_htf.empty or len(df_htf) < 30:
        return None
    if df_ltf is None or df_ltf.empty or len(df_ltf) < 30:
        return None

    # Work on reset-index copies for positional consistency
    df_htf = df_htf.reset_index(drop=True)
    df_ltf = df_ltf.reset_index(drop=True)

    # ── Step 4: Structure detection on HTF ──────────────────────────────────
    swings    = detect_swings(df_htf, pivot=5)
    structure = classify_structure(swings, df_htf)

    # ── Step 5: Structure filter ─────────────────────────────────────────────
    if not is_uptrend_confirmed(structure):
        return None  # state is UNDEFINED, DOWNTREND, or CHOCH

    # ── Step 6: Zone detection on HTF ────────────────────────────────────────
    smart_obs = detect_smart_obs(df_htf, structure)
    fvgs      = detect_fvgs(df_htf, structure)
    fibo      = detect_fibo_levels(df_htf, structure)
    sr_levels = detect_sr_levels(df_htf, lookback=lookback)

    # Early exit if no zones at all (saves EMA computation)
    zones_exist = bool(smart_obs or fvgs or fibo or sr_levels)
    if not zones_exist:
        return None

    # ── Step 7: EMA tier ─────────────────────────────────────────────────────
    current_price = float(df_htf["close"].iloc[-1])
    ema_tier = classify_ema_tier(df_htf, current_price)
    if ema_tier == "SKIP":
        return None

    # ── Step 8: Zone classification ──────────────────────────────────────────
    zone_cls = classify_current_price_in_zones(
        current_price, smart_obs, fvgs, fibo, sr_levels
    )

    # ── Step 9: Require at least one actionable zone ─────────────────────────
    if not zone_cls.get("any", False):
        return None

    # ── Step 10: Pick primary zone ───────────────────────────────────────────
    primary_zone = pick_primary_zone(zone_cls, fibo=fibo)
    if primary_zone is None:
        return None

    # ── Step 11: LTF confirmation ─────────────────────────────────────────────
    ltf_result = check_ltf_confirmation(df_ltf, primary_zone)
    ltf_confirmation = ltf_result["status"]
    ltf_signal_bar   = ltf_result.get("signal_bar")

    # ── Step 12: Backtest for primary zone setup ──────────────────────────────
    zone_type   = primary_zone["type"]
    bt_params   = _ZONE_BACKTEST_PARAMS.get(zone_type, {"sl_method": "structural", "tp_method": "fixed_R"})
    sl_method   = bt_params["sl_method"]
    tp_method   = bt_params["tp_method"]
    tp_R        = _DEFAULT_TP_R

    per_coin_bt = backtest_per_coin(
        df_htf=df_htf,
        df_ltf=None,
        entry_zone_type=zone_type,
        sl_method=sl_method,
        tp_method=tp_method,
        tp_R=tp_R,
        max_setups_to_replay=200,
        simplified=False,
    )

    # Mark low_confidence if backtest returned zero setups
    if per_coin_bt["n_setups"] == 0:
        per_coin_bt["low_confidence"] = True

    universe_bt = backtest_universe_baseline(
        entry_zone_type=zone_type,
        sl_method=sl_method,
        tp_method=tp_method,
        tp_R=tp_R,
        timeframe=htf_interval,
        n_sample_coins=50,
    )

    blended = bayesian_blend(per_coin_bt, universe_bt, prior_strength=30)

    trades_raw = per_coin_bt.get("trades_raw", [])
    recent_check = compute_recent_check(trades_raw, n_df=len(df_htf))

    backtest_summary = {
        "per_coin":     per_coin_bt,
        "universe":     universe_bt,
        "blended":      blended,
        "recent_check": recent_check,
    }

    # ── Step 13: Macro flag ───────────────────────────────────────────────────
    fights_macro = btc_regime in {"BEAR"}

    # ── Step 14: Trade plan ───────────────────────────────────────────────────
    trade_plan = build_trade_plan(primary_zone, current_price, df_htf, structure)

    # ── Step 15: Build result dict ────────────────────────────────────────────
    return {
        "symbol":       symbol,
        "mode":         mode,
        "htf_tf":       htf_interval,
        "ltf_tf":       ltf_interval,
        "btc_regime":   btc_regime,
        "fights_macro": fights_macro,
        "ema_tier":     ema_tier,
        "structure":    structure,
        "zones": {
            "smart_obs":  smart_obs,
            "fvgs":       fvgs,
            "fibo":       fibo,
            "sr_levels":  sr_levels,
        },
        "current_price":              current_price,
        "current_zone_classification": zone_cls,
        "primary_zone":               primary_zone,
        "ltf_confirmation":           ltf_confirmation,
        "ltf_signal_bar":             ltf_signal_bar,
        "backtest":                   backtest_summary,
        "trade_plan":                 trade_plan,
        "timestamp_utc":              datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# LTF CONFIRMATION
# ============================================================================

def check_ltf_confirmation(
    df_ltf: pd.DataFrame,
    primary_zone: Dict[str, Any],
) -> Dict[str, Any]:
    """
    On the LTF, look for one of these confirmations inside the HTF primary_zone:
      - A bullish CHoCH on LTF (price closed above LTF swing high) WITHIN zone
      - A strong bullish reversal candle (close in upper third, vol >= 1.5x avg)
        AND inside the HTF zone range in last 5 LTF bars

    Args:
        df_ltf: LTF OHLCV DataFrame
        primary_zone: {"type": str, "data": dict}

    Returns:
        {
            "status": "CONFIRMED" | "PENDING" | "NONE",
            "signal_bar": int | None,
            "reason": str,
        }
    """
    _no_zone = {
        "status":     "NONE",
        "signal_bar": None,
        "reason":     "Price not inside HTF zone on LTF.",
    }

    if df_ltf is None or df_ltf.empty or len(df_ltf) < 6:
        return _no_zone

    zone_data = primary_zone.get("data", {})
    zone_type = primary_zone.get("type", "")

    # ── Extract zone bounds from primary_zone type ───────────────────────────
    zone_low, zone_high = _zone_bounds(zone_data, zone_type)
    if zone_low is None or zone_high is None:
        return _no_zone

    # Ensure the most recent LTF bar is inside the HTF zone
    ltf_close = df_ltf["close"].iloc[-1]
    ltf_low   = df_ltf["low"].iloc[-1]
    ltf_high  = df_ltf["high"].iloc[-1]

    price_in_zone = (ltf_low <= zone_high) and (ltf_high >= zone_low)
    if not price_in_zone:
        return _no_zone

    n = len(df_ltf)
    closes  = df_ltf["close"].to_numpy()
    opens   = df_ltf["open"].to_numpy()
    highs   = df_ltf["high"].to_numpy()
    lows    = df_ltf["low"].to_numpy()
    volumes = df_ltf["volume"].to_numpy()

    # ── Test 1: Bullish CHoCH on LTF within HTF zone ─────────────────────────
    # Detect LTF swing highs (simplified: local high over 3-bar window)
    # A bullish CHoCH = close above a prior LTF swing high while price was
    # within the HTF zone at the time.
    ltf_swing_highs = []
    for i in range(2, n - 2):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] \
                and highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]:
            # Only count swing highs inside or near the HTF zone
            if lows[i] <= zone_high * 1.01:
                ltf_swing_highs.append((i, float(highs[i])))

    # Scan forward from each swing high to look for CHoCH close
    for sh_bar, sh_price in ltf_swing_highs:
        for bar_i in range(sh_bar + 1, n):
            if closes[bar_i] > sh_price and lows[bar_i] <= zone_high:
                return {
                    "status":     "CONFIRMED",
                    "signal_bar": bar_i,
                    "reason":     (
                        f"LTF CHoCH: closed above swing high {sh_price:.6g} "
                        f"at LTF bar {bar_i} while inside HTF zone."
                    ),
                }

    # ── Test 2: Strong bullish reversal candle in last 5 LTF bars ────────────
    # Conditions:
    #   - close in upper third of candle range  (bullish close)
    #   - volume >= 1.5× 20-bar average
    #   - bar overlaps the HTF zone
    vol_avg_period = min(20, n - 6)
    check_start    = max(0, n - 5)

    for bar_i in range(check_start, n):
        bar_lo = float(lows[bar_i])
        bar_hi = float(highs[bar_i])
        bar_cl = float(closes[bar_i])
        bar_op = float(opens[bar_i])

        # Must overlap zone
        if bar_lo > zone_high or bar_hi < zone_low:
            continue

        rng = bar_hi - bar_lo
        if rng <= 0:
            continue

        # Close in upper third
        if bar_cl < bar_lo + rng * (2.0 / 3.0):
            continue

        # Bullish candle (close > open)
        if bar_cl <= bar_op:
            continue

        # Volume check
        vol_start = max(0, bar_i - vol_avg_period)
        vol_slice = volumes[vol_start:bar_i]
        if len(vol_slice) == 0:
            continue
        vol_avg = float(np.mean(vol_slice))
        if vol_avg <= 0 or float(volumes[bar_i]) < 1.5 * vol_avg:
            continue

        return {
            "status":     "CONFIRMED",
            "signal_bar": bar_i,
            "reason":     (
                f"Strong bullish reversal candle at LTF bar {bar_i} "
                f"(close in upper third, vol {volumes[bar_i]:.0f} >= 1.5× avg {vol_avg:.0f})."
            ),
        }

    # ── PENDING: price is inside zone but no confirmation yet ─────────────────
    return {
        "status":     "PENDING",
        "signal_bar": None,
        "reason":     "Price is inside HTF zone on LTF; no LTF confirmation signal yet.",
    }


def _zone_bounds(
    zone_data: Dict[str, Any],
    zone_type: str,
) -> tuple:
    """
    Return (zone_low, zone_high) for any primary zone type.
    Returns (None, None) on failure.
    """
    try:
        if zone_type == "smart_ob":
            return float(zone_data["ob_low"]), float(zone_data["ob_high"])

        elif zone_type == "fvg":
            return float(zone_data["bottom"]), float(zone_data["top"])

        elif zone_type == "fibo_786":
            z_bottom = zone_data.get("fib_786_zone_bottom")
            z_top    = zone_data.get("fib_786_zone_top")
            if z_bottom is None or z_top is None:
                fib786 = float(zone_data["fib_786"])
                return fib786 * 0.995, fib786 * 1.005
            return float(z_bottom), float(z_top)

        elif zone_type == "sr":
            sr_price = float(zone_data["price"])
            return sr_price * 0.995, sr_price * 1.005

    except (KeyError, TypeError, ValueError):
        pass

    return None, None


# ============================================================================
# TRADE PLAN BUILDER
# ============================================================================

def build_trade_plan(
    primary_zone: Dict[str, Any],
    current_price: float,
    df_htf: pd.DataFrame,
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute entry, SL, TPs based on the primary zone.

    Entry logic by zone type:
        smart_ob:  entry_price = ob_high; sl = ob_low * 0.998
        fvg:       entry_price = fvg.mid; sl = fvg.bottom * 0.998
        fibo_786:  entry_price = fib_786; sl = fib_786 * 0.985
        sr:        entry_price = sr.price * 1.002; sl = sr.price * 0.985

    TP logic:
        risk = entry_price - sl
        tp1 = entry_price + 2.0 * risk
        tp2 = entry_price + 2.5 * risk
        tp3 = entry_price + 3.0 * risk

    Returns:
        trade_plan dict with keys:
            entry_zone_type, entry_price, sl,
            tp1_R, tp1_price, tp2_R, tp2_price, tp3_R, tp3_price,
            rr_to_tp2
    """
    zone_type = primary_zone["type"]
    zone_data = primary_zone["data"]

    # ── Compute entry and SL by zone type ────────────────────────────────────
    try:
        if zone_type == "smart_ob":
            entry_price = float(zone_data["ob_high"])
            sl          = float(zone_data["ob_low"]) * 0.998

        elif zone_type == "fvg":
            entry_price = float(zone_data["mid"])
            sl          = float(zone_data["bottom"]) * 0.998

        elif zone_type == "fibo_786":
            fib786      = float(zone_data["fib_786"])
            entry_price = fib786
            sl          = fib786 * 0.985

        elif zone_type == "sr":
            sr_price    = float(zone_data["price"])
            entry_price = sr_price * 1.002
            sl          = sr_price * 0.985

        else:
            # Fallback: use current_price with a 1.5% SL
            entry_price = current_price
            sl          = current_price * 0.985

    except (KeyError, TypeError, ValueError) as e:
        print(f"[scanner] build_trade_plan fallback ({zone_type}): {e}")
        entry_price = current_price
        sl          = current_price * 0.985

    # Safety: SL must be strictly below entry
    if sl >= entry_price:
        sl = entry_price * 0.985
    if sl <= 0:
        sl = entry_price * 0.985

    risk = entry_price - sl
    if risk <= 0:
        risk = entry_price * 0.015   # 1.5% fallback

    tp1_price = entry_price + 2.0 * risk
    tp2_price = entry_price + 2.5 * risk
    tp3_price = entry_price + 3.0 * risk

    rr_to_tp2 = (tp2_price - entry_price) / risk if risk > 0 else 2.5

    return {
        "entry_zone_type": zone_type,
        "entry_price":     round(entry_price, 10),
        "sl":              round(sl, 10),
        "tp1_R":           2.0,
        "tp1_price":       round(tp1_price, 10),
        "tp2_R":           2.5,
        "tp2_price":       round(tp2_price, 10),
        "tp3_R":           3.0,
        "tp3_price":       round(tp3_price, 10),
        "rr_to_tp2":       round(rr_to_tp2, 4),
    }


# ============================================================================
# EMA TIER CLASSIFIER
# ============================================================================

def classify_ema_tier(df_htf: pd.DataFrame, current_price: float) -> str:
    """
    Determine bullish-regime tier per spec.

    - STRONG: current_price > EMA50 AND current_price > EMA200
    - MEDIUM: current_price > one of them but not both
    - SKIP:   below both

    If df_htf has < 200 bars: use only EMA 50 → STRONG if above, else SKIP.

    Returns: "STRONG" | "MEDIUM" | "SKIP"
    """
    if len(df_htf) < 52:
        # Not enough data even for EMA50 to be meaningful
        return "SKIP"

    ema50 = calculate_ema(df_htf, 50)
    ema50_val = float(ema50.iloc[-1])

    if len(df_htf) < 200:
        # Short history: use EMA50 only
        if np.isnan(ema50_val):
            return "SKIP"
        if current_price > ema50_val:
            return "STRONG"
        return "SKIP"

    ema200 = calculate_ema(df_htf, 200)
    ema200_val = float(ema200.iloc[-1])

    # Handle NaN (insufficient warm-up)
    ema50_ok  = not np.isnan(ema50_val)
    ema200_ok = not np.isnan(ema200_val)

    above_50  = ema50_ok  and current_price > ema50_val
    above_200 = ema200_ok and current_price > ema200_val

    if above_50 and above_200:
        return "STRONG"
    if above_50 or above_200:
        return "MEDIUM"
    return "SKIP"


# ============================================================================
# PRIMARY ZONE PICKER
# ============================================================================

def pick_primary_zone(
    zone_classification: Dict[str, Any],
    fibo: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    From the dict output of classify_current_price_in_zones, pick the PRIMARY
    zone for trade-planning purposes.

    Priority order (highest to lowest):
        1. STRONG smart_ob   (highest conviction)
        2. fvg FRESH
        3. REGULAR smart_ob
        4. fvg PARTIAL_<50
        5. fibo_786
        6. sr_support
        7. fvg PARTIAL_>50

    Args:
        zone_classification: output of classify_current_price_in_zones()
        fibo: the full fibo dict from detect_fibo_levels() — needed so that
              the returned zone data contains the actual fib_786 price.

    Returns:
        {"type": "smart_ob"|"fvg"|"fibo_786"|"sr", "data": <zone_dict>}
        or None if no zone in classification.
    """
    in_obs  = zone_classification.get("in_smart_ob", [])
    in_fvgs = zone_classification.get("in_fvg", [])
    in_fibo = zone_classification.get("in_fibo_786", False)
    in_srs  = zone_classification.get("at_sr_support", [])

    # Priority 1: STRONG smart_ob
    strong_obs = [ob for ob in in_obs if ob.get("tier") == "STRONG"]
    if strong_obs:
        # Pick the freshest (highest freshness_score) among STRONG OBs
        best = max(strong_obs, key=lambda x: x.get("freshness_score", 0.0))
        return {"type": "smart_ob", "data": best}

    # Priority 2: FRESH fvg
    fresh_fvgs = [f for f in in_fvgs if f.get("status") == "FRESH"]
    if fresh_fvgs:
        # Pick the most recently created FRESH FVG
        best = max(fresh_fvgs, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    # Priority 3: REGULAR smart_ob
    regular_obs = [ob for ob in in_obs if ob.get("tier") == "REGULAR"]
    if regular_obs:
        best = max(regular_obs, key=lambda x: x.get("freshness_score", 0.0))
        return {"type": "smart_ob", "data": best}

    # Priority 4: fvg PARTIAL_<50
    partial_lt50 = [f for f in in_fvgs if f.get("status") == "PARTIAL_<50"]
    if partial_lt50:
        best = max(partial_lt50, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    # Priority 5: fibo_786
    if in_fibo and fibo:
        return {"type": "fibo_786", "data": fibo}

    # Priority 6: sr_support
    if in_srs:
        # Pick the strongest S/R level
        best = max(in_srs, key=lambda x: x.get("strength", 0.0))
        return {"type": "sr", "data": best}

    # Priority 7: fvg PARTIAL_>50 (least preferred)
    partial_gt50 = [f for f in in_fvgs if f.get("status") == "PARTIAL_>50"]
    if partial_gt50:
        best = max(partial_gt50, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    return None
