"""
qf_smc_short/scanner.py — Multi-TF SMC scanner orchestrator (SHORT)
====================================================================
Mirror of qf_smc/scanner.py for SHORT setups.

Same architecture (ThreadPoolExecutor, hard per-coin timeout, 3 modes),
but with every directional component flipped:

    LONG (qf_smc.scanner)              SHORT (qf_smc_short.scanner)
    ──────────────────────             ──────────────────────────────
    is_uptrend_confirmed()             is_downtrend_confirmed()
    detect_smart_obs()                 detect_smart_obs_short()
    detect_fvgs()                      detect_fvgs_short()
    detect_fibo_levels()               detect_fibo_levels_short()
    detect_sr_levels()                 detect_sr_levels_short()
    LTF bullish CHoCH / reversal       LTF bearish CHoCH / reversal
    EMA tier: above EMA50/200          EMA tier: below EMA50/200
    Trade plan: entry low, SL low,     Trade plan: entry high, SL high,
      TP = entry + R                     TP = entry - R
    fights_macro: BEAR                 fights_macro: BULL

Fast scan by default (no backtest). Phase 2 will add backtest_short module.
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
from qf_smc.structure import detect_swings   # direction-neutral, reused
from qf_smc_short.structure import classify_structure_short, is_downtrend_confirmed
from qf_smc_short.zones import (
    detect_smart_obs_short,
    detect_fvgs_short,
    detect_fibo_levels_short,
    detect_sr_levels_short,
    classify_current_price_in_zones_short,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "SWING": {"htf": "1d",  "ltf": "4h",  "lookback": 50, "htf_label": "1D", "ltf_label": "4H"},
    "DAY":   {"htf": "4h",  "ltf": "15m", "lookback": 50, "htf_label": "4H", "ltf_label": "15m"},
    "SCALP": {"htf": "1h",  "ltf": "5m",  "lookback": 50, "htf_label": "1H", "ltf_label": "5m"},
}


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_scan_short(
    mode: str,
    symbols: List[str],
    btc_regime: str,
    atr_multiplier: float = 0.5,
    run_backtest: bool = False,        # Phase 2 — opt-in 24-variant grid
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Run a full SMC SHORT scan over the provided symbol list.

    Args:
        mode: "SWING" | "DAY" | "SCALP"
        symbols: list of symbol strings
        btc_regime: "BULL" | "CHOP" | "BEAR" | "UNKNOWN"
                    (fights_macro = True when BULL — shorts fight bullish macro)
        atr_multiplier: ATR×N tolerance for Fibo 0.786 zone (default 0.5)
        run_backtest: if True, attach 24-variant backtest grid to each result.
                      Default False — scan is fast; full backtest is deferred
                      to the Deep Dive button per coin.
        progress_callback: optional callback(current_idx, total, symbol_name)

    Returns:
        List of result dicts (one per symbol with a short setup found).
    """
    results: List[Dict[str, Any]] = []

    import time as _time
    import threading as _threading
    import traceback as _traceback
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    total = len(symbols)
    if total == 0:
        return results

    PER_COIN_TIMEOUT = 10.0   # seconds of actual run time
    max_workers = min(8, max(1, total))
    skipped_slow: list = []

    # Per-future bookkeeping (mirror of LONG implementation)
    future_to_symbol: Dict[Any, str] = {}
    future_started_at: Dict[Any, float] = {}
    future_done_flag: Dict[Any, _threading.Event] = {}

    def _wrapped_scan(symbol: str, started_evt: _threading.Event) -> Optional[Dict]:
        started_evt.set()
        try:
            return scan_one_symbol_short(
                symbol=symbol,
                mode=mode,
                btc_regime=btc_regime,
                atr_multiplier=atr_multiplier,
                run_backtest=run_backtest,
            )
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sym in symbols:
            started_evt = _threading.Event()
            fut = executor.submit(_wrapped_scan, sym, started_evt)
            future_to_symbol[fut] = sym
            future_done_flag[fut] = started_evt

        pending = set(future_to_symbol.keys())
        completed_count = 0

        while pending:
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)

            # Check for slow coins
            now = _time.monotonic()
            to_cancel = []
            for fut in list(pending):
                if fut in done:
                    continue
                started_evt = future_done_flag.get(fut)
                if started_evt is None or not started_evt.is_set():
                    continue  # not started yet
                if fut not in future_started_at:
                    future_started_at[fut] = now
                    continue
                elapsed = now - future_started_at[fut]
                if elapsed > PER_COIN_TIMEOUT:
                    to_cancel.append(fut)

            for fut in to_cancel:
                pending.discard(fut)
                sym = future_to_symbol.get(fut, "?")
                skipped_slow.append(sym)
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total, f"SKIP-SLOW {sym}")
                    except Exception:
                        pass

            for fut in done:
                pending.discard(fut)
                sym = future_to_symbol.get(fut, "?")
                try:
                    result = fut.result(timeout=0)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total, sym)
                    except Exception:
                        pass

    if skipped_slow:
        print(f"[scanner_short] Skipped {len(skipped_slow)} slow coins: {skipped_slow[:5]}...")

    return results


def scan_one_symbol_short(
    symbol: str,
    mode: str,
    btc_regime: str,
    atr_multiplier: float = 0.5,
    run_backtest: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Scan ONE symbol for a SHORT setup. Returns None if filters fail.

    When run_backtest=True, also computes the 24-variant backtest grid
    (slower; deferred to Deep Dive in the standard scan flow).
    """
    cfg = MODE_CONFIG.get(mode)
    if cfg is None:
        return None

    htf_interval = cfg["htf"]
    ltf_interval = cfg["ltf"]
    lookback     = cfg["lookback"]

    # ── Step 1: Fetch candles ───────────────────────────────────────────────
    df_htf = _scanner_fetch_candles(symbol, htf_interval, limit=500)
    df_ltf = _scanner_fetch_candles(symbol, ltf_interval, limit=lookback + 50)

    if df_htf is None or df_htf.empty or len(df_htf) < 30:
        return None
    if df_ltf is None or df_ltf.empty or len(df_ltf) < 30:
        return None

    df_htf = df_htf.reset_index(drop=True)
    df_ltf = df_ltf.reset_index(drop=True)

    # ── Step 2: Structure detection on HTF (SHORT classifier) ───────────────
    swings    = detect_swings(df_htf, pivot=5)
    structure = classify_structure_short(swings, df_htf)

    # ── Step 3: Structure filter — must be confirmed downtrend ─────────────
    if not is_downtrend_confirmed(structure):
        return None

    # ── Step 4: Fibo zone FIRST ─────────────────────────────────────────────
    fibo_zone = detect_fibo_levels_short(df_htf, structure, atr_multiplier=atr_multiplier)
    if not fibo_zone:
        return None

    # ── Step 5: Zone detectors WITH Fibo 0.786 filter ───────────────────────
    smart_obs = detect_smart_obs_short(df_htf, structure, fibo_786_zone=fibo_zone, require_in_zone=True)
    fvgs      = detect_fvgs_short(df_htf, structure, fibo_786_zone=fibo_zone, require_in_zone=True)
    sr_levels = detect_sr_levels_short(df_htf, lookback=lookback, fibo_786_zone=fibo_zone, require_in_zone=True)

    # ── Step 6: Require at least one zone inside Fibo 0.786 ─────────────────
    if not smart_obs and not fvgs and not sr_levels:
        if "fib_786_zone_top" not in fibo_zone:
            return None
        # Fall through — price-into-fibo zone is itself the trigger

    # ── Step 7: EMA tier (SHORT — must be BELOW EMAs) ───────────────────────
    current_price = float(df_htf["close"].iloc[-1])
    ema_tier = classify_ema_tier_short(df_htf, current_price)
    if ema_tier == "SKIP":
        return None

    # ── Step 8: Zone classification ─────────────────────────────────────────
    zone_cls = classify_current_price_in_zones_short(
        current_price, smart_obs, fvgs, fibo_zone, sr_levels
    )

    # ── Step 9: Require actionable zone ─────────────────────────────────────
    if not zone_cls.get("any", False):
        return None

    # ── Step 10: Pick primary zone (same priority logic, bearish content) ──
    primary_zone = pick_primary_zone_short(zone_cls, fibo=fibo_zone)
    if primary_zone is None:
        return None

    ob_tier: Optional[str] = None
    if primary_zone["type"] == "smart_ob":
        ob_tier = primary_zone["data"].get("tier")

    # ── Step 11: LTF confirmation (bearish) ─────────────────────────────────
    ltf_result       = check_ltf_confirmation_short(df_ltf, primary_zone)
    ltf_confirmation = ltf_result["status"]
    ltf_signal_bar   = ltf_result.get("signal_bar")

    # ── Step 12: Macro flag — shorts fight BULL regime ──────────────────────
    fights_macro = btc_regime in {"BULL"}

    # ── Step 13: Trade plan (SHORT) ─────────────────────────────────────────
    trade_plan = build_trade_plan_short(primary_zone, current_price, df_htf, structure)

    # ── Step 14: Transparency fields ────────────────────────────────────────
    wick_adjustments         = swings.get("wick_adjustments", [])
    overall_bearish_verified = structure.get("overall_bearish_verified", False)

    # ── Step 14b: Optional 24-variant backtest grid (Phase 2) ───────────────
    # v1.0 / Phase 2: the scan is FAST by default (run_backtest=False), so the
    # variant grid is empty and Deep Dive runs it on demand for one coin.
    # When run_backtest=True, we run the full grid here for every coin.
    _EMPTY_GRID_COLS = [
        "entry_type", "ltf_mode", "tp_R",
        "n_setups", "n_filled", "n_wins",
        "wr", "mean_r", "median_r", "pf", "max_dd_R",
        "avg_entry_price", "avg_sl_pct", "avg_rr_to_tp",
        "recent_check", "trades_raw",
    ]

    if run_backtest:
        # Lazy import — backtest module pulls qf_shared decay-bucket helpers
        # which we don't need to load on a fast scan
        try:
            from qf_smc_short.backtest import run_variant_grid_short
            variant_grid: pd.DataFrame = run_variant_grid_short(
                df_htf=df_htf,
                df_ltf=df_ltf,
                structure=structure,
                fibo_zone=fibo_zone,
                smart_obs=smart_obs,
                fvgs=fvgs,
                sr_levels=sr_levels,
            )
            low_confidence = len(df_htf) < 100
            if low_confidence and not variant_grid.empty:
                variant_grid["low_confidence"] = True
            best_variant: Optional[Dict[str, Any]] = None
            if not variant_grid.empty:
                best_variant = variant_grid.iloc[0].to_dict()
        except Exception as exc:
            print(f"[scanner_short] backtest error for {symbol}: {exc}")
            variant_grid = pd.DataFrame(columns=_EMPTY_GRID_COLS)
            best_variant = None
    else:
        variant_grid = pd.DataFrame(columns=_EMPTY_GRID_COLS)
        best_variant = None

    # ── Step 15: Build result dict (same shape as LONG for UI compat) ───────
    return {
        # identity
        "symbol":       symbol,
        "mode":         mode,
        "direction":    "short",       # flag for UI / shared consumers
        "htf_tf":       htf_interval,
        "ltf_tf":       ltf_interval,
        "btc_regime":   btc_regime,
        "fights_macro": fights_macro,

        # structure
        "structure":                  structure,
        "overall_bearish_verified":   overall_bearish_verified,
        "wick_adjustments":           wick_adjustments,

        # zones
        "fibo_zone":           fibo_zone,
        "atr_multiplier_used": atr_multiplier,
        "zones": {
            "smart_obs":  smart_obs,
            "fvgs":       fvgs,
            "fibo":       fibo_zone,
            "sr_levels":  sr_levels,
        },

        # price + classification
        "current_price":               current_price,
        "current_zone_classification": zone_cls,
        "primary_zone":                primary_zone,
        "ob_tier":                     ob_tier,
        "ema_tier":                    ema_tier,

        # LTF
        "ltf_confirmation": ltf_confirmation,
        "ltf_signal_bar":   ltf_signal_bar,

        # backtest (Phase 2 — populated when run_backtest=True, else empty)
        "variant_grid":        variant_grid,
        "best_variant":        best_variant,
        "deep_dive_available": True,    # Phase 2: Deep Dive button enabled

        # candle cache (for Deep Dive on demand)
        "_df_htf_cache": df_htf,
        "_df_ltf_cache": df_ltf,

        # trade plan
        "trade_plan": trade_plan,

        # metadata
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# LTF CONFIRMATION (mirror — bearish version)
# ============================================================================

def check_ltf_confirmation_short(
    df_ltf: pd.DataFrame,
    primary_zone: Dict[str, Any],
) -> Dict[str, Any]:
    """
    On the LTF, look for a bearish confirmation inside the HTF primary_zone:
      - A bearish CHoCH on LTF (price closed below an LTF swing low) within zone
      - A strong bearish reversal candle (close in lower third, vol >= 1.5×)
        AND inside the HTF zone in last 5 LTF bars
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

    zone_low, zone_high = _zone_bounds(zone_data, zone_type)
    if zone_low is None or zone_high is None:
        return _no_zone

    ltf_low  = df_ltf["low"].iloc[-1]
    ltf_high = df_ltf["high"].iloc[-1]

    price_in_zone = (ltf_low <= zone_high) and (ltf_high >= zone_low)
    if not price_in_zone:
        return _no_zone

    n = len(df_ltf)
    closes  = df_ltf["close"].to_numpy()
    opens   = df_ltf["open"].to_numpy()
    highs   = df_ltf["high"].to_numpy()
    lows    = df_ltf["low"].to_numpy()
    volumes = df_ltf["volume"].to_numpy()

    # ── Test 1: Bearish CHoCH on LTF — close BELOW a prior LTF swing low ────
    ltf_swing_lows = []
    for i in range(2, n - 2):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] \
                and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
            # Only count swing lows inside or near the HTF zone
            if highs[i] >= zone_low * 0.99:
                ltf_swing_lows.append((i, float(lows[i])))

    for sl_bar, sl_price in ltf_swing_lows:
        for bar_i in range(sl_bar + 1, n):
            if closes[bar_i] < sl_price and highs[bar_i] >= zone_low:
                return {
                    "status":     "CONFIRMED",
                    "signal_bar": bar_i,
                    "reason":     (
                        f"LTF bearish CHoCH: closed below swing low {sl_price:.6g} "
                        f"at LTF bar {bar_i} while inside HTF zone."
                    ),
                }

    # ── Test 2: Strong bearish reversal candle in last 5 LTF bars ───────────
    vol_avg_period = min(20, n - 6)
    check_start    = max(0, n - 5)

    for bar_i in range(check_start, n):
        bar_lo = float(lows[bar_i])
        bar_hi = float(highs[bar_i])
        bar_cl = float(closes[bar_i])
        bar_op = float(opens[bar_i])

        if bar_lo > zone_high or bar_hi < zone_low:
            continue

        rng = bar_hi - bar_lo
        if rng <= 0:
            continue

        # FLIPPED: close in LOWER third
        if bar_cl > bar_lo + rng * (1.0 / 3.0):
            continue

        # FLIPPED: bearish candle (close < open)
        if bar_cl >= bar_op:
            continue

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
                f"Strong bearish reversal candle at LTF bar {bar_i} "
                f"(close in lower third, vol {volumes[bar_i]:.0f} >= 1.5× avg {vol_avg:.0f})."
            ),
        }

    return {
        "status":     "PENDING",
        "signal_bar": None,
        "reason":     "Price inside HTF zone on LTF; no LTF bearish confirmation yet.",
    }


def _zone_bounds(zone_data: Dict[str, Any], zone_type: str) -> tuple:
    """Return (zone_low, zone_high) for any primary zone type. Same shape for both directions."""
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
# TRADE PLAN BUILDER (SHORT — inverted entry/SL/TP)
# ============================================================================

def build_trade_plan_short(
    primary_zone: Dict[str, Any],
    current_price: float,
    df_htf: pd.DataFrame,
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute SHORT entry, SL, TPs based on the primary zone.

    Flipped entry logic:
        smart_ob:  entry = ob_low;     sl = ob_high * 1.002    (SL above OB)
        fvg:       entry = fvg.mid;    sl = fvg.top * 1.002    (SL above FVG)
        fibo_786:  entry = fib_786;    sl = fib_786 * 1.015    (SL 1.5% above)
        sr:        entry = sr * 0.998; sl = sr * 1.015         (SL above S/R)

    Flipped TP logic:
        risk = sl - entry_price          (positive for SHORT — SL above entry)
        tp1 = entry_price - 2.0 * risk
        tp2 = entry_price - 2.5 * risk
        tp3 = entry_price - 3.0 * risk
    """
    zone_type = primary_zone["type"]
    zone_data = primary_zone["data"]

    try:
        if zone_type == "smart_ob":
            entry_price = float(zone_data["ob_low"])
            sl          = float(zone_data["ob_high"]) * 1.002

        elif zone_type == "fvg":
            entry_price = float(zone_data["mid"])
            sl          = float(zone_data["top"]) * 1.002

        elif zone_type == "fibo_786":
            fib786      = float(zone_data["fib_786"])
            entry_price = fib786
            sl          = fib786 * 1.015

        elif zone_type == "sr":
            sr_price    = float(zone_data["price"])
            entry_price = sr_price * 0.998
            sl          = sr_price * 1.015

        else:
            entry_price = current_price
            sl          = current_price * 1.015

    except (KeyError, TypeError, ValueError) as e:
        print(f"[scanner_short] build_trade_plan fallback ({zone_type}): {e}")
        entry_price = current_price
        sl          = current_price * 1.015

    # ── Risk and TPs (SHORT: profit = entry - exit) ─────────────────────────
    risk = sl - entry_price
    if risk <= 0:
        # Degenerate (SL not above entry) — force a tight stop
        risk = entry_price * 0.015
        sl   = entry_price + risk

    tp1_price = entry_price - 2.0 * risk
    tp2_price = entry_price - 2.5 * risk
    tp3_price = entry_price - 3.0 * risk

    rr_to_tp2 = (entry_price - tp2_price) / risk if risk > 0 else 0.0

    return {
        "direction":       "short",
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
# EMA TIER (SHORT — flipped: must be BELOW EMAs)
# ============================================================================

def classify_ema_tier_short(df_htf: pd.DataFrame, current_price: float) -> str:
    """
    Determine bearish-regime tier per spec (mirror of LONG classify_ema_tier).

    - STRONG: current_price < EMA50 AND current_price < EMA200
    - MEDIUM: current_price < one of them but not both
    - SKIP:   above both
    """
    if len(df_htf) < 52:
        return "SKIP"

    ema50 = calculate_ema(df_htf, 50)
    ema50_val = float(ema50.iloc[-1])

    if len(df_htf) < 200:
        if np.isnan(ema50_val):
            return "SKIP"
        if current_price < ema50_val:
            return "STRONG"
        return "SKIP"

    ema200 = calculate_ema(df_htf, 200)
    ema200_val = float(ema200.iloc[-1])

    ema50_ok  = not np.isnan(ema50_val)
    ema200_ok = not np.isnan(ema200_val)

    below_50  = ema50_ok  and current_price < ema50_val
    below_200 = ema200_ok and current_price < ema200_val

    if below_50 and below_200:
        return "STRONG"
    if below_50 or below_200:
        return "MEDIUM"
    return "SKIP"


# ============================================================================
# PRIMARY ZONE PICKER (same priority order, operates on bearish zones)
# ============================================================================

def pick_primary_zone_short(
    zone_classification: Dict[str, Any],
    fibo: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Same priority order as LONG version (LIQUIDITY_SWEEP first → ... → PARTIAL_>50).
    Operates on bearish zone classifications.
    """
    in_obs  = zone_classification.get("in_smart_ob", [])
    in_fvgs = zone_classification.get("in_fvg", [])
    in_fibo = zone_classification.get("in_fibo_786", False)
    in_srs  = zone_classification.get("at_sr_support", [])  # semantically: at resistance

    liq_sweep_obs = [ob for ob in in_obs if ob.get("tier") == "LIQUIDITY_SWEEP"]
    if liq_sweep_obs:
        best = max(liq_sweep_obs, key=lambda x: x.get("freshness_score", 0.0))
        return {"type": "smart_ob", "data": best}

    strong_obs = [ob for ob in in_obs if ob.get("tier") == "STRONG"]
    if strong_obs:
        best = max(strong_obs, key=lambda x: x.get("freshness_score", 0.0))
        return {"type": "smart_ob", "data": best}

    fresh_fvgs = [f for f in in_fvgs if f.get("status") == "FRESH"]
    if fresh_fvgs:
        best = max(fresh_fvgs, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    regular_obs = [ob for ob in in_obs if ob.get("tier") == "REGULAR"]
    if regular_obs:
        best = max(regular_obs, key=lambda x: x.get("freshness_score", 0.0))
        return {"type": "smart_ob", "data": best}

    partial_lt50 = [f for f in in_fvgs if f.get("status") == "PARTIAL_<50"]
    if partial_lt50:
        best = max(partial_lt50, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    if in_fibo and fibo:
        return {"type": "fibo_786", "data": fibo}

    if in_srs:
        best = max(in_srs, key=lambda x: x.get("strength", 0.0))
        return {"type": "sr", "data": best}

    partial_gt50 = [f for f in in_fvgs if f.get("status") == "PARTIAL_>50"]
    if partial_gt50:
        best = max(partial_gt50, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    return None
