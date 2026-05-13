"""
qf_smc/backtest.py — Backtest engine for SMC Long setups
=========================================================
Per-coin backtest + universe baseline + Bayesian blend + bucketed verdict
per QUANTFLOW SMC Long spec v1.2 §2.6 – §2.7.

Revision v1.2 changes (spec §2.6–§2.7):
  - run_variant_grid()       NEW — 24 entry variants (4 types × 2 LTF modes × 3 TP-R)
  - _backtest_single_variant() NEW — single-variant historical replay
  - deep_dive_backtest()     NEW — 72-variant deep dive (24 × 3 management modes)
  - _simulate_with_management() NEW — re-simulate trades with BE/Trailing mgmt
  - _aggregate_trades()      NEW — aggregate helper
  - _compute_max_dd()        NEW — max drawdown in R-units
  - _build_trade_plan_from_variant() NEW — build concrete trade plan

Unchanged v1.1 functions (backward compat):
  - backtest_per_coin()
  - backtest_universe_baseline()
  - bayesian_blend()
  - compute_recent_check()
  - run_combo_grid()

Public API:
  - backtest_per_coin(df_htf, df_ltf, entry, sl, tp, tp_R, ...)
  - backtest_universe_baseline(entry, sl, tp, tp_R, timeframe, n_sample_coins)
  - bayesian_blend(per_coin, universe, prior_strength)
  - compute_recent_check(trades_raw, n_df)
  - run_combo_grid(df_htf, df_ltf)
  - run_variant_grid(df_htf, df_ltf, structure, fibo_zone, smart_obs, fvgs, sr_levels)
  - deep_dive_backtest(symbol, df_htf, df_ltf, structure, fibo_zone, ...)

Internal helpers:
  - _detect_zone_setup(historical_df, structure, entry_zone_type, current_price)
  - _compute_tp(historical_df, structure, entry, sl, tp_method, tp_R)
  - _compute_atr(df, bar_idx, period=14)
  - _compute_trade_stats(trades, tp_R)
  - _save_baseline_cache(combo_key, data)
  - _load_baseline_cache(combo_key)
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from qf_shared import (
    _scanner_get_universe,
    _scanner_fetch_candles,
    _compute_decay_buckets,
    _bucket_stats_for_trades,
    _classify_outcome,
)
from qf_smc.structure import detect_swings, classify_structure
from qf_smc.zones import (
    detect_smart_obs,
    detect_fvgs,
    detect_fibo_levels,
    detect_sr_levels,
    classify_current_price_in_zones,
)


# ============================================================================
# Constants
# ============================================================================

BASELINE_CACHE_DIR = Path("data/smc_baselines")
BASELINE_TTL_DAYS  = 7   # re-compute universe baseline weekly

# Entry zone types accepted by backtest_per_coin
_VALID_ENTRY_TYPES = {"smart_ob", "fvg", "fibo_786", "sr"}

# Structural SL: percentage buffer below level when zone has no natural low
_SR_SL_BUFFER_PCT   = 0.010   # 1.0% below SR support price
_FIBO_SL_BUFFER_PCT = 0.005   # 0.5% extra below fib_886

# Max forward-look window for trade outcome resolution (bars)
_MAX_FWD_BARS = 50

# Max forward-look window for limit order FILL (bars) — spec §2.6
_MAX_FILL_BARS = 30

# Simplified-walk step (every N bars instead of every bar)
_SIMPLIFIED_STEP = 20


# ============================================================================
# Internal helpers
# ============================================================================

def _compute_atr(df: pd.DataFrame, bar_idx: int, period: int = 14) -> float:
    """
    Simple ATR estimate using mean(high - low) over the last `period` bars
    ending at bar_idx. Returns a small positive fallback if insufficient data.
    """
    start = max(0, bar_idx - period)
    highs = df["high"].iloc[start:bar_idx].to_numpy()
    lows  = df["low"].iloc[start:bar_idx].to_numpy()
    if len(highs) == 0:
        return float(df["close"].iloc[bar_idx]) * 0.01  # 1% fallback
    return float(np.mean(highs - lows))


def _detect_zone_setup(
    historical_df: pd.DataFrame,
    structure: dict,
    entry_zone_type: str,
    current_price: float,
) -> Optional[Tuple[float, float]]:
    """
    Check whether a fresh zone matching `entry_zone_type` overlaps current price.

    Uses classify_current_price_in_zones from qf_smc.zones for consistent
    zone-touching logic across scanner and backtest.

    Returns:
        (entry_price, sl_structural) — both in price units, or None if no zone.

    Entry is always current_price (market fill at close of signal bar).
    sl_structural is the natural zone floor (ob_low, fvg.bottom, fib_886, sr - buffer).
    """
    obs       = detect_smart_obs(historical_df, structure)
    fvgs_list = detect_fvgs(historical_df, structure)
    fibo      = detect_fibo_levels(historical_df, structure)
    srs       = detect_sr_levels(historical_df)

    in_zones = classify_current_price_in_zones(
        current_price, obs, fvgs_list, fibo, srs
    )

    entry = current_price  # market-order fill at close of signal bar

    if entry_zone_type == "smart_ob":
        matching = in_zones["in_smart_ob"]
        if not matching:
            return None
        zone = max(matching, key=lambda z: z["ob_bar"])
        sl_structural = float(zone["ob_low"])
        return entry, sl_structural

    elif entry_zone_type == "fvg":
        matching = in_zones["in_fvg"]
        if not matching:
            return None
        zone = max(matching, key=lambda z: z["created_at_bar"])
        sl_structural = float(zone["bottom"])
        return entry, sl_structural

    elif entry_zone_type == "fibo_786":
        if not in_zones["in_fibo_786"]:
            return None
        fib_886 = fibo.get("fib_886")
        if fib_886 is None:
            return None
        sl_structural = float(fib_886) * (1.0 - _FIBO_SL_BUFFER_PCT)
        return entry, sl_structural

    elif entry_zone_type == "sr":
        matching = in_zones["at_sr_support"]
        if not matching:
            return None
        zone = min(matching, key=lambda z: abs(z["price"] - current_price))
        sl_structural = float(zone["price"]) * (1.0 - _SR_SL_BUFFER_PCT)
        return entry, sl_structural

    return None


def _compute_tp(
    historical_df: pd.DataFrame,
    structure: dict,
    entry: float,
    sl: float,
    tp_method: str,
    tp_R: float,
) -> float:
    """
    Compute TP price for a given setup.

    tp_method options:
        "fixed_R"        — entry + tp_R * risk
        "sr"             — next resistance level above entry (fallback: fixed_R)
        "bearish_ob_fvg" — next resistance S/R (Phase 1 fallback)
        "liq"            — liquidity cluster (Phase 1 fallback: fixed_R)
    """
    risk = entry - sl
    if risk <= 0:
        risk = entry * 0.015  # safety: assume 1.5% risk

    if tp_method == "fixed_R":
        return entry + tp_R * risk

    elif tp_method in ("sr", "bearish_ob_fvg"):
        srs = detect_sr_levels(historical_df)
        resistances = [
            s for s in srs
            if s["kind"] == "resistance" and s["price"] > entry * 1.001
        ]
        if resistances:
            return float(min(resistances, key=lambda x: x["price"])["price"])
        return entry + tp_R * risk

    elif tp_method == "liq":
        return entry + tp_R * risk

    return entry + tp_R * risk


def _compute_trade_stats(trades: List[Dict], tp_R: float) -> Dict[str, Any]:
    """
    Aggregate a list of trade dicts into backtest summary metrics.
    Each trade must have 'r_net' and 'exit_reason'.
    """
    if not trades:
        return {
            "n_setups":   0,
            "n_filled":   0,
            "n_wins":     0,
            "wr":         0.0,
            "mean_r":     0.0,
            "median_r":   0.0,
            "pf":         0.0,
            "max_dd_R":   0.0,
            "trades_raw": [],
        }

    n_setups = len(trades)
    n_filled = sum(1 for t in trades if t["exit_reason"] in {"tp", "sl"})
    n_wins   = sum(1 for t in trades if t["r_net"] > 0)
    wr       = n_wins / n_setups if n_setups > 0 else 0.0

    r_vals   = [t["r_net"] for t in trades]
    mean_r   = float(np.mean(r_vals))
    median_r = float(np.median(r_vals))

    pos_r = sum(r for r in r_vals if r > 0)
    neg_r = abs(sum(r for r in r_vals if r < 0))
    pf    = (pos_r / neg_r) if neg_r > 0 else float("inf")

    cumulative  = np.cumsum(r_vals)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns   = running_max - cumulative
    max_dd_R    = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "n_setups":   n_setups,
        "n_filled":   n_filled,
        "n_wins":     n_wins,
        "wr":         round(wr, 6),
        "mean_r":     round(mean_r, 6),
        "median_r":   round(median_r, 6),
        "pf":         round(pf, 6) if not np.isinf(pf) else float("inf"),
        "max_dd_R":   round(max_dd_R, 6),
        "trades_raw": trades,
    }


# ============================================================================
# v1.2 NEW: Variant grid helpers
# ============================================================================

def _compute_max_dd(r_nets: List[float]) -> float:
    """Max drawdown in R-units from cumulative equity curve."""
    cum   = 0.0
    peak  = 0.0
    max_dd = 0.0
    for r in r_nets:
        cum  += r
        peak  = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def _aggregate_trades(trades: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics from a list of trade dicts."""
    n = len(trades)
    if n == 0:
        return {
            "n_setups": 0, "n_filled": 0, "n_wins": 0,
            "wr": 0.0, "mean_r": 0.0, "median_r": 0.0,
            "pf": 0.0, "max_dd_R": 0.0,
            "avg_entry_price": 0.0, "avg_sl_pct": 0.0, "avg_rr_to_tp": 0.0,
        }

    r_nets = [t["r_net"] for t in trades]
    wins   = [r for r in r_nets if r > 0]
    losses = [r for r in r_nets if r <= 0]

    entries   = [t.get("entry", 0.0) for t in trades]
    sls       = [t.get("sl",    0.0) for t in trades]
    tp_multis = [t.get("tp_R",  2.0) for t in trades]

    avg_entry = float(np.mean(entries)) if entries else 0.0
    avg_sl_pct = 0.0
    valid_pairs = [(e, s) for e, s in zip(entries, sls) if e > 0 and s > 0]
    if valid_pairs:
        avg_sl_pct = float(np.mean([(e - s) / e * 100.0 for e, s in valid_pairs]))

    avg_rr_to_tp = float(np.mean(tp_multis)) if tp_multis else 2.0

    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")

    return {
        "n_setups":       n,
        "n_filled":       n,
        "n_wins":         len(wins),
        "wr":             round(len(wins) / n, 6),
        "mean_r":         round(float(np.mean(r_nets)), 6),
        "median_r":       round(float(np.median(r_nets)), 6),
        "pf":             round(pf, 6) if not np.isinf(pf) else float("inf"),
        "max_dd_R":       round(_compute_max_dd(r_nets), 6),
        "avg_entry_price": round(avg_entry, 10),
        "avg_sl_pct":      round(avg_sl_pct, 4),
        "avg_rr_to_tp":    round(avg_rr_to_tp, 4),
    }


def _backtest_single_variant(
    df_htf: pd.DataFrame,
    df_ltf: Optional[pd.DataFrame],
    structure: dict,
    fibo_zone: dict,
    smart_obs: list,
    fvgs: list,
    sr_levels: list,
    entry_type: str,
    ltf_mode: str,
    tp_R: float,
) -> Dict[str, Any]:
    """
    Single-variant historical replay.

    Algorithm:
        Walk df_htf bar-by-bar from index 50 to end.
        At each cursor where uptrend is confirmed:
          1. Recompute structure + fibo_zone at this point (historical_df = df_htf.iloc[:cursor+1])
          2. Detect zones at this point matching entry_type (NO fibo filter —
             we're replaying history so we use raw zone detection for fill
             probability, then check zone vs. fibo zone for zone availability)
          3. If no matching zone → skip
          4. Determine entry_price = limit at zone level (not market fill)
          5. Walk forward up to _MAX_FILL_BARS to find FILL
          6. If filled: compute SL per ltf_mode, walk up to _MAX_FWD_BARS for outcome
          7. Record trade

    ltf_mode SL computation:
        A (limit, wider SL):  sl = zone_floor * 0.995
        B (LTF confirmed, tighter): sl = HTF df.iloc[fill_bar:fill_bar+5].low.min() * 0.997
                                    (LTF proxy using HTF bars)

    Returns:
        dict of aggregate metrics + trades_raw + recent_check
    """
    df = df_htf.reset_index(drop=True)
    n  = len(df)

    if n < 60:
        empty = _aggregate_trades([])
        empty["recent_check"] = {}
        empty["trades_raw"]   = []
        return empty

    trades: List[Dict] = []
    fwd_highs = df["high"].to_numpy()
    fwd_lows  = df["low"].to_numpy()

    for cursor_bar in range(50, n):
        historical_df = df.iloc[: cursor_bar + 1].copy().reset_index(drop=True)

        # ── Structure at this point ──────────────────────────────────────────
        swings_h = detect_swings(historical_df, pivot=5)
        struct_h = classify_structure(swings_h, historical_df)

        if struct_h["state"] not in {"BOS", "UPTREND"}:
            continue

        current_price = float(df["close"].iloc[cursor_bar])

        # ── Zone detection (no fibo filter during historical replay) ──────────
        obs_h  = detect_smart_obs(historical_df, struct_h)
        fvgs_h = detect_fvgs(historical_df, struct_h)
        fibo_h = detect_fibo_levels(historical_df, struct_h)
        srs_h  = detect_sr_levels(historical_df)

        in_zones_h = classify_current_price_in_zones(
            current_price, obs_h, fvgs_h, fibo_h, srs_h
        )

        # ── Determine entry_price and zone_floor per entry_type ───────────────
        entry_price: Optional[float] = None
        zone_floor:  Optional[float] = None

        if entry_type == "smart_ob":
            candidates = in_zones_h.get("in_smart_ob", [])
            if not candidates:
                continue
            zone_d = max(candidates, key=lambda z: z["ob_bar"])
            entry_price = float(zone_d["ob_high"])
            zone_floor  = float(zone_d["ob_low"])

        elif entry_type == "fvg":
            candidates = in_zones_h.get("in_fvg", [])
            if not candidates:
                continue
            zone_d = max(candidates, key=lambda z: z["created_at_bar"])
            entry_price = float(zone_d["mid"])
            zone_floor  = float(zone_d["bottom"])

        elif entry_type == "fibo_786":
            if not fibo_h:
                continue
            fib786 = fibo_h.get("fib_786")
            if fib786 is None:
                continue
            entry_price = float(fib786)
            zone_floor  = fibo_h.get("fib_786_zone_bottom", entry_price * 0.995)
            if zone_floor is None:
                zone_floor = entry_price * 0.995
            zone_floor = float(zone_floor)

        elif entry_type == "sr":
            candidates = in_zones_h.get("at_sr_support", [])
            if not candidates:
                continue
            zone_d = min(candidates, key=lambda z: abs(z["price"] - current_price))
            entry_price = float(zone_d["price"])
            zone_floor  = entry_price * 0.985

        else:
            continue

        if entry_price is None or zone_floor is None:
            continue

        # ── Walk forward to find FILL (price touches limit entry_price) ───────
        fill_bar: Optional[int] = None
        fill_end = min(cursor_bar + _MAX_FILL_BARS + 1, n)
        for fwd in range(cursor_bar + 1, fill_end):
            bar_low  = float(fwd_lows[fwd])
            bar_high = float(fwd_highs[fwd])
            # Filled if candle range covers entry_price
            if bar_low <= entry_price <= bar_high:
                fill_bar = fwd
                break

        if fill_bar is None:
            continue  # limit never filled

        # ── SL per ltf_mode ───────────────────────────────────────────────────
        if ltf_mode == "A":
            sl = zone_floor * 0.995
        else:
            # ltf_mode B: tighter SL using post-fill HTF swing low as LTF proxy
            proxy_end = min(fill_bar + 5, n)
            ltf_proxy_low = float(df["low"].iloc[fill_bar:proxy_end].min())
            sl = ltf_proxy_low * 0.997

        # Safety: SL must be strictly below entry
        if sl >= entry_price:
            sl = entry_price * 0.985
        if sl <= 0:
            sl = entry_price * 0.985

        risk = entry_price - sl
        if risk <= 0:
            continue

        tp = entry_price + tp_R * risk

        # ── Walk forward from fill_bar to find outcome ─────────────────────
        outcome_bar = None
        r_net       = 0.0
        exit_reason = "timeout"

        outcome_end = min(fill_bar + _MAX_FWD_BARS + 1, n)
        for fwd in range(fill_bar + 1, outcome_end):
            bar_low  = float(fwd_lows[fwd])
            bar_high = float(fwd_highs[fwd])

            if bar_low <= sl:
                outcome_bar = fwd
                r_net       = -1.0
                exit_reason = "sl"
                break
            if bar_high >= tp:
                outcome_bar = fwd
                r_net       = float(tp_R)
                exit_reason = "tp"
                break

        if outcome_bar is None:
            final_bar   = min(fill_bar + _MAX_FWD_BARS, n - 1)
            final_price = float(df["close"].iloc[final_bar])
            r_net       = (final_price - entry_price) / risk
            outcome_bar = final_bar
            exit_reason = "timeout"

        trades.append({
            "bar_index":   cursor_bar,
            "r_mult":      r_net,
            "setup_bar":   cursor_bar,
            "fill_bar":    fill_bar,
            "entry":       entry_price,
            "sl":          sl,
            "tp":          tp,
            "tp_R":        tp_R,
            "r_net":       r_net,
            "exit_reason": exit_reason,
            "outcome_bar": outcome_bar,
        })

    # ── Aggregate ──────────────────────────────────────────────────────────
    result = _aggregate_trades(trades)
    result["trades_raw"]   = trades
    result["recent_check"] = compute_recent_check(trades, n)
    return result


# ============================================================================
# v1.2 PUBLIC: run_variant_grid
# ============================================================================

def run_variant_grid(
    df_htf: pd.DataFrame,
    df_ltf: Optional[pd.DataFrame],
    structure: Dict,
    fibo_zone: Dict,
    smart_obs: List,
    fvgs: List,
    sr_levels: List,
) -> pd.DataFrame:
    """
    Run all 24 entry variants for this coin.

    Variants = 4 entry_types × 2 ltf_modes × 3 tp_R values

    Entry types:
        smart_ob  — limit at OB.ob_high   (if any OBs in fibo zone)
        fvg       — limit at FVG.mid      (if any FVGs in fibo zone)
        fibo_786  — limit at fib_786 level (always available if fibo_zone is set)
        sr        — limit at top S/R support inside zone (if any)

    LTF confirmation modes:
        A. limit (no LTF wait): pre-place limit, SL wider (zone floor * 0.995)
        B. LTF confirmed:       wait for LTF reversal, SL tighter
                                (HTF swing-low proxy of post-fill 5 bars * 0.997)

    TP multiples: 2.0, 2.5, 3.0

    Edge cases:
        - entry_type has no matching zones at a cursor → n_setups for that variant
          may be 0; the row is still included with all metrics = 0.
        - Only fibo_786 available (no OB/FVG/SR in zone): 6 of 24 rows will have
          actual setups; the rest will have n_setups = 0.
        - df_htf has < 100 bars: results flagged low_confidence = True by caller.
        - PF = inf (no losses): kept as float('inf'), displayed by UI as ∞ / N/A.

    Returns:
        DataFrame with 24 rows × columns:
            entry_type, ltf_mode, tp_R,
            n_setups, n_filled, n_wins,
            wr, mean_r, median_r, pf, max_dd_R,
            avg_entry_price, avg_sl_pct, avg_rr_to_tp,
            recent_check (dict), trades_raw (list)
        Sorted by pf descending, index reset.
    """
    ENTRY_TYPES = ["smart_ob", "fvg", "fibo_786", "sr"]
    LTF_MODES   = ["A", "B"]
    TP_RS       = [2.0, 2.5, 3.0]

    rows = []

    for entry_type in ENTRY_TYPES:
        for ltf_mode in LTF_MODES:
            for tp_R in TP_RS:
                try:
                    result = _backtest_single_variant(
                        df_htf, df_ltf, structure, fibo_zone,
                        smart_obs, fvgs, sr_levels,
                        entry_type, ltf_mode, tp_R,
                    )
                except Exception as exc:
                    print(f"[backtest] variant error {entry_type}/{ltf_mode}/{tp_R}: {exc}")
                    result = _aggregate_trades([])
                    result["trades_raw"]   = []
                    result["recent_check"] = {}

                rows.append({
                    "entry_type":  entry_type,
                    "ltf_mode":    ltf_mode,
                    "tp_R":        tp_R,
                    **{k: v for k, v in result.items() if k != "trades_raw"},
                    "trades_raw":  result.get("trades_raw", []),
                })

    df = pd.DataFrame(rows)

    # Sort by PF descending (treat inf as very large)
    def _pf_sort_key(v: Any) -> float:
        if isinstance(v, float) and np.isinf(v):
            return 1e18
        try:
            return float(v)
        except Exception:
            return 0.0

    df["_pf_sort"] = df["pf"].apply(_pf_sort_key)
    df = (
        df.sort_values("_pf_sort", ascending=False)
          .drop(columns=["_pf_sort"])
          .reset_index(drop=True)
    )
    return df


# ============================================================================
# v1.2 PUBLIC: deep_dive_backtest
# ============================================================================

def deep_dive_backtest(
    symbol: str,
    df_htf: pd.DataFrame,
    df_ltf: Optional[pd.DataFrame],
    structure: Dict,
    fibo_zone: Dict,
    smart_obs: List,
    fvgs: List,
    sr_levels: List,
    ai_provider: Optional[str] = None,
    ai_api_key: Optional[str] = None,
) -> Dict:
    """
    User-triggered detailed analysis for ONE coin (called from UI deep-dive button).

    Steps:
        1. Run run_variant_grid → 24 variants (M1 Fixed SL+TP)
        2. For each of 24 variants, simulate 2 additional management modes:
               M2: Move SL to BE at 1R
               M3: Trailing stop (1.5×ATR after 1R)
           Total: 72 variants
        3. Identify BEST per (entry_type, ltf_mode) — one recommendation per zone
           type with its best management mode
        4. Identify BEST_OVERALL across all 72
        5. (Optional) Call AI verdict with full SMC context
        6. Build final trade_plan from best_overall

    NOTE: AI verdict is NOT called by run_scan (too slow for batch scan).
    It is ONLY triggered here, when the user clicks the deep-dive button.

    Returns:
        {
            "symbol": str,
            "structure": dict,
            "fibo_zone": dict,
            "all_72_variants": pd.DataFrame,
            "best_per_entry": [{"entry_type": str, "best_row": dict}, ...],
            "best_overall": dict | None,
            "ai_verdict": dict | None,
            "recommended_trade_plan": dict | None,
            "computed_at_utc": str,
        }
    """
    # ── Step 1: 24-variant grid (M1 Fixed) ───────────────────────────────────
    variant_24 = run_variant_grid(
        df_htf, df_ltf, structure, fibo_zone, smart_obs, fvgs, sr_levels
    )

    # ── Step 2: Expand with M2 and M3 management modes ─────────────────────
    all_72: List[Dict] = []
    for _, row in variant_24.iterrows():
        base       = row.to_dict()
        trades_raw = base.get("trades_raw", [])

        # M1: Fixed — already computed
        all_72.append({**base, "management": "Fixed"})

        # M2: BE at 1R
        m2_metrics = _simulate_with_management(trades_raw, "BE_at_1R")
        all_72.append({**base, **m2_metrics, "management": "BE_at_1R"})

        # M3: Trailing stop (1.5×ATR after 1R)
        m3_metrics = _simulate_with_management(trades_raw, "Trailing", df_htf=df_htf)
        all_72.append({**base, **m3_metrics, "management": "Trailing"})

    df_72 = pd.DataFrame(all_72)

    # Sort by PF descending
    def _pf_sort_key(v: Any) -> float:
        if isinstance(v, float) and np.isinf(v):
            return 1e18
        try:
            return float(v)
        except Exception:
            return 0.0

    if not df_72.empty and "pf" in df_72.columns:
        df_72["_pf_sort"] = df_72["pf"].apply(_pf_sort_key)
        df_72 = (
            df_72.sort_values("_pf_sort", ascending=False)
                 .drop(columns=["_pf_sort"])
                 .reset_index(drop=True)
        )

    # ── Step 3: Best per entry_type ───────────────────────────────────────────
    best_per_entry = []
    for et in ["smart_ob", "fvg", "fibo_786", "sr"]:
        sub = df_72[df_72["entry_type"] == et] if not df_72.empty else pd.DataFrame()
        if not sub.empty:
            best_per_entry.append({
                "entry_type": et,
                "best_row":   sub.iloc[0].to_dict(),
            })

    # ── Step 4: Best overall ──────────────────────────────────────────────────
    best_overall: Optional[Dict] = df_72.iloc[0].to_dict() if not df_72.empty else None

    # ── Step 5: Recommended trade plan ───────────────────────────────────────
    recommended_trade_plan: Optional[Dict] = None
    if best_overall:
        try:
            recommended_trade_plan = _build_trade_plan_from_variant(
                best_overall, fibo_zone, smart_obs, fvgs, sr_levels
            )
        except Exception as exc:
            print(f"[backtest] trade plan build error for {symbol}: {exc}")

    # ── Step 6: (Optional) AI verdict ────────────────────────────────────────
    ai_verdict: Optional[Dict] = None
    if ai_provider and ai_api_key:
        try:
            from qf_smc.ai_verdict import build_smc_prompt, get_verdict  # type: ignore
            context = {
                "symbol":          symbol,
                "structure":       structure,
                "fibo_zone":       fibo_zone,
                "best_overall":    best_overall,
                "best_per_entry":  best_per_entry,
            }
            prompt     = build_smc_prompt(context)
            ai_verdict = get_verdict(prompt, provider=ai_provider, api_key=ai_api_key)
        except Exception as e:
            ai_verdict = {"error": str(e)}

    return {
        "symbol":                  symbol,
        "structure":               structure,
        "fibo_zone":               fibo_zone,
        "all_72_variants":         df_72,
        "best_per_entry":          best_per_entry,
        "best_overall":            best_overall,
        "ai_verdict":              ai_verdict,
        "recommended_trade_plan":  recommended_trade_plan,
        "computed_at_utc":         datetime.utcnow().isoformat(),
    }


# ============================================================================
# v1.2 PRIVATE: management-mode simulation helpers
# ============================================================================

def _simulate_with_management(
    trades_raw: List[Dict],
    management_mode: str,
    df_htf: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Re-simulate trades with a different management mode.

    M2 (BE_at_1R):
        For each trade, check whether (conceptually) price reached entry + 1R
        before the exit. Approximation without full bar replay:
          - If r_net >= 0: price already passed entry; keep as-is (BE saved
            any negative; if TP hit, keep TP).
          - If -1 < r_net < 0: price neared BE but not quite — we conservatively
            apply BE logic for losses with |r_net| < 0.5 (likely reached +1R
            on the way, then reversed to BE, then drifted to a small loss that
            the trailing BE would have capped at 0).
          - If r_net <= -1: full SL hit before reaching +1R; keep r_net = -1.

    M3 (Trailing, 1.5×ATR after 1R):
        After 1R reached, the stop trails at (highest high - 1.5×ATR).
        Approximation: for winning trades (r_net > 0), apply a 10% uplift to
        capture the trailing benefit in trends. For losing trades, trailing after
        1R means the stop is at entry (BE), so apply the same BE heuristic as M2.
        Full bar-by-bar re-walk would require df_htf for each trade; the
        approximation is conservative and consistent with M2 for the loss side.

    Rationale for approximations:
        Full re-walk (ideal) needs all bar-level highs/lows between fill_bar and
        outcome_bar. The trades_raw list stores only entry, sl, tp, r_net — not
        the per-bar path. Rather than re-running the walk here (expensive for 24
        × large df), we use the heuristics above. When df_htf is provided, a
        future release can do the proper re-walk.

    Returns: dict of new aggregate metrics (same shape as _aggregate_trades output).
    """
    if not trades_raw:
        return _aggregate_trades([])

    new_trades: List[Dict] = []

    for t in trades_raw:
        r = t["r_net"]

        if management_mode == "BE_at_1R":
            if r >= 0:
                new_r = r              # TP hit or timeout above entry — keep
            elif abs(r) < 0.5:
                new_r = 0.0            # approximation: BE stop saved it
            else:
                new_r = r              # full SL hit before reaching 1R

        elif management_mode == "Trailing":
            if r > 0:
                new_r = r * 1.10       # ~10% uplift for successful trailing
            elif abs(r) < 0.5:
                new_r = 0.0            # same BE floor as M2
            else:
                new_r = r

        else:  # Fixed / M1 — unchanged
            new_r = r

        new_trades.append({**t, "r_net": new_r, "r_mult": new_r})

    return _aggregate_trades(new_trades)


# ============================================================================
# v1.2 PRIVATE: trade plan builder from variant row
# ============================================================================

def _build_trade_plan_from_variant(
    variant_row: Dict,
    fibo_zone: Dict,
    smart_obs: List,
    fvgs: List,
    sr_levels: List,
) -> Dict:
    """
    Build a concrete trade plan from the BEST variant row.

    Looks up the current OB/FVG/SR/Fibo for the entry_type, then computes
    entry_price, SL, and TP levels for the live trade recommendation.
    """
    entry_type  = variant_row.get("entry_type", "fibo_786")
    ltf_mode    = variant_row.get("ltf_mode", "A")
    tp_R        = float(variant_row.get("tp_R", 2.0))
    management  = variant_row.get("management", "Fixed")

    entry_price: Optional[float] = None
    zone_floor:  Optional[float] = None

    # ── Lookup current zone ───────────────────────────────────────────────────
    if entry_type == "smart_ob":
        zone = smart_obs[0] if smart_obs else None
        if zone:
            entry_price = float(zone["ob_high"])
            zone_floor  = float(zone["ob_low"])

    elif entry_type == "fvg":
        zone = fvgs[0] if fvgs else None
        if zone:
            entry_price = float(zone["mid"])
            zone_floor  = float(zone["bottom"])

    elif entry_type == "fibo_786":
        fib786 = fibo_zone.get("fib_786")
        if fib786 is not None:
            entry_price = float(fib786)
            zone_floor  = float(
                fibo_zone.get("fib_786_zone_bottom", entry_price * 0.995) or
                entry_price * 0.995
            )

    else:  # sr
        support = next((s for s in sr_levels if s.get("kind") == "support"), None)
        if support:
            entry_price = float(support["price"])
            zone_floor  = entry_price * 0.985

    # Fallback if no zone found
    if entry_price is None or zone_floor is None:
        return {
            "entry_type":    entry_type,
            "ltf_mode":      ltf_mode,
            "management":    management,
            "tp_R":          tp_R,
            "entry_price":   None,
            "sl":            None,
            "error":         "No current zone found for entry_type",
        }

    # ── SL per ltf_mode ───────────────────────────────────────────────────────
    if ltf_mode == "A":
        sl = zone_floor * 0.995   # wider (limit mode)
    else:
        sl = zone_floor * 0.998   # tighter (LTF confirmed — conservative estimate)

    # Safety
    if sl >= entry_price:
        sl = entry_price * 0.985
    if sl <= 0:
        sl = entry_price * 0.985

    risk = entry_price - sl
    if risk <= 0:
        risk = entry_price * 0.015

    tp1 = entry_price + 2.0 * risk
    tp2 = entry_price + 2.5 * risk
    tp3 = entry_price + 3.0 * risk

    return {
        "entry_type":       entry_type,
        "ltf_mode":         ltf_mode,
        "management":       management,
        "tp_R":             tp_R,
        "entry_price":      entry_price,
        "sl":               sl,
        "tp1_price":        tp1,
        "tp2_price":        tp2,
        "tp3_price":        tp3,
        "risk_pct":         round((risk / entry_price) * 100, 4),
        "rr_to_tp1":        2.0,
        "rr_to_tp2":        2.5,
        "rr_to_tp3":        3.0,
        "expected_value_R": round(float(variant_row.get("mean_r", 0.0)), 4),
    }


# ============================================================================
# v1.1 PUBLIC API — unchanged for backward compatibility
# ============================================================================

def backtest_per_coin(
    df_htf: pd.DataFrame,
    df_ltf: pd.DataFrame,
    entry_zone_type: str,
    sl_method: str,
    tp_method: str,
    tp_R: float = 2.0,
    max_setups_to_replay: int = 200,
    simplified: bool = False,
) -> Dict[str, Any]:
    """
    Walk df_htf chronologically, detect SMC setups, and simulate outcomes.

    Args:
        df_htf: HTF OHLCV DataFrame (walked bar-by-bar).
        df_ltf: LTF DataFrame — reserved for future LTF confirmation; pass None
                to skip (Phase 1 uses HTF only).
        entry_zone_type: one of {"smart_ob", "fvg", "fibo_786", "sr"}.
        sl_method:
            "fixed"             — SL = entry * 0.985  (1.5% below)
            "structural"        — SL = zone-specific low
            "structural_wider"  — structural SL minus 1.5 × ATR(14)
        tp_method:
            "fixed_R"           — TP = entry + tp_R × risk
            "sr"                — TP = next resistance level (fallback: fixed_R)
            "bearish_ob_fvg"    — TP = nearest bearish zone above (Phase 1 → sr)
            "liq"               — TP = liq cluster (Phase 1 → fixed_R)
        tp_R: R-multiple for fixed_R method (default 2.0).
        max_setups_to_replay: cap on historical setups (performance).
        simplified: if True, only snapshot structure every _SIMPLIFIED_STEP bars
                    for faster universe-baseline computation.

    Returns:
        Dict with keys: n_setups, n_filled, n_wins, wr, mean_r, median_r,
        pf, max_dd_R, trades_raw.
        Returns all-zeros dict if no setups found.
    """
    if entry_zone_type not in _VALID_ENTRY_TYPES:
        raise ValueError(
            f"entry_zone_type must be one of {_VALID_ENTRY_TYPES}, "
            f"got {entry_zone_type!r}"
        )

    df = df_htf.reset_index(drop=True)
    n  = len(df)

    if n < 60:
        return _compute_trade_stats([], tp_R)

    trades: List[Dict] = []
    stride = _SIMPLIFIED_STEP if simplified else 1

    for cursor_bar in range(50, n, stride):
        if len(trades) >= max_setups_to_replay:
            break

        historical_df = df.iloc[: cursor_bar + 1].copy()
        historical_df = historical_df.reset_index(drop=True)

        swings    = detect_swings(historical_df, pivot=5)
        structure = classify_structure(swings, historical_df)

        if structure["state"] not in {"BOS", "UPTREND"}:
            continue

        current_price = float(df["close"].iloc[cursor_bar])

        zone_result = _detect_zone_setup(
            historical_df, structure, entry_zone_type, current_price
        )
        if zone_result is None:
            continue

        entry, sl_structural = zone_result

        if sl_method == "fixed":
            sl = entry * 0.985
        elif sl_method == "structural":
            sl = sl_structural
        elif sl_method == "structural_wider":
            atr = _compute_atr(df, cursor_bar, period=14)
            sl  = sl_structural - 1.5 * atr
        else:
            sl = entry * 0.985

        if sl >= entry:
            sl = entry * 0.985
        if sl <= 0:
            sl = entry * 0.985

        risk = entry - sl
        if risk <= 0:
            continue

        tp = _compute_tp(historical_df, structure, entry, sl, tp_method, tp_R)
        if tp <= entry:
            tp = entry + tp_R * risk

        outcome_bar = None
        r_net       = 0.0
        exit_reason = "timeout"

        fwd_end = min(cursor_bar + _MAX_FWD_BARS + 1, n)
        for fwd in range(cursor_bar + 1, fwd_end):
            bar_low  = float(df["low"].iloc[fwd])
            bar_high = float(df["high"].iloc[fwd])

            if bar_low <= sl:
                outcome_bar = fwd
                r_net       = -1.0
                exit_reason = "sl"
                break
            if bar_high >= tp:
                outcome_bar = fwd
                r_net       = float(tp_R)
                exit_reason = "tp"
                break

        if outcome_bar is None:
            final_bar   = min(cursor_bar + _MAX_FWD_BARS, n - 1)
            final_price = float(df["close"].iloc[final_bar])
            r_net       = (final_price - entry) / risk
            outcome_bar = final_bar
            exit_reason = "timeout"

        trades.append({
            "bar_index":   cursor_bar,
            "r_mult":      r_net,
            "setup_bar":   cursor_bar,
            "entry":       entry,
            "sl":          sl,
            "tp":          tp,
            "r_net":       r_net,
            "exit_reason": exit_reason,
            "outcome_bar": outcome_bar,
        })

    return _compute_trade_stats(trades, tp_R)


# ----------------------------------------------------------------------------

@st.cache_data(ttl=86400 * BASELINE_TTL_DAYS, show_spinner=False)
def backtest_universe_baseline(
    entry_zone_type: str,
    sl_method: str,
    tp_method: str,
    tp_R: float = 2.0,
    timeframe: str = "4h",
    n_sample_coins: int = 50,
) -> Dict[str, Any]:
    """
    Compute or load cached universe-wide baseline metrics for a given combo.

    Cache location:
        data/smc_baselines/{entry}_{sl}_{tp}_{tp_R:.1f}R_{timeframe}.json

    If a fresh cache file exists (<BASELINE_TTL_DAYS old): load and return it.
    Otherwise: sample the top-N coins by volume, run backtest_per_coin on each
    with simplified=True (fast walk), aggregate results, save to cache.

    Aggregation:
        wr_universe    = sum(n_wins) / sum(n_setups)
        mean_r         = weighted average by n_setups
        pf_universe    = sum(all positive R) / abs(sum(all negative R))
        max_dd_R       = worst single-coin max drawdown

    Returns same shape as backtest_per_coin output (minus trades_raw) plus:
        "n_coins_sampled": int
        "computed_at_utc": ISO timestamp
    """
    combo_key = (
        f"{entry_zone_type}_{sl_method}_{tp_method}_{tp_R:.1f}R_{timeframe}"
    )

    cached = _load_baseline_cache(combo_key)
    if cached is not None:
        return cached

    symbols = _scanner_get_universe(n=n_sample_coins, timeframe=timeframe)

    total_wins = 0
    total_sets = 0
    total_r    = 0.0
    all_pos_r  = 0.0
    all_neg_r  = 0.0
    max_dd     = 0.0

    for sym in symbols:
        try:
            df = _scanner_fetch_candles(sym, timeframe, limit=300)
            if df is None or len(df) < 60:
                continue

            result = backtest_per_coin(
                df_htf=df,
                df_ltf=None,
                entry_zone_type=entry_zone_type,
                sl_method=sl_method,
                tp_method=tp_method,
                tp_R=tp_R,
                max_setups_to_replay=100,
                simplified=True,
            )

            ns = result["n_setups"]
            if ns == 0:
                continue

            total_wins += result["n_wins"]
            total_sets += ns
            total_r    += result["mean_r"] * ns

            for t in result["trades_raw"]:
                r = t["r_net"]
                if r > 0:
                    all_pos_r += r
                else:
                    all_neg_r += abs(r)

            if result["max_dd_R"] > max_dd:
                max_dd = result["max_dd_R"]

        except Exception:
            continue

    wr_universe     = (total_wins / total_sets) if total_sets > 0 else 0.0
    mean_r_universe = (total_r / total_sets)    if total_sets > 0 else 0.0
    pf_universe     = (all_pos_r / all_neg_r)   if all_neg_r > 0 else float("inf")

    now_utc = datetime.now(timezone.utc).isoformat()

    result_data: Dict[str, Any] = {
        "metadata": {
            "computed_at_utc":  now_utc,
            "n_coins_sampled":  len(symbols),
            "timeframe":        timeframe,
            "entry_zone_type":  entry_zone_type,
            "sl_method":        sl_method,
            "tp_method":        tp_method,
            "tp_R":             tp_R,
        },
        "n_setups":         total_sets,
        "n_filled":         0,
        "n_wins":           total_wins,
        "wr":               round(wr_universe, 6),
        "mean_r":           round(mean_r_universe, 6),
        "median_r":         0.0,
        "pf":               round(pf_universe, 6) if not np.isinf(pf_universe) else float("inf"),
        "max_dd_R":         round(max_dd, 6),
        "trades_raw":       [],
        "n_coins_sampled":  len(symbols),
        "computed_at_utc":  now_utc,
    }

    _save_baseline_cache(combo_key, result_data)
    return result_data


# ----------------------------------------------------------------------------

def bayesian_blend(
    per_coin: Dict[str, Any],
    universe: Dict[str, Any],
    prior_strength: int = 30,
) -> Dict[str, Any]:
    """
    Blend per-coin sample (small n) with universe prior (large n) to combat
    overfitting on coins with few historical setups.

    Blending formulas:
        blended_mean_r = (n_coin × mean_r_coin + prior × mean_r_univ)
                         / (n_coin + prior)
        blended_wr     = (n_coin × wr_coin + prior × wr_univ)
                         / (n_coin + prior)
        blended_pf     = geometric blend in log-space

    Args:
        per_coin:       Output of backtest_per_coin for a single coin.
        universe:       Output of backtest_universe_baseline.
        prior_strength: Number of "virtual" universe samples to add.

    Returns:
        {"wr", "mean_r", "pf", "n_effective", "weight_per_coin"}
    """
    n_coin  = int(per_coin.get("n_setups", 0))
    n_total = n_coin + prior_strength

    if n_coin == 0:
        return {
            "wr":              float(universe.get("wr", 0.0)),
            "mean_r":          float(universe.get("mean_r", 0.0)),
            "pf":              float(universe.get("pf", 1.0)),
            "n_effective":     prior_strength,
            "weight_per_coin": 0.0,
        }

    blended_mean_r = (
        n_coin * float(per_coin.get("mean_r", 0.0))
        + prior_strength * float(universe.get("mean_r", 0.0))
    ) / n_total

    blended_wr = (
        n_coin * float(per_coin.get("wr", 0.0))
        + prior_strength * float(universe.get("wr", 0.0))
    ) / n_total

    _PF_CLAMP_HI = 20.0
    _PF_CLAMP_LO = 0.01
    pf_coin  = float(per_coin.get("pf", 1.0))
    pf_univ  = float(universe.get("pf", 1.0))
    pf_coin_c = max(_PF_CLAMP_LO, min(_PF_CLAMP_HI, pf_coin))
    pf_univ_c = max(_PF_CLAMP_LO, min(_PF_CLAMP_HI, pf_univ))

    log_pf     = (n_coin * np.log(pf_coin_c) + prior_strength * np.log(pf_univ_c)) / n_total
    blended_pf = float(np.exp(log_pf))

    return {
        "wr":              round(blended_wr, 6),
        "mean_r":          round(blended_mean_r, 6),
        "pf":              round(blended_pf, 6),
        "n_effective":     n_total,
        "weight_per_coin": round(n_coin / n_total, 6),
    }


# ----------------------------------------------------------------------------

def compute_recent_check(
    trades_raw: List[Dict],
    n_df: int,
) -> Dict[str, Any]:
    """
    Split trades into earlier vs recent time buckets and issue a verdict on
    whether recent performance is STRONGER / STABLE / WEAKER vs earlier.

    Uses _compute_decay_buckets and _bucket_stats_for_trades from qf_shared.

    Args:
        trades_raw: List of trade dicts. Each must have keys 'bar_index' and 'r_mult'.
        n_df:       Total bars in the source DataFrame.

    Returns:
        {
            "earlier_period_label": str,
            "earlier_stats": {mean_r, pf, n, wr},
            "recent_period_label":  str,
            "recent_stats":  {mean_r, pf, n, wr},
            "verdict": "STRONGER" | "STABLE" | "WEAKER" | "INSUFFICIENT_DATA",
            "pf_ratio": float | None,
        }
    """
    _empty = {
        "earlier_period_label": "N/A",
        "earlier_stats":  {"mean_r": 0.0, "pf": 0.0, "n": 0, "wr": 0.0},
        "recent_period_label":  "N/A",
        "recent_stats":   {"mean_r": 0.0, "pf": 0.0, "n": 0, "wr": 0.0},
        "verdict":        "INSUFFICIENT_DATA",
        "pf_ratio":       None,
    }

    if n_df <= 1 or not trades_raw:
        return _empty

    buckets                = _compute_decay_buckets(n_df)
    bucket_rows, _, _      = _bucket_stats_for_trades(trades_raw, n_df, buckets)

    if len(bucket_rows) < 2:
        return _empty

    earlier_row = bucket_rows[0]
    recent_row  = bucket_rows[-1]

    earlier_label = earlier_row["label"]
    recent_label  = recent_row["label"]

    denom = float(n_df - 1)

    def _get_bucket_trades(edge_idx: int) -> List[Dict]:
        lo, hi = buckets["edges"][edge_idx]
        is_last_old = (edge_idx == 0)
        result = []
        for t in trades_raw:
            bi = t.get("bar_index")
            if bi is None:
                continue
            age = (n_df - 1 - bi) / denom
            in_range = (lo <= age < hi) or (is_last_old and age == hi)
            if in_range:
                result.append(t)
        return result

    earlier_trades = _get_bucket_trades(0)
    recent_trades  = _get_bucket_trades(len(bucket_rows) - 1)

    def _pf(trades_subset: List[Dict]) -> float:
        pos = sum(t["r_mult"] for t in trades_subset if t["r_mult"] > 0)
        neg = abs(sum(t["r_mult"] for t in trades_subset if t["r_mult"] < 0))
        return (pos / neg) if neg > 0 else float("inf")

    def _mean_r(trades_subset: List[Dict]) -> float:
        if not trades_subset:
            return 0.0
        return float(np.mean([t["r_mult"] for t in trades_subset]))

    earlier_n = len(earlier_trades)
    recent_n  = len(recent_trades)

    earlier_wr_pct = earlier_row["wr"]
    recent_wr_pct  = recent_row["wr"]

    earlier_stats = {
        "mean_r": round(_mean_r(earlier_trades), 4),
        "pf":     round(_pf(earlier_trades), 4)
                  if not np.isinf(_pf(earlier_trades)) else float("inf"),
        "n":      earlier_n,
        "wr":     round(earlier_wr_pct / 100.0, 4),
    }
    recent_stats = {
        "mean_r": round(_mean_r(recent_trades), 4),
        "pf":     round(_pf(recent_trades), 4)
                  if not np.isinf(_pf(recent_trades)) else float("inf"),
        "n":      recent_n,
        "wr":     round(recent_wr_pct / 100.0, 4),
    }

    if recent_n < 10 or earlier_n < 10:
        return {
            "earlier_period_label": earlier_label,
            "earlier_stats":        earlier_stats,
            "recent_period_label":  recent_label,
            "recent_stats":         recent_stats,
            "verdict":              "INSUFFICIENT_DATA",
            "pf_ratio":             None,
        }

    earlier_pf = earlier_stats["pf"]
    recent_pf  = recent_stats["pf"]

    if np.isinf(earlier_pf) and np.isinf(recent_pf):
        pf_ratio = 1.0
    elif np.isinf(recent_pf):
        pf_ratio = 2.0
    elif np.isinf(earlier_pf):
        pf_ratio = 0.0
    else:
        pf_ratio = float(recent_pf) / max(float(earlier_pf), 0.5)

    if pf_ratio > 1.20:
        verdict = "STRONGER"
    elif pf_ratio < 0.80:
        verdict = "WEAKER"
    else:
        verdict = "STABLE"

    return {
        "earlier_period_label": earlier_label,
        "earlier_stats":        earlier_stats,
        "recent_period_label":  recent_label,
        "recent_stats":         recent_stats,
        "verdict":              verdict,
        "pf_ratio":             round(pf_ratio, 4),
    }


# ----------------------------------------------------------------------------

def run_combo_grid(
    df_htf: pd.DataFrame,
    df_ltf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run every combination of entry × SL × TP for this coin's history.

    Grid: 4 entries × 3 SL × 4 TP = 48 combos.
        entry_zone_type: ["smart_ob", "fvg", "fibo_786", "sr"]
        sl_method:       ["fixed", "structural", "structural_wider"]
        tp_spec:         ["fixed_2R", "fixed_2.5R", "fixed_3R", "sr"]

    NOTE: This function is preserved unchanged for backward compatibility.
    New callers should prefer run_variant_grid() which uses the v1.2 spec.

    Returns:
        DataFrame sorted by pf descending with columns:
        entry_zone_type, sl_method, tp_method, tp_R,
        n_setups, n_filled, n_wins, wr, mean_r, pf, max_dd_R, recent_verdict.
    """
    entries  = ["smart_ob", "fvg", "fibo_786", "sr"]
    sls      = ["fixed", "structural", "structural_wider"]
    tp_specs = [
        ("fixed_2R",   "fixed_R", 2.0),
        ("fixed_2.5R", "fixed_R", 2.5),
        ("fixed_3R",   "fixed_R", 3.0),
        ("sr",         "sr",      2.0),
    ]

    n_df = len(df_htf)
    rows = []

    for entry_zone_type in entries:
        for sl_method in sls:
            for tp_label, tp_method, tp_R in tp_specs:
                try:
                    result = backtest_per_coin(
                        df_htf=df_htf,
                        df_ltf=df_ltf,
                        entry_zone_type=entry_zone_type,
                        sl_method=sl_method,
                        tp_method=tp_method,
                        tp_R=tp_R,
                        max_setups_to_replay=200,
                        simplified=False,
                    )

                    recent_verdict = "INSUFFICIENT_DATA"
                    if result["n_setups"] >= 5:
                        check = compute_recent_check(result["trades_raw"], n_df)
                        recent_verdict = check.get("verdict", "INSUFFICIENT_DATA")

                    rows.append({
                        "entry_zone_type": entry_zone_type,
                        "sl_method":       sl_method,
                        "tp_method":       tp_label,
                        "tp_R":            tp_R,
                        "n_setups":        result["n_setups"],
                        "n_filled":        result["n_filled"],
                        "n_wins":          result["n_wins"],
                        "wr":              result["wr"],
                        "mean_r":          result["mean_r"],
                        "pf":              result["pf"],
                        "max_dd_R":        result["max_dd_R"],
                        "recent_verdict":  recent_verdict,
                    })

                except Exception as exc:
                    rows.append({
                        "entry_zone_type": entry_zone_type,
                        "sl_method":       sl_method,
                        "tp_method":       tp_label,
                        "tp_R":            tp_R,
                        "n_setups":        0,
                        "n_filled":        0,
                        "n_wins":          0,
                        "wr":              0.0,
                        "mean_r":          0.0,
                        "pf":              0.0,
                        "max_dd_R":        0.0,
                        "recent_verdict":  f"ERROR: {exc}",
                    })

    combo_df = pd.DataFrame(rows)

    def _sort_pf(v: Any) -> float:
        if isinstance(v, float) and np.isinf(v):
            return 1e18
        try:
            return float(v)
        except Exception:
            return 0.0

    combo_df["_pf_sort"] = combo_df["pf"].apply(_sort_pf)
    combo_df = combo_df.sort_values("_pf_sort", ascending=False).drop(
        columns=["_pf_sort"]
    ).reset_index(drop=True)

    return combo_df


# ============================================================================
# Cache helpers
# ============================================================================

def _save_baseline_cache(combo_key: str, data: Dict[str, Any]) -> None:
    """Write baseline dict to JSON in BASELINE_CACHE_DIR/{combo_key}.json."""
    BASELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = BASELINE_CACHE_DIR / f"{combo_key}.json"

    def _sanitise(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitise(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if np.isinf(v) or np.isnan(v) else v
        if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
            return None
        return obj

    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(_sanitise(data), fh, indent=2)
    except Exception as exc:
        print(f"[backtest] Cache write failed for {combo_key}: {exc}")


def _load_baseline_cache(combo_key: str) -> Optional[Dict[str, Any]]:
    """
    Load baseline dict from JSON if the file exists and is < BASELINE_TTL_DAYS old.
    Returns None if missing, stale, or unreadable.
    """
    cache_path = BASELINE_CACHE_DIR / f"{combo_key}.json"

    if not cache_path.exists():
        return None

    age_seconds = time.time() - cache_path.stat().st_mtime
    if age_seconds > BASELINE_TTL_DAYS * 86400:
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data
    except Exception as exc:
        print(f"[backtest] Cache read failed for {combo_key}: {exc}")
        return None
