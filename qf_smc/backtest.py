"""
qf_smc/backtest.py — Backtest engine for SMC Long setups
=========================================================
Per-coin backtest + universe baseline + Bayesian blend + bucketed verdict
per QUANTFLOW SMC Long spec v1.1 §5.6.

Public API:
  - backtest_per_coin(df_htf, df_ltf, entry, sl, tp, tp_R, ...)
  - backtest_universe_baseline(entry, sl, tp, tp_R, timeframe, n_sample_coins)
  - bayesian_blend(per_coin, universe, prior_strength)
  - compute_recent_check(trades_raw, n_df)
  - run_combo_grid(df_htf, df_ltf)

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
_SR_SL_BUFFER_PCT = 0.010   # 1.0% below SR support price
_FIBO_SL_BUFFER_PCT = 0.005  # 0.5% extra below fib_886

# Max forward-look window for trade outcome resolution (bars)
_MAX_FWD_BARS = 50

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
    # Detect all zone types in one pass so we can reuse for TP calculation later
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
        # Prefer the most recently formed OB (highest ob_bar index)
        zone = max(matching, key=lambda z: z["ob_bar"])
        sl_structural = float(zone["ob_low"])
        return entry, sl_structural

    elif entry_zone_type == "fvg":
        matching = in_zones["in_fvg"]
        if not matching:
            return None
        # Most recently created FVG
        zone = max(matching, key=lambda z: z["created_at_bar"])
        sl_structural = float(zone["bottom"])
        return entry, sl_structural

    elif entry_zone_type == "fibo_786":
        if not in_zones["in_fibo_786"]:
            return None
        # SL = fib_886 level with a small extra buffer
        fib_886 = fibo.get("fib_886")
        if fib_886 is None:
            return None
        sl_structural = float(fib_886) * (1.0 - _FIBO_SL_BUFFER_PCT)
        return entry, sl_structural

    elif entry_zone_type == "sr":
        matching = in_zones["at_sr_support"]
        if not matching:
            return None
        # Closest support level
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
        "bearish_ob_fvg" — next resistance S/R (Phase 1 fallback; full version
                           would scan for bearish OB/FVG above)
        "liq"            — liquidity cluster (Phase 1 fallback: fixed_R)
    """
    risk = entry - sl
    if risk <= 0:
        risk = entry * 0.015  # safety: assume 1.5% risk

    if tp_method == "fixed_R":
        return entry + tp_R * risk

    elif tp_method in ("sr", "bearish_ob_fvg"):
        # Find the nearest resistance above entry
        srs = detect_sr_levels(historical_df)
        # Filter: resistance strictly above entry (at least 0.1% gap to avoid
        # capturing the current bar itself as a resistance)
        resistances = [
            s for s in srs
            if s["kind"] == "resistance" and s["price"] > entry * 1.001
        ]
        if resistances:
            return float(min(resistances, key=lambda x: x["price"])["price"])
        # Fallback to fixed R
        return entry + tp_R * risk

    elif tp_method == "liq":
        # Phase 1 fallback — full implementation would use OI/liquidation heatmap
        return entry + tp_R * risk

    # Default
    return entry + tp_R * risk


def _compute_trade_stats(trades: List[Dict], tp_R: float) -> Dict[str, Any]:
    """
    Aggregate a list of trade dicts into backtest summary metrics.

    Each trade must have 'r_net' and 'exit_reason'.
    Returns the full result dict as specified in backtest_per_coin's docstring.
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

    # Max drawdown in R units (peak-to-trough on the cumulative R curve)
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
# Public API
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

    # Work on a positional-integer-indexed copy so iloc and [] are equivalent
    df = df_htf.reset_index(drop=True)
    n  = len(df)

    # Need at least 50 bars to start detecting swings and zones
    if n < 60:
        return _compute_trade_stats([], tp_R)

    trades: List[Dict] = []

    # Determine iteration stride
    stride = _SIMPLIFIED_STEP if simplified else 1

    for cursor_bar in range(50, n, stride):
        if len(trades) >= max_setups_to_replay:
            break

        # ── 1. Simulate what the scanner sees at this bar ────────────────────
        historical_df = df.iloc[: cursor_bar + 1].copy()
        historical_df = historical_df.reset_index(drop=True)

        swings    = detect_swings(historical_df, pivot=5)
        structure = classify_structure(swings, historical_df)

        if structure["state"] not in {"BOS", "UPTREND"}:
            continue

        current_price = float(df["close"].iloc[cursor_bar])

        # ── 2. Zone detection ────────────────────────────────────────────────
        zone_result = _detect_zone_setup(
            historical_df, structure, entry_zone_type, current_price
        )
        if zone_result is None:
            continue

        entry, sl_structural = zone_result

        # ── 3. SL calculation ────────────────────────────────────────────────
        if sl_method == "fixed":
            sl = entry * 0.985
        elif sl_method == "structural":
            sl = sl_structural
        elif sl_method == "structural_wider":
            atr = _compute_atr(df, cursor_bar, period=14)
            sl  = sl_structural - 1.5 * atr
        else:
            sl = entry * 0.985  # safe fallback

        # Safety: SL must be strictly below entry
        if sl >= entry:
            sl = entry * 0.985
        if sl <= 0:
            sl = entry * 0.985

        risk = entry - sl
        if risk <= 0:
            continue

        # ── 4. TP calculation ────────────────────────────────────────────────
        tp = _compute_tp(historical_df, structure, entry, sl, tp_method, tp_R)

        # Safety: TP must be strictly above entry
        if tp <= entry:
            tp = entry + tp_R * risk

        # ── 5. Walk forward to find outcome ──────────────────────────────────
        outcome_bar  = None
        r_net        = 0.0
        exit_reason  = "timeout"

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
            # Keys used by qf_shared bucket utilities
            "bar_index":  cursor_bar,
            "r_mult":     r_net,
            # Backtest-specific keys
            "setup_bar":  cursor_bar,
            "entry":      entry,
            "sl":         sl,
            "tp":         tp,
            "r_net":      r_net,
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

    # ── Try disk cache first ─────────────────────────────────────────────────
    cached = _load_baseline_cache(combo_key)
    if cached is not None:
        return cached

    # ── Fetch universe ───────────────────────────────────────────────────────
    # Minimum $10 M 24h volume to filter micro-caps
    universe = _scanner_get_universe(min_volume_usdt=10_000_000)
    symbols  = [coin["symbol"] for coin in universe[:n_sample_coins]]

    # Map app-level timeframe string to Binance interval string
    _TF_MAP = {"1D": "1d", "4H": "4h", "1H": "1h", "4h": "4h", "1h": "1h", "1d": "1d"}
    interval = _TF_MAP.get(timeframe, "4h")

    # ── Per-coin backtest (simplified walk) ──────────────────────────────────
    all_pos_r  = 0.0
    all_neg_r  = 0.0
    total_wins = 0
    total_sets = 0
    total_r    = 0.0
    max_dd     = 0.0

    for symbol in symbols:
        try:
            df_coin = _scanner_fetch_candles(symbol, interval, limit=1000)
            if df_coin is None or df_coin.empty or len(df_coin) < 60:
                continue

            result = backtest_per_coin(
                df_htf=df_coin,
                df_ltf=None,
                entry_zone_type=entry_zone_type,
                sl_method=sl_method,
                tp_method=tp_method,
                tp_R=tp_R,
                max_setups_to_replay=100,  # cap per coin for speed
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
            # Silently skip any coin that fails (network, data quality, etc.)
            continue

    # ── Aggregate ────────────────────────────────────────────────────────────
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
        # Flat fields (same shape as backtest_per_coin)
        "n_setups":         total_sets,
        "n_filled":         0,   # not tracked at aggregate level
        "n_wins":           total_wins,
        "wr":               round(wr_universe, 6),
        "mean_r":           round(mean_r_universe, 6),
        "median_r":         0.0,  # not computed at aggregate level
        "pf":               round(pf_universe, 6) if not np.isinf(pf_universe) else float("inf"),
        "max_dd_R":         round(max_dd, 6),
        "trades_raw":       [],   # not stored at universe level
        # Extra fields
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

        blended_pf     = (pf_coin^n_coin × pf_univ^prior)^(1/(n_coin+prior))
                         [geometric blend; handles inf via log-space]

    Args:
        per_coin:      Output of backtest_per_coin for a single coin.
        universe:      Output of backtest_universe_baseline.
        prior_strength: Number of "virtual" universe samples to add.

    Returns:
        {
            "wr":              float,   # blended win-rate  (0–1)
            "mean_r":          float,   # blended mean R
            "pf":              float,   # blended profit factor
            "n_effective":     int,     # n_coin + prior_strength
            "weight_per_coin": float,   # n_coin / (n_coin + prior_strength)
        }
    """
    n_coin = int(per_coin.get("n_setups", 0))
    n_total = n_coin + prior_strength

    if n_coin == 0:
        # No per-coin data → return universe metrics with n_effective = prior
        return {
            "wr":              float(universe.get("wr", 0.0)),
            "mean_r":          float(universe.get("mean_r", 0.0)),
            "pf":              float(universe.get("pf", 1.0)),
            "n_effective":     prior_strength,
            "weight_per_coin": 0.0,
        }

    # ── Linear blend for mean_r and wr ──────────────────────────────────────
    blended_mean_r = (
        n_coin * float(per_coin.get("mean_r", 0.0))
        + prior_strength * float(universe.get("mean_r", 0.0))
    ) / n_total

    blended_wr = (
        n_coin * float(per_coin.get("wr", 0.0))
        + prior_strength * float(universe.get("wr", 0.0))
    ) / n_total

    # ── Geometric blend for PF (log-space to handle inf safely) ─────────────
    pf_coin  = float(per_coin.get("pf", 1.0))
    pf_univ  = float(universe.get("pf", 1.0))

    # Clamp to avoid log(0) or log(inf); inf PF → treat as large finite value
    _PF_CLAMP_HI = 20.0
    _PF_CLAMP_LO = 0.01
    pf_coin_c = max(_PF_CLAMP_LO, min(_PF_CLAMP_HI, pf_coin))
    pf_univ_c = max(_PF_CLAMP_LO, min(_PF_CLAMP_HI, pf_univ))

    log_pf = (n_coin * np.log(pf_coin_c) + prior_strength * np.log(pf_univ_c)) / n_total
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
        trades_raw: List of trade dicts from backtest_per_coin["trades_raw"].
                    Each must have keys 'bar_index' and 'r_mult'.
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

    buckets     = _compute_decay_buckets(n_df)
    bucket_rows, _, _ = _bucket_stats_for_trades(trades_raw, n_df, buckets)

    # Need at least 2 buckets to compare earlier vs recent
    if len(bucket_rows) < 2:
        return _empty

    # bucket_rows is ordered oldest→newest (aligned with buckets["edges"])
    earlier_row = bucket_rows[0]   # oldest bucket
    recent_row  = bucket_rows[-1]  # newest bucket

    earlier_label = earlier_row["label"]
    recent_label  = recent_row["label"]

    # ── Re-bucket trades to compute PF (not returned by _bucket_stats_for_trades) ──
    denom = float(n_df - 1)

    def _get_bucket_trades(edge_idx: int) -> List[Dict]:
        lo, hi  = buckets["edges"][edge_idx]
        is_last_old = (edge_idx == 0)  # oldest bucket includes upper bound
        result  = []
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

    earlier_n  = len(earlier_trades)
    recent_n   = len(recent_trades)

    # _bucket_stats_for_trades returns wr as 0-100; convert to 0-1
    earlier_wr_pct = earlier_row["wr"]   # already 0-100
    recent_wr_pct  = recent_row["wr"]

    earlier_stats = {
        "mean_r": round(_mean_r(earlier_trades), 4),
        "pf":     round(_pf(earlier_trades), 4)
                  if not np.isinf(_pf(earlier_trades)) else float("inf"),
        "n":      earlier_n,
        "wr":     round(earlier_wr_pct / 100.0, 4),  # convert to 0-1
    }
    recent_stats = {
        "mean_r": round(_mean_r(recent_trades), 4),
        "pf":     round(_pf(recent_trades), 4)
                  if not np.isinf(_pf(recent_trades)) else float("inf"),
        "n":      recent_n,
        "wr":     round(recent_wr_pct / 100.0, 4),
    }

    # ── Verdict ──────────────────────────────────────────────────────────────
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

    # Guard against inf in ratio: if earlier_pf is inf and recent_pf is also inf,
    # treat as STABLE; if only recent is inf, that's clearly STRONGER.
    if np.isinf(earlier_pf) and np.isinf(recent_pf):
        pf_ratio = 1.0
    elif np.isinf(recent_pf):
        pf_ratio = 2.0   # classify as STRONGER
    elif np.isinf(earlier_pf):
        pf_ratio = 0.0   # classify as WEAKER (earlier was "perfect")
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
            → tp_spec maps to (tp_method, tp_R):
              "fixed_2R"   → ("fixed_R", 2.0)
              "fixed_2.5R" → ("fixed_R", 2.5)
              "fixed_3R"   → ("fixed_R", 3.0)
              "sr"         → ("sr",      2.0)

    Returns:
        DataFrame sorted by pf descending with columns:
        entry_zone_type, sl_method, tp_method, tp_R,
        n_setups, n_filled, n_wins, wr, mean_r, pf, max_dd_R, recent_verdict.
    """
    entries = ["smart_ob", "fvg", "fibo_786", "sr"]
    sls     = ["fixed", "structural", "structural_wider"]
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

                    # Recent verdict
                    recent_verdict = "INSUFFICIENT_DATA"
                    if result["n_setups"] >= 5:
                        check = compute_recent_check(result["trades_raw"], n_df)
                        recent_verdict = check.get("verdict", "INSUFFICIENT_DATA")

                    rows.append({
                        "entry_zone_type": entry_zone_type,
                        "sl_method":       sl_method,
                        "tp_method":       tp_label,    # human-readable label
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
                    # Don't let one bad combo abort the entire grid
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

    # Sort by pf descending; treat inf as the largest finite value seen + 1
    def _sort_pf(v):
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

    # Convert any numpy / inf values to JSON-safe types
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

    # Freshness check via file mtime
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
