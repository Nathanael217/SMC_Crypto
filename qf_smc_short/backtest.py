"""
qf_smc_short/backtest.py — Backtest engine for SMC SHORT setups
================================================================
Mirror of qf_smc/backtest.py for SHORT direction.

P&L math flipped vs LONG:
    LONG: profit = exit - entry   (entry low, exit higher)
    SHORT: profit = entry - exit  (entry high, exit lower)

    LONG SL hits when:  bar_low  <= sl  (price dropped to stop below entry)
    SHORT SL hits when: bar_high >= sl  (price rallied to stop above entry)

    LONG TP hits when:  bar_high >= tp  (price rallied to target above entry)
    SHORT TP hits when: bar_low  <= tp  (price dropped to target below entry)

Zone geometry flipped:
    LONG: entry at zone bottom, SL below zone floor (ob_low, fvg.bottom, fib_886)
    SHORT: entry at zone top, SL above zone ceiling (ob_high, fvg.top, anchor area)

Public API:
  - run_variant_grid_short(df_htf, df_ltf, structure, fibo_zone,
                           smart_obs, fvgs, sr_levels)
  - deep_dive_backtest_short(symbol, df_htf, df_ltf, structure, fibo_zone, ...)

Reuses from qf_smc.backtest (direction-neutral helpers):
  - compute_recent_check  (bucketing logic, only uses r_mult sign)
  - _compute_max_dd       (drawdown math)
  - _aggregate_trades     (stats aggregation)
  - _compute_atr          (volatility math)

NOTE: We re-define those helpers locally rather than importing them, to keep
qf_smc_short self-contained and prevent cross-package coupling for r_net
which has flipped semantics.
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
from qf_smc.structure import detect_swings
from qf_smc_short.structure import classify_structure_short
from qf_smc_short.zones import (
    detect_smart_obs_short,
    detect_fvgs_short,
    detect_fibo_levels_short,
    detect_sr_levels_short,
    classify_current_price_in_zones_short,
)


# ============================================================================
# Constants
# ============================================================================

BASELINE_CACHE_DIR_SHORT = Path("data/smc_short_baselines")
BASELINE_TTL_DAYS        = 7

# Entry zone types accepted
_VALID_ENTRY_TYPES = {"smart_ob", "fvg", "fibo_786", "sr"}

# Structural SL: buffer for SR / Fibo (ABOVE the level for SHORT)
_SR_SL_BUFFER_PCT   = 0.010   # 1.0% above SR resistance price
_FIBO_SL_BUFFER_PCT = 0.005   # 0.5% extra above the leg-start anchor

# Forward-look windows
_MAX_FWD_BARS  = 50
_MAX_FILL_BARS = 30


# ============================================================================
# Internal helpers (direction-neutral, kept local)
# ============================================================================

def _compute_atr(df: pd.DataFrame, bar_idx: int, period: int = 14) -> float:
    """ATR estimate over the last `period` bars ending at bar_idx."""
    if bar_idx < period:
        return float(df["high"].iloc[: bar_idx + 1].mean()
                     - df["low"].iloc[: bar_idx + 1].mean()) or 0.001
    sub = df.iloc[bar_idx - period + 1: bar_idx + 1]
    return float((sub["high"] - sub["low"]).mean()) or 0.001


def _compute_max_dd(r_nets: List[float]) -> float:
    """Max drawdown in R-units from cumulative equity curve."""
    cum    = 0.0
    peak   = 0.0
    max_dd = 0.0
    for r in r_nets:
        cum    += r
        peak    = max(peak, cum)
        max_dd  = max(max_dd, peak - cum)
    return max_dd


def _aggregate_trades(trades: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics from a list of SHORT trade dicts."""
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

    # ── FLIPPED for SHORT: SL pct = (sl - entry) / entry × 100 ──────────────
    # (SL is ABOVE entry for shorts, so this is a positive percentage)
    avg_sl_pct = 0.0
    valid_pairs = [(e, s) for e, s in zip(entries, sls) if e > 0 and s > 0]
    if valid_pairs:
        avg_sl_pct = float(np.mean([(s - e) / e * 100.0 for e, s in valid_pairs]))

    avg_rr_to_tp = float(np.mean(tp_multis)) if tp_multis else 2.0

    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")

    return {
        "n_setups":        n,
        "n_filled":        n,
        "n_wins":          len(wins),
        "wr":              round(len(wins) / n, 6),
        "mean_r":          round(float(np.mean(r_nets)), 6),
        "median_r":        round(float(np.median(r_nets)), 6),
        "pf":              round(pf, 6) if not np.isinf(pf) else float("inf"),
        "max_dd_R":        round(_compute_max_dd(r_nets), 6),
        "avg_entry_price": round(avg_entry, 10),
        "avg_sl_pct":      round(avg_sl_pct, 4),
        "avg_rr_to_tp":    round(avg_rr_to_tp, 4),
    }


def _compute_recent_check_short(trades_raw: List[Dict], n_df: int) -> Dict[str, Any]:
    """
    Earlier-vs-recent verdict on SHORT trades. Direction-neutral: the math
    just looks at r_mult sign and timing. Re-implementation of the LONG version
    (kept local to avoid a circular-ish reliance on qf_smc.backtest).
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

    buckets           = _compute_decay_buckets(n_df)
    bucket_rows, _, _ = _bucket_stats_for_trades(trades_raw, n_df, buckets)

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

    def _pf(ts):
        pos = sum(t["r_mult"] for t in ts if t["r_mult"] > 0)
        neg = abs(sum(t["r_mult"] for t in ts if t["r_mult"] < 0))
        return (pos / neg) if neg > 0 else float("inf")

    def _mean_r(ts):
        if not ts: return 0.0
        return float(np.mean([t["r_mult"] for t in ts]))

    earlier_n = len(earlier_trades)
    recent_n  = len(recent_trades)

    earlier_stats = {
        "mean_r": round(_mean_r(earlier_trades), 4),
        "pf":     round(_pf(earlier_trades), 4) if not np.isinf(_pf(earlier_trades)) else float("inf"),
        "n":      earlier_n,
        "wr":     round(earlier_row["wr"] / 100.0, 4),
    }
    recent_stats = {
        "mean_r": round(_mean_r(recent_trades), 4),
        "pf":     round(_pf(recent_trades), 4) if not np.isinf(_pf(recent_trades)) else float("inf"),
        "n":      recent_n,
        "wr":     round(recent_row["wr"] / 100.0, 4),
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


# ============================================================================
# Single-variant SHORT backtest replay
# ============================================================================

def _backtest_single_variant_short(
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
    Single-variant historical replay for SHORT.

    Walks df_htf bar-by-bar from index 50 to end:
      1. Recompute structure at this cursor (must be confirmed DOWNTREND)
      2. Detect zones (no fibo filter — replay uses raw)
      3. Find a matching zone for entry_type
      4. Determine entry_price = limit at zone top (for shorts)
      5. Walk forward up to _MAX_FILL_BARS to find FILL (price rallies up
         into the limit)
      6. If filled: compute SL per ltf_mode (ABOVE entry)
      7. Walk forward to _MAX_FWD_BARS — outcome:
         - SL hit:  bar_high >= sl  → r_net = -1.0
         - TP hit:  bar_low  <= tp  → r_net = tp_R
         - timeout: r_net = (entry - final_close) / risk

    ltf_mode SL (SHORT):
      A (limit, wider SL):  sl = zone_ceiling * 1.005
      B (LTF confirmed):    sl = HTF df.iloc[fill_bar:fill_bar+5].high.max() * 1.003
                            (HTF proxy for LTF; tighter)
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

        # ── Structure at this point (must be DOWNTREND for short) ────────────
        swings_h = detect_swings(historical_df, pivot=5)
        struct_h = classify_structure_short(swings_h, historical_df)

        if struct_h["state"] not in {"BOS", "DOWNTREND"}:
            continue

        current_price = float(df["close"].iloc[cursor_bar])

        # ── Zone detection (no fibo filter during historical replay) ─────────
        obs_h  = detect_smart_obs_short(historical_df, struct_h)
        fvgs_h = detect_fvgs_short(historical_df, struct_h)
        fibo_h = detect_fibo_levels_short(historical_df, struct_h)
        srs_h  = detect_sr_levels_short(historical_df)

        in_zones_h = classify_current_price_in_zones_short(
            current_price, obs_h, fvgs_h, fibo_h, srs_h
        )

        # ── Determine entry_price + zone_ceiling per entry_type (SHORT) ──────
        entry_price:   Optional[float] = None
        zone_ceiling:  Optional[float] = None

        if entry_type == "smart_ob":
            candidates = in_zones_h.get("in_smart_ob", [])
            if not candidates:
                continue
            zone_d = max(candidates, key=lambda z: z["ob_bar"])
            # FLIPPED: short entry at OB low (price rallies UP to fill the limit)
            entry_price  = float(zone_d["ob_low"])
            zone_ceiling = float(zone_d["ob_high"])

        elif entry_type == "fvg":
            candidates = in_zones_h.get("in_fvg", [])
            if not candidates:
                continue
            zone_d = max(candidates, key=lambda z: z["created_at_bar"])
            # SHORT FVG: entry at mid, zone_ceiling = top
            entry_price  = float(zone_d["mid"])
            zone_ceiling = float(zone_d["top"])

        elif entry_type == "fibo_786":
            if not fibo_h:
                continue
            fib786 = fibo_h.get("fib_786")
            if fib786 is None:
                continue
            entry_price  = float(fib786)
            # SHORT Fibo ceiling: top of zone (= fib_786 + tolerance)
            zone_ceiling = fibo_h.get("fib_786_zone_top", entry_price * 1.005)
            if zone_ceiling is None:
                zone_ceiling = entry_price * 1.005
            zone_ceiling = float(zone_ceiling)

        elif entry_type == "sr":
            candidates = in_zones_h.get("at_sr_support", [])  # is resistance in SHORT
            if not candidates:
                continue
            zone_d = min(candidates, key=lambda z: abs(z["price"] - current_price))
            entry_price  = float(zone_d["price"])
            zone_ceiling = entry_price * 1.015

        else:
            continue

        if entry_price is None or zone_ceiling is None:
            continue

        # ── Walk forward to find FILL (price covers limit entry_price) ───────
        # Limit fills when intra-bar range covers entry_price (same as LONG —
        # a limit order fills when price hits it, regardless of direction)
        fill_bar: Optional[int] = None
        fill_end = min(cursor_bar + _MAX_FILL_BARS + 1, n)
        for fwd in range(cursor_bar + 1, fill_end):
            bar_low  = float(fwd_lows[fwd])
            bar_high = float(fwd_highs[fwd])
            if bar_low <= entry_price <= bar_high:
                fill_bar = fwd
                break

        if fill_bar is None:
            continue  # limit never filled

        # ── SL per ltf_mode (ABOVE entry for shorts) ─────────────────────────
        if ltf_mode == "A":
            sl = zone_ceiling * 1.005   # wider buffer above zone
        else:
            # ltf_mode B: tighter SL using post-fill HTF swing HIGH proxy
            proxy_end     = min(fill_bar + 5, n)
            ltf_proxy_high = float(df["high"].iloc[fill_bar:proxy_end].max())
            sl = ltf_proxy_high * 1.003

        # Safety: SL must be strictly ABOVE entry for SHORT
        if sl <= entry_price:
            sl = entry_price * 1.015
        if entry_price <= 0:
            continue

        risk = sl - entry_price
        if risk <= 0:
            continue

        # ── FLIPPED: TP is BELOW entry ───────────────────────────────────────
        tp = entry_price - tp_R * risk
        if tp <= 0:
            continue  # unrealistic — skip

        # ── Walk forward from fill_bar to find outcome ─────────────────────
        outcome_bar = None
        r_net       = 0.0
        exit_reason = "timeout"

        outcome_end = min(fill_bar + _MAX_FWD_BARS + 1, n)
        for fwd in range(fill_bar + 1, outcome_end):
            bar_low  = float(fwd_lows[fwd])
            bar_high = float(fwd_highs[fwd])

            # FLIPPED: SL hits when price rallies UP to sl
            if bar_high >= sl:
                outcome_bar = fwd
                r_net       = -1.0
                exit_reason = "sl"
                break
            # FLIPPED: TP hits when price drops DOWN to tp
            if bar_low <= tp:
                outcome_bar = fwd
                r_net       = float(tp_R)
                exit_reason = "tp"
                break

        if outcome_bar is None:
            final_bar   = min(fill_bar + _MAX_FWD_BARS, n - 1)
            final_price = float(df["close"].iloc[final_bar])
            # FLIPPED R-multiple: profit = entry - exit for shorts
            r_net       = (entry_price - final_price) / risk
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

    # ── Aggregate ───────────────────────────────────────────────────────────
    result = _aggregate_trades(trades)
    result["trades_raw"]   = trades
    result["recent_check"] = _compute_recent_check_short(trades, n)
    return result


# ============================================================================
# Public API: 24-variant grid for SHORT
# ============================================================================

def run_variant_grid_short(
    df_htf: pd.DataFrame,
    df_ltf: Optional[pd.DataFrame],
    structure: Dict,
    fibo_zone: Dict,
    smart_obs: List,
    fvgs: List,
    sr_levels: List,
) -> pd.DataFrame:
    """
    Run all 24 entry variants for this coin in SHORT direction.

    Variants = 4 entry_types × 2 ltf_modes × 3 tp_R values
    Returns DataFrame sorted by PF descending.
    """
    ENTRY_TYPES = ["smart_ob", "fvg", "fibo_786", "sr"]
    LTF_MODES   = ["A", "B"]
    TP_RS       = [2.0, 2.5, 3.0]

    rows = []

    for entry_type in ENTRY_TYPES:
        for ltf_mode in LTF_MODES:
            for tp_R in TP_RS:
                try:
                    result = _backtest_single_variant_short(
                        df_htf, df_ltf, structure, fibo_zone,
                        smart_obs, fvgs, sr_levels,
                        entry_type, ltf_mode, tp_R,
                    )
                except Exception as exc:
                    print(f"[backtest_short] variant error {entry_type}/{ltf_mode}/{tp_R}: {exc}")
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

    def _pf_sort_key(v):
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
# Management mode simulation (mirror of LONG — direction-neutral math on r_net)
# ============================================================================

def _simulate_with_management_short(
    trades_raw: List[Dict],
    management_mode: str,
    df_htf: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Re-simulate trades with a different management mode.

    The math is direction-neutral (operates on r_net signs), but kept local
    to qf_smc_short for clarity & isolation. Same heuristic as LONG version.
    """
    if not trades_raw:
        return _aggregate_trades([])

    new_trades: List[Dict] = []

    for t in trades_raw:
        r = t["r_net"]

        if management_mode == "BE_at_1R":
            if r >= 0:
                new_r = r
            elif abs(r) < 0.5:
                new_r = 0.0           # BE stop approximation
            else:
                new_r = r             # full SL before 1R

        elif management_mode == "Trailing":
            if r > 0:
                new_r = r * 1.10      # ~10% uplift for trailing capture
            elif abs(r) < 0.5:
                new_r = 0.0
            else:
                new_r = r

        else:  # Fixed / M1
            new_r = r

        new_trades.append({**t, "r_net": new_r, "r_mult": new_r})

    return _aggregate_trades(new_trades)


# ============================================================================
# Trade plan builder from best variant (SHORT)
# ============================================================================

def _build_trade_plan_from_variant_short(
    variant_row: Dict,
    fibo_zone: Dict,
    smart_obs: List,
    fvgs: List,
    sr_levels: List,
) -> Dict:
    """
    Build a concrete SHORT trade plan from the BEST variant row.
    Inverted geometry: entry at zone top/ceiling, SL above, TP below.
    """
    entry_type = variant_row.get("entry_type", "fibo_786")
    ltf_mode   = variant_row.get("ltf_mode", "A")
    tp_R       = float(variant_row.get("tp_R", 2.0))
    management = variant_row.get("management", "Fixed")

    entry_price:  Optional[float] = None
    zone_ceiling: Optional[float] = None

    # ── Lookup current zone (SHORT geometry) ─────────────────────────────────
    if entry_type == "smart_ob":
        zone = smart_obs[0] if smart_obs else None
        if zone:
            entry_price  = float(zone["ob_low"])
            zone_ceiling = float(zone["ob_high"])

    elif entry_type == "fvg":
        zone = fvgs[0] if fvgs else None
        if zone:
            entry_price  = float(zone["mid"])
            zone_ceiling = float(zone["top"])

    elif entry_type == "fibo_786":
        fib786 = fibo_zone.get("fib_786")
        if fib786 is not None:
            entry_price  = float(fib786)
            zone_ceiling = float(
                fibo_zone.get("fib_786_zone_top", entry_price * 1.005) or
                entry_price * 1.005
            )

    else:  # sr (resistance)
        resistance = next((s for s in sr_levels if s.get("kind") == "resistance"), None)
        if resistance:
            entry_price  = float(resistance["price"])
            zone_ceiling = entry_price * 1.015

    # Fallback if no zone found
    if entry_price is None or zone_ceiling is None:
        return {
            "entry_type":  entry_type,
            "ltf_mode":    ltf_mode,
            "management":  management,
            "tp_R":        tp_R,
            "entry_price": None,
            "sl":          None,
            "error":       "No current zone found for entry_type",
        }

    # ── SL per ltf_mode (ABOVE entry for shorts) ─────────────────────────────
    if ltf_mode == "A":
        sl = zone_ceiling * 1.005   # wider
    else:
        sl = zone_ceiling * 1.002   # tighter (LTF confirmed)

    # Safety
    if sl <= entry_price:
        sl = entry_price * 1.015

    risk = sl - entry_price
    if risk <= 0:
        risk = entry_price * 0.015
        sl   = entry_price + risk

    # TPs BELOW entry
    tp1 = entry_price - 2.0 * risk
    tp2 = entry_price - 2.5 * risk
    tp3 = entry_price - 3.0 * risk

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
        "direction":        "short",
    }


# ============================================================================
# Public API: deep_dive_backtest_short
# ============================================================================

def deep_dive_backtest_short(
    symbol: str,
    df_htf: pd.DataFrame,
    df_ltf: Optional[pd.DataFrame],
    structure: Dict,
    fibo_zone: Dict,
    smart_obs: List,
    fvgs: List,
    sr_levels: List,
    ai_provider: Optional[str] = None,
    ai_api_key: Optional[str]  = None,
) -> Dict:
    """
    User-triggered detailed SHORT analysis for ONE coin.

    Steps:
        1. Run run_variant_grid_short → 24 variants (M1 Fixed)
        2. Expand with M2 (BE at 1R) and M3 (Trailing) → 72 variants
        3. Best per (entry_type, ltf_mode)
        4. Best overall
        5. AI verdict (if key provided)
        6. Recommended trade plan
    """
    # ── Step 1: 24-variant grid ──────────────────────────────────────────────
    variant_24 = run_variant_grid_short(
        df_htf, df_ltf, structure, fibo_zone, smart_obs, fvgs, sr_levels
    )

    # ── Step 2: Expand with M2 and M3 ────────────────────────────────────────
    all_72: List[Dict] = []
    for _, row in variant_24.iterrows():
        base       = row.to_dict()
        trades_raw = base.get("trades_raw", [])

        # M1 Fixed
        all_72.append({**base, "management": "Fixed"})

        # M2 BE_at_1R
        m2_metrics = _simulate_with_management_short(trades_raw, "BE_at_1R")
        all_72.append({**base, **m2_metrics, "management": "BE_at_1R"})

        # M3 Trailing
        m3_metrics = _simulate_with_management_short(trades_raw, "Trailing", df_htf=df_htf)
        all_72.append({**base, **m3_metrics, "management": "Trailing"})

    df_72 = pd.DataFrame(all_72)

    def _pf_sort_key(v):
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

    # ── Step 3: Best per entry_type ──────────────────────────────────────────
    best_per_entry = []
    for et in ["smart_ob", "fvg", "fibo_786", "sr"]:
        sub = df_72[df_72["entry_type"] == et] if not df_72.empty else pd.DataFrame()
        if not sub.empty:
            best_per_entry.append({
                "entry_type": et,
                "best_row":   sub.iloc[0].to_dict(),
            })

    # ── Step 4: Best overall ─────────────────────────────────────────────────
    best_overall: Optional[Dict] = df_72.iloc[0].to_dict() if not df_72.empty else None

    # ── Step 5: Recommended trade plan ───────────────────────────────────────
    recommended_trade_plan: Optional[Dict] = None
    if not df_72.empty:
        # Try best_overall first, then fall back through PF-sorted rows
        _candidate_rows: List[Dict] = []
        if best_overall:
            _candidate_rows.append(best_overall)
        for _, _row in df_72.iterrows():
            _rd = _row.to_dict()
            if best_overall and (
                _rd.get("entry_type") == best_overall.get("entry_type")
                and _rd.get("ltf_mode") == best_overall.get("ltf_mode")
                and _rd.get("tp_R") == best_overall.get("tp_R")
                and _rd.get("management") == best_overall.get("management")
            ):
                continue
            _candidate_rows.append(_rd)

        for _cand in _candidate_rows:
            try:
                _plan = _build_trade_plan_from_variant_short(
                    _cand, fibo_zone, smart_obs, fvgs, sr_levels
                )
            except Exception as exc:
                print(f"[backtest_short] trade plan error for {symbol}: {exc}")
                continue
            if _plan and _plan.get("entry_price") is not None:
                recommended_trade_plan = _plan
                break

        if recommended_trade_plan is None and best_overall:
            try:
                recommended_trade_plan = _build_trade_plan_from_variant_short(
                    best_overall, fibo_zone, smart_obs, fvgs, sr_levels
                )
            except Exception as exc:
                print(f"[backtest_short] trade plan fallback error for {symbol}: {exc}")
                recommended_trade_plan = None

    # ── Step 6: AI verdict (optional) ────────────────────────────────────────
    ai_verdict: Optional[Dict] = None
    if ai_provider and ai_api_key:
        try:
            from qf_smc_short.ai_verdict import build_smc_short_prompt_v2
            from qf_smc.ai_verdict import get_verdict  # reuse — LLM calling is direction-neutral

            _htf_label = str(structure.get("htf_tf", structure.get("htf_label", "?")))
            _ltf_label = str(structure.get("ltf_tf", structure.get("ltf_label", "?")))

            try:
                _current_price = float(df_htf["close"].iloc[-1])
            except Exception:
                _current_price = 0.0

            _wick_adj = structure.get("wick_adjustments", []) or []

            prompt = build_smc_short_prompt_v2(
                symbol=symbol,
                mode=str(structure.get("mode", "?")),
                htf_label=_htf_label,
                ltf_label=_ltf_label,
                btc_regime=str(structure.get("btc_regime", "UNKNOWN")),
                fights_macro=bool(structure.get("fights_macro", False)),
                structure=structure,
                fibo_zone=fibo_zone,
                smart_obs=smart_obs,
                fvgs=fvgs,
                sr_levels=sr_levels,
                wick_adjustments=_wick_adj,
                ltf_confirmation=str(structure.get("ltf_confirmation", "NONE")),
                variant_grid=df_72,
                best_overall=best_overall or {},
                current_price=_current_price,
                scan_timestamp=datetime.utcnow().isoformat(),
            )
            ai_verdict = get_verdict(prompt, provider=ai_provider, api_key=ai_api_key)
        except Exception as e:
            import traceback as _tb
            print(f"[backtest_short] AI verdict error for {symbol}: {e}")
            print(_tb.format_exc())
            ai_verdict = {"error": f"{type(e).__name__}: {e}"}

    return {
        "symbol":                  symbol,
        "direction":               "short",
        "structure":               structure,
        "fibo_zone":               fibo_zone,
        "all_72_variants":         df_72,
        "best_per_entry":          best_per_entry,
        "best_overall":            best_overall,
        "ai_verdict":              ai_verdict,
        "recommended_trade_plan":  recommended_trade_plan,
        "computed_at_utc":         datetime.utcnow().isoformat(),
    }
