"""
qf_smc_short/zones.py — Zone detection for SHORT (bearish) setups
==================================================================
Mirror of qf_smc/zones.py for SHORT direction.

Flipped logic:
    LONG version                       SHORT version
    ───────────────                    ───────────────
    Bullish OB (down-candle then       Bearish OB (up-candle then
      bullish impulse breaks                bearish impulse breaks
      above OB high)                        below OB low)
    Bullish FVG: low[n] > high[n-2]    Bearish FVG: high[n] < low[n-2]
    Fibo of UP-leg: anchor_low →       Fibo of DOWN-leg: anchor_high →
      anchor_high → retrace DOWN          anchor_low → retrace UP
    S/R support (price below current)  S/R resistance (price above current)
    LIQUIDITY_SWEEP = lower_wick       LIQUIDITY_SWEEP = upper_wick
      / body > 0.5                        / body > 0.5

Reuses direction-neutral helpers from qf_smc.zones (_compute_atr,
_detect_swings_minimal, _check_fibo_zone_overlap, _check_price_in_fibo_zone).

Public API:
  - detect_smart_obs_short(df, structure, vol_avg_period=20,
                           fibo_786_zone=None, require_in_zone=True)
  - detect_fvgs_short(df, structure, fibo_786_zone=None, require_in_zone=True)
  - detect_fibo_levels_short(df, structure, atr_multiplier=0.5, atr_period=14)
  - detect_sr_levels_short(df, lookback=50, cluster_pct=0.5,
                           fibo_786_zone=None, require_in_zone=True)
  - classify_current_price_in_zones_short(current_price, smart_obs,
                                          fvgs, fibo, srs)
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Reuse direction-neutral helpers from LONG package
from qf_smc.zones import (
    _compute_atr,
    _detect_swings_minimal,
    _check_fibo_zone_overlap,
    _check_price_in_fibo_zone,
)


# ============================================================================
# Constants (mirror of qf_smc.zones)
# ============================================================================

_OB_IMPULSE_WINDOW   = 5
_MIN_VOL_MULT        = 1.2
_STRONG_VOL_MULT     = 2.0
_MIN_BODY_RATIO      = 0.30
_FRESHNESS_DECAY     = 0.10
_PIVOT_N             = 5
_SR_MIN_CLUSTER_SIZE = 2
_SR_MAX_STRENGTH     = 5.0
_LIQUIDITY_SWEEP_WICK_RATIO = 0.5   # upper_wick / body > this → LIQUIDITY_SWEEP (short version)


# ============================================================================
# Private helpers (mirror)
# ============================================================================

def _nearest_swing_low_before(
    bar_idx: int,
    swing_lows: List[Tuple[int, float]],
) -> Optional[float]:
    """Mirror of _nearest_swing_high_before — for bearish BOS detection."""
    candidates = [(b, p) for b, p in swing_lows if b < bar_idx]
    if not candidates:
        return None
    return candidates[-1][1]   # most recent


def _is_valid_swing_high(
    b_idx: int,
    swing_highs: List[Tuple[int, float]],
    b_high: float,
    tolerance_pct: float = 0.001,
) -> bool:
    """Mirror of _is_valid_swing_low — checks if b_idx aligns with a swing high."""
    for bar, price in swing_highs:
        if bar == b_idx:
            return True
        if abs(price - b_high) / max(b_high, 1e-12) <= tolerance_pct:
            return True
    return False


def _count_swing_lows_broken(
    ob_bar: int,
    bos_bar: int,
    bos_close_price: float,
    swing_lows: List[Tuple[int, float]],
) -> int:
    """
    Mirror of _count_swing_highs_broken — counts swing LOWS broken by the
    bearish impulse close (bos_close_price falls below them).
    """
    count = 0
    for bar, price in swing_lows:
        if bar < ob_bar or bar > bos_bar:
            continue
        if bos_close_price < price:
            count += 1
    return count


# ============================================================================
# Public API
# ============================================================================

def detect_smart_obs_short(
    df: pd.DataFrame,
    structure: dict,
    vol_avg_period: int = 20,
    fibo_786_zone: Optional[Dict] = None,
    require_in_zone: bool = True,
) -> List[Dict[str, Any]]:
    """
    Detect Bearish Smart Order Blocks (mirror of detect_smart_obs).

    A bearish OB is a BULLISH candle (close > open) immediately followed within
    _OB_IMPULSE_WINDOW bars by a bearish impulse that:
      1. Closes BELOW the OB's low
      2. Sweeps the OB's high (mitigation) before/at the impulse bar
      3. Confirms BOS by closing below the nearest swing low

    Tier classification (first match wins):
        1. LIQUIDITY_SWEEP  — upper_wick / body > 0.5 (long upper wick = trapped longs)
        2. STRONG           — vol_mult >= 2.0 AND at-swing-high AND multi-bar BOS↓
        3. REGULAR          — vol_mult >= 1.2

    Returns:
        List of OB dicts sorted by ob_bar ascending. Each dict has the same
        SHAPE as the LONG version (keys ob_high, ob_low, etc.) so render.py
        can be reused — but semantically these are bearish OBs (entry SHORT,
        SL ABOVE ob_high, TP BELOW).
    """
    results: List[Dict[str, Any]] = []

    current_leg = structure.get("current_leg")
    if current_leg is None:
        return results

    leg_start_bar:  int = current_leg["leg_start_bar"]
    bos_bar_struct: Optional[int] = structure.get("bos_bar")
    # For SHORT, the leg goes from leg_start (HIGH) down to leg_low_bar
    leg_low_bar: int = current_leg.get("leg_low_bar", len(df) - 1)
    scan_end_bar: int = bos_bar_struct if bos_bar_struct is not None else leg_low_bar
    scan_end_bar = min(scan_end_bar, len(df) - 1)

    n       = len(df)
    last    = n - 1
    opens   = df["open"].to_numpy()
    highs   = df["high"].to_numpy()
    lows    = df["low"].to_numpy()
    closes  = df["close"].to_numpy()
    volumes = df["volume"].to_numpy()

    swing_highs, swing_lows = _detect_swings_minimal(df)
    current_price = float(closes[last])

    apply_zone_filter = (fibo_786_zone is not None) and require_in_zone

    for b_idx in range(leg_start_bar, scan_end_bar + 1):
        if b_idx < vol_avg_period:
            continue

        # ── FLIPPED: Bullish candle filter (OB is the up-candle before drop) ─
        if closes[b_idx] <= opens[b_idx]:
            continue

        b_high  = float(highs[b_idx])
        b_low   = float(lows[b_idx])
        b_open  = float(opens[b_idx])
        b_close = float(closes[b_idx])
        b_vol   = float(volumes[b_idx])

        # ── Body filter (unchanged) ─────────────────────────────────────────
        body  = abs(b_close - b_open)
        rng   = b_high - b_low
        if rng <= 0 or body / rng < _MIN_BODY_RATIO:
            continue

        # ── Volume filter (unchanged) ───────────────────────────────────────
        vol_window = volumes[b_idx - vol_avg_period : b_idx]
        if len(vol_window) == 0:
            continue
        vol_avg = float(np.mean(vol_window))
        if vol_avg <= 0:
            continue
        vol_mult = b_vol / vol_avg
        if vol_mult < _MIN_VOL_MULT:
            continue

        # ── FLIPPED: Bearish impulse — next bar closes BELOW OB low ─────────
        impulse_bar: Optional[int] = None
        window_end = min(b_idx + _OB_IMPULSE_WINDOW, n - 1)
        for k in range(b_idx + 1, window_end + 1):
            if closes[k] < b_low:
                impulse_bar = k
                break
        if impulse_bar is None:
            continue

        # ── FLIPPED: Mitigation — sweep of OB.high before/at impulse ────────
        swept_at_bar: Optional[int] = None
        for k in range(b_idx + 1, impulse_bar + 1):
            if highs[k] >= b_high:
                swept_at_bar = k
                break
        if swept_at_bar is None:
            continue

        # ── FLIPPED: BOS↓ — impulse closes below nearest swing LOW ──────────
        nearest_sl_price = _nearest_swing_low_before(impulse_bar, swing_lows)
        if nearest_sl_price is not None:
            if closes[impulse_bar] >= nearest_sl_price:
                continue
        bos_at_bar = impulse_bar

        # ── FLIPPED tier classification ──────────────────────────────────────
        lower_wick = min(b_open, b_close) - b_low
        upper_wick = b_high - max(b_open, b_close)
        wick_lower_ratio = lower_wick / body if body > 0 else 0.0
        wick_upper_ratio = upper_wick / body if body > 0 else 0.0

        impulse_close_price = float(closes[bos_at_bar])
        is_at_swing_high = _is_valid_swing_high(b_idx, swing_highs, b_high)
        broken_sl_count = _count_swing_lows_broken(
            b_idx, bos_at_bar, impulse_close_price, swing_lows
        )
        multi_bar_bos = broken_sl_count >= 2

        # LIQUIDITY_SWEEP for shorts = long UPPER wick (trapped longs at the top)
        if wick_upper_ratio > _LIQUIDITY_SWEEP_WICK_RATIO:
            tier = "LIQUIDITY_SWEEP"
        elif vol_mult >= _STRONG_VOL_MULT and is_at_swing_high and multi_bar_bos:
            tier = "STRONG"
        else:
            tier = "REGULAR"

        # ── Status (mirror) ──────────────────────────────────────────────────
        # For SHORT OB: price BELOW ob_low = FRESH (haven't retraced up yet)
        if current_price < b_low:
            status = "FRESH"
        elif b_low <= current_price <= b_high:
            status = "TESTING"
        else:  # current_price > b_high → moved past the zone
            status = "MITIGATED"

        # ── Freshness (mirror — count touches into OB range) ────────────────
        touch_count = 0
        for k in range(bos_at_bar, last + 1):
            if lows[k] <= b_high and highs[k] >= b_low:
                touch_count += 1
        freshness_score = max(0.0, 1.0 - touch_count * _FRESHNESS_DECAY)

        # ── Fibo 0.786 zone check (unchanged — overlap test) ─────────────────
        in_zone = _check_fibo_zone_overlap(b_high, b_low, fibo_786_zone)
        if apply_zone_filter and not in_zone:
            continue

        results.append({
            "type":                "smart_ob",
            "direction":           "bearish",          # NEW vs LONG
            "tier":                tier,
            "ob_bar":              b_idx,
            "ob_high":             b_high,
            "ob_low":              b_low,
            "ob_open":             b_open,
            "ob_close":            b_close,
            "volume_mult":         round(vol_mult, 4),
            "swept_at_bar":        swept_at_bar,
            "bos_at_bar":          bos_at_bar,
            "status":              status,
            "freshness_score":     round(freshness_score, 4),
            "wick_lower_ratio":    round(wick_lower_ratio, 6),
            "wick_upper_ratio":    round(wick_upper_ratio, 6),
            "is_in_fibo_786_zone": in_zone,
        })

    results.sort(key=lambda x: x["ob_bar"])
    return results


# ----------------------------------------------------------------------------

def detect_fvgs_short(
    df: pd.DataFrame,
    structure: dict,
    fibo_786_zone: Optional[Dict] = None,
    require_in_zone: bool = True,
) -> List[Dict[str, Any]]:
    """
    Detect Bearish Fair Value Gaps (mirror of detect_fvgs).

    A bearish FVG forms when high(bar_n) < low(bar_n-2), creating a gap
    that wasn't filled by bar n-1. The gap is a zone of inefficiency where
    price moved DOWN aggressively, leaving an unfilled imbalance ABOVE.
    Price often returns up to fill it before continuing lower.

    For SHORT: entry inside the FVG, SL above FVG top, TP below.
    """
    results: List[Dict[str, Any]] = []

    current_leg = structure.get("current_leg")
    if current_leg is None:
        return results

    leg_start_bar: int = current_leg["leg_start_bar"]
    n      = len(df)
    last   = n - 1
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    apply_zone_filter = (fibo_786_zone is not None) and require_in_zone
    scan_start = max(leg_start_bar + 2, 2)

    for bar_n in range(scan_start, n):
        bar_n_minus_2 = bar_n - 2

        # ── FLIPPED: bearish FVG → high[n] < low[n-2] ────────────────────────
        # Gap: top = low[n-2], bottom = high[n]
        top    = float(lows[bar_n_minus_2])
        bottom = float(highs[bar_n])

        if bottom >= top:
            continue   # No bearish gap

        gap_height = top - bottom
        mid = (top + bottom) / 2.0

        # ── Status: track if price returned UP into the gap ─────────────────
        status   = "FRESH"
        fill_pct = 0.0
        max_high_in_fvg = float("-inf")

        for k in range(bar_n + 1, last + 1):
            # Did this bar touch the FVG range?
            if highs[k] >= bottom and lows[k] <= top:
                # Mitigation: candle CLOSES above the FVG top
                if closes[k] > top:
                    status   = "MITIGATED"
                    fill_pct = 1.0
                    break
                if highs[k] > max_high_in_fvg:
                    max_high_in_fvg = highs[k]

        if status != "MITIGATED":
            if max_high_in_fvg == float("-inf"):
                status   = "FRESH"
                fill_pct = 0.0
            else:
                fill_depth = max_high_in_fvg - bottom
                fill_pct   = min(fill_depth / gap_height, 1.0)
                status = "PARTIAL_<50" if fill_pct < 0.5 else "PARTIAL_>50"

        # ── Fibo 0.786 zone check (unchanged) ────────────────────────────────
        in_zone = _check_fibo_zone_overlap(top, bottom, fibo_786_zone)
        if apply_zone_filter and not in_zone:
            continue

        results.append({
            "type":                "fvg",
            "direction":           "bearish",          # FLIPPED
            "top":                 round(top, 10),
            "bottom":              round(bottom, 10),
            "mid":                 round(mid, 10),
            "created_at_bar":      bar_n,
            "status":              status,
            "fill_pct":            round(fill_pct, 6),
            "is_in_fibo_786_zone": in_zone,
        })

    results.sort(key=lambda x: x["created_at_bar"])
    return results


# ----------------------------------------------------------------------------

def detect_fibo_levels_short(
    df: pd.DataFrame,
    structure: dict,
    atr_multiplier: float = 0.5,
    atr_period: int = 14,
) -> dict:
    """
    Compute Fibonacci retracement levels for the current DOWN leg.

    Anchor (from structure['current_leg']):
        leg_start_HIGH (the LH)  →  leg_low_LOW (the LL)

    Retracement levels measure how far price has bounced UP from the LL
    toward the LH. Fib 0.786 is the DEEPEST allowable retracement before
    structure breaks (price returning to 78.6% of the prior down-leg).

    For SHORT entries:
        - Price retracing UP into Fib 0.786 zone = optimal short entry zone
        - Above 0.886 = too deep, structure likely flipping
        - Below 0.618 = early, might not have completed pullback

    Returns dict with same keys as LONG version so render.py can be shared,
    but the semantics are inverted (levels build UP from leg_low).
    """
    current_leg = structure.get("current_leg")
    if current_leg is None:
        return {}

    anchor_high_bar: int   = current_leg["leg_start_bar"]    # the LH
    anchor_low_bar:  int   = current_leg["leg_low_bar"]      # the LL
    anchor_high:     float = float(current_leg["leg_start_price"])
    anchor_low:      float = float(current_leg["leg_low_price"])

    leg_range = anchor_high - anchor_low
    if leg_range <= 0:
        return {}

    # ── FLIPPED: levels measured UPWARD from anchor_low ──────────────────────
    # Fib 0% = anchor_low (just bounced); Fib 100% = anchor_high (full retrace)
    fib_236 = anchor_low + leg_range * 0.236
    fib_382 = anchor_low + leg_range * 0.382
    fib_500 = anchor_low + leg_range * 0.500
    fib_618 = anchor_low + leg_range * 0.618
    fib_705 = anchor_low + leg_range * 0.705
    fib_786 = anchor_low + leg_range * 0.786
    fib_886 = anchor_low + leg_range * 0.886

    # ATR-based zone tolerance (same as LONG)
    atr_value = _compute_atr(df, period=atr_period)
    tolerance = atr_value * atr_multiplier

    fib_786_zone_top    = fib_786 + tolerance
    fib_786_zone_bottom = fib_786 - tolerance

    if fib_786 > 0:
        zone_width_pct = (fib_786_zone_top - fib_786_zone_bottom) / fib_786 * 100.0
    else:
        zone_width_pct = 0.0

    return {
        "anchor_high_bar":        anchor_high_bar,
        "anchor_high":            round(anchor_high, 10),
        "anchor_low_bar":         anchor_low_bar,
        "anchor_low":             round(anchor_low, 10),
        "leg_range":              round(leg_range, 10),
        "fib_236":                round(fib_236, 10),
        "fib_382":                round(fib_382, 10),
        "fib_500":                round(fib_500, 10),
        "fib_618":                round(fib_618, 10),
        "fib_705":                round(fib_705, 10),
        "fib_786":                round(fib_786, 10),
        "fib_886":                round(fib_886, 10),
        "fib_786_zone_top":       round(fib_786_zone_top, 10),
        "fib_786_zone_bottom":    round(fib_786_zone_bottom, 10),
        "fib_786_zone_width_pct": round(zone_width_pct, 6),
        "atr_used":               round(atr_value, 10),
        "atr_multiplier":         atr_multiplier,
    }


# ----------------------------------------------------------------------------

def detect_sr_levels_short(
    df: pd.DataFrame,
    lookback: int = 50,
    cluster_pct: float = 0.5,
    fibo_786_zone: Optional[Dict] = None,
    require_in_zone: bool = True,
) -> List[Dict[str, Any]]:
    """
    Detect horizontal S/R levels. For SHORT setups we care about RESISTANCE
    levels — clusters above current_price where the rally is likely to stall.

    Algorithm identical to LONG version, but only RESISTANCE levels are
    returned (kind == 'resistance').
    """
    n = len(df)
    if n < 2 * _PIVOT_N + 1:
        return []

    start_bar = max(0, n - lookback)
    slice_df  = df.iloc[start_bar:].copy()

    swing_highs, swing_lows = _detect_swings_minimal(slice_df, pivot=_PIVOT_N)

    abs_highs = [(start_bar + bar, price) for bar, price in swing_highs]
    abs_lows  = [(start_bar + bar, price) for bar, price in swing_lows]

    pivots: List[Tuple[float, int, str]] = []
    for bar, price in abs_highs:
        pivots.append((price, bar, "high"))
    for bar, price in abs_lows:
        pivots.append((price, bar, "low"))

    if not pivots:
        return []

    pivots.sort(key=lambda x: x[0])

    tol_fraction = cluster_pct / 100.0
    clusters: List[List[Dict[str, Any]]] = []

    for price, bar_idx, kind in pivots:
        if (
            clusters
            and abs(price - clusters[-1][-1]["price"]) / max(price, 1e-12) < tol_fraction
        ):
            clusters[-1].append({"price": price, "bar_idx": bar_idx, "kind": kind})
        else:
            clusters.append([{"price": price, "bar_idx": bar_idx, "kind": kind}])

    current_price = float(df["close"].iloc[-1])
    apply_zone_filter = (fibo_786_zone is not None) and require_in_zone

    results: List[Dict[str, Any]] = []

    for cluster in clusters:
        if len(cluster) < _SR_MIN_CLUSTER_SIZE:
            continue

        prices_in_cluster = [m["price"] for m in cluster]
        bar_idxs          = [m["bar_idx"] for m in cluster]
        sr_price          = float(np.mean(prices_in_cluster))
        touches           = len(cluster)
        last_touch_bar    = int(max(bar_idxs))
        kind              = "support" if sr_price < current_price else "resistance"
        strength          = min(1.0, touches / _SR_MAX_STRENGTH)

        # ── FLIPPED FILTER: for SHORT, only return RESISTANCE levels ────────
        # (a level above current_price that price is rallying back to)
        if kind != "resistance":
            continue

        in_zone = _check_price_in_fibo_zone(sr_price, fibo_786_zone)
        if apply_zone_filter and not in_zone:
            continue

        results.append({
            "price":               round(sr_price, 10),
            "touches":             touches,
            "kind":                kind,
            "last_touch_bar":      last_touch_bar,
            "strength":            round(strength, 4),
            "is_in_fibo_786_zone": in_zone,
        })

    results.sort(key=lambda x: x["price"])
    return results


# ----------------------------------------------------------------------------

def classify_current_price_in_zones_short(
    current_price: float,
    smart_obs: List[Dict[str, Any]],
    fvgs: List[Dict[str, Any]],
    fibo: dict,
    srs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Mirror of classify_current_price_in_zones — output shape kept identical
    so render.py can be reused. The only semantic difference:
      'at_sr_support'  →  meaning 'at S/R resistance' for SHORT setups.
    """
    in_smart_ob = [
        ob for ob in smart_obs
        if ob["ob_low"] <= current_price <= ob["ob_high"]
    ]

    tradeable_fvg_statuses = {"FRESH", "PARTIAL_<50"}
    in_fvg = [
        fvg for fvg in fvgs
        if (
            fvg["bottom"] <= current_price <= fvg["top"]
            and fvg["status"] in tradeable_fvg_statuses
        )
    ]

    in_fibo_786 = False
    if fibo:
        zone_bottom = fibo.get("fib_786_zone_bottom")
        zone_top    = fibo.get("fib_786_zone_top")
        if zone_bottom is not None and zone_top is not None:
            in_fibo_786 = bool(zone_bottom <= current_price <= zone_top)

    sr_proximity_pct = 0.005
    # For SHORT: "at resistance" replaces "at support" — we kept the key name
    # 'at_sr_support' for render.py compat. Down-stream code treats this as
    # "actionable S/R", direction-aware.
    at_sr_support = [
        sr for sr in srs
        if (
            sr["kind"] == "resistance"
            and abs(sr["price"] - current_price) / max(current_price, 1e-12)
            <= sr_proximity_pct
        )
    ]

    any_zone = bool(in_smart_ob or in_fvg or in_fibo_786 or at_sr_support)

    return {
        "in_smart_ob":   in_smart_ob,
        "in_fvg":        in_fvg,
        "in_fibo_786":   in_fibo_786,
        "at_sr_support": at_sr_support,   # semantically: at resistance
        "any":           any_zone,
    }
