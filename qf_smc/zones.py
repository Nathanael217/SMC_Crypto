"""
qf_smc/zones.py — Zone detection for SMC Long
==============================================
Detects Smart OBs, FVGs, Fibonacci levels, and Classic S/R from OHLCV data
per QUANTFLOW SMC Long spec v1.2 (revised from v1.1 §5.3).

Public API:
  - detect_smart_obs(df, structure, vol_avg_period=20, fibo_786_zone=None, require_in_zone=True)
  - detect_fvgs(df, structure, fibo_786_zone=None, require_in_zone=True)
  - detect_fibo_levels(df, structure, atr_multiplier=0.5, atr_period=14)
  - detect_sr_levels(df, lookback=50, cluster_pct=0.5, fibo_786_zone=None, require_in_zone=True)
  - classify_current_price_in_zones(current_price, smart_obs, fvgs, fibo, srs)

Revisions v1.2 (spec §2.3 – §2.5):
  - _compute_atr helper (NEW)
  - detect_fibo_levels(): ATR-based zone tolerance (replaces fixed ±0.5%)
  - detect_smart_obs(): fibo_786_zone filter + LIQUIDITY_SWEEP tier
  - detect_fvgs(): fibo_786_zone filter
  - detect_sr_levels(): fibo_786_zone filter

Standalone module — does NOT import qf_shared or structure.
The caller (scanner.py) passes the structure dict explicitly.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


# ============================================================================
# Constants
# ============================================================================

_OB_IMPULSE_WINDOW   = 5    # max bars forward to look for bullish impulse
_MIN_VOL_MULT        = 1.2  # minimum volume multiplier for OB qualification
_STRONG_VOL_MULT     = 2.0  # volume multiplier threshold for STRONG tier
_MIN_BODY_RATIO      = 0.30 # minimum body/range for non-doji filter
_FRESHNESS_DECAY     = 0.10 # freshness score decreases by this per touch
_PIVOT_N             = 5    # default pivot bars for internal swing detection
_SR_MIN_CLUSTER_SIZE = 2    # minimum pivots per S/R cluster
_SR_MAX_STRENGTH     = 5.0  # strength normaliser (touches / this, capped 1.0)
_LIQUIDITY_SWEEP_WICK_RATIO = 0.5  # lower_wick / body > this → LIQUIDITY_SWEEP


# ============================================================================
# Internal helpers
# ============================================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute the latest ATR (Average True Range) value.

    Returns the most recent ATR value as a float.
    Returns 0.0 if df has fewer than `period` + 1 bars (can't compute a full
    rolling window) or if the result is NaN.

    Per spec v1.2 §2.3: used as the basis for Fibo 0.786 zone tolerance.
    """
    if len(df) < period + 1:
        return 0.0

    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values

    # True range components (indices 1..n aligned with close[:-1])
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:]  - close[:-1])
    true_range = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3))

    atr = true_range.rolling(period).mean()
    last_val = atr.iloc[-1]
    return float(last_val) if not pd.isna(last_val) else 0.0


def _detect_swings_minimal(
    df: pd.DataFrame,
    pivot: int = _PIVOT_N,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Lightweight swing detector (no labeled_df overhead).

    Returns (swing_highs, swing_lows) as sorted lists of (bar_idx, price).
    bar_idx is always a positional integer (0-based).
    """
    n = len(df)
    min_bars = 2 * pivot + 1
    if n < min_bars:
        return [], []

    highs_arr = df["high"].to_numpy()
    lows_arr  = df["low"].to_numpy()

    swing_highs: List[Tuple[int, float]] = []
    swing_lows:  List[Tuple[int, float]] = []

    for i in range(pivot, n - pivot):
        window_h = highs_arr[i - pivot : i + pivot + 1]
        window_l = lows_arr[i - pivot  : i + pivot + 1]

        if highs_arr[i] >= window_h.max():
            swing_highs.append((i, float(highs_arr[i])))
        if lows_arr[i] <= window_l.min():
            swing_lows.append((i, float(lows_arr[i])))

    return swing_highs, swing_lows


def _is_valid_swing_low(
    bar_idx: int,
    swing_lows: List[Tuple[int, float]],
    price: float,
    tol: float = 1e-9,
) -> bool:
    """Return True if `price` at `bar_idx` coincides with a known swing low."""
    for sl_bar, sl_price in swing_lows:
        if sl_bar == bar_idx and abs(sl_price - price) <= tol:
            return True
    return False


def _count_swing_highs_broken(
    start_bar: int,
    end_bar: int,
    break_price: float,
    swing_highs: List[Tuple[int, float]],
) -> int:
    """
    Count how many swing highs between start_bar and end_bar have price
    strictly below break_price (i.e. the impulse closed above them).
    """
    count = 0
    for sh_bar, sh_price in swing_highs:
        if start_bar <= sh_bar <= end_bar and sh_price < break_price:
            count += 1
    return count


def _nearest_swing_high_before(
    bar_idx: int,
    swing_highs: List[Tuple[int, float]],
) -> Optional[float]:
    """Return the price of the most recent swing high strictly before bar_idx."""
    best_bar   = -1
    best_price = None
    for sh_bar, sh_price in swing_highs:
        if sh_bar < bar_idx and sh_bar > best_bar:
            best_bar   = sh_bar
            best_price = sh_price
    return best_price


def _check_fibo_zone_overlap(
    price_high: float,
    price_low: float,
    fibo_786_zone: Optional[Dict],
) -> bool:
    """
    Returns True if [price_low, price_high] overlaps the Fibo 0.786 zone.
    Overlap condition: price_high >= zone_bottom AND price_low <= zone_top.
    Returns True (no filter) if fibo_786_zone is None or zone keys missing.
    """
    if fibo_786_zone is None:
        return True
    zone_top    = fibo_786_zone.get("fib_786_zone_top")
    zone_bottom = fibo_786_zone.get("fib_786_zone_bottom")
    if zone_top is None or zone_bottom is None:
        return True
    # Edge case: zero-width zone (ATR = 0) — keep all (per spec edge case table)
    if zone_top == zone_bottom:
        return True
    return (price_high >= zone_bottom) and (price_low <= zone_top)


def _check_price_in_fibo_zone(
    price: float,
    fibo_786_zone: Optional[Dict],
) -> bool:
    """Returns True if price is within [zone_bottom, zone_top]."""
    if fibo_786_zone is None:
        return True
    zone_top    = fibo_786_zone.get("fib_786_zone_top")
    zone_bottom = fibo_786_zone.get("fib_786_zone_bottom")
    if zone_top is None or zone_bottom is None:
        return True
    if zone_top == zone_bottom:
        return True
    return bool(zone_bottom <= price <= zone_top)


# ============================================================================
# Public API
# ============================================================================

def detect_smart_obs(
    df: pd.DataFrame,
    structure: dict,
    vol_avg_period: int = 20,
    fibo_786_zone: Optional[Dict] = None,   # NEW v1.2
    require_in_zone: bool = True,            # NEW v1.2
) -> List[Dict[str, Any]]:
    """
    Detect Smart Order Blocks per QUANTFLOW spec v1.2.
    SMART OB HYBRID = mitigation block + volume filter, 3-tier
    (LIQUIDITY_SWEEP / STRONG / REGULAR).

    Args:
        df: OHLCV DataFrame with at minimum [open, high, low, close, volume].
        structure: dict from classify_structure() — uses 'state' and 'current_leg'.
        vol_avg_period: bars for trailing volume average (default 20).
        fibo_786_zone: output of detect_fibo_levels(). When provided AND
            require_in_zone=True, only OBs whose body overlaps the zone are
            returned. Pass None to skip filtering (backward-compatible default).
        require_in_zone: if True and fibo_786_zone is provided, filter to zone
            overlapping OBs only. Defaults to True; ignored when fibo_786_zone
            is None.

    Tier classification order (first match wins):
        1. LIQUIDITY_SWEEP  — lower_wick / body > 0.5
        2. STRONG           — vol_mult >= 2.0 AND swing_low_aligned AND multi_bar_BOS
        3. REGULAR          — vol_mult >= 1.2 (minimum qualifier)
        Skip if vol_mult < 1.2 OR body < 30% of range.

    Returns:
        List of OB dicts, sorted by ob_bar ascending. Each:
        {
            "type": "smart_ob",
            "tier": "LIQUIDITY_SWEEP" | "STRONG" | "REGULAR",
            "ob_bar":      int,
            "ob_high":     float,
            "ob_low":      float,
            "ob_open":     float,
            "ob_close":    float,
            "volume_mult": float,
            "swept_at_bar": int,
            "bos_at_bar":   int,
            "status":       "FRESH" | "TESTING" | "MITIGATED",
            "freshness_score": float,
            "wick_lower_ratio": float,   # NEW v1.2 — lower_wick / body
            "wick_upper_ratio": float,   # NEW v1.2 — upper_wick / body
            "is_in_fibo_786_zone": bool, # NEW v1.2 — for transparency
        }
    """
    results: List[Dict[str, Any]] = []

    current_leg = structure.get("current_leg")
    if current_leg is None:
        return results

    leg_start_bar: int = current_leg["leg_start_bar"]
    bos_bar_struct: Optional[int] = structure.get("bos_bar")
    leg_high_bar: int = current_leg.get("leg_high_bar", len(df) - 1)
    scan_end_bar: int = bos_bar_struct if bos_bar_struct is not None else leg_high_bar
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

    # Determine effective filter state
    apply_zone_filter = (fibo_786_zone is not None) and require_in_zone

    for b_idx in range(leg_start_bar, scan_end_bar + 1):
        if b_idx < vol_avg_period:
            continue

        # Bearish candle filter
        if closes[b_idx] >= opens[b_idx]:
            continue

        b_high  = float(highs[b_idx])
        b_low   = float(lows[b_idx])
        b_open  = float(opens[b_idx])
        b_close = float(closes[b_idx])
        b_vol   = float(volumes[b_idx])

        # ── Body filter ──────────────────────────────────────────────────────
        body  = abs(b_close - b_open)
        rng   = b_high - b_low
        if rng <= 0 or body / rng < _MIN_BODY_RATIO:
            continue

        # ── Volume filter ────────────────────────────────────────────────────
        vol_window = volumes[b_idx - vol_avg_period : b_idx]
        if len(vol_window) == 0:
            continue
        vol_avg = float(np.mean(vol_window))
        if vol_avg <= 0:
            continue
        vol_mult = b_vol / vol_avg
        if vol_mult < _MIN_VOL_MULT:
            continue

        # ── Bullish impulse in next _OB_IMPULSE_WINDOW bars ─────────────────
        impulse_bar: Optional[int] = None
        window_end = min(b_idx + _OB_IMPULSE_WINDOW, n - 1)
        for k in range(b_idx + 1, window_end + 1):
            if closes[k] > b_high:
                impulse_bar = k
                break
        if impulse_bar is None:
            continue

        # ── Mitigation (sweep of B.low before/at impulse) ───────────────────
        swept_at_bar: Optional[int] = None
        for k in range(b_idx + 1, impulse_bar + 1):
            if lows[k] <= b_low:
                swept_at_bar = k
                break
        if swept_at_bar is None:
            continue

        # ── BOS — impulse bar closes above nearest swing high ────────────────
        nearest_sh_price = _nearest_swing_high_before(impulse_bar, swing_highs)
        if nearest_sh_price is not None:
            if closes[impulse_bar] <= nearest_sh_price:
                continue
        bos_at_bar = impulse_bar

        # ── Tier classification (v1.2: LIQUIDITY_SWEEP first) ───────────────
        lower_wick = min(b_open, b_close) - b_low
        upper_wick = b_high - max(b_open, b_close)
        wick_lower_ratio = lower_wick / body if body > 0 else 0.0
        wick_upper_ratio = upper_wick / body if body > 0 else 0.0

        impulse_close_price = float(closes[bos_at_bar])
        is_at_swing_low = _is_valid_swing_low(b_idx, swing_lows, b_low)
        broken_sh_count = _count_swing_highs_broken(
            b_idx, bos_at_bar, impulse_close_price, swing_highs
        )
        multi_bar_bos = broken_sh_count >= 2

        if wick_lower_ratio > _LIQUIDITY_SWEEP_WICK_RATIO:
            tier = "LIQUIDITY_SWEEP"
        elif vol_mult >= _STRONG_VOL_MULT and is_at_swing_low and multi_bar_bos:
            tier = "STRONG"
        else:
            tier = "REGULAR"

        # ── Status ───────────────────────────────────────────────────────────
        if current_price > b_high:
            status = "FRESH"
        elif b_low <= current_price <= b_high:
            status = "TESTING"
        else:
            status = "MITIGATED"

        # ── Freshness score ──────────────────────────────────────────────────
        touch_count = 0
        for k in range(bos_at_bar, last + 1):
            if lows[k] <= b_high and highs[k] >= b_low:
                touch_count += 1
        freshness_score = max(0.0, 1.0 - touch_count * _FRESHNESS_DECAY)

        # ── Fibo 0.786 zone check ─────────────────────────────────────────────
        in_zone = _check_fibo_zone_overlap(b_high, b_low, fibo_786_zone)

        if apply_zone_filter and not in_zone:
            continue  # filtered out — OB outside the Fibo 0.786 zone

        results.append({
            "type":               "smart_ob",
            "tier":               tier,
            "ob_bar":             b_idx,
            "ob_high":            b_high,
            "ob_low":             b_low,
            "ob_open":            b_open,
            "ob_close":           b_close,
            "volume_mult":        round(vol_mult, 4),
            "swept_at_bar":       swept_at_bar,
            "bos_at_bar":         bos_at_bar,
            "status":             status,
            "freshness_score":    round(freshness_score, 4),
            "wick_lower_ratio":   round(wick_lower_ratio, 6),
            "wick_upper_ratio":   round(wick_upper_ratio, 6),
            "is_in_fibo_786_zone": in_zone,
        })

    results.sort(key=lambda x: x["ob_bar"])
    return results


# ----------------------------------------------------------------------------

def detect_fvgs(
    df: pd.DataFrame,
    structure: dict,
    fibo_786_zone: Optional[Dict] = None,   # NEW v1.2
    require_in_zone: bool = True,            # NEW v1.2
) -> List[Dict[str, Any]]:
    """
    Detect bullish Fair Value Gaps (3-candle imbalance).

    A bullish FVG forms when high(bar_n-2) < low(bar_n), creating a gap
    between bar n-2 and bar n that wasn't filled by bar n-1.

    Args:
        df: OHLCV DataFrame.
        structure: dict from classify_structure() — only detect FVGs within
                   the current uptrend leg (from leg_start_bar onwards).
        fibo_786_zone: output of detect_fibo_levels(). When provided AND
            require_in_zone=True, only FVGs overlapping the zone are returned.
        require_in_zone: see detect_smart_obs() for semantics.

    Returns:
        List of FVG dicts, sorted by created_at_bar ascending. Each:
        {
            "type": "fvg",
            "direction": "bullish",
            "top":            float,
            "bottom":         float,
            "mid":            float,
            "created_at_bar": int,
            "status": "FRESH" | "PARTIAL_<50" | "PARTIAL_>50" | "MITIGATED",
            "fill_pct":       float,
            "is_in_fibo_786_zone": bool,   # NEW v1.2
        }
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

        top    = float(lows[bar_n])
        bottom = float(highs[bar_n_minus_2])

        if bottom >= top:
            continue

        gap_height = top - bottom
        mid = (top + bottom) / 2.0

        # ── Status tracking ──────────────────────────────────────────────────
        status   = "FRESH"
        fill_pct = 0.0
        min_low_in_fvg = float("inf")

        for k in range(bar_n + 1, last + 1):
            if lows[k] <= top and highs[k] >= bottom:
                if closes[k] < bottom:
                    status   = "MITIGATED"
                    fill_pct = 1.0
                    break
                if lows[k] < min_low_in_fvg:
                    min_low_in_fvg = lows[k]

        if status != "MITIGATED":
            if min_low_in_fvg == float("inf"):
                status   = "FRESH"
                fill_pct = 0.0
            else:
                fill_depth = top - min_low_in_fvg
                fill_pct   = min(fill_depth / gap_height, 1.0)
                status = "PARTIAL_<50" if fill_pct < 0.5 else "PARTIAL_>50"

        # ── Fibo 0.786 zone check ─────────────────────────────────────────────
        in_zone = _check_fibo_zone_overlap(top, bottom, fibo_786_zone)

        if apply_zone_filter and not in_zone:
            continue

        results.append({
            "type":                "fvg",
            "direction":           "bullish",
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

def detect_fibo_levels(
    df: pd.DataFrame,
    structure: dict,
    atr_multiplier: float = 0.5,   # NEW v1.2 — user-adjustable, typical 0.3–2.0
    atr_period: int = 14,           # NEW v1.2
) -> dict:
    """
    Compute Fibonacci retracement levels for the current uptrend leg.

    v1.2: anchor is now the MOST RECENT HL → HH pair from classify_structure()
    (the `current_leg` dict is built by v1.2 classify_structure). Zone tolerance
    is ATR-based instead of a fixed ±0.5%.

    Anchor: leg_start_low (HL) → leg_high_high (HH) from structure['current_leg'].

    Args:
        df: OHLCV DataFrame.
        structure: dict from classify_structure(). Returns {} if
                   structure['current_leg'] is None.
        atr_multiplier: multiplier applied to ATR for the ±tolerance around
            fib_786. Default 0.5 (narrow band). Range 0.3–2.0 in practice.
        atr_period: ATR rolling period. Default 14.

    Returns:
        {
            "anchor_low_bar":  int,
            "anchor_low":      float,
            "anchor_high_bar": int,
            "anchor_high":     float,
            "leg_range":       float,

            "fib_236":  float,
            "fib_382":  float,
            "fib_500":  float,
            "fib_618":  float,
            "fib_705":  float,
            "fib_786":  float,
            "fib_886":  float,

            # ATR-based 0.786 zone (NEW v1.2)
            "fib_786_zone_top":      float,   # fib_786 + atr_used * atr_multiplier
            "fib_786_zone_bottom":   float,   # fib_786 - atr_used * atr_multiplier
            "fib_786_zone_width_pct": float,  # zone width as % of fib_786 (for UI)
            "atr_used":              float,   # actual ATR value used
            "atr_multiplier":        float,   # multiplier applied
        }

    Returns {} if structure has no current_leg.

    Edge cases:
        - df has < atr_period+1 bars: ATR = 0.0 → tolerance = 0 → zone collapses to
          fib_786 point (degenerate but valid; downstream handles zero-width zone).
        - leg_range <= 0: returns {} immediately.
    """
    current_leg = structure.get("current_leg")
    if current_leg is None:
        return {}

    anchor_low_bar:  int   = current_leg["leg_start_bar"]
    anchor_high_bar: int   = current_leg["leg_high_bar"]
    anchor_low:      float = float(current_leg["leg_start_price"])
    anchor_high:     float = float(current_leg["leg_high_price"])

    leg_range = anchor_high - anchor_low
    if leg_range <= 0:
        return {}

    fib_236 = anchor_high - leg_range * 0.236
    fib_382 = anchor_high - leg_range * 0.382
    fib_500 = anchor_high - leg_range * 0.500
    fib_618 = anchor_high - leg_range * 0.618
    fib_705 = anchor_high - leg_range * 0.705
    fib_786 = anchor_high - leg_range * 0.786
    fib_886 = anchor_high - leg_range * 0.886

    # ATR-based zone tolerance (v1.2 — replaces fixed ±0.5%)
    atr_value = _compute_atr(df, period=atr_period)
    tolerance = atr_value * atr_multiplier

    fib_786_zone_top    = fib_786 + tolerance
    fib_786_zone_bottom = fib_786 - tolerance

    # Width as percentage (for UI display)
    if fib_786 > 0:
        zone_width_pct = (fib_786_zone_top - fib_786_zone_bottom) / fib_786 * 100.0
    else:
        zone_width_pct = 0.0

    return {
        "anchor_low_bar":         anchor_low_bar,
        "anchor_low":             round(anchor_low, 10),
        "anchor_high_bar":        anchor_high_bar,
        "anchor_high":            round(anchor_high, 10),
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

def detect_sr_levels(
    df: pd.DataFrame,
    lookback: int = 50,
    cluster_pct: float = 0.5,
    fibo_786_zone: Optional[Dict] = None,   # NEW v1.2
    require_in_zone: bool = True,            # NEW v1.2
) -> List[Dict[str, Any]]:
    """
    Detect horizontal support/resistance levels from clustered swing pivots.

    Algorithm:
        1. Collect all swing highs and lows in the last `lookback` bars.
        2. Cluster pivot prices: two pivots within `cluster_pct`% of each
           other belong to the same cluster.
        3. Each cluster with >=2 pivots = one S/R level.
        4. Level price = mean of cluster pivot prices.
        5. Kind: "support" if below current price, else "resistance".

    Args:
        df: OHLCV DataFrame.
        lookback: bars to look back (default 50).
        cluster_pct: percent tolerance for grouping (default 0.5%).
        fibo_786_zone: when provided AND require_in_zone=True, only S/R levels
            whose price falls within [zone_bottom, zone_top] are returned.
        require_in_zone: see detect_smart_obs() for semantics.

    Returns:
        List of S/R dicts:
        [
            {
                "price":          float,
                "touches":        int,
                "kind":           "support" | "resistance",
                "last_touch_bar": int,
                "strength":       float,
                "is_in_fibo_786_zone": bool,   # NEW v1.2
            },
            ...
        ]
        Sorted by `price` ascending.
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

def classify_current_price_in_zones(
    current_price: float,
    smart_obs: List[Dict[str, Any]],
    fvgs: List[Dict[str, Any]],
    fibo: dict,
    srs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Determine which zones the current price is INSIDE right now.

    This function operates on already-filtered zone lists; no changes needed
    for v1.2 (per spec: classify_current_price_in_zones — unchanged).

    Args:
        current_price: latest close from df.
        smart_obs: output of detect_smart_obs().
        fvgs:      output of detect_fvgs().
        fibo:      output of detect_fibo_levels(). Pass {} if no current leg.
        srs:       output of detect_sr_levels().

    Returns:
        {
            "in_smart_ob":   [<ob_dict>, ...],
            "in_fvg":        [<fvg_dict>, ...],
            "in_fibo_786":   bool,
            "at_sr_support": [<sr_dict>, ...],
            "any":           bool,
        }
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
    at_sr_support = [
        sr for sr in srs
        if (
            sr["kind"] == "support"
            and abs(sr["price"] - current_price) / max(current_price, 1e-12)
            <= sr_proximity_pct
        )
    ]

    any_zone = bool(in_smart_ob or in_fvg or in_fibo_786 or at_sr_support)

    return {
        "in_smart_ob":   in_smart_ob,
        "in_fvg":        in_fvg,
        "in_fibo_786":   in_fibo_786,
        "at_sr_support": at_sr_support,
        "any":           any_zone,
    }
