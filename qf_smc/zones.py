"""
qf_smc/zones.py — Zone detection for SMC Long
==============================================
Detects Smart OBs, FVGs, Fibonacci levels, and Classic S/R from OHLCV data
per QUANTFLOW SMC Long spec v1.1 §5.3.

Public API:
  - detect_smart_obs(df, structure, vol_avg_period=20)
  - detect_fvgs(df, structure)
  - detect_fibo_levels(df, structure)
  - detect_sr_levels(df, lookback=50, cluster_pct=0.5)
  - classify_current_price_in_zones(current_price, smart_obs, fvgs, fibo, srs)

Standalone module — does NOT import qf_shared or structure.
The caller (Session 05 scanner) passes the structure dict explicitly.
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


# ============================================================================
# Internal helpers
# ============================================================================

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


# ============================================================================
# Public API
# ============================================================================

def detect_smart_obs(
    df: pd.DataFrame,
    structure: dict,
    vol_avg_period: int = 20,
) -> List[Dict[str, Any]]:
    """
    Detect Smart Order Blocks per QUANTFLOW spec.
    SMART OB HYBRID = mitigation block + volume filter, 2-tier (REGULAR / STRONG).

    Args:
        df: OHLCV DataFrame with at minimum [open, high, low, close, volume].
        structure: dict from classify_structure() — uses 'state' and 'current_leg'.
        vol_avg_period: bars for trailing volume average (default 20).

    Returns:
        List of OB dicts, sorted by ob_bar ascending. Each:
        {
            "type": "smart_ob",
            "tier": "STRONG" | "REGULAR",
            "ob_bar":      int,    # positional index of OB candle
            "ob_high":     float,
            "ob_low":      float,
            "ob_open":     float,
            "ob_close":    float,
            "volume_mult": float,  # OB volume / 20-bar trailing avg
            "swept_at_bar": int,   # bar where OB.low was swept (sweep happens before BOS)
            "bos_at_bar":   int,   # bar where impulse confirmed BOS
            "status":       "FRESH" | "TESTING" | "MITIGATED",
            "freshness_score": float,  # 0-1, decreases with each touch since formation
        }
    """
    results: List[Dict[str, Any]] = []

    current_leg = structure.get("current_leg")
    if current_leg is None:
        return results

    leg_start_bar: int = current_leg["leg_start_bar"]
    # Upper scan bound: use structure bos_bar if available; fall back to leg_high_bar;
    # never exceed last bar.
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

    # Pre-compute swing highs & lows for the full df (needed for tier/BOS checks)
    swing_highs, swing_lows = _detect_swings_minimal(df)

    current_price = float(closes[last])

    for b_idx in range(leg_start_bar, scan_end_bar + 1):
        # Must have enough history for volume average
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

        # ── Step 5: Body filter (check early — cheap) ───────────────────────
        body  = abs(b_close - b_open)
        rng   = b_high - b_low
        if rng <= 0 or body / rng < _MIN_BODY_RATIO:
            continue

        # ── Step 4: Volume filter ────────────────────────────────────────────
        vol_window = volumes[b_idx - vol_avg_period : b_idx]
        if len(vol_window) == 0:
            continue
        vol_avg = float(np.mean(vol_window))
        if vol_avg <= 0:
            continue
        vol_mult = b_vol / vol_avg
        if vol_mult < _MIN_VOL_MULT:
            continue

        # ── Step 1: Find bullish impulse in next _OB_IMPULSE_WINDOW bars ────
        impulse_bar: Optional[int] = None
        window_end = min(b_idx + _OB_IMPULSE_WINDOW, n - 1)
        for k in range(b_idx + 1, window_end + 1):
            if closes[k] > b_high:
                impulse_bar = k
                break
        if impulse_bar is None:
            continue

        # ── Step 2: Verify mitigation (sweep of B.low before/at impulse) ────
        swept_at_bar: Optional[int] = None
        for k in range(b_idx + 1, impulse_bar + 1):
            if lows[k] <= b_low:
                swept_at_bar = k
                break
        if swept_at_bar is None:
            continue

        # ── Step 3: Verify BOS — impulse bar closes above nearest swing high ─
        nearest_sh_price = _nearest_swing_high_before(impulse_bar, swing_highs)
        if nearest_sh_price is None:
            # No swing high to validate against; still treat as BOS if the
            # impulse close exceeded the OB high (already confirmed in step 1).
            pass  # accept — the close > b_high already acts as confirmation
        else:
            if closes[impulse_bar] <= nearest_sh_price:
                continue  # impulse didn't clear the nearest swing high → skip
        bos_at_bar = impulse_bar

        # ── Step 6: Tier classification ──────────────────────────────────────
        impulse_close_price = float(closes[bos_at_bar])
        is_at_swing_low = _is_valid_swing_low(b_idx, swing_lows, b_low)
        broken_sh_count = _count_swing_highs_broken(
            b_idx, bos_at_bar, impulse_close_price, swing_highs
        )
        is_strong = (
            vol_mult >= _STRONG_VOL_MULT
            and is_at_swing_low
            and broken_sh_count >= 2
        )
        tier = "STRONG" if is_strong else "REGULAR"

        # ── Step 7: Status ───────────────────────────────────────────────────
        if current_price > b_high:
            status = "FRESH"
        elif b_low <= current_price <= b_high:
            status = "TESTING"
        else:
            status = "MITIGATED"

        # ── Step 8: Freshness score ──────────────────────────────────────────
        touch_count = 0
        for k in range(bos_at_bar, last + 1):
            if lows[k] <= b_high and highs[k] >= b_low:
                touch_count += 1
        freshness_score = max(0.0, 1.0 - touch_count * _FRESHNESS_DECAY)

        results.append({
            "type":            "smart_ob",
            "tier":            tier,
            "ob_bar":          b_idx,
            "ob_high":         b_high,
            "ob_low":          b_low,
            "ob_open":         b_open,
            "ob_close":        b_close,
            "volume_mult":     round(vol_mult, 4),
            "swept_at_bar":    swept_at_bar,
            "bos_at_bar":      bos_at_bar,
            "status":          status,
            "freshness_score": round(freshness_score, 4),
        })

    results.sort(key=lambda x: x["ob_bar"])
    return results


# ----------------------------------------------------------------------------

def detect_fvgs(df: pd.DataFrame, structure: dict) -> List[Dict[str, Any]]:
    """
    Detect bullish Fair Value Gaps (3-candle imbalance).

    A bullish FVG forms when high(bar_n-2) < low(bar_n), creating a gap
    between bar n-2 and bar n that wasn't filled by bar n-1.

    Args:
        df: OHLCV DataFrame.
        structure: dict from classify_structure() — only detect FVGs within
                   the current uptrend leg (from leg_start_bar onwards).

    Returns:
        List of FVG dicts, sorted by created_at_bar ascending. Each:
        {
            "type": "fvg",
            "direction": "bullish",
            "top":            float,   # = low(bar_n)        (top of imbalance)
            "bottom":         float,   # = high(bar_n-2)     (bottom of imbalance)
            "mid":            float,   # = (top + bottom) / 2
            "created_at_bar": int,     # bar n (the candle that confirms the gap)
            "status": "FRESH" | "PARTIAL_<50" | "PARTIAL_>50" | "MITIGATED",
            "fill_pct":       float,   # 0.0 (untouched) → 1.0 (fully filled)
        }

    Status logic:
        - FRESH:        no bar after creation has touched the FVG range
        - PARTIAL_<50:  some fill has occurred, but <50% of the gap depth filled
        - PARTIAL_>50:  more than 50% filled but FVG bottom not yet broken
        - MITIGATED:    price has closed below FVG bottom OR fully filled the gap
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

    # FVG scan: bar_n goes from leg_start_bar+2 onwards (needs bar_n-2, bar_n-1, bar_n)
    scan_start = max(leg_start_bar + 2, 2)

    for bar_n in range(scan_start, n):
        bar_n_minus_2 = bar_n - 2
        # bar_n_minus_1 = bar_n - 1  # implicitly checked by the gap condition

        top    = float(lows[bar_n])
        bottom = float(highs[bar_n_minus_2])

        # Bullish FVG: high(n-2) < low(n)  — genuine gap, positive height
        if bottom >= top:
            continue

        gap_height = top - bottom
        mid = (top + bottom) / 2.0

        # ── Status tracking ──────────────────────────────────────────────────
        status   = "FRESH"
        fill_pct = 0.0
        min_low_in_fvg = float("inf")  # deepest penetration from top (lowest low that entered range)

        for k in range(bar_n + 1, last + 1):
            # Bar k touches the FVG if it overlaps [bottom, top]
            if lows[k] <= top and highs[k] >= bottom:
                if closes[k] < bottom:
                    # Closed below the FVG bottom → MITIGATED
                    status   = "MITIGATED"
                    fill_pct = 1.0
                    break
                # Partial fill: track deepest low that entered the range
                if lows[k] < min_low_in_fvg:
                    min_low_in_fvg = lows[k]

        if status != "MITIGATED":
            if min_low_in_fvg == float("inf"):
                status   = "FRESH"
                fill_pct = 0.0
            else:
                # fill_depth measured downward from the top of the gap
                fill_depth = top - min_low_in_fvg
                fill_pct   = min(fill_depth / gap_height, 1.0)
                if fill_pct < 0.5:
                    status = "PARTIAL_<50"
                else:
                    status = "PARTIAL_>50"

        results.append({
            "type":           "fvg",
            "direction":      "bullish",
            "top":            round(top, 10),
            "bottom":         round(bottom, 10),
            "mid":            round(mid, 10),
            "created_at_bar": bar_n,
            "status":         status,
            "fill_pct":       round(fill_pct, 6),
        })

    results.sort(key=lambda x: x["created_at_bar"])
    return results


# ----------------------------------------------------------------------------

def detect_fibo_levels(df: pd.DataFrame, structure: dict) -> dict:
    """
    Compute Fibonacci retracement levels for the current uptrend leg.

    Anchor: leg_start_low → leg_high_high (from structure['current_leg']).

    Args:
        df: OHLCV DataFrame.
        structure: dict from classify_structure(). Returns empty dict if
                   structure['current_leg'] is None.

    Returns:
        {
            "anchor_low_bar":  int,
            "anchor_low":      float,
            "anchor_high_bar": int,
            "anchor_high":     float,
            "leg_range":       float,    # high - low

            "fib_236":  float,
            "fib_382":  float,
            "fib_500":  float,
            "fib_618":  float,
            "fib_705":  float,
            "fib_786":  float,           # THE headline level for SMC long entry
            "fib_886":  float,

            # 0.786 entry zone window: ±0.5% around fib_786
            "fib_786_zone_top":    float,
            "fib_786_zone_bottom": float,
        }
    """
    current_leg = structure.get("current_leg")
    if current_leg is None:
        return {}

    anchor_low_bar:  int   = current_leg["leg_start_bar"]
    anchor_high_bar: int   = current_leg["leg_high_bar"]
    anchor_low:      float = float(current_leg["leg_start_price"])
    anchor_high:     float = float(current_leg["leg_high_price"])

    # Safety: if anchor values are inverted or equal, return empty
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

    fib_786_zone_top    = fib_786 * 1.005
    fib_786_zone_bottom = fib_786 * 0.995

    return {
        "anchor_low_bar":       anchor_low_bar,
        "anchor_low":           round(anchor_low, 10),
        "anchor_high_bar":      anchor_high_bar,
        "anchor_high":          round(anchor_high, 10),
        "leg_range":            round(leg_range, 10),
        "fib_236":              round(fib_236, 10),
        "fib_382":              round(fib_382, 10),
        "fib_500":              round(fib_500, 10),
        "fib_618":              round(fib_618, 10),
        "fib_705":              round(fib_705, 10),
        "fib_786":              round(fib_786, 10),
        "fib_886":              round(fib_886, 10),
        "fib_786_zone_top":     round(fib_786_zone_top, 10),
        "fib_786_zone_bottom":  round(fib_786_zone_bottom, 10),
    }


# ----------------------------------------------------------------------------

def detect_sr_levels(
    df: pd.DataFrame,
    lookback: int = 50,
    cluster_pct: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Detect horizontal support/resistance levels from clustered swing pivots.

    Algorithm:
        1. Collect all swing highs and swing lows in the last `lookback` bars
           (uses pivot=5 internally).
        2. Cluster pivot prices: two pivots within `cluster_pct`% of each other
           belong to the same cluster.
        3. Each cluster with >=2 pivots = one S/R level.
        4. Level price = mean of cluster pivot prices.
        5. Kind: "support" if cluster mostly below current price, else "resistance".

    Args:
        df: OHLCV DataFrame.
        lookback: bars to look back (default 50).
        cluster_pct: percent tolerance for grouping (default 0.5%).

    Returns:
        List of S/R dicts:
        [
            {
                "price":          float,
                "touches":        int,        # number of pivots in cluster
                "kind":           "support" | "resistance",
                "last_touch_bar": int,        # bar of most recent pivot in cluster
                "strength":       float,      # min(1.0, touches / 5.0)
            },
            ...
        ]
        Sorted by `price` ascending.
    """
    n = len(df)
    if n < 2 * _PIVOT_N + 1:
        return []

    # Restrict to the last `lookback` bars (offset so bar_idx is still absolute)
    start_bar = max(0, n - lookback)
    slice_df  = df.iloc[start_bar:].copy()

    swing_highs, swing_lows = _detect_swings_minimal(slice_df, pivot=_PIVOT_N)

    # Convert slice-relative indices back to absolute df indices
    abs_highs = [(start_bar + bar, price) for bar, price in swing_highs]
    abs_lows  = [(start_bar + bar, price) for bar, price in swing_lows]

    # Build flat pivot list: (price, bar_idx, kind)
    pivots: List[Tuple[float, int, str]] = []
    for bar, price in abs_highs:
        pivots.append((price, bar, "high"))
    for bar, price in abs_lows:
        pivots.append((price, bar, "low"))

    if not pivots:
        return []

    # Sort ascending by price for sequential clustering
    pivots.sort(key=lambda x: x[0])

    # ── Greedy cluster build (per spec algorithm) ────────────────────────────
    # clusters: list of lists, each inner list is dicts {price, bar_idx, kind}
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

    # ── Build S/R level from each qualifying cluster ─────────────────────────
    current_price = float(df["close"].iloc[-1])
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

        results.append({
            "price":          round(sr_price, 10),
            "touches":        touches,
            "kind":           kind,
            "last_touch_bar": last_touch_bar,
            "strength":       round(strength, 4),
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

    Args:
        current_price: latest close from df.
        smart_obs: output of detect_smart_obs().
        fvgs:      output of detect_fvgs().
        fibo:      output of detect_fibo_levels(). Pass {} if no current leg.
        srs:       output of detect_sr_levels().

    Returns:
        {
            "in_smart_ob":   [<ob_dict>, ...],   # OBs whose low <= price <= high
            "in_fvg":        [<fvg_dict>, ...],  # FVGs whose bottom <= price <= top
                                                  # AND status in {FRESH, PARTIAL_<50}
            "in_fibo_786":   bool,                # True if price within fib_786 ±0.5%
            "at_sr_support": [<sr_dict>, ...],    # support levels within 0.5% of price
            "any":           bool,                # True if any of above is non-empty/True
        }
    """
    # ── Smart OBs: price within [ob_low, ob_high] ────────────────────────────
    in_smart_ob = [
        ob for ob in smart_obs
        if ob["ob_low"] <= current_price <= ob["ob_high"]
    ]

    # ── FVGs: price within [bottom, top] AND status is FRESH or PARTIAL_<50 ──
    tradeable_fvg_statuses = {"FRESH", "PARTIAL_<50"}
    in_fvg = [
        fvg for fvg in fvgs
        if (
            fvg["bottom"] <= current_price <= fvg["top"]
            and fvg["status"] in tradeable_fvg_statuses
        )
    ]

    # ── Fibo 0.786 zone ──────────────────────────────────────────────────────
    in_fibo_786 = False
    if fibo:
        zone_bottom = fibo.get("fib_786_zone_bottom")
        zone_top    = fibo.get("fib_786_zone_top")
        if zone_bottom is not None and zone_top is not None:
            in_fibo_786 = bool(zone_bottom <= current_price <= zone_top)

    # ── S/R support within 0.5% of current price ─────────────────────────────
    sr_proximity_pct = 0.005  # ±0.5%
    at_sr_support = [
        sr for sr in srs
        if (
            sr["kind"] == "support"
            and abs(sr["price"] - current_price) / max(current_price, 1e-12)
            <= sr_proximity_pct
        )
    ]

    any_zone = bool(
        in_smart_ob
        or in_fvg
        or in_fibo_786
        or at_sr_support
    )

    return {
        "in_smart_ob":   in_smart_ob,
        "in_fvg":        in_fvg,
        "in_fibo_786":   in_fibo_786,
        "at_sr_support": at_sr_support,
        "any":           any_zone,
    }
