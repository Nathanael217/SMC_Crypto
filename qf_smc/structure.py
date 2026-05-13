"""
qf_smc/structure.py — Market structure detection
=================================================
Detects swing highs/lows and classifies market structure state per
QUANTFLOW SMC Long spec v1.2 (revised from v1.1 §5.2).

Public API:
  - detect_swings(df, pivot=5)
  - classify_structure(swings, df)
  - is_uptrend_confirmed(state)

Revisions v1.2 (spec §2.1 – §2.2):
  - _is_wick_too_long / _adjust_swing_high / _adjust_swing_low helpers (NEW)
  - detect_swings() applies wick adjustment and returns expanded dict with
    raw_swing_highs, raw_swing_lows, wick_adjustments fields
  - classify_structure() uses MOST RECENT HL+HH pair for current_leg anchor
    when state is BOS or UPTREND; adds overall_bullish_verified field

No external dependencies beyond numpy/pandas. Do NOT import from qf_shared.
"""

from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np


# ============================================================================
# Constants
# ============================================================================

_RECENCY_WINDOW = 30        # bars — BOS/CHoCH within this range is "fresh"
_MIN_DOWNTREND_PAIRS = 2    # consecutive LH + LL pairs required before CHoCH
_STATE_UNDEFINED  = "UNDEFINED"
_STATE_DOWNTREND  = "DOWNTREND"
_STATE_CHOCH      = "CHOCH"
_STATE_BOS        = "BOS"
_STATE_UPTREND    = "UPTREND"


# ============================================================================
# Private helpers — wick adjustment (spec v1.2 §2.1)
# ============================================================================

def _is_wick_too_long(bar: pd.Series, total_range_threshold: float = 0.5) -> bool:
    """
    Returns True if the LONGER wick (upper or lower) takes up more than
    `total_range_threshold` (default 0.5 = 50%) of the TOTAL candle range
    (high - low).

    Per spec v1.2 §2.1, this implements the "Q2a-C" rule: wick takes
    >50% of total candle range.

    Edge case: zero-range bars return False (can't classify).
    """
    total_range = bar["high"] - bar["low"]
    if total_range <= 0:
        return False
    upper_wick = bar["high"] - max(bar["open"], bar["close"])
    lower_wick = min(bar["open"], bar["close"]) - bar["low"]
    return (max(upper_wick, lower_wick) / total_range) > total_range_threshold


def _adjust_swing_high(df: pd.DataFrame, raw_idx: int) -> tuple:
    """
    For a raw swing high at bar `raw_idx`:
    - If candle's wick is not extreme (< 50% of range) → return as-is
    - Else look at next bar (idx+1):
        - If next bar's wick is also extreme → no adjustment (both unreliable)
        - Else → use next bar's HIGH as effective swing high

    Returns: (effective_price, effective_bar_idx, was_adjusted: bool)
    """
    bar = df.iloc[raw_idx]
    if not _is_wick_too_long(bar):
        return (float(bar["high"]), int(raw_idx), False)

    next_idx = raw_idx + 1
    if next_idx >= len(df):
        return (float(bar["high"]), int(raw_idx), False)

    next_bar = df.iloc[next_idx]
    if _is_wick_too_long(next_bar):
        # Both bars have extreme wicks → unreliable, keep original
        return (float(bar["high"]), int(raw_idx), False)

    return (float(next_bar["high"]), int(next_idx), True)


def _adjust_swing_low(df: pd.DataFrame, raw_idx: int) -> tuple:
    """
    Symmetric to _adjust_swing_high — for swing lows.
    - If candle's wick is not extreme (< 50% of range) → return as-is
    - Else look at next bar (idx+1):
        - If next bar's wick is also extreme → no adjustment (both unreliable)
        - Else → use next bar's LOW as effective swing low

    Returns: (effective_price, effective_bar_idx, was_adjusted: bool)
    """
    bar = df.iloc[raw_idx]
    if not _is_wick_too_long(bar):
        return (float(bar["low"]), int(raw_idx), False)

    next_idx = raw_idx + 1
    if next_idx >= len(df):
        return (float(bar["low"]), int(raw_idx), False)

    next_bar = df.iloc[next_idx]
    if _is_wick_too_long(next_bar):
        return (float(bar["low"]), int(raw_idx), False)

    return (float(next_bar["low"]), int(next_idx), True)


# ============================================================================
# Public API
# ============================================================================

def detect_swings(df: pd.DataFrame, pivot: int = 5) -> dict:
    """
    Detect swing highs and swing lows using the pivot-N method, with wick
    adjustment applied to each raw swing (spec v1.2 §2.1).

    A swing high at bar i: high[i] >= max(high[i-pivot : i+pivot+1]).
    A swing low  at bar i: low[i]  <= min(low[i-pivot  : i+pivot+1]).

    After raw detection, each swing is passed through _adjust_swing_high /
    _adjust_swing_low. If a swing candle has an extreme wick (>50% of total
    range) AND the next bar does not, the next bar's high/low is used instead.

    Args:
        df:    DataFrame with at minimum columns
               ['open', 'high', 'low', 'close', 'volume'].
               Index may be sequential int or datetime; function operates on
               positional .iloc throughout.
        pivot: Number of bars left/right required for confirmation. Default 5.

    Returns:
        {
            "swing_highs":     [(adj_bar_idx, adj_price), ...],  # ADJUSTED, used downstream
            "swing_lows":      [(adj_bar_idx, adj_price), ...],  # ADJUSTED, used downstream
            "raw_swing_highs": [(raw_bar_idx, raw_price), ...],  # pre-adjustment, for overlay/UI
            "raw_swing_lows":  [(raw_bar_idx, raw_price), ...],
            "wick_adjustments": [
                {"type": "high"|"low", "raw_idx": int, "adj_idx": int,
                 "raw_price": float, "adj_price": float},
                ...
            ],
            "labeled_df": df.copy() with boolean columns 'is_sh', 'is_sl'
                          (True at ADJUSTED bar indices)
        }

    Edge cases:
        - Fewer than 2*pivot+1 rows → empty lists, is_sh/is_sl all False.
        - Last `pivot` bars cannot be confirmed (no right-side bars) → False.
        - bar_idx is the POSITIONAL integer index (0-based), not df.index value.
    """
    n = len(df)
    min_bars = 2 * pivot + 1

    labeled = df.copy()
    labeled["is_sh"] = False
    labeled["is_sl"] = False

    swing_highs: List[Tuple[int, float]] = []
    swing_lows:  List[Tuple[int, float]] = []
    raw_swing_highs: List[Tuple[int, float]] = []
    raw_swing_lows:  List[Tuple[int, float]] = []
    wick_adjustments: List[Dict[str, Any]] = []

    if n < min_bars:
        return {
            "swing_highs":      swing_highs,
            "swing_lows":       swing_lows,
            "raw_swing_highs":  raw_swing_highs,
            "raw_swing_lows":   raw_swing_lows,
            "wick_adjustments": wick_adjustments,
            "labeled_df":       labeled,
        }

    highs_arr = df["high"].to_numpy()
    lows_arr  = df["low"].to_numpy()

    # Classifiable range: pivot <= i <= n-1-pivot
    for i in range(pivot, n - pivot):
        window_h = highs_arr[i - pivot : i + pivot + 1]
        window_l = lows_arr[i  - pivot : i + pivot + 1]

        # ── Swing high ─────────────────────────────────────────────────────
        if highs_arr[i] >= window_h.max():
            raw_price = float(highs_arr[i])
            raw_swing_highs.append((i, raw_price))

            adj_price, adj_idx, was_adj = _adjust_swing_high(df, i)
            swing_highs.append((adj_idx, adj_price))

            if was_adj:
                wick_adjustments.append({
                    "type":      "high",
                    "raw_idx":   i,
                    "adj_idx":   adj_idx,
                    "raw_price": raw_price,
                    "adj_price": adj_price,
                })

            # Mark adjusted bar in labeled_df
            if adj_idx < n:
                labeled.iloc[adj_idx, labeled.columns.get_loc("is_sh")] = True

        # ── Swing low ──────────────────────────────────────────────────────
        if lows_arr[i] <= window_l.min():
            raw_price = float(lows_arr[i])
            raw_swing_lows.append((i, raw_price))

            adj_price, adj_idx, was_adj = _adjust_swing_low(df, i)
            swing_lows.append((adj_idx, adj_price))

            if was_adj:
                wick_adjustments.append({
                    "type":      "low",
                    "raw_idx":   i,
                    "adj_idx":   adj_idx,
                    "raw_price": raw_price,
                    "adj_price": adj_price,
                })

            if adj_idx < n:
                labeled.iloc[adj_idx, labeled.columns.get_loc("is_sl")] = True

    return {
        "swing_highs":      swing_highs,
        "swing_lows":       swing_lows,
        "raw_swing_highs":  raw_swing_highs,
        "raw_swing_lows":   raw_swing_lows,
        "wick_adjustments": wick_adjustments,
        "labeled_df":       labeled,
    }


def classify_structure(swings: dict, df: pd.DataFrame) -> dict:
    """
    Classify the current market structure state based on the swing sequence.

    Walks the combined swing-high and swing-low sequences to determine trend,
    then checks for CHoCH (Change of Character) and BOS (Break of Structure)
    events per the QUANTFLOW SMC Long spec v1.2 §5.2.

    State priority (highest → lowest):
      BOS      — BOS confirmed within last 30 bars
      CHOCH    — CHoCH confirmed, no subsequent BOS (and no re-downtrend)
      UPTREND  — HH/HL pattern in last ≥2 pairs (mature uptrend)
      DOWNTREND— LH/LL pattern in last ≥2 pairs
      UNDEFINED— mixed, insufficient, or choppy data

    CHoCH bullish (strict, spec v1.1 §2):
      Precondition : ≥2 prior LH/LL pairs confirmed.
      Trigger      : any candle AFTER the most recent swing high CLOSES above it.
      choch_bar    : positional index of that candle (NOT the swing high's bar).

    BOS bullish (strict):
      Precondition : CHoCH has occurred.
      Trigger      : a NEW swing high forms ABOVE the CHoCH-trigger high AND a
                     subsequent candle closes above that new swing high.
      bos_bar      : positional index of the confirming candle.

    v1.2 revision — current_leg anchor for BOS / UPTREND states:
      Uses the MOST RECENT HL+HH pair instead of overall min/max swing.
      Adds 'overall_bullish_verified' bool to output (spec §2.2).

    Args:
        swings: Output dict of detect_swings().
        df:     Same OHLCV DataFrame passed to detect_swings().

    Returns:
        {
            "state":      one of DOWNTREND | CHOCH | BOS | UPTREND | UNDEFINED,
            "choch_bar":  int | None,
            "bos_bar":    int | None,
            "current_leg": {
                "leg_start_bar":   int,
                "leg_high_bar":    int,
                "leg_start_price": float,
                "leg_high_price":  float,
            } | None,
            "overall_bullish_verified": bool,   # NEW v1.2
            "reason":     str,  # one-sentence human-readable explanation
        }
    """
    highs: List[Tuple[int, float]] = swings["swing_highs"]
    lows:  List[Tuple[int, float]] = swings["swing_lows"]
    n = len(df)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _undefined(reason: str) -> dict:
        return {
            "state": _STATE_UNDEFINED,
            "choch_bar": None,
            "bos_bar": None,
            "current_leg": None,
            "overall_bullish_verified": False,
            "reason": reason,
        }

    def _downtrend(reason: str) -> dict:
        return {
            "state": _STATE_DOWNTREND,
            "choch_bar": None,
            "bos_bar": None,
            "current_leg": None,
            "overall_bullish_verified": False,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Early-exit: too few swings for any meaningful analysis
    # ------------------------------------------------------------------
    if len(highs) < 2 or len(lows) < 2:
        return _undefined("Fewer than 2 swing highs or lows detected — insufficient data.")

    # ------------------------------------------------------------------
    # Label each swing high/low relative to its predecessor
    # sh_labels[i] describes highs[i+1] compared with highs[i]
    # ------------------------------------------------------------------
    sh_labels: List[Tuple[int, float, str]] = []
    for i in range(1, len(highs)):
        lbl = "HH" if highs[i][1] > highs[i - 1][1] else "LH"
        sh_labels.append((highs[i][0], highs[i][1], lbl))

    sl_labels: List[Tuple[int, float, str]] = []
    for i in range(1, len(lows)):
        lbl = "HL" if lows[i][1] > lows[i - 1][1] else "LL"
        sl_labels.append((lows[i][0], lows[i][1], lbl))

    # ------------------------------------------------------------------
    # Small-dataset edge cases (only 2 highs / 2 lows → 1 label each)
    # ------------------------------------------------------------------
    if len(sh_labels) < 2 or len(sl_labels) < 2:
        all_hh = all(x[2] == "HH" for x in sh_labels)
        all_hl = all(x[2] == "HL" for x in sl_labels)
        all_lh = all(x[2] == "LH" for x in sh_labels)
        all_ll = all(x[2] == "LL" for x in sl_labels)

        if all_hh and all_hl:
            leg_start = min(lows, key=lambda x: x[1])
            leg_high  = highs[-1]
            return {
                "state": _STATE_UPTREND,
                "choch_bar": None,
                "bos_bar": None,
                "current_leg": {
                    "leg_start_bar":   leg_start[0],
                    "leg_high_bar":    leg_high[0],
                    "leg_start_price": leg_start[1],
                    "leg_high_price":  leg_high[1],
                },
                "overall_bullish_verified": False,
                "reason": "All swing highs are HH and lows are HL — sparse but consistent uptrend.",
            }
        if all_lh and all_ll:
            return _downtrend("All swing highs are LH and lows are LL — downtrend (sparse data).")

        return _undefined("Only one pair of swing labels available — mixed or insufficient pattern.")

    # ------------------------------------------------------------------
    # Find the most recent confirmed DOWNTREND block
    # Requires ≥2 consecutive LH in sh_labels AND ≥2 consecutive LL in sl_labels
    # ------------------------------------------------------------------

    def _last_run_end_idx(labels: List[Tuple[int, float, str]], target: str) -> Optional[int]:
        """
        Return the index in `labels` of the final element in the most recent
        trailing run of length ≥ _MIN_DOWNTREND_PAIRS with label == target.
        Returns None if no such run exists.
        """
        for i in range(len(labels) - 1, _MIN_DOWNTREND_PAIRS - 2, -1):
            run_ok = True
            for j in range(_MIN_DOWNTREND_PAIRS):
                if i - j < 0 or labels[i - j][2] != target:
                    run_ok = False
                    break
            if run_ok:
                return i
        return None

    last_lh_idx = _last_run_end_idx(sh_labels, "LH")
    last_ll_idx = _last_run_end_idx(sl_labels, "LL")

    # ------------------------------------------------------------------
    # CHoCH detection
    # ------------------------------------------------------------------
    choch_bar: Optional[int]                  = None
    choch_trigger_high_bar: Optional[int]     = None
    choch_trigger_high_price: Optional[float] = None
    bos_bar: Optional[int]                    = None

    close_arr = df["close"].to_numpy()

    if last_lh_idx is not None and last_ll_idx is not None:
        trigger_sh_bar   = sh_labels[last_lh_idx][0]
        trigger_sh_price = sh_labels[last_lh_idx][1]

        for bar_i in range(trigger_sh_bar + 1, n):
            if close_arr[bar_i] > trigger_sh_price:
                choch_bar                = bar_i
                choch_trigger_high_bar   = trigger_sh_bar
                choch_trigger_high_price = trigger_sh_price
                break

        # ------------------------------------------------------------------
        # BOS detection (only possible after CHoCH)
        # ------------------------------------------------------------------
        if choch_bar is not None:
            post_choch_highs = [
                (bar, price) for bar, price in highs
                if bar > choch_bar and price > choch_trigger_high_price  # type: ignore[operator]
            ]

            for new_sh_bar, new_sh_price in post_choch_highs:
                for bar_i in range(new_sh_bar + 1, n):
                    if close_arr[bar_i] > new_sh_price:
                        bos_bar = bar_i
                        break
                if bos_bar is not None:
                    break

    # ------------------------------------------------------------------
    # Check whether recent swings re-established a downtrend AFTER CHoCH
    # ------------------------------------------------------------------
    re_downtrend_after_choch = False
    if choch_bar is not None and bos_bar is None:
        post_sh = [(bar, p, lbl) for bar, p, lbl in sh_labels if bar > choch_bar]
        post_sl = [(bar, p, lbl) for bar, p, lbl in sl_labels if bar > choch_bar]
        if len(post_sh) >= _MIN_DOWNTREND_PAIRS and len(post_sl) >= _MIN_DOWNTREND_PAIRS:
            recent_lh = all(x[2] == "LH" for x in post_sh[-_MIN_DOWNTREND_PAIRS:])
            recent_ll = all(x[2] == "LL" for x in post_sl[-_MIN_DOWNTREND_PAIRS:])
            if recent_lh and recent_ll:
                re_downtrend_after_choch = True

    # ------------------------------------------------------------------
    # Evaluate recent pattern for UPTREND / DOWNTREND baseline
    # ------------------------------------------------------------------
    recent_sh_lbls = [x[2] for x in sh_labels[-_MIN_DOWNTREND_PAIRS:]]
    recent_sl_lbls = [x[2] for x in sl_labels[-_MIN_DOWNTREND_PAIRS:]]

    pure_recent_uptrend   = (all(l == "HH" for l in recent_sh_lbls)
                             and all(l == "HL" for l in recent_sl_lbls))
    pure_recent_downtrend = (all(l == "LH" for l in recent_sh_lbls)
                             and all(l == "LL" for l in recent_sl_lbls))

    # ------------------------------------------------------------------
    # Determine final state (priority order)
    # ------------------------------------------------------------------
    bars_since_bos   = (n - 1 - bos_bar)   if bos_bar   is not None else None
    bars_since_choch = (n - 1 - choch_bar) if choch_bar is not None else None

    if bos_bar is not None and bars_since_bos <= _RECENCY_WINDOW:   # type: ignore[operator]
        state  = _STATE_BOS
        reason = (
            f"BOS confirmed at bar {bos_bar} — new swing high broke above "
            f"CHoCH-trigger level ({choch_trigger_high_price:.4g}); "
            f"fresh continuation setup ({bars_since_bos} bars ago)."
        )

    elif bos_bar is not None:
        state  = _STATE_UPTREND
        reason = (
            f"BOS confirmed at bar {bos_bar} (>{_RECENCY_WINDOW} bars ago) — "
            f"structure has matured into an established uptrend."
        )

    elif choch_bar is not None and not re_downtrend_after_choch:
        state  = _STATE_CHOCH
        reason = (
            f"CHoCH at bar {choch_bar} — close broke above swing high at "
            f"{choch_trigger_high_price:.4g}; awaiting BOS confirmation."
        )

    elif re_downtrend_after_choch or pure_recent_downtrend:
        state  = _STATE_DOWNTREND
        reason = (
            "Recent swings show LH/LL — downtrend pattern active"
            + (" (re-established after failed CHoCH)." if re_downtrend_after_choch
               else ".")
        )

    elif pure_recent_uptrend:
        state  = _STATE_UPTREND
        reason = (
            "Recent swings show HH/HL pattern — mature uptrend; "
            "no CHoCH required to confirm."
        )

    else:
        state  = _STATE_UNDEFINED
        reason = "Mixed swing pattern — no clear trend or structure event detected."

    # ------------------------------------------------------------------
    # Build current_leg anchor points (used by zones.py Fibo calculations)
    # v1.2: BOS / UPTREND use MOST RECENT HL+HH pair (spec §2.2)
    # CHOCH keeps v1.1 behavior (no HL pattern confirmed yet)
    # ------------------------------------------------------------------
    current_leg: Optional[Dict[str, Any]] = None
    overall_bullish_verified: bool = False

    if state in {_STATE_BOS, _STATE_UPTREND}:
        # ── v1.2 HL/HH anchor ───────────────────────────────────────────
        swing_lows_sorted  = sorted(lows,  key=lambda x: x[0])
        swing_highs_sorted = sorted(highs, key=lambda x: x[0])

        # Find Higher Lows (HL): a swing low that is higher than the previous one
        hl_candidates: List[Tuple[int, float]] = []
        for i in range(1, len(swing_lows_sorted)):
            prev_bar, prev_price = swing_lows_sorted[i - 1]
            curr_bar, curr_price = swing_lows_sorted[i]
            if curr_price > prev_price:
                hl_candidates.append((curr_bar, curr_price))

        if not hl_candidates:
            # Degraded uptrend — no HL found; use oldest low → most recent high
            if swing_lows_sorted and swing_highs_sorted:
                leg_start_bar, leg_start_price = swing_lows_sorted[0]
                leg_high_bar,  leg_high_price  = swing_highs_sorted[-1]
                current_leg = {
                    "leg_start_bar":   leg_start_bar,
                    "leg_start_price": leg_start_price,
                    "leg_high_bar":    leg_high_bar,
                    "leg_high_price":  leg_high_price,
                }
        else:
            # Use MOST RECENT HL
            leg_start_bar, leg_start_price = hl_candidates[-1]

            # Find swing highs that came AFTER this HL
            highs_after_hl = [
                (b, p) for b, p in swing_highs_sorted if b > leg_start_bar
            ]
            if highs_after_hl:
                leg_high_bar, leg_high_price = highs_after_hl[-1]   # most recent HH
            else:
                # No swing high formed after HL yet — use current bar's high
                leg_high_bar  = len(df) - 1
                leg_high_price = float(df["high"].iloc[-1])

            current_leg = {
                "leg_start_bar":   leg_start_bar,
                "leg_start_price": leg_start_price,
                "leg_high_bar":    leg_high_bar,
                "leg_high_price":  leg_high_price,
            }

        # Verify: overall bullish structure must still hold
        # (leg_start_price >= overall lowest swing low × 1.02)
        if swing_lows_sorted and current_leg is not None:
            overall_lowest_price = min(p for _, p in swing_lows_sorted)
            overall_bullish_verified = (
                current_leg["leg_start_price"] >= overall_lowest_price * 1.02
            )

    elif state == _STATE_CHOCH:
        # ── v1.1 behavior for CHOCH ─────────────────────────────────────
        reference_bar = choch_bar if choch_bar is not None else n
        pre_lows = [(bar, price) for bar, price in lows if bar < reference_bar]
        if not pre_lows:
            pre_lows = list(lows)

        if pre_lows and highs:
            leg_start_bar, leg_start_price = min(pre_lows, key=lambda x: x[1])
            leg_high_bar,  leg_high_price  = highs[-1]

            current_leg = {
                "leg_start_bar":   leg_start_bar,
                "leg_high_bar":    leg_high_bar,
                "leg_start_price": leg_start_price,
                "leg_high_price":  leg_high_price,
            }

    return {
        "state":                    state,
        "choch_bar":                choch_bar,
        "bos_bar":                  bos_bar,
        "current_leg":              current_leg,
        "overall_bullish_verified": overall_bullish_verified,
        "reason":                   reason,
    }


def is_uptrend_confirmed(state: dict) -> bool:
    """
    Convenience helper — returns True iff the market structure state is
    either 'BOS' or 'UPTREND', indicating a confirmed long bias.

    Args:
        state: Output dict of classify_structure().

    Returns:
        True if state["state"] in {"BOS", "UPTREND"}, False otherwise.
    """
    return state.get("state") in {_STATE_BOS, _STATE_UPTREND}
