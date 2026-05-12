"""
qf_smc/structure.py — Market structure detection
=================================================
Detects swing highs/lows and classifies market structure state per
QUANTFLOW SMC Long spec v1.1 §5.2.

Public API:
  - detect_swings(df, pivot=5)
  - classify_structure(swings, df)
  - is_uptrend_confirmed(state)

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
# Public API
# ============================================================================

def detect_swings(df: pd.DataFrame, pivot: int = 5) -> dict:
    """
    Detect swing highs and swing lows using the pivot-N method.

    A swing high at bar i: high[i] >= max(high[i-pivot : i+pivot+1]).
    A swing low  at bar i: low[i]  <= min(low[i-pivot  : i+pivot+1]).

    Args:
        df:    DataFrame with at minimum columns
               ['open', 'high', 'low', 'close', 'volume'].
               Index may be sequential int or datetime; function operates on
               positional .iloc throughout.
        pivot: Number of bars left/right required for confirmation. Default 5.

    Returns:
        {
            "swing_highs": [(bar_idx, price), ...],  # sorted by bar_idx asc
            "swing_lows":  [(bar_idx, price), ...],  # sorted by bar_idx asc
            "labeled_df":  df.copy() with boolean columns 'is_sh' and 'is_sl'
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

    if n < min_bars:
        return {
            "swing_highs": swing_highs,
            "swing_lows":  swing_lows,
            "labeled_df":  labeled,
        }

    highs_arr = df["high"].to_numpy()
    lows_arr  = df["low"].to_numpy()

    # Classifiable range: pivot <= i <= n-1-pivot
    # (needs pivot bars on both sides)
    for i in range(pivot, n - pivot):
        window_h = highs_arr[i - pivot : i + pivot + 1]
        window_l = lows_arr[i  - pivot : i + pivot + 1]

        # Swing high: bar's high is at least as high as every bar in window
        if highs_arr[i] >= window_h.max():
            swing_highs.append((i, float(highs_arr[i])))
            labeled.iloc[i, labeled.columns.get_loc("is_sh")] = True

        # Swing low: bar's low is at most as low as every bar in window
        if lows_arr[i] <= window_l.min():
            swing_lows.append((i, float(lows_arr[i])))
            labeled.iloc[i, labeled.columns.get_loc("is_sl")] = True

    return {
        "swing_highs": swing_highs,
        "swing_lows":  swing_lows,
        "labeled_df":  labeled,
    }


def classify_structure(swings: dict, df: pd.DataFrame) -> dict:
    """
    Classify the current market structure state based on the swing sequence.

    Walks the combined swing-high and swing-low sequences to determine trend,
    then checks for CHoCH (Change of Character) and BOS (Break of Structure)
    events per the QUANTFLOW SMC Long spec v1.1 §5.2.

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
            "reason": reason,
        }

    def _downtrend(reason: str) -> dict:
        return {
            "state": _STATE_DOWNTREND,
            "choch_bar": None,
            "bos_bar": None,
            "current_leg": None,
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
    # sh_labels: list of (bar_idx, price, "HH"|"LH")
    sh_labels: List[Tuple[int, float, str]] = []
    for i in range(1, len(highs)):
        lbl = "HH" if highs[i][1] > highs[i - 1][1] else "LH"
        sh_labels.append((highs[i][0], highs[i][1], lbl))

    # sl_labels: list of (bar_idx, price, "HL"|"LL")
    sl_labels: List[Tuple[int, float, str]] = []
    for i in range(1, len(lows)):
        lbl = "HL" if lows[i][1] > lows[i - 1][1] else "LL"
        sl_labels.append((lows[i][0], lows[i][1], lbl))

    # ------------------------------------------------------------------
    # Small-dataset edge cases (only 2 highs / 2 lows → 1 label each)
    # Handle pure uptrend / pure downtrend with sparse data
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
                "reason": "All swing highs are HH and lows are HL — sparse but consistent uptrend.",
            }
        if all_lh and all_ll:
            return _downtrend("All swing highs are LH and lows are LL — downtrend (sparse data).")

        return _undefined("Only one pair of swing labels available — mixed or insufficient pattern.")

    # ------------------------------------------------------------------
    # Find the most recent confirmed DOWNTREND block
    # Requires ≥2 consecutive LH in sh_labels AND ≥2 consecutive LL in sl_labels
    # We search backward so we find the MOST RECENT such block.
    # ------------------------------------------------------------------

    def _last_run_end_idx(labels: List[Tuple[int, float, str]], target: str) -> Optional[int]:
        """
        Return the index in `labels` of the final element in the most recent
        trailing run of length ≥ _MIN_DOWNTREND_PAIRS with label == target.
        Returns None if no such run exists.
        """
        for i in range(len(labels) - 1, _MIN_DOWNTREND_PAIRS - 2, -1):
            # Check if labels[i] ends a run of ≥ _MIN_DOWNTREND_PAIRS
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
        # Downtrend was confirmed. The CHoCH trigger is the swing high at
        # sh_labels[last_lh_idx] — the most recent LH in the confirmed run.
        trigger_sh_bar   = sh_labels[last_lh_idx][0]
        trigger_sh_price = sh_labels[last_lh_idx][1]

        # Scan every bar AFTER the trigger swing high for a close above it
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
            # Find the FIRST swing high after choch_bar that is HIGHER than
            # the CHoCH-trigger high. Then look for a close above it.
            post_choch_highs = [
                (bar, price) for bar, price in highs
                if bar > choch_bar and price > choch_trigger_high_price  # type: ignore[operator]
            ]

            for new_sh_bar, new_sh_price in post_choch_highs:
                # Candle that closes above this new swing high
                for bar_i in range(new_sh_bar + 1, n):
                    if close_arr[bar_i] > new_sh_price:
                        bos_bar = bar_i
                        break
                if bos_bar is not None:
                    break  # Use the FIRST (earliest) valid BOS

    # ------------------------------------------------------------------
    # Check whether recent swings re-established a downtrend AFTER CHoCH
    # (price fell back — invalidate CHoCH state, revert to DOWNTREND)
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
        # BOS is older than the recency window — treat as mature uptrend
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
    # ------------------------------------------------------------------
    current_leg: Optional[Dict[str, Any]] = None

    if state in {_STATE_BOS, _STATE_UPTREND, _STATE_CHOCH}:
        # leg_start = lowest swing low BEFORE choch_bar (bottom of the move)
        # If no choch_bar, use all lows (pure uptrend case)
        reference_bar = choch_bar if choch_bar is not None else n

        pre_lows = [(bar, price) for bar, price in lows if bar < reference_bar]
        if not pre_lows:
            # Edge case: choch_bar is very early; fall back to all lows
            pre_lows = list(lows)

        if pre_lows and highs:
            leg_start_bar, leg_start_price = min(pre_lows, key=lambda x: x[1])
            leg_high_bar,  leg_high_price  = highs[-1]  # most recent swing high

            current_leg = {
                "leg_start_bar":   leg_start_bar,
                "leg_high_bar":    leg_high_bar,
                "leg_start_price": leg_start_price,
                "leg_high_price":  leg_high_price,
            }

    return {
        "state":       state,
        "choch_bar":   choch_bar,
        "bos_bar":     bos_bar,
        "current_leg": current_leg,
        "reason":      reason,
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
