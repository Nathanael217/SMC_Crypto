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


# ============================================================================
# Hierarchical View Detection — LONG (UPTREND)
# ============================================================================

def find_hierarchical_views_long(
    swings: dict,
    df: pd.DataFrame,
    min_bounce_pct: float = 0.236,
) -> Dict[str, Any]:
    """
    Detect two nested HL→HH legs (BROAD and NARROW) inside a confirmed uptrend.

    Hierarchy concept
    -----------------
    In an uptrend, price makes Higher Lows (HL) and Higher Highs (HH).  Each
    pullback from a HH creates a new HL.  Two distinct views exist simultaneously:

      BROAD VIEW  — the larger, more significant HL→HH pair.  This is the
                    *most-recent still-valid* HL that has NOT been broken
                    downward.  It represents the dominant up-leg that traders
                    use for Fibonacci zone placement.

      NARROW VIEW — a *later* (more recent) HL→HH pair that sits *inside* the
                    broad leg: the narrow HL is above the broad HL, and the
                    narrow HL-HH pair formed AFTER the broad HH.  It is the
                    smaller pullback setup used as a fallback entry.

    Example (BTCUSDT 4h in uptrend):
      Broad:  HL @ $94,000 → HH @ $99,000   (main up-leg)
      Narrow: HL @ $96,000 → HH @ $98,000   (smaller pullback inside the broad leg)

    min_bounce_pct (default 0.236 = 23.6% Fibonacci)
    --------------------------------------------------
    A HL is only considered structurally meaningful if the *pullback that
    created it* retraced at least ``min_bounce_pct`` of the prior up-leg.
    Tiny noise-pullbacks are excluded.

      prior_up_leg_range = prior_HH_price - prior_HL_price
      pullback_pct       = (prior_HH_price - HL_price) / prior_up_leg_range
      Reject if pullback_pct < min_bounce_pct

    Wick rule (liquidity sweep filter)
    -----------------------------------
    When checking whether a HL has been "broken" by a later bar, a candle
    whose wick pierced the HL but whose body stayed above it may be a
    liquidity sweep rather than a genuine break.  The rule:

      wick_below_HL  = min(open, close, HL_price) − candle_low
      candle_range   = candle_high − candle_low
      wick_ratio     = wick_below_HL / candle_range

      If wick_ratio > 0.5 AND a next candle exists:
          if next_candle.close > HL_price → sweep, HL still valid
          else                            → truly broken
      Else (wick_ratio ≤ 0.5 OR last bar) → HL is broken (strict)

    Parameters
    ----------
    swings : dict returned by detect_swings()
        Must contain keys ``swing_highs`` and ``swing_lows``, each a list
        of (bar_idx, price) tuples.  The existing classify_structure()
        in the same file uses the same format.
    df : pd.DataFrame
        Raw OHLCV data (columns: open, high, low, close, volume).
        Index must be integer-positional (0-based).
    min_bounce_pct : float, optional
        Minimum retrace fraction of the prior up-leg for a pullback to qualify
        as a meaningful HL.  Default 0.236 (23.6%).

    Returns
    -------
    dict with keys:
      ``broad``               — leg dict or None
      ``narrow``              — leg dict or None
      ``broad_invalidated``   — True if the broad HL was broken downward
      ``narrow_invalidated``  — True if the narrow HL was broken downward
      ``reason``              — human-readable explanation string

    Each leg dict matches the ``current_leg`` shape consumed by zones.py:
      {
        "leg_start_bar":   int,    # bar index of HL (the low anchor)
        "leg_start_price": float,  # HL price
        "leg_high_bar":    int,    # bar index of HH
        "leg_high_price":  float,  # HH price
      }

    This naming is identical to classify_structure()'s current_leg keys for
    UPTREND/BOS states (leg_start_price = HL, leg_high_price = HH).

    Edge cases
    ----------
    * swings empty OR df has fewer than 20 bars → all None, reason="Insufficient data"
    * Only 1 valid pair  → broad = that pair, narrow = None
    * Broad == Narrow    → broad = pair, narrow = None (no duplicates)

    # -----------------------------------------------------------------------
    # UNIT-TEST ASSERTIONS (commented out — reference only)
    # -----------------------------------------------------------------------
    # Given synthetic swings: SL@9.50, SH@10.50, SL@10.10 (HL), SH@10.70 (HH),
    #                          SL@10.40 (HL2), SH@11.00 (HH2)
    #
    # HL candidates:
    #   HL-A: 10.10 > 9.50 → prior_up = 10.50-9.50=1.00, pullback=(10.50-10.10)/1.00=40% ✓
    #   HL-B: 10.40 > 10.10 → prior_up = 10.70-10.10=0.60, pullback=(10.70-10.40)/0.60=50% ✓
    #
    # Most-recent still-valid with HH after it:
    #   HL-B (10.40) → HH @ 11.00 = might be BROAD (most recent)
    #   HL-A (10.10) → HH @ 10.70 = BROAD if HL-B came after HH of HL-A
    #
    # assert result["broad"]["leg_start_price"] >= result["broad"]["leg_high_price"] is False
    # assert result["broad"]["leg_high_price"] > result["broad"]["leg_start_price"]
    # assert result["broad_invalidated"] is False
    """

    # ------------------------------------------------------------------
    # Guard: insufficient data
    # ------------------------------------------------------------------
    _EMPTY: Dict[str, Any] = {
        "broad": None,
        "narrow": None,
        "broad_invalidated": False,
        "narrow_invalidated": False,
        "reason": "Insufficient data",
    }

    if not isinstance(swings, dict) or not swings.get("swing_lows") or len(df) < 20:
        return _EMPTY

    # ------------------------------------------------------------------
    # Step 1: separate swing highs and lows in bar order
    # ------------------------------------------------------------------
    all_highs: List[Tuple[int, float]] = sorted(
        swings.get("swing_highs", []),
        key=lambda x: x[0],
    )
    all_lows: List[Tuple[int, float]] = sorted(
        swings.get("swing_lows", []),
        key=lambda x: x[0],
    )

    if len(all_lows) < 2 or len(all_highs) < 1:
        return _EMPTY

    # ------------------------------------------------------------------
    # Step 2: merge all swings into one chronological sequence
    # ------------------------------------------------------------------
    all_swings_sorted: List[Tuple[int, float, str]] = sorted(
        [(b, p, "high") for b, p in all_highs]
        + [(b, p, "low")  for b, p in all_lows],
        key=lambda x: x[0],
    )

    # ------------------------------------------------------------------
    # Step 3: build HL candidates with pullback metrics
    # ------------------------------------------------------------------
    # Walk the merged swing list.  When we encounter a swing low that is
    # HIGHER than the previous swing low, it is a Higher Low (HL) candidate.
    # We also track the last swing high seen before the HL so we can compute
    # the prior up-leg range for the pullback-percentage filter.

    last_sh_bar:   Optional[int]   = None
    last_sh_price: Optional[float] = None
    last_sl_bar:   Optional[int]   = None
    last_sl_price: Optional[float] = None

    hl_candidates: List[Dict[str, Any]] = []

    for bar, price, stype in all_swings_sorted:
        if stype == "low":
            if last_sl_price is not None and price > last_sl_price:
                # This swing low is higher than the previous → HL candidate
                if last_sh_price is not None and last_sh_bar is not None:
                    # prior up-leg: from last_sl_price up to last_sh_price
                    prior_up_range = last_sh_price - last_sl_price
                    if prior_up_range > 0:
                        pullback_pct = (last_sh_price - price) / prior_up_range
                        hl_candidates.append({
                            "hl_bar":         bar,
                            "hl_price":       price,
                            "prior_sl_price": last_sl_price,
                            "prior_sh_bar":   last_sh_bar,
                            "prior_sh_price": last_sh_price,
                            "pullback_pct":   pullback_pct,
                        })
            # Update last-seen swing low regardless
            last_sl_bar   = bar
            last_sl_price = price
        else:  # high
            last_sh_bar   = bar
            last_sh_price = price

    if not hl_candidates:
        return {**_EMPTY, "reason": "No HL candidates found"}

    # ------------------------------------------------------------------
    # Step 4: pair each HL with its subsequent HH, apply pullback filter
    # ------------------------------------------------------------------
    valid_pairs: List[Dict[str, Any]] = []

    for cand in hl_candidates:
        hl_bar   = cand["hl_bar"]
        hl_price = cand["hl_price"]

        # Skip if pullback too small (noise)
        if cand["pullback_pct"] < min_bounce_pct:
            continue

        # Find the most recent swing high AFTER this HL
        paired_hh: Optional[Tuple[int, float]] = None
        for sh_bar, sh_price in all_highs:
            if sh_bar > hl_bar:
                if paired_hh is None or sh_bar > paired_hh[0]:
                    paired_hh = (sh_bar, sh_price)

        if paired_hh is None:
            continue  # no HH after this HL yet

        hh_bar, hh_price = paired_hh

        if hh_price <= hl_price:
            continue  # degenerate: HH must be strictly above HL

        valid_pairs.append({
            "hl_bar":       hl_bar,
            "hl_price":     hl_price,
            "hh_bar":       hh_bar,
            "hh_price":     hh_price,
            "pullback_pct": cand["pullback_pct"],
        })

    if not valid_pairs:
        return {**_EMPTY, "reason": "No HL-HH pairs passed pullback filter"}

    # Sort chronologically by HL bar
    valid_pairs.sort(key=lambda x: x["hl_bar"])

    # ------------------------------------------------------------------
    # Helper: is_hl_broken(hl_price, after_bar)
    # ------------------------------------------------------------------
    def is_hl_broken(hl_price: float, after_bar: int) -> bool:
        """
        Return True if any bar AFTER ``after_bar`` genuinely breaks BELOW
        ``hl_price`` (i.e. the HL support has failed).

        Applies the wick rule: a candle whose low dips below HL but whose
        body remains above it is treated as a liquidity sweep if the next
        candle closes back above HL.
        """
        max_bar = len(df) - 1
        for idx in range(after_bar + 1, max_bar + 1):
            if df["low"].iloc[idx] < hl_price:
                c_high  = df["high"].iloc[idx]
                c_low   = df["low"].iloc[idx]
                c_open  = df["open"].iloc[idx]
                c_close = df["close"].iloc[idx]
                c_range = c_high - c_low

                if c_range == 0:
                    return True  # doji at/below HL → broken

                # wick_below_HL: portion of the candle range that is a wick
                # below the lower of (open, close, HL_price)
                wick_below = min(c_open, c_close, hl_price) - c_low
                wick_ratio = wick_below / c_range

                if wick_ratio > 0.5 and idx < max_bar:
                    # Long lower wick — potential sweep; check next candle
                    next_close = df["close"].iloc[idx + 1]
                    if next_close > hl_price:
                        continue  # confirmed sweep: HL still valid
                    else:
                        return True  # price continued lower → broken
                else:
                    # Body break, or wick ≤ 50%, or last bar → strict break
                    return True
        return False

    # ------------------------------------------------------------------
    # Step 5: find BROAD VIEW = most-recent still-valid HL-HH pair
    # ------------------------------------------------------------------
    broad_pair: Optional[Dict[str, Any]] = None
    broad_invalidated = False

    for pair in reversed(valid_pairs):
        if not is_hl_broken(pair["hl_price"], pair["hh_bar"]):
            broad_pair = pair
            break

    # If no valid pair found, the most-recent pair is considered invalidated
    if broad_pair is None and valid_pairs:
        broad_invalidated = True

    # ------------------------------------------------------------------
    # Step 6: find NARROW VIEW
    # ------------------------------------------------------------------
    narrow_pair: Optional[Dict[str, Any]] = None
    narrow_invalidated = False

    if broad_pair is not None:
        # Narrow must:
        #   - have HL bar AFTER broad_pair["hh_bar"]  (formed after broad HH)
        #   - have HL price ABOVE broad_pair["hl_price"]  (nested inside broad)
        #   - not be the same object as broad_pair
        #   - not be broken downward
        narrow_candidates = [
            p for p in valid_pairs
            if p["hl_bar"] > broad_pair["hh_bar"]
            and p["hl_price"] > broad_pair["hl_price"]
            and p is not broad_pair
        ]

        for pair in reversed(narrow_candidates):
            if not is_hl_broken(pair["hl_price"], pair["hh_bar"]):
                narrow_pair = pair
                break
            else:
                narrow_invalidated = True

    # ------------------------------------------------------------------
    # Step 7: build output leg dicts (key names match classify_structure)
    # ------------------------------------------------------------------
    def _to_leg(p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "leg_start_bar":   int(p["hl_bar"]),
            "leg_start_price": float(p["hl_price"]),
            "leg_high_bar":    int(p["hh_bar"]),
            "leg_high_price":  float(p["hh_price"]),
        }

    broad_leg  = _to_leg(broad_pair)  if broad_pair  is not None else None
    narrow_leg = _to_leg(narrow_pair) if narrow_pair is not None else None

    # ------------------------------------------------------------------
    # Step 8: compose reason string
    # ------------------------------------------------------------------
    parts: List[str] = []
    if broad_invalidated:
        parts.append("broad HL broken (trend reversed or insufficient valid pairs)")
    elif broad_leg:
        parts.append(
            f"broad: HL@{broad_leg['leg_start_price']:.4f} bar={broad_leg['leg_start_bar']}"
            f" → HH@{broad_leg['leg_high_price']:.4f} bar={broad_leg['leg_high_bar']}"
        )
    else:
        parts.append("no broad HL-HH pair found")

    if narrow_invalidated and narrow_leg is None:
        parts.append("narrow: all candidates broken")
    elif narrow_leg:
        parts.append(
            f"narrow: HL@{narrow_leg['leg_start_price']:.4f} bar={narrow_leg['leg_start_bar']}"
            f" → HH@{narrow_leg['leg_high_price']:.4f} bar={narrow_leg['leg_high_bar']}"
        )
    else:
        parts.append("narrow: none (no qualifying sub-leg after broad HH)")

    reason = "; ".join(parts)

    return {
        "broad":              broad_leg,
        "narrow":             narrow_leg,
        "broad_invalidated":  broad_invalidated,
        "narrow_invalidated": narrow_invalidated,
        "reason":             reason,
    }
