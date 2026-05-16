"""
qf_smc_short/structure.py — Market structure detection for SHORT (downtrend)
=============================================================================
Mirror of qf_smc/structure.py for SHORT direction.

Same swing detection algorithm (pivot-N with wick adjustment), but the
classifier flips every comparison:

    LONG version              SHORT version
    ──────────────            ──────────────
    LH/LL pattern             HH/HL pattern
    CHoCH = close > swing_H   CHoCH = close < swing_L
    BOS   = close > new SH    BOS   = close < new SL
    Recent: HH+HL → UPTREND   Recent: LH+LL → DOWNTREND
    current_leg = HL→HH       current_leg = LH→LL
    overall_bullish_verified  overall_bearish_verified

Public API:
  - detect_swings(df, pivot=5)      [REUSED from qf_smc.structure]
  - classify_structure_short(swings, df)
  - is_downtrend_confirmed(state)

No external dependencies beyond numpy/pandas + qf_smc.structure (for the
shared swing detector — wick adjustment is direction-neutral).
"""

from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np

# Reuse swing detection from LONG package — wick adjustment is direction-neutral
from qf_smc.structure import detect_swings  # noqa: F401  (re-exported)


# ============================================================================
# Constants (mirror of qf_smc.structure)
# ============================================================================

_RECENCY_WINDOW = 30
_MIN_UPTREND_PAIRS = 2          # consecutive HH+HL pairs needed before CHoCH↓
_STATE_UNDEFINED  = "UNDEFINED"
_STATE_UPTREND    = "UPTREND"
_STATE_CHOCH      = "CHOCH"     # bearish CHoCH
_STATE_BOS        = "BOS"       # bearish BOS
_STATE_DOWNTREND  = "DOWNTREND"


# ============================================================================
# Public API
# ============================================================================

def classify_structure_short(swings: dict, df: pd.DataFrame) -> dict:
    """
    Classify the current market structure for SHORT-direction setups.

    State priority (highest → lowest):
      BOS       — Bearish BOS confirmed within last 30 bars
      CHOCH     — Bearish CHoCH confirmed, no subsequent BOS
      DOWNTREND — LH/LL pattern in last >=2 pairs (mature downtrend)
      UPTREND   — HH/HL pattern in last >=2 pairs (skip — wrong direction)
      UNDEFINED — mixed / insufficient

    CHoCH bearish (strict):
      Precondition : >=2 prior HH/HL pairs confirmed (uptrend was established).
      Trigger      : any candle AFTER the most recent swing LOW CLOSES below it.
      choch_bar    : positional index of that candle.

    BOS bearish (strict):
      Precondition : bearish CHoCH has occurred.
      Trigger      : a NEW swing low forms BELOW the CHoCH-trigger low AND a
                     subsequent candle closes below that new swing low.

    current_leg anchor for BOS / DOWNTREND states:
      Uses the MOST RECENT LH+LL pair. leg_start_HIGH → leg_low_LOW for Fibo.

    Returns:
        {
            "state":      DOWNTREND | CHOCH | BOS | UPTREND | UNDEFINED,
            "choch_bar":  int | None,
            "bos_bar":    int | None,
            "current_leg": {
                "leg_start_bar":   int,    # bar of the LH (peak)
                "leg_low_bar":     int,    # bar of the LL (trough)
                "leg_start_price": float,  # the LH price (anchor HIGH for Fibo)
                "leg_low_price":   float,  # the LL price (anchor LOW for Fibo)
            } | None,
            "overall_bearish_verified": bool,
            "reason":     str,
        }

    NOTE: The current_leg keys differ from the LONG version:
        LONG  uses leg_start_price (HL) and leg_high_price (HH)
        SHORT uses leg_start_price (LH) and leg_low_price  (LL)
        This matters for zones_short.detect_fibo_levels_short().
    """
    highs: List[Tuple[int, float]] = swings["swing_highs"]
    lows:  List[Tuple[int, float]] = swings["swing_lows"]
    n = len(df)

    def _undefined(reason: str) -> dict:
        return {
            "state": _STATE_UNDEFINED,
            "choch_bar": None,
            "bos_bar": None,
            "current_leg": None,
            "overall_bearish_verified": False,
            "reason": reason,
        }

    def _uptrend(reason: str) -> dict:
        return {
            "state": _STATE_UPTREND,
            "choch_bar": None,
            "bos_bar": None,
            "current_leg": None,
            "overall_bearish_verified": False,
            "reason": reason,
        }

    # Early-exit: too few swings
    if len(highs) < 2 or len(lows) < 2:
        return _undefined("Fewer than 2 swing highs/lows — insufficient data.")

    # ── Label each swing relative to predecessor ─────────────────────────────
    # sh_labels[i] describes highs[i+1] vs highs[i]
    sh_labels: List[Tuple[int, float, str]] = []
    for i in range(1, len(highs)):
        lbl = "HH" if highs[i][1] > highs[i - 1][1] else "LH"
        sh_labels.append((highs[i][0], highs[i][1], lbl))

    sl_labels: List[Tuple[int, float, str]] = []
    for i in range(1, len(lows)):
        lbl = "HL" if lows[i][1] > lows[i - 1][1] else "LL"
        sl_labels.append((lows[i][0], lows[i][1], lbl))

    # ── Small-dataset edge case (only 1 label pair each) ────────────────────
    if len(sh_labels) < 2 or len(sl_labels) < 2:
        all_hh = all(x[2] == "HH" for x in sh_labels)
        all_hl = all(x[2] == "HL" for x in sl_labels)
        all_lh = all(x[2] == "LH" for x in sh_labels)
        all_ll = all(x[2] == "LL" for x in sl_labels)

        if all_lh and all_ll:
            # Sparse downtrend — use highest swing high and most recent low
            leg_start = max(highs, key=lambda x: x[1])
            leg_low   = lows[-1]
            return {
                "state": _STATE_DOWNTREND,
                "choch_bar": None,
                "bos_bar": None,
                "current_leg": {
                    "leg_start_bar":   leg_start[0],
                    "leg_low_bar":     leg_low[0],
                    "leg_start_price": leg_start[1],
                    "leg_low_price":   leg_low[1],
                },
                "overall_bearish_verified": False,
                "reason": "All swing highs LH and lows LL — sparse but consistent downtrend.",
            }
        if all_hh and all_hl:
            return _uptrend("All swing highs HH and lows HL — uptrend (sparse data).")
        return _undefined("Only one label pair — mixed or insufficient pattern.")

    # ── Find most recent confirmed UPTREND block (precondition for CHoCH↓) ─
    def _last_run_end_idx(labels, target: str) -> Optional[int]:
        for i in range(len(labels) - 1, _MIN_UPTREND_PAIRS - 2, -1):
            run_ok = True
            for j in range(_MIN_UPTREND_PAIRS):
                if i - j < 0 or labels[i - j][2] != target:
                    run_ok = False
                    break
            if run_ok:
                return i
        return None

    last_hh_idx = _last_run_end_idx(sh_labels, "HH")
    last_hl_idx = _last_run_end_idx(sl_labels, "HL")

    # ── Bearish CHoCH detection ─────────────────────────────────────────────
    choch_bar:                Optional[int]   = None
    choch_trigger_low_bar:    Optional[int]   = None
    choch_trigger_low_price:  Optional[float] = None
    bos_bar:                  Optional[int]   = None

    close_arr = df["close"].to_numpy()

    if last_hh_idx is not None and last_hl_idx is not None:
        # The trigger for bearish CHoCH is the most recent SWING LOW
        # (after >=2 HH/HL pairs, a close BELOW that low is the change of character)
        trigger_sl_bar   = sl_labels[last_hl_idx][0]
        trigger_sl_price = sl_labels[last_hl_idx][1]

        for bar_i in range(trigger_sl_bar + 1, n):
            if close_arr[bar_i] < trigger_sl_price:
                choch_bar               = bar_i
                choch_trigger_low_bar   = trigger_sl_bar
                choch_trigger_low_price = trigger_sl_price
                break

        # ── BOS bearish detection (only possible after CHoCH↓) ──────────────
        if choch_bar is not None:
            # Look for a new swing LOW that printed lower than the CHoCH-trigger low,
            # and a subsequent candle that closes below it
            post_choch_lows = [
                (bar, price) for bar, price in lows
                if bar > choch_bar and price < choch_trigger_low_price  # type: ignore[operator]
            ]
            for new_sl_bar, new_sl_price in post_choch_lows:
                for bar_i in range(new_sl_bar + 1, n):
                    if close_arr[bar_i] < new_sl_price:
                        bos_bar = bar_i
                        break
                if bos_bar is not None:
                    break

    # ── Check if uptrend re-established after CHoCH (failed CHoCH↓) ────────
    re_uptrend_after_choch = False
    if choch_bar is not None and bos_bar is None:
        post_sh = [(bar, p, lbl) for bar, p, lbl in sh_labels if bar > choch_bar]
        post_sl = [(bar, p, lbl) for bar, p, lbl in sl_labels if bar > choch_bar]
        if len(post_sh) >= _MIN_UPTREND_PAIRS and len(post_sl) >= _MIN_UPTREND_PAIRS:
            recent_hh = all(x[2] == "HH" for x in post_sh[-_MIN_UPTREND_PAIRS:])
            recent_hl = all(x[2] == "HL" for x in post_sl[-_MIN_UPTREND_PAIRS:])
            if recent_hh and recent_hl:
                re_uptrend_after_choch = True

    # ── Recent pattern baseline ─────────────────────────────────────────────
    recent_sh_lbls = [x[2] for x in sh_labels[-_MIN_UPTREND_PAIRS:]]
    recent_sl_lbls = [x[2] for x in sl_labels[-_MIN_UPTREND_PAIRS:]]
    pure_recent_downtrend = (all(l == "LH" for l in recent_sh_lbls)
                             and all(l == "LL" for l in recent_sl_lbls))
    pure_recent_uptrend   = (all(l == "HH" for l in recent_sh_lbls)
                             and all(l == "HL" for l in recent_sl_lbls))

    # ── Determine final state (priority order) ──────────────────────────────
    bars_since_bos   = (n - 1 - bos_bar)   if bos_bar   is not None else None
    bars_since_choch = (n - 1 - choch_bar) if choch_bar is not None else None

    if bos_bar is not None and bars_since_bos <= _RECENCY_WINDOW:  # type: ignore[operator]
        state  = _STATE_BOS
        reason = (
            f"Bearish BOS at bar {bos_bar} — new swing low broke below "
            f"CHoCH-trigger level ({choch_trigger_low_price:.4g}); "
            f"fresh continuation setup ({bars_since_bos} bars ago)."
        )

    elif bos_bar is not None:
        state  = _STATE_DOWNTREND
        reason = (
            f"Bearish BOS at bar {bos_bar} (>{_RECENCY_WINDOW} bars ago) — "
            f"structure has matured into an established downtrend."
        )

    elif choch_bar is not None and not re_uptrend_after_choch:
        state  = _STATE_CHOCH
        reason = (
            f"Bearish CHoCH at bar {choch_bar} — close broke below swing low at "
            f"{choch_trigger_low_price:.4g}; awaiting BOS confirmation."
        )

    elif re_uptrend_after_choch or pure_recent_uptrend:
        state  = _STATE_UPTREND
        reason = (
            "Recent swings show HH/HL — uptrend pattern active"
            + (" (re-established after failed CHoCH)." if re_uptrend_after_choch
               else ".")
        )

    elif pure_recent_downtrend:
        state  = _STATE_DOWNTREND
        reason = "Recent swings show LH/LL — mature downtrend; no CHoCH required."

    else:
        state  = _STATE_UNDEFINED
        reason = "Mixed swing pattern — no clear trend or structure event detected."

    # ── Build current_leg for BOS / DOWNTREND states ────────────────────────
    # MOST RECENT LH+LL pair: leg_start_HIGH → leg_low_LOW
    current_leg: Optional[Dict[str, Any]] = None
    overall_bearish_verified: bool = False

    if state in {_STATE_BOS, _STATE_DOWNTREND}:
        swing_highs_sorted = sorted(highs, key=lambda x: x[0])
        swing_lows_sorted  = sorted(lows,  key=lambda x: x[0])

        # Find Lower Highs (LH): a swing high lower than the previous one
        lh_candidates: List[Tuple[int, float]] = []
        for i in range(1, len(swing_highs_sorted)):
            prev_bar, prev_price = swing_highs_sorted[i - 1]
            curr_bar, curr_price = swing_highs_sorted[i]
            if curr_price < prev_price:
                lh_candidates.append((curr_bar, curr_price))

        if not lh_candidates:
            # Degraded downtrend — no LH found; use oldest high → most recent low
            if swing_highs_sorted and swing_lows_sorted:
                leg_start_bar, leg_start_price = swing_highs_sorted[0]
                leg_low_bar,   leg_low_price   = swing_lows_sorted[-1]
                current_leg = {
                    "leg_start_bar":   leg_start_bar,
                    "leg_start_price": leg_start_price,
                    "leg_low_bar":     leg_low_bar,
                    "leg_low_price":   leg_low_price,
                }
        else:
            # Use MOST RECENT LH as the leg start
            leg_start_bar, leg_start_price = lh_candidates[-1]

            # Find swing lows that came AFTER this LH
            lows_after_lh = [(b, p) for b, p in swing_lows_sorted if b > leg_start_bar]
            if lows_after_lh:
                leg_low_bar, leg_low_price = lows_after_lh[-1]  # most recent LL
            else:
                leg_low_bar  = len(df) - 1
                leg_low_price = float(df["low"].iloc[-1])

            current_leg = {
                "leg_start_bar":   leg_start_bar,
                "leg_start_price": leg_start_price,
                "leg_low_bar":     leg_low_bar,
                "leg_low_price":   leg_low_price,
            }

        # Verify overall bearish structure still holds
        # (leg_start_price <= overall highest swing high × 0.98)
        if swing_highs_sorted and current_leg is not None:
            overall_highest_price = max(p for _, p in swing_highs_sorted)
            overall_bearish_verified = (
                current_leg["leg_start_price"] <= overall_highest_price * 0.98
            )

    elif state == _STATE_CHOCH:
        # CHoCH↓ leg: from most recent (or highest) swing high BEFORE choch_bar,
        # down to the most recent swing low
        reference_bar = choch_bar if choch_bar is not None else n
        pre_highs = [(bar, price) for bar, price in highs if bar < reference_bar]
        if not pre_highs:
            pre_highs = list(highs)

        if pre_highs and lows:
            leg_start_bar, leg_start_price = max(pre_highs, key=lambda x: x[1])
            leg_low_bar,   leg_low_price   = lows[-1]

            current_leg = {
                "leg_start_bar":   leg_start_bar,
                "leg_low_bar":     leg_low_bar,
                "leg_start_price": leg_start_price,
                "leg_low_price":   leg_low_price,
            }

    return {
        "state":                    state,
        "choch_bar":                choch_bar,
        "bos_bar":                  bos_bar,
        "current_leg":              current_leg,
        "overall_bearish_verified": overall_bearish_verified,
        "reason":                   reason,
    }


def is_downtrend_confirmed(state: dict) -> bool:
    """
    Mirror of is_uptrend_confirmed. Returns True iff state is BOS or DOWNTREND.
    Both indicate a confirmed bearish bias suitable for short setups.
    """
    return state.get("state") in {_STATE_BOS, _STATE_DOWNTREND}


# ============================================================================
# Hierarchical View Detection
# ============================================================================

def find_hierarchical_views_short(
    swings: dict,
    df: pd.DataFrame,
    min_bounce_pct: float = 0.236,
) -> Dict[str, Any]:
    """
    Detect two nested LH→LL legs (BROAD and NARROW) inside a confirmed downtrend.

    Hierarchy concept
    -----------------
    In a downtrend, price makes Lower Highs (LH) and Lower Lows (LL).  Each
    bounce off a LL creates a new LH.  Two distinct views exist simultaneously:

      BROAD VIEW  — the larger, more significant LH→LL pair.  This is the
                    *most-recent still-valid* LH that has NOT been broken
                    upward.  It represents the dominant down-leg that traders
                    use for Fibonacci zone placement.

      NARROW VIEW — a *later* (more recent) LH→LL pair that sits *inside* the
                    broad leg: the narrow LH is below the broad LH, and the
                    narrow LH-LL pair formed AFTER the broad LL.  It is the
                    smaller counter-bounce setup used as a fallback entry.

    Example (LINKUSDT 1h):
      Broad:  LH @ $10.76 → LL @ $9.73   (main down-leg)
      Narrow: LH @ ~$10.20 → LL @ $9.73  (small bounce inside the broad leg)

    min_bounce_pct (default 0.236 = 23.6% Fibonacci)
    --------------------------------------------------
    A LH is only considered structurally meaningful if the *bounce that
    created it* retraced at least ``min_bounce_pct`` of the prior down-leg.
    Tiny noise-bounces are excluded.

    Wick rule (liquidity sweep filter)
    -----------------------------------
    When checking whether a LH has been "broken" by a later bar, a candle
    whose wick pierced the LH but whose body stayed below it may be a
    liquidity sweep rather than a genuine break.  The rule:

      wick_above_LH  = candle_high − max(open, close, LH_price)
      candle_range   = candle_high − candle_low
      wick_ratio     = wick_above_LH / candle_range

      If wick_ratio > 0.5 AND a next candle exists:
          if next_candle.close < LH_price → sweep, LH still valid
          else                            → truly broken
      Else (wick_ratio ≤ 0.5 OR last bar) → LH is broken (strict)

    Parameters
    ----------
    swings : dict returned by detect_swings()
        Must contain keys ``swing_highs`` and ``swing_lows``, each a list
        of (bar_idx, price) tuples.  The existing classify_structure_short()
        in the same file uses the same format.
    df : pd.DataFrame
        Raw OHLCV data (columns: open, high, low, close, volume).
        Index must be integer-positional (0-based).
    min_bounce_pct : float, optional
        Minimum retrace fraction of the prior down-leg for a bounce to qualify
        as a meaningful LH.  Default 0.236 (23.6%).

    Returns
    -------
    dict with keys:
      ``broad``               — leg dict or None
      ``narrow``              — leg dict or None
      ``broad_invalidated``   — True if the broad LH was broken upward
      ``narrow_invalidated``  — True if the narrow LH was broken upward
      ``reason``              — human-readable explanation string

    Each leg dict matches the ``current_leg`` shape consumed by zones.py:
      {
        "leg_start_bar":   int,    # bar index of LH
        "leg_start_price": float,  # LH price
        "leg_low_bar":     int,    # bar index of LL
        "leg_low_price":   float,  # LL price
      }

    Edge cases
    ----------
    * swings empty OR df has fewer than 20 bars → all None, reason="Insufficient data"
    * Only 1 valid pair  → broad = that pair, narrow = None
    * Broad == Narrow    → broad = pair, narrow = None (no duplicates)

    # -----------------------------------------------------------------------
    # UNIT-TEST ASSERTIONS (commented out — reference only)
    # -----------------------------------------------------------------------
    # Given synthetic swings: SH@10.87, SL@10.10, SH@10.81, SL@10.03,
    #                          SH@10.76, SL@9.73
    #
    # LH candidates:
    #   LH-A: 10.81 < 10.87 → prior_down = 10.87-10.10=0.77, bounce=10.81-10.10=0.71/0.77=92% ✓
    #   LH-B: 10.76 < 10.81 → prior_down = 10.81-10.03=0.78, bounce=10.76-10.03=0.73/0.78=94% ✓
    #
    # Most-recent still-valid with LL after it:
    #   LH-B (10.76) → LL @ 9.73  = BROAD
    #   LH-A (10.81) is OLDER than broad LL (9.73) so it cannot be narrow
    #
    # Narrow must come AFTER broad LL (9.73) → none in this synthetic set → narrow=None
    #
    # assert result["broad"]["leg_start_price"] == 10.76
    # assert result["broad"]["leg_low_price"]   == 9.73
    # assert result["narrow"] is None
    # assert result["broad_invalidated"] is False
    """

    # ------------------------------------------------------------------
    # Guard: insufficient data
    # ------------------------------------------------------------------
    _EMPTY = {
        "broad": None,
        "narrow": None,
        "broad_invalidated": False,
        "narrow_invalidated": False,
        "reason": "Insufficient data",
    }

    if not isinstance(swings, dict) or not swings.get("swing_highs") or len(df) < 20:
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

    if len(all_highs) < 2 or len(all_lows) < 1:
        return _EMPTY

    # ------------------------------------------------------------------
    # Step 2: merge all swings into one chronological sequence
    # ------------------------------------------------------------------
    all_swings_sorted: List[Tuple[int, float, str]] = sorted(
        [(b, p, "high") for b, p in all_highs]
        + [(b, p, "low") for b, p in all_lows],
        key=lambda x: x[0],
    )

    # ------------------------------------------------------------------
    # Step 3: build LH candidates with paired LL and bounce metrics
    # ------------------------------------------------------------------
    # We walk the merged swing list.  When we encounter a swing high that is
    # lower than the *previous* swing high, it is a Lower High candidate.

    # Track the last-seen swing high in the merged sequence
    last_sh_bar: Optional[int]   = None
    last_sh_price: Optional[float] = None
    last_sl_bar: Optional[int]   = None
    last_sl_price: Optional[float] = None

    # For each LH, we need the swing high BEFORE it and the SL before it
    # to compute the prior down-leg.
    # We store: (lh_bar, lh_price, prior_sh_price, prior_sl_price)
    lh_candidates: List[Dict[str, Any]] = []

    for bar, price, stype in all_swings_sorted:
        if stype == "high":
            if last_sh_price is not None and price < last_sh_price:
                # This is a LH
                if last_sl_price is not None and last_sl_bar is not None:
                    # We have prior SH and prior SL → can compute bounce
                    prior_down_range = last_sh_price - last_sl_price
                    if prior_down_range > 0:
                        bounce_pct = (price - last_sl_price) / prior_down_range
                        lh_candidates.append({
                            "lh_bar":         bar,
                            "lh_price":       price,
                            "prior_sh_price": last_sh_price,
                            "prior_sl_bar":   last_sl_bar,
                            "prior_sl_price": last_sl_price,
                            "bounce_pct":     bounce_pct,
                        })
            # Update last seen SH regardless
            last_sh_bar   = bar
            last_sh_price = price
        else:  # low
            last_sl_bar   = bar
            last_sl_price = price

    if not lh_candidates:
        return {**_EMPTY, "reason": "No LH candidates found"}

    # ------------------------------------------------------------------
    # Step 4: pair each LH with its subsequent LL, apply bounce filter
    # ------------------------------------------------------------------
    valid_pairs: List[Dict[str, Any]] = []

    for cand in lh_candidates:
        lh_bar   = cand["lh_bar"]
        lh_price = cand["lh_price"]

        # Skip if bounce too small
        if cand["bounce_pct"] < min_bounce_pct:
            continue

        # Find the most recent swing low AFTER this LH
        paired_ll = None
        for sl_bar, sl_price in all_lows:
            if sl_bar > lh_bar:
                if paired_ll is None or sl_bar > paired_ll[0]:
                    paired_ll = (sl_bar, sl_price)

        if paired_ll is None:
            continue  # no LL after this LH yet

        ll_bar, ll_price = paired_ll

        if ll_price >= lh_price:
            continue  # LL must be below LH

        valid_pairs.append({
            "lh_bar":     lh_bar,
            "lh_price":   lh_price,
            "ll_bar":     ll_bar,
            "ll_price":   ll_price,
            "bounce_pct": cand["bounce_pct"],
        })

    if not valid_pairs:
        return {**_EMPTY, "reason": "No LH-LL pairs passed bounce filter"}

    # Sort by LH bar (chronological)
    valid_pairs.sort(key=lambda x: x["lh_bar"])

    # ------------------------------------------------------------------
    # Helper: is_lh_broken(lh_bar, lh_price, ll_bar)
    # ------------------------------------------------------------------
    def is_lh_broken(lh_price: float, after_bar: int) -> bool:
        """Return True if any bar after ``after_bar`` genuinely breaks above lh_price."""
        max_bar = len(df) - 1
        check_bars = [i for i in range(after_bar + 1, max_bar + 1)]
        for idx in check_bars:
            if df["high"].iloc[idx] > lh_price:
                # Potential break — apply wick rule
                c_high  = df["high"].iloc[idx]
                c_low   = df["low"].iloc[idx]
                c_open  = df["open"].iloc[idx]
                c_close = df["close"].iloc[idx]
                c_range = c_high - c_low
                if c_range == 0:
                    return True  # doji above LH → broken
                wick_above = c_high - max(c_open, c_close, lh_price)
                wick_ratio = wick_above / c_range if c_range > 0 else 1.0
                if wick_ratio > 0.5 and idx < max_bar:
                    # Potential sweep — check next candle
                    next_close = df["close"].iloc[idx + 1]
                    if next_close < lh_price:
                        continue  # sweep — LH still valid
                    else:
                        return True  # continued higher → broken
                else:
                    return True  # body break or last bar → broken
        return False

    # ------------------------------------------------------------------
    # Step 5: find BROAD VIEW = most-recent still-valid LH-LL pair
    # ------------------------------------------------------------------
    broad_pair: Optional[Dict] = None
    broad_invalidated = False

    for pair in reversed(valid_pairs):
        broken = is_lh_broken(pair["lh_price"], pair["ll_bar"])
        if not broken:
            broad_pair = pair
            break

    # If no valid pair found, the most-recent pair is invalidated
    if broad_pair is None and valid_pairs:
        broad_invalidated = True

    # ------------------------------------------------------------------
    # Step 6: find NARROW VIEW
    # ------------------------------------------------------------------
    narrow_pair: Optional[Dict] = None
    narrow_invalidated = False

    if broad_pair is not None:
        # Narrow must:
        #   - have LH bar > broad_pair["ll_bar"]  (came AFTER broad LL)
        #   - have LH price < broad_pair["lh_price"]  (fits inside broad)
        #   - not be the same pair as broad
        #   - not be broken
        narrow_candidates = [
            p for p in valid_pairs
            if p["lh_bar"] > broad_pair["ll_bar"]
            and p["lh_price"] < broad_pair["lh_price"]
            and p is not broad_pair
        ]

        for pair in reversed(narrow_candidates):
            broken = is_lh_broken(pair["lh_price"], pair["ll_bar"])
            if not broken:
                narrow_pair = pair
                break
            else:
                narrow_invalidated = True

    # ------------------------------------------------------------------
    # Step 7: build output leg dicts
    # ------------------------------------------------------------------
    def _to_leg(p: Dict) -> Dict[str, Any]:
        return {
            "leg_start_bar":   int(p["lh_bar"]),
            "leg_start_price": float(p["lh_price"]),
            "leg_low_bar":     int(p["ll_bar"]),
            "leg_low_price":   float(p["ll_price"]),
        }

    broad_leg  = _to_leg(broad_pair)  if broad_pair  is not None else None
    narrow_leg = _to_leg(narrow_pair) if narrow_pair is not None else None

    # ------------------------------------------------------------------
    # Step 8: compose reason string
    # ------------------------------------------------------------------
    parts: List[str] = []
    if broad_invalidated:
        parts.append("broad LH broken (trend reversed or insufficient valid pairs)")
    elif broad_leg:
        parts.append(
            f"broad: LH@{broad_leg['leg_start_price']:.4f} bar={broad_leg['leg_start_bar']}"
            f" → LL@{broad_leg['leg_low_price']:.4f} bar={broad_leg['leg_low_bar']}"
        )
    else:
        parts.append("no broad LH-LL pair found")

    if narrow_invalidated and narrow_leg is None:
        parts.append("narrow: all candidates broken")
    elif narrow_leg:
        parts.append(
            f"narrow: LH@{narrow_leg['leg_start_price']:.4f} bar={narrow_leg['leg_start_bar']}"
            f" → LL@{narrow_leg['leg_low_price']:.4f} bar={narrow_leg['leg_low_bar']}"
        )
    else:
        parts.append("narrow: none (no qualifying sub-leg after broad LL)")

    reason = "; ".join(parts)

    return {
        "broad":             broad_leg,
        "narrow":            narrow_leg,
        "broad_invalidated": broad_invalidated,
        "narrow_invalidated": narrow_invalidated,
        "reason":            reason,
    }
