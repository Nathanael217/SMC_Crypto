"""
qf_smc/scanner.py — Multi-TF SMC scanner orchestrator
======================================================
Coordinates structure detection, zone detection, LTF confirmation, variant-grid
backtest, and trade plan building per QUANTFLOW SMC Long spec v1.2 §2.6.

Revision v1.2 changes (spec §2.6):
  - scan_one_symbol / run_scan accept atr_multiplier (float, default 0.5)
  - Fibo zone is computed FIRST; coin skipped if no current leg
  - All zone detectors receive fibo_786_zone + require_in_zone=True
  - Coin skipped if no zones land in Fibo 0.786 area
  - pick_primary_zone: LIQUIDITY_SWEEP added as priority 1
  - Old backtest_per_coin call replaced by run_variant_grid (24 variants)
  - Result dict enriched with fibo_zone, atr_multiplier_used, wick_adjustments,
    overall_bullish_verified, ob_tier, variant_grid, best_variant,
    deep_dive_available

Public API:
  - MODE_CONFIG: dict
  - run_scan(mode, symbols, btc_regime, atr_multiplier, progress_callback)
  - scan_one_symbol(symbol, mode, btc_regime, atr_multiplier)
  - helpers: check_ltf_confirmation, build_trade_plan, classify_ema_tier,
             pick_primary_zone
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable

from qf_shared import (
    _scanner_fetch_candles,
    calculate_ema,
    _clean_df,
)
from qf_smc.structure import detect_swings, classify_structure, is_uptrend_confirmed, find_hierarchical_views_long
from qf_smc.zones import (
    detect_smart_obs, detect_fvgs, detect_fibo_levels, detect_sr_levels,
    classify_current_price_in_zones,
)
from qf_smc.backtest import (
    backtest_per_coin, backtest_universe_baseline,
    bayesian_blend, compute_recent_check,
    run_variant_grid,                # NEW v1.2
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "SWING": {"htf": "1d",  "ltf": "4h",  "lookback": 100, "htf_label": "1D", "ltf_label": "4H"},
    "DAY":   {"htf": "4h",  "ltf": "15m", "lookback": 100, "htf_label": "4H", "ltf_label": "15m"},
    "SCALP": {"htf": "1h",  "ltf": "5m",  "lookback": 100, "htf_label": "1H", "ltf_label": "5m"},
}

# Default TP R-multiple for legacy backtest_per_coin calls (still available)
_DEFAULT_TP_R = 2.0


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_scan(
    mode: str,
    symbols: List[str],
    btc_regime: str,
    atr_multiplier: float = 0.5,                              # NEW v1.2
    run_backtest: bool = False,                               # NEW v1.2b — scan is fast by default
    min_bounce_pct: float = 0.236,                            # NEW Session 5
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Run a full SMC scan over the provided symbol list in the chosen mode.

    Args:
        mode: "SWING" | "DAY" | "SCALP"
        symbols: list of symbol strings to scan (from Session 04 screener output)
        btc_regime: "BULL" | "CHOP" | "BEAR" | "UNKNOWN"
                    (used for BADGE + AI context only — NOT a skip gate.
                    Scanner runs symbols regardless of BTC regime.)
        atr_multiplier: ATR×N tolerance for Fibo 0.786 zone (default 0.5).
                        Streamlit UI exposes a slider 0.3–2.0 for this value.
        progress_callback: optional callback(current_idx, total, symbol_name)
                          fired once per symbol for UI progress display

    Returns:
        List of result dicts (one per symbol that produced a setup).
        Symbols that fail HTF structure check or have no zone retracement
        return None and are excluded from the result list.
    """
    results: List[Dict[str, Any]] = []

    import time as _time
    import threading as _threading
    import traceback as _traceback
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    total = len(symbols)
    if total == 0:
        return results

    # ── HARD per-coin timeout ────────────────────────────────────────────────
    # The ONLY thing that ever takes long is a slow / hung Binance API call.
    # A coin whose worker has been RUNNING (not just queued) longer than this
    # is abandoned: the scan drops it and moves on. Slow coins can never block.
    #
    # Measured from when the worker ACTUALLY STARTED running this coin — a coin
    # still waiting in the queue for a free worker is not "slow", so it is
    # never wrongly skipped.
    PER_COIN_TIMEOUT = 10.0  # seconds of actual run time

    # 8 workers: the work is network-bound (2 HTTP calls per coin), so threads
    # overlap their waiting. Capped at 8 to stay friendly to Binance.
    max_workers = min(8, max(1, total))

    skipped_slow: list = []

    # symbol -> monotonic time its worker began executing. Written by worker
    # threads, read by the main reaper loop — guarded by a lock.
    _start_times: Dict[str, float] = {}
    _start_lock = _threading.Lock()

    def _worker(sym: str) -> Optional[Dict[str, Any]]:
        # Record the real start time so the reaper can tell running-too-long
        # apart from still-waiting-in-queue.
        with _start_lock:
            _start_times[sym] = _time.monotonic()
        return scan_one_symbol(
            sym, mode, btc_regime,
            atr_multiplier=atr_multiplier,
            run_backtest=run_backtest,
            min_bounce_pct=min_bounce_pct,
        )

    completed = 0
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_symbol: Dict[Any, str] = {
            executor.submit(_worker, sym): sym for sym in symbols
        }
        pending = set(future_to_symbol.keys())

        # Poll in short slices: update progress smoothly AND reap any worker
        # that has been running past the per-coin budget.
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

            # ── finished futures ─────────────────────────────────────────────
            for future in done:
                symbol = future_to_symbol[future]
                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, total, symbol)
                    except Exception:
                        pass  # a UI callback error must never kill the scan
                try:
                    result = future.result()  # already done — never blocks
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"[scanner] error scanning {symbol}: "
                          f"{type(e).__name__}: {e}")
                    print(_traceback.format_exc())

            # ── reap workers that have RUN too long ──────────────────────────
            now = _time.monotonic()
            to_abandon = []
            with _start_lock:
                for future in list(pending):
                    symbol = future_to_symbol[future]
                    started_at = _start_times.get(symbol)
                    if started_at is not None and (now - started_at) > PER_COIN_TIMEOUT:
                        to_abandon.append(future)

            for future in to_abandon:
                symbol = future_to_symbol[future]
                completed += 1
                skipped_slow.append(symbol)
                pending.discard(future)
                # Do NOT call future.result() — it would block. Just stop
                # tracking it. The orphaned thread ends on its own; its return
                # value is discarded. cancel() is a harmless no-op if running.
                future.cancel()
                if progress_callback:
                    try:
                        progress_callback(completed, total, f"{symbol} (skipped \u2014 slow)")
                    except Exception:
                        pass
                print(f"[scanner] SKIPPED {symbol}: worker ran > {PER_COIN_TIMEOUT}s "
                      f"\u2014 slow/hung response, coin dropped from results.")
    finally:
        # Non-blocking shutdown: run_scan returns immediately. Any orphaned
        # slow threads finish in the background and never delay the user.
        executor.shutdown(wait=False)

    if skipped_slow:
        print(f"[scanner] {len(skipped_slow)} coin(s) skipped for slow response: "
              f"{skipped_slow[:15]}{'...' if len(skipped_slow) > 15 else ''}")

    return results


# ============================================================================
# SINGLE-SYMBOL PIPELINE
# ============================================================================

def scan_one_symbol(
    symbol: str,
    mode: str,
    btc_regime: str,
    atr_multiplier: float = 0.5,    # NEW v1.2
    run_backtest: bool = False,     # NEW v1.2b — False = fast scan, skip 24-variant grid
    min_bounce_pct: float = 0.236,  # NEW Session 5
) -> Optional[Dict[str, Any]]:
    """
    Full pipeline for ONE symbol.

    Steps:
        1.  Fetch HTF candles (mode-specific TF, lookback bars)
        2.  Fetch LTF candles (mode-specific TF, lookback bars)
        3.  If either fetch fails or returns < 30 bars → return None
        4.  Run structure.detect_swings + classify_structure on HTF
        5.  If state not in {"BOS", "UPTREND"} → return None
        6.  NEW v1.2: compute Fibo zone FIRST (requires structure.current_leg)
              → return None if current_leg is absent
        7.  Run zone detectors WITH Fibo 0.786 filter
        8.  If no zones found in Fibo 0.786 area → check Fibo 0.786 itself;
              if even that is absent from the fibo_zone dict → return None
        9.  Check EMA 50 & EMA 200 confidence tier
       10.  classify_current_price_in_zones(current_price, ...)
       11.  If classification['any'] is False → return None
       12.  Pick PRIMARY zone
       13.  Run LTF confirmation check
       14.  Run 24-variant grid backtest (run_variant_grid)
       15.  Compute fights_macro = btc_regime in {"BEAR"}
       16.  Build trade_plan from primary_zone
       17.  Return enriched result dict

    Args:
        atr_multiplier: ATR×N tolerance for Fibo 0.786 zone (default 0.5).
                        Streamlit UI exposes a slider 0.3–2.0 for this.

    Returns:
        Full result dict, or None if filtered out.
    """
    cfg = MODE_CONFIG.get(mode)
    if cfg is None:
        print(f"[scanner] Unknown mode {mode!r}; valid: {list(MODE_CONFIG.keys())}")
        return None

    htf_interval = cfg["htf"]
    ltf_interval = cfg["ltf"]
    lookback     = cfg["lookback"]

    # ── Step 1-2: Fetch candles ──────────────────────────────────────────────
    # Deep fetch for HTF — gives variant grid meaningful history (≥200 bars)
    df_htf = _scanner_fetch_candles(symbol, htf_interval, limit=500)
    df_ltf = _scanner_fetch_candles(symbol, ltf_interval, limit=lookback + 50)

    # ── Step 3: Validate data ────────────────────────────────────────────────
    if df_htf is None or df_htf.empty or len(df_htf) < 30:
        return None
    if df_ltf is None or df_ltf.empty or len(df_ltf) < 30:
        return None

    # Work on reset-index copies for positional consistency
    df_htf = df_htf.reset_index(drop=True)
    df_ltf = df_ltf.reset_index(drop=True)

    # ── Step 4: Structure detection on HTF ──────────────────────────────────
    swings    = detect_swings(df_htf, pivot=5)
    structure = classify_structure(swings, df_htf)

    # ── Step 5: Structure filter ─────────────────────────────────────────────
    if not is_uptrend_confirmed(structure):
        return None  # state is UNDEFINED, DOWNTREND, or CHOCH

    # ── Step 5b: Hierarchical views (Session 5) ───────────────────────────────
    views = find_hierarchical_views_long(swings, df_htf, min_bounce_pct=min_bounce_pct)

    # Hard stop: broad view structure has been invalidated
    if views.get("broad_invalidated"):
        return None

    broad_leg  = views.get("broad")
    narrow_leg = views.get("narrow")

    # Must have at least one valid leg
    if broad_leg is None and narrow_leg is None:
        return None

    def _compute_view_zone_data(leg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Given one hierarchical leg dict (broad or narrow), compute fibo zone,
        smart OBs, FVGs, SR levels, retracement_status, zone_classification,
        and primary_zone for that view.

        Mirrors the SHORT scanner's _compute_view_zone_data so LONG/SHORT
        scanners produce structurally-identical per-view dicts. Returns None
        if the leg is None or the Fibo zone calculation fails.
        """
        if leg is None:
            return None

        # Build a minimal structure-like dict for the zone detectors
        structure_for_view = dict(structure)
        structure_for_view["current_leg"] = {
            "leg_start_bar":   leg.get("leg_start_bar"),
            "leg_start_price": leg.get("leg_start_price"),   # HL
            "leg_high_bar":    leg.get("leg_high_bar"),
            "leg_high_price":  leg.get("leg_high_price"),    # HH
        }

        fibo_z = detect_fibo_levels(df_htf, structure_for_view,
                                    atr_multiplier=atr_multiplier)
        if not fibo_z:
            return None

        s_obs   = detect_smart_obs(df_htf, structure_for_view,
                                   fibo_786_zone=fibo_z, require_in_zone=True)
        _fvgs   = detect_fvgs(df_htf, structure_for_view,
                              fibo_786_zone=fibo_z, require_in_zone=True)
        sr_lvls = detect_sr_levels(df_htf, lookback=lookback,
                                   fibo_786_zone=fibo_z, require_in_zone=True)

        cp = float(df_htf["close"].iloc[-1])

        # Retracement status — LONG: price pulled back DOWN from HH
        fib_618  = float(fibo_z.get("fib_618", 0))
        zone_top = float(fibo_z.get("fib_786_zone_top", 0))
        zone_bot = float(fibo_z.get("fib_786_zone_bottom", 0))
        fib_886  = float(fibo_z.get("fib_886", zone_bot * 0.99 if zone_bot else 0))

        if cp > fib_618:
            ret_status = "WAITING"
        elif cp > zone_top:
            ret_status = "APPROACHING"
        elif cp >= zone_bot:
            ret_status = "ACTIONABLE"
        elif cp >= fib_886:
            ret_status = "OVERSHOOT"
        else:
            ret_status = "INVALIDATED"

        # Per-view zone classification + primary pick (mirrors SHORT)
        zone_cls_view = classify_current_price_in_zones(
            cp, s_obs, _fvgs, fibo_z, sr_lvls
        )
        primary_zone_view = pick_primary_zone(zone_cls_view, fibo=fibo_z)

        return {
            "leg":                 leg,
            "fibo_zone":           fibo_z,
            "smart_obs":           s_obs,
            "fvgs":                _fvgs,
            "sr_levels":           sr_lvls,
            "current_price":       cp,
            "retracement_status":  ret_status,
            "zone_classification": zone_cls_view,
            "primary_zone":        primary_zone_view,
            "fib_618":             fib_618,
            "zone_bot":            zone_bot,
            "zone_top":            zone_top,
        }

    def _is_view_tradeable_long(view_data: Optional[Dict[str, Any]]) -> bool:
        """
        Mirror of SHORT scanner's _is_view_tradeable: a view is tradeable only
        when retracement_status is ACTIONABLE or OVERSHOOT AND primary_zone
        is not None. WAITING / APPROACHING / INVALIDATED are not tradeable.
        """
        if view_data is None:
            return False
        if view_data["retracement_status"] not in {"ACTIONABLE", "OVERSHOOT"}:
            return False
        if view_data["primary_zone"] is None:
            return False
        return True

    # Compute zone data for each available view
    broad_data  = _compute_view_zone_data(broad_leg)
    narrow_data = _compute_view_zone_data(narrow_leg)

    # ── Step 5: View selection — prefer BROAD, fall back to NARROW ──────────
    # Mirrored from SHORT scanner so behaviour is identical between sides.
    active_view      = None
    active_view_data = None

    if _is_view_tradeable_long(broad_data):
        active_view      = "BROAD"
        active_view_data = broad_data
    elif _is_view_tradeable_long(narrow_data):
        active_view      = "NARROW"
        active_view_data = narrow_data

    # Return non-None only if at least one view is APPROACHING / ACTIONABLE /
    # OVERSHOOT (so the UI can display the approaching-zone warning card).
    # WAITING and INVALIDATED across both views → skip entirely.
    has_any_view_in_play = False
    for vd in (broad_data, narrow_data):
        if vd and vd["retracement_status"] in {"ACTIONABLE", "OVERSHOOT", "APPROACHING"}:
            has_any_view_in_play = True
            break

    if not has_any_view_in_play:
        return None   # WAITING / INVALIDATED across both views → nothing to show

    # ── Resolve the active dict for downstream steps ──────────────────────────
    # If no tradeable view yet (only APPROACHING), still use best available
    # for EMA / LTF / trade-plan logic but mark the result as not-yet-tradeable
    # (active_view stays None → result falls into "Other Setups" bucket in UI).
    active = active_view_data or broad_data or narrow_data

    retracement_status = active["retracement_status"]
    fibo_zone          = active["fibo_zone"]
    smart_obs          = active["smart_obs"]
    fvgs               = active["fvgs"]
    sr_levels          = active["sr_levels"]
    current_price      = active["current_price"]
    zone_cls           = active["zone_classification"]
    primary_zone       = active["primary_zone"]   # may be None when APPROACHING

    if not fibo_zone:
        return None  # defensive — should never happen if has_any_view_in_play

    # ── Step 8: Require at least one zone inside Fibo 0.786 area ─────────────
    # Only enforce when a tradeable view exists. APPROACHING views are allowed
    # to pass through with empty zones — they show in "Other Setups" until
    # price retraces deeper.
    if active_view_data is not None:
        if not smart_obs and not fvgs and not sr_levels:
            if "fib_786_zone_top" not in fibo_zone:
                return None

    # ── Step 9: EMA tier ─────────────────────────────────────────────────────
    ema_tier = classify_ema_tier(df_htf, current_price)
    if ema_tier == "SKIP":
        return None

    # ── Step 11: Require at least one actionable zone (tradeable views only) ─
    if active_view_data is not None:
        if not zone_cls.get("any", False):
            return None
        if primary_zone is None:
            return None

    # ── ob_tier: tier of primary zone if it is a smart_ob ────────────────────
    ob_tier: Optional[str] = None
    if primary_zone is not None and primary_zone["type"] == "smart_ob":
        ob_tier = primary_zone["data"].get("tier")  # "LIQUIDITY_SWEEP"|"STRONG"|"REGULAR"

    # ── Step 13: LTF confirmation (defensive against primary_zone=None) ──────
    ltf_result = (
        check_ltf_confirmation(df_ltf, primary_zone)
        if primary_zone is not None
        else {"status": "NONE", "signal_bar": None}
    )
    ltf_confirmation = ltf_result["status"]
    ltf_signal_bar   = ltf_result.get("signal_bar")

    # ── Step 14: 24-variant backtest grid ─────────────────────────────────────
    # v1.2b: the SCAN is FAST by default — it does structure + zone detection
    # only and SKIPS the expensive 24-variant backtest. The full backtest is
    # deferred to the Deep Dive button (backtest.deep_dive_backtest), which
    # runs its own grid on demand for the single coin the user picks.
    low_confidence = len(df_htf) < 100

    _EMPTY_GRID_COLS = [
        "entry_type", "ltf_mode", "tp_R",
        "n_setups", "n_filled", "n_wins",
        "wr", "mean_r", "median_r", "pf", "max_dd_R",
        "avg_entry_price", "avg_sl_pct", "avg_rr_to_tp",
        "recent_check", "trades_raw",
    ]

    if run_backtest:
        # Full path — used only if a caller explicitly asks for the grid.
        variant_grid: pd.DataFrame = run_variant_grid(
            df_htf=df_htf,
            df_ltf=df_ltf,
            structure=structure,
            fibo_zone=fibo_zone,
            smart_obs=smart_obs,
            fvgs=fvgs,
            sr_levels=sr_levels,
        )
        if low_confidence and not variant_grid.empty:
            variant_grid["low_confidence"] = True
        best_variant: Optional[Dict[str, Any]] = None
        if not variant_grid.empty:
            best_variant = variant_grid.iloc[0].to_dict()
    else:
        # Fast scan — no backtest. Empty grid; the UI shows a "Deep Dive for
        # backtest" message and the Deep Dive button runs the real grid.
        variant_grid = pd.DataFrame(columns=_EMPTY_GRID_COLS)
        best_variant = None

    # ── Step 15: Macro flag ───────────────────────────────────────────────────
    fights_macro = btc_regime in {"BEAR"}

    # ── Step 16: Trade plan (defensive against primary_zone=None) ────────────
    trade_plan = (
        build_trade_plan(primary_zone, current_price, df_htf, structure)
        if primary_zone is not None
        else {}
    )

    # ── Step 17: Transparency fields from structure ───────────────────────────
    wick_adjustments        = swings.get("wick_adjustments", [])
    overall_bullish_verified = structure.get("overall_bullish_verified", False)

    # ── Step 18: Build result dict ────────────────────────────────────────────
    return {
        # ── identity ──────────────────────────────────────────────────────────
        "symbol":       symbol,
        "mode":         mode,
        "htf_tf":       htf_interval,
        "ltf_tf":       ltf_interval,
        "btc_regime":   btc_regime,
        "fights_macro": fights_macro,

        # ── structure ─────────────────────────────────────────────────────────
        "structure":                 structure,
        "overall_bullish_verified":  overall_bullish_verified,   # NEW v1.2
        "wick_adjustments":          wick_adjustments,           # NEW v1.2

        # ── hierarchical views (Session 5; now mirrored LONG/SHORT) ───────────
        "view_used":           active_view,                # "BROAD" | "NARROW" | None
        "broad_data":          broad_data,
        "narrow_data":         narrow_data,
        "broad_invalidated":   views.get("broad_invalidated", False),
        "min_bounce_pct_used": min_bounce_pct,
        "retracement_status":  retracement_status,

        # ── fibo zone (NEW v1.2) ───────────────────────────────────────────────
        "fibo_zone":           fibo_zone,
        "atr_multiplier_used": atr_multiplier,

        # ── other zones ───────────────────────────────────────────────────────
        "zones": {
            "smart_obs":  smart_obs,
            "fvgs":       fvgs,
            "fibo":       fibo_zone,        # kept under legacy key for UI compat
            "sr_levels":  sr_levels,
        },

        # ── price + zone classification ───────────────────────────────────────
        "current_price":              current_price,
        "current_zone_classification": zone_cls,
        "primary_zone":               primary_zone,
        "ob_tier":                    ob_tier,           # NEW v1.2
        "ema_tier":                   ema_tier,

        # ── LTF confirmation ──────────────────────────────────────────────────
        "ltf_confirmation": ltf_confirmation,
        "ltf_signal_bar":   ltf_signal_bar,

        # ── backtest (v1.2: variant grid replaces single backtest) ─────────────
        "variant_grid":        variant_grid,      # pd.DataFrame (empty on fast scan)
        "best_variant":        best_variant,       # dict or None (None on fast scan)
        "deep_dive_available": True,               # UI deep-dive button flag

        # ── candle cache for Deep Dive (render.py needs these to run the
        #    full backtest on demand — see render.py render_deep_dive_results)
        "_df_htf_cache": df_htf,
        "_df_ltf_cache": df_ltf,

        # ── trade plan ────────────────────────────────────────────────────────
        "trade_plan": trade_plan,

        # ── metadata ──────────────────────────────────────────────────────────
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# LTF CONFIRMATION
# ============================================================================

def check_ltf_confirmation(
    df_ltf: pd.DataFrame,
    primary_zone: Dict[str, Any],
) -> Dict[str, Any]:
    """
    On the LTF, look for one of these confirmations inside the HTF primary_zone:
      - A bullish CHoCH on LTF (price closed above LTF swing high) WITHIN zone
      - A strong bullish reversal candle (close in upper third, vol >= 1.5x avg)
        AND inside the HTF zone range in last 5 LTF bars

    Args:
        df_ltf: LTF OHLCV DataFrame
        primary_zone: {"type": str, "data": dict}

    Returns:
        {
            "status": "CONFIRMED" | "PENDING" | "NONE",
            "signal_bar": int | None,
            "reason": str,
        }
    """
    _no_zone = {
        "status":     "NONE",
        "signal_bar": None,
        "reason":     "Price not inside HTF zone on LTF.",
    }

    if df_ltf is None or df_ltf.empty or len(df_ltf) < 6:
        return _no_zone

    zone_data = primary_zone.get("data", {})
    zone_type = primary_zone.get("type", "")

    # ── Extract zone bounds from primary_zone type ───────────────────────────
    zone_low, zone_high = _zone_bounds(zone_data, zone_type)
    if zone_low is None or zone_high is None:
        return _no_zone

    # Ensure the most recent LTF bar is inside the HTF zone
    ltf_close = df_ltf["close"].iloc[-1]
    ltf_low   = df_ltf["low"].iloc[-1]
    ltf_high  = df_ltf["high"].iloc[-1]

    price_in_zone = (ltf_low <= zone_high) and (ltf_high >= zone_low)
    if not price_in_zone:
        return _no_zone

    n = len(df_ltf)
    closes  = df_ltf["close"].to_numpy()
    opens   = df_ltf["open"].to_numpy()
    highs   = df_ltf["high"].to_numpy()
    lows    = df_ltf["low"].to_numpy()
    volumes = df_ltf["volume"].to_numpy()

    # ── Test 1: Bullish CHoCH on LTF within HTF zone ─────────────────────────
    # Detect LTF swing highs (simplified: local high over 3-bar window)
    # A bullish CHoCH = close above a prior LTF swing high while price was
    # within the HTF zone at the time.
    ltf_swing_highs = []
    for i in range(2, n - 2):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] \
                and highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]:
            # Only count swing highs inside or near the HTF zone
            if lows[i] <= zone_high * 1.01:
                ltf_swing_highs.append((i, float(highs[i])))

    # Scan forward from each swing high to look for CHoCH close
    for sh_bar, sh_price in ltf_swing_highs:
        for bar_i in range(sh_bar + 1, n):
            if closes[bar_i] > sh_price and lows[bar_i] <= zone_high:
                return {
                    "status":     "CONFIRMED",
                    "signal_bar": bar_i,
                    "reason":     (
                        f"LTF CHoCH: closed above swing high {sh_price:.6g} "
                        f"at LTF bar {bar_i} while inside HTF zone."
                    ),
                }

    # ── Test 2: Strong bullish reversal candle in last 5 LTF bars ────────────
    vol_avg_period = min(20, n - 6)
    check_start    = max(0, n - 5)

    for bar_i in range(check_start, n):
        bar_lo = float(lows[bar_i])
        bar_hi = float(highs[bar_i])
        bar_cl = float(closes[bar_i])
        bar_op = float(opens[bar_i])

        # Must overlap zone
        if bar_lo > zone_high or bar_hi < zone_low:
            continue

        rng = bar_hi - bar_lo
        if rng <= 0:
            continue

        # Close in upper third
        if bar_cl < bar_lo + rng * (2.0 / 3.0):
            continue

        # Bullish candle (close > open)
        if bar_cl <= bar_op:
            continue

        # Volume check
        vol_start = max(0, bar_i - vol_avg_period)
        vol_slice = volumes[vol_start:bar_i]
        if len(vol_slice) == 0:
            continue
        vol_avg = float(np.mean(vol_slice))
        if vol_avg <= 0 or float(volumes[bar_i]) < 1.5 * vol_avg:
            continue

        return {
            "status":     "CONFIRMED",
            "signal_bar": bar_i,
            "reason":     (
                f"Strong bullish reversal candle at LTF bar {bar_i} "
                f"(close in upper third, vol {volumes[bar_i]:.0f} >= 1.5× avg {vol_avg:.0f})."
            ),
        }

    # ── PENDING: price is inside zone but no confirmation yet ─────────────────
    return {
        "status":     "PENDING",
        "signal_bar": None,
        "reason":     "Price is inside HTF zone on LTF; no LTF confirmation signal yet.",
    }


def _zone_bounds(
    zone_data: Dict[str, Any],
    zone_type: str,
) -> tuple:
    """
    Return (zone_low, zone_high) for any primary zone type.
    Returns (None, None) on failure.
    """
    try:
        if zone_type == "smart_ob":
            return float(zone_data["ob_low"]), float(zone_data["ob_high"])

        elif zone_type == "fvg":
            return float(zone_data["bottom"]), float(zone_data["top"])

        elif zone_type == "fibo_786":
            z_bottom = zone_data.get("fib_786_zone_bottom")
            z_top    = zone_data.get("fib_786_zone_top")
            if z_bottom is None or z_top is None:
                fib786 = float(zone_data["fib_786"])
                return fib786 * 0.995, fib786 * 1.005
            return float(z_bottom), float(z_top)

        elif zone_type == "sr":
            sr_price = float(zone_data["price"])
            return sr_price * 0.995, sr_price * 1.005

    except (KeyError, TypeError, ValueError):
        pass

    return None, None


# ============================================================================
# TRADE PLAN BUILDER
# ============================================================================

def build_trade_plan(
    primary_zone: Dict[str, Any],
    current_price: float,
    df_htf: pd.DataFrame,
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute entry, SL, TPs based on the primary zone.

    Entry logic by zone type:
        smart_ob:  entry_price = ob_high; sl = ob_low * 0.998
        fvg:       entry_price = fvg.mid; sl = fvg.bottom * 0.998
        fibo_786:  entry_price = fib_786; sl = fib_786 * 0.985
        sr:        entry_price = sr.price * 1.002; sl = sr.price * 0.985

    TP logic:
        risk = entry_price - sl
        tp1 = entry_price + 2.0 * risk
        tp2 = entry_price + 2.5 * risk
        tp3 = entry_price + 3.0 * risk

    Returns:
        trade_plan dict with keys:
            entry_zone_type, entry_price, sl,
            tp1_R, tp1_price, tp2_R, tp2_price, tp3_R, tp3_price,
            rr_to_tp2
    """
    zone_type = primary_zone["type"]
    zone_data = primary_zone["data"]

    # ── Compute entry and SL by zone type ────────────────────────────────────
    try:
        if zone_type == "smart_ob":
            entry_price = float(zone_data["ob_high"])
            sl          = float(zone_data["ob_low"]) * 0.998

        elif zone_type == "fvg":
            entry_price = float(zone_data["mid"])
            sl          = float(zone_data["bottom"]) * 0.998

        elif zone_type == "fibo_786":
            fib786      = float(zone_data["fib_786"])
            entry_price = fib786
            sl          = fib786 * 0.985

        elif zone_type == "sr":
            sr_price    = float(zone_data["price"])
            entry_price = sr_price * 1.002
            sl          = sr_price * 0.985

        else:
            # Fallback: use current_price with a 1.5% SL
            entry_price = current_price
            sl          = current_price * 0.985

    except (KeyError, TypeError, ValueError) as e:
        print(f"[scanner] build_trade_plan fallback ({zone_type}): {e}")
        entry_price = current_price
        sl          = current_price * 0.985

    # Safety: SL must be strictly below entry
    if sl >= entry_price:
        sl = entry_price * 0.985
    if sl <= 0:
        sl = entry_price * 0.985

    risk = entry_price - sl
    if risk <= 0:
        risk = entry_price * 0.015   # 1.5% fallback

    tp1_price = entry_price + 2.0 * risk
    tp2_price = entry_price + 2.5 * risk
    tp3_price = entry_price + 3.0 * risk

    rr_to_tp2 = (tp2_price - entry_price) / risk if risk > 0 else 2.5

    return {
        "entry_zone_type": zone_type,
        "entry_price":     round(entry_price, 10),
        "sl":              round(sl, 10),
        "tp1_R":           2.0,
        "tp1_price":       round(tp1_price, 10),
        "tp2_R":           2.5,
        "tp2_price":       round(tp2_price, 10),
        "tp3_R":           3.0,
        "tp3_price":       round(tp3_price, 10),
        "rr_to_tp2":       round(rr_to_tp2, 4),
    }


# ============================================================================
# EMA TIER CLASSIFIER
# ============================================================================

def classify_ema_tier(df_htf: pd.DataFrame, current_price: float) -> str:
    """
    Determine bullish-regime tier per spec.

    - STRONG: current_price > EMA50 AND current_price > EMA200
    - MEDIUM: current_price > one of them but not both
    - SKIP:   below both

    If df_htf has < 200 bars: use only EMA 50 → STRONG if above, else SKIP.

    Returns: "STRONG" | "MEDIUM" | "SKIP"
    """
    if len(df_htf) < 52:
        # Not enough data even for EMA50 to be meaningful
        return "SKIP"

    ema50 = calculate_ema(df_htf, 50)
    ema50_val = float(ema50.iloc[-1])

    if len(df_htf) < 200:
        # Short history: use EMA50 only
        if np.isnan(ema50_val):
            return "SKIP"
        if current_price > ema50_val:
            return "STRONG"
        return "SKIP"

    ema200 = calculate_ema(df_htf, 200)
    ema200_val = float(ema200.iloc[-1])

    # Handle NaN (insufficient warm-up)
    ema50_ok  = not np.isnan(ema50_val)
    ema200_ok = not np.isnan(ema200_val)

    above_50  = ema50_ok  and current_price > ema50_val
    above_200 = ema200_ok and current_price > ema200_val

    if above_50 and above_200:
        return "STRONG"
    if above_50 or above_200:
        return "MEDIUM"
    return "SKIP"


# ============================================================================
# PRIMARY ZONE PICKER
# ============================================================================

def pick_primary_zone(
    zone_classification: Dict[str, Any],
    fibo: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    From the dict output of classify_current_price_in_zones, pick the PRIMARY
    zone for trade-planning purposes.

    Priority order v1.2 (highest to lowest):
        1. LIQUIDITY_SWEEP smart_ob  (NEW v1.2 — strongest reversal signal)
        2. STRONG smart_ob
        3. FVG FRESH
        4. REGULAR smart_ob
        5. FVG PARTIAL_<50
        6. Fibo 0.786 (price within zone)
        7. S/R support
        8. FVG PARTIAL_>50

    Args:
        zone_classification: output of classify_current_price_in_zones()
        fibo: the full fibo dict from detect_fibo_levels() — needed so that
              the returned zone data contains the actual fib_786 price.

    Returns:
        {"type": "smart_ob"|"fvg"|"fibo_786"|"sr", "data": <zone_dict>}
        or None if no zone in classification.
    """
    in_obs  = zone_classification.get("in_smart_ob", [])
    in_fvgs = zone_classification.get("in_fvg", [])
    in_fibo = zone_classification.get("in_fibo_786", False)
    in_srs  = zone_classification.get("at_sr_support", [])

    # Priority 1 (NEW v1.2): LIQUIDITY_SWEEP smart_ob — strongest reversal signal
    liq_sweep_obs = [ob for ob in in_obs if ob.get("tier") == "LIQUIDITY_SWEEP"]
    if liq_sweep_obs:
        best = max(liq_sweep_obs, key=lambda x: x.get("freshness_score", 0.0))
        return {"type": "smart_ob", "data": best}

    # Priority 2: STRONG smart_ob
    strong_obs = [ob for ob in in_obs if ob.get("tier") == "STRONG"]
    if strong_obs:
        best = max(strong_obs, key=lambda x: x.get("freshness_score", 0.0))
        return {"type": "smart_ob", "data": best}

    # Priority 3: FRESH fvg
    fresh_fvgs = [f for f in in_fvgs if f.get("status") == "FRESH"]
    if fresh_fvgs:
        best = max(fresh_fvgs, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    # Priority 4: REGULAR smart_ob
    regular_obs = [ob for ob in in_obs if ob.get("tier") == "REGULAR"]
    if regular_obs:
        best = max(regular_obs, key=lambda x: x.get("freshness_score", 0.0))
        return {"type": "smart_ob", "data": best}

    # Priority 5: fvg PARTIAL_<50
    partial_lt50 = [f for f in in_fvgs if f.get("status") == "PARTIAL_<50"]
    if partial_lt50:
        best = max(partial_lt50, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    # Priority 6: fibo_786
    if in_fibo and fibo:
        return {"type": "fibo_786", "data": fibo}

    # Priority 7: sr_support
    if in_srs:
        best = max(in_srs, key=lambda x: x.get("strength", 0.0))
        return {"type": "sr", "data": best}

    # Priority 8: fvg PARTIAL_>50 (least preferred)
    partial_gt50 = [f for f in in_fvgs if f.get("status") == "PARTIAL_>50"]
    if partial_gt50:
        best = max(partial_gt50, key=lambda x: x.get("created_at_bar", 0))
        return {"type": "fvg", "data": best}

    return None
