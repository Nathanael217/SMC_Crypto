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
from qf_smc.structure import detect_swings, classify_structure, is_uptrend_confirmed
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
    "SWING": {"htf": "1d",  "ltf": "4h",  "lookback": 50, "htf_label": "1D", "ltf_label": "4H"},
    "DAY":   {"htf": "4h",  "ltf": "15m", "lookback": 50, "htf_label": "4H", "ltf_label": "15m"},
    "SCALP": {"htf": "1h",  "ltf": "5m",  "lookback": 50, "htf_label": "1H", "ltf_label": "5m"},
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
    import traceback
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

    error_log: List[Dict[str, str]] = []
    skipped_slow: List[str] = []
    total = len(symbols)

    # Worker count: network-bound work benefits from oversubscription, but we
    # cap at 8 to stay friendly to Binance rate limits and avoid thread churn.
    max_workers = min(8, max(1, total))

    # ── Per-coin wall-clock budget ───────────────────────────────────────────
    # A healthy coin completes structure+zone analysis in well under a second
    # once its candles are fetched. The only thing that takes long is a slow /
    # hung Binance API response. If a coin hasn't finished within this many
    # seconds we ABANDON it: stop waiting, mark it skipped, move on. The
    # orphaned worker thread finishes on its own in the background and its
    # result is discarded.
    PER_COIN_TIMEOUT = 12.0  # seconds — generous, but kills true anomalies

    def _worker(sym: str) -> Optional[Dict[str, Any]]:
        """Run one symbol scan. Exceptions propagate to the future."""
        return scan_one_symbol(
            sym, mode, btc_regime,
            atr_multiplier=atr_multiplier,
            run_backtest=run_backtest,
        )

    completed = 0
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        # Submit everything up front. Record submit time per future so we can
        # enforce a per-coin deadline.
        future_to_symbol: Dict[Any, str] = {}
        submit_time: Dict[Any, float] = {}
        now0 = _time.monotonic()
        for sym in symbols:
            fut = executor.submit(_worker, sym)
            future_to_symbol[fut] = sym
            submit_time[fut] = now0  # all submitted ~now; pool runs 8 at a time

        pending = set(future_to_symbol.keys())

        # Poll loop: wait in short slices so we can (a) update progress smoothly
        # and (b) reap any future that has blown past PER_COIN_TIMEOUT.
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)

            # ── Handle finished futures ──────────────────────────────────────
            for future in done:
                symbol = future_to_symbol[future]
                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, total, symbol)
                    except Exception:
                        pass
                try:
                    result = future.result()  # already done — won't block
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[scanner] error scanning {symbol}: {type(e).__name__}: {e}")
                    print(f"[scanner] traceback for {symbol}:\n{tb}")
                    error_log.append({"symbol": symbol, "error": f"{type(e).__name__}: {e}"})

            # ── Reap futures that have exceeded the per-coin budget ──────────
            # A future can't realistically run until earlier ones free a worker
            # slot, so we add grace proportional to how deep the queue still is.
            now = _time.monotonic()
            to_abandon = []
            for future in list(pending):
                age = now - submit_time[future]
                queue_grace = (len(pending) / max_workers) * 2.0
                if age > (PER_COIN_TIMEOUT + queue_grace):
                    to_abandon.append(future)

            for future in to_abandon:
                symbol = future_to_symbol[future]
                completed += 1
                skipped_slow.append(symbol)
                pending.discard(future)
                # Do NOT call future.result() — that would block. Just stop
                # tracking it; the orphaned worker thread finishes on its own
                # (its short 5s HTTP timeout guarantees it ends soon) and the
                # return value is garbage-collected. Try cancel in case it
                # hasn't started (no-op if already running).
                future.cancel()
                if progress_callback:
                    try:
                        progress_callback(completed, total, f"{symbol} (skipped — slow)")
                    except Exception:
                        pass
                print(f"[scanner] SKIPPED {symbol}: exceeded per-coin timeout "
                      f"({PER_COIN_TIMEOUT}s budget) — likely slow/hung API response.")
    finally:
        # Non-blocking shutdown: return to the caller immediately. Any orphaned
        # slow threads finish in the background — their short 5s HTTP timeout
        # means they die within a few seconds and never delay the user.
        executor.shutdown(wait=False)

    # ── Summary logging ──────────────────────────────────────────────────────
    if skipped_slow:
        print(f"[scanner] {len(skipped_slow)} coin(s) skipped for slow response: "
              f"{skipped_slow[:10]}{'...' if len(skipped_slow) > 10 else ''}")

    if not results and (error_log or skipped_slow):
        print(f"[scanner] No setups. errors={len(error_log)} skipped_slow={len(skipped_slow)}")

    return results


# ============================================================================
# SINGLE-SYMBOL PIPELINE
# ============================================================================

def scan_one_symbol(
    symbol: str,
    mode: str,
    btc_regime: str,
    atr_multiplier: float = 0.5,    # NEW v1.2
    run_backtest: bool = False,     # NEW v1.2b — when False, skip 24-variant grid (fast scan)
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
    # HTF fetch depth depends on whether we will run the backtest:
    #   - fast scan (run_backtest=False): 250 bars is plenty for structure
    #     (pivot-5 swings) + zone detection. Smaller payload = faster.
    #   - full scan (run_backtest=True): 500 bars gives the variant grid
    #     meaningful history for the historical replay.
    _htf_limit = 500 if run_backtest else 250
    # Short per-request timeout (5s): a slow/hung coin fails fast here instead
    # of stalling the worker thread. run_scan's per-coin budget then skips it.
    _fetch_timeout = 5.0
    df_htf = _scanner_fetch_candles(
        symbol, htf_interval, limit=_htf_limit, timeout=_fetch_timeout,
    )
    df_ltf = _scanner_fetch_candles(
        symbol, ltf_interval, limit=lookback + 50, timeout=_fetch_timeout,
    )

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

    # ── Step 6: Fibo zone FIRST (v1.2) ───────────────────────────────────────
    # detect_fibo_levels returns {} when current_leg is absent
    fibo_zone = detect_fibo_levels(df_htf, structure, atr_multiplier=atr_multiplier)
    if not fibo_zone:
        return None  # no current leg → can't define zone

    # ── Step 7: Zone detectors WITH Fibo 0.786 filter ────────────────────────
    smart_obs = detect_smart_obs(df_htf, structure, fibo_786_zone=fibo_zone, require_in_zone=True)
    fvgs      = detect_fvgs(df_htf, structure, fibo_786_zone=fibo_zone, require_in_zone=True)
    sr_levels = detect_sr_levels(df_htf, lookback=lookback, fibo_786_zone=fibo_zone, require_in_zone=True)

    # ── Step 8: Require at least one zone inside Fibo 0.786 area ─────────────
    if not smart_obs and not fvgs and not sr_levels:
        # Last-resort: accept if fib_786_zone_top is defined (price may enter
        # the zone itself as the trade trigger — fibo_786 entry variant)
        if "fib_786_zone_top" not in fibo_zone:
            return None
        # Fall through — run_variant_grid will populate only fibo_786 variants

    # ── Step 9: EMA tier ─────────────────────────────────────────────────────
    current_price = float(df_htf["close"].iloc[-1])
    ema_tier = classify_ema_tier(df_htf, current_price)
    if ema_tier == "SKIP":
        return None

    # ── Step 10: Zone classification ─────────────────────────────────────────
    zone_cls = classify_current_price_in_zones(
        current_price, smart_obs, fvgs, fibo_zone, sr_levels
    )

    # ── Step 11: Require at least one actionable zone ────────────────────────
    if not zone_cls.get("any", False):
        return None

    # ── Step 12: Pick primary zone ───────────────────────────────────────────
    primary_zone = pick_primary_zone(zone_cls, fibo=fibo_zone)
    if primary_zone is None:
        return None

    # ── ob_tier: tier of primary zone if it is a smart_ob ────────────────────
    ob_tier: Optional[str] = None
    if primary_zone["type"] == "smart_ob":
        ob_tier = primary_zone["data"].get("tier")  # "LIQUIDITY_SWEEP"|"STRONG"|"REGULAR"

    # ── Step 13: LTF confirmation ─────────────────────────────────────────────
    ltf_result       = check_ltf_confirmation(df_ltf, primary_zone)
    ltf_confirmation = ltf_result["status"]
    ltf_signal_bar   = ltf_result.get("signal_bar")

    # ── Step 14: 24-variant backtest grid ─────────────────────────────────────
    # v1.2b: by default the SCAN is FAST — it skips the backtest entirely.
    # The full 24/72-variant grid is computed only when the user clicks the
    # Deep Dive button (deep_dive_backtest runs its own grid).
    low_confidence = len(df_htf) < 100

    _EMPTY_GRID_COLS = [
        "entry_type", "ltf_mode", "tp_R",
        "n_setups", "n_filled", "n_wins",
        "wr", "mean_r", "median_r", "pf", "max_dd_R",
        "avg_entry_price", "avg_sl_pct", "avg_rr_to_tp",
        "recent_check", "trades_raw",
    ]

    if run_backtest:
        try:
            variant_grid: pd.DataFrame = run_variant_grid(
                df_htf=df_htf,
                df_ltf=df_ltf,
                structure=structure,
                fibo_zone=fibo_zone,
                smart_obs=smart_obs,
                fvgs=fvgs,
                sr_levels=sr_levels,
            )
        except Exception as e:
            import traceback as _tb
            print(f"[scanner] {symbol}: run_variant_grid failed: {type(e).__name__}: {e}")
            print(_tb.format_exc())
            variant_grid = pd.DataFrame(columns=_EMPTY_GRID_COLS)

        if low_confidence and not variant_grid.empty:
            variant_grid["low_confidence"] = True

        best_variant: Optional[Dict[str, Any]] = None
        if not variant_grid.empty:
            best_variant = variant_grid.iloc[0].to_dict()
    else:
        # Fast scan path — no backtest. Empty grid; UI shows "Deep Dive for backtest".
        variant_grid = pd.DataFrame(columns=_EMPTY_GRID_COLS)
        best_variant = None

    # ── Step 15: Macro flag ───────────────────────────────────────────────────
    fights_macro = btc_regime in {"BEAR"}

    # ── Step 16: Trade plan ───────────────────────────────────────────────────
    trade_plan = build_trade_plan(primary_zone, current_price, df_htf, structure)

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
        "variant_grid":        variant_grid,      # pd.DataFrame 24 rows
        "best_variant":        best_variant,       # dict: row with highest PF
        "deep_dive_available": True,               # UI deep-dive button flag

        # ── trade plan ────────────────────────────────────────────────────────
        "trade_plan": trade_plan,

        # ── df cache for deep-dive (NEW v1.2 — required by render.py) ─────────
        "_df_htf_cache": df_htf,
        "_df_ltf_cache": df_ltf,

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
