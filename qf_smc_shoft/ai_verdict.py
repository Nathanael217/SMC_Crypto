"""
qf_smc_short/ai_verdict.py — AI verdict layer for SMC SHORT setups
====================================================================
Builds SHORT-flavored prompt → calls LLM → parses JSON verdict.

Reuses the LLM calling infrastructure from qf_smc.ai_verdict (get_verdict,
_call_groq, _call_anthropic, _parse_verdict_json) since those are direction-
neutral. The only SHORT-specific piece is the prompt template + builder.

Public API:
  - PROMPT_TEMPLATE_V2_SHORT
  - build_smc_short_prompt_v2(...)
  - score_setup_short(result, provider, api_key)
"""

import os
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st

# Reuse LLM calling + JSON parser from LONG package — direction-neutral
from qf_smc.ai_verdict import get_verdict, _parse_verdict_json  # noqa: F401


# ============================================================================
# PROMPT TEMPLATE V2 — SHORT-aware (mirror of LONG v2)
# ============================================================================

PROMPT_TEMPLATE_V2_SHORT = """You are an SMC (Smart Money Concepts) analyst for the QUANTFLOW SMC SHORT system. Assess this SHORT setup using ICT/SMC methodology and output a structured JSON verdict.

══════════════════════════════════════════════════════════════════════════════
SMC METHODOLOGY CONTEXT (SHORT direction — read this first):

The system uses TOP-DOWN ICT methodology, mirrored for short setups:
1. HTF identifies DOWNTREND via LH+LL pattern (lower highs + lower lows)
2. Fibo retracement anchored from MOST RECENT LH (peak) DOWN to MOST RECENT LL (trough)
3. Entry zones ONLY accepted inside Fibo 0.786 ± ATR-based tolerance (the deepest
   retracement up from the trough before structure breaks)
4. Entry types: Bearish Smart OB (best), Bearish FVG (good), Fibo 0.786 (acceptable),
   Classic resistance S/R (weakest)
5. OB freshness: FRESH (never tapped from below) > TESTING (tap 1, still valid) >
   MITIGATED (price already broke above and invalidated it)
6. LIQUIDITY_SWEEP OB = UPPER wick > 50% body → swept buy-stops above then
   reversed down → STRONG bearish signal (traps longs at the top)
7. Wick adjustment: swing points with extreme wicks replaced with next bar's
   close for accurate Fibo anchoring

LTF entry modes:
  A. Limit, no confirmation — pre-place limit, accept wider SL, accept risk
     of being filled then SL'd
  B. LTF confirmed — wait for LTF bearish CHoCH or strong bearish reversal
     candle inside HTF zone, tighter SL

══════════════════════════════════════════════════════════════════════════════
SETUP DETAILS:

Symbol: {symbol}
Mode: {mode} (HTF={htf_label}, LTF={ltf_label})
BTC Regime: {btc_regime}{fights_macro_note}
Scan Time: {scan_timestamp}

MARKET STRUCTURE (bearish):
  State: {state}
  Bearish CHoCH bar: {choch_bar} | Bearish BOS bar: {bos_bar}
  Overall bearish verified: {overall_bearish_verified}

CURRENT LEG (anchor for Fibo, down-leg):
  LH (leg start, peak): bar {leg_start_bar}, price ${leg_start_price:.6f}
  LL (leg low, trough): bar {leg_low_bar}, price ${leg_low_price:.6f}
  Leg range: ${leg_range:.6f} ({leg_range_pct:.2f}%)

FIBONACCI 0.786 ZONE (retracement up from LL toward LH):
  Level: ${fib_786:.6f}
  Tolerance: ATR×{atr_multiplier} = ±${tolerance:.6f}
  Zone: [${fib_786_zone_bottom:.6f}, ${fib_786_zone_top:.6f}]
  Current price: ${current_price:.6f}
  Distance to zone: {dist_to_zone_pct:.2f}% (negative = already inside zone)

ENTRY ZONES IN FIBO 0.786 AREA (bearish):
  Bearish Smart OBs: {n_smart_obs} found
    Best OB: {best_ob_summary}
  Bearish FVGs: {n_fvgs} found
    Best FVG: {best_fvg_summary}
  Resistance S/R: {n_srs} found

WICK ADJUSTMENTS (transparency):
  {wick_adjustments_summary}

LTF CONFIRMATION:
  Current status: {ltf_confirmation_status}
  (CONFIRMED = LTF bearish CHoCH/reversal inside zone | PENDING = inside zone,
   no signal yet | NONE = not in zone)

══════════════════════════════════════════════════════════════════════════════
BACKTEST RESULTS (24 SHORT variants):

TOP 5 BY PROFIT FACTOR:
{top_5_variants_table}

BEST PER ENTRY TYPE:
{best_per_entry_summary}

DEEP-DIVE OVERALL BEST: {best_overall_summary}

══════════════════════════════════════════════════════════════════════════════
YOUR TASK:

Assess this SHORT setup using SMC principles. Consider:
  - Is the LH+LL structure clean and respected (bearish character confirmed)?
  - Is the Fibo 0.786 zone well-defined or noisy (e.g. multiple wick adjustments)?
  - Are there high-quality fresh bearish OBs/FVGs in the zone?
  - Does a LIQUIDITY_SWEEP OB exist? (long upper wick — highest conviction for
    shorts, indicates longs got trapped above before markup down)
  - Is the macro (BTC regime) supportive (BEAR) or fighting (BULL)?
  - Does backtest evidence corroborate the structure?
  - Which entry/management combo is highest expected value for a short?

Output ONLY a JSON object (no markdown):
{{
  "verdict": "TAKE" | "WAIT" | "SKIP",
  "confidence": 0-100,
  "smc_quality_score": 0-100,
  "preferred_entry": "smart_ob" | "fvg" | "fibo_786" | "sr",
  "preferred_ltf_mode": "A" | "B",
  "preferred_tp_R": 2.0 | 2.5 | 3.0,
  "preferred_management": "Fixed" | "BE_at_1R" | "Trailing",
  "reasoning": "<3-5 sentences: explain bearish structural read + backtest read + chosen variant>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "fights_macro_note": "<if BTC bull, address whether setup quality overrides macro>",
  "wick_concerns": "<if wick adjustments occurred or wick-heavy bars in zone, flag>"
}}

DECISION FRAMEWORK (SHORT):
  - TAKE: SMC quality >= 70, best variant PF >= 1.4, recent verdict not WEAKER,
          BTC not strongly bullish (no severe macro fight)
  - WAIT: structure good (>=60 quality) but LTF not confirmed yet, or PF
          marginal, or price not yet in zone
  - SKIP: SMC quality < 60, or best variant PF < 1.0, or recent verdict
          WEAKER, or wick concerns severe, or strongly bullish macro

Shorts in crypto are typically HARDER than longs (long-term upward drift). Be
extra conservative — when in doubt, WAIT or SKIP. The market punishes premature
shorts. A solid short needs CLEAN structure + CONFIRMED ltf + supportive macro
to earn a TAKE."""


# ============================================================================
# PUBLIC API — build_smc_short_prompt_v2
# ============================================================================

def build_smc_short_prompt_v2(
    symbol: str,
    mode: str,
    htf_label: str,
    ltf_label: str,
    btc_regime: str,
    fights_macro: bool,
    structure: Dict,
    fibo_zone: Dict,
    smart_obs: List,
    fvgs: List,
    sr_levels: List,
    wick_adjustments: List,
    ltf_confirmation: str,
    variant_grid: pd.DataFrame,
    best_overall: Dict,
    current_price: float,
    scan_timestamp: str,
) -> str:
    """
    Build the SHORT v2 SMC-aware prompt.

    Mirror of qf_smc.ai_verdict.build_smc_prompt_v2 with:
      - "bearish_verified" instead of "bullish_verified"
      - LH/LL anchor labels instead of HL/HH
      - Bearish OB / FVG framing
      - "leg_low_*" keys in structure instead of "leg_high_*"
    """

    # ── Format top 5 variants table ──────────────────────────────────────────
    if isinstance(variant_grid, pd.DataFrame) and not variant_grid.empty:
        top_5 = variant_grid.head(5)
        top_5_table = "\n".join([
            f"  {str(row.get('entry_type', '?')):10} | {str(row.get('ltf_mode', '?'))} "
            f"| TP{float(row.get('tp_R', 0)):.1f}R "
            f"| WR {float(row.get('wr', 0)):.0%} "
            f"| PF {float(row.get('pf', 0)):.2f} "
            f"| n={row.get('n_setups', 0)}"
            for _, row in top_5.iterrows()
        ])
    else:
        top_5_table = "  (no variant data)"

    # ── Best per entry type ──────────────────────────────────────────────────
    best_per_entry_lines = []
    if isinstance(variant_grid, pd.DataFrame) and not variant_grid.empty:
        for et in ["smart_ob", "fvg", "fibo_786", "sr"]:
            sub = variant_grid[variant_grid["entry_type"] == et]
            if not sub.empty:
                r = sub.iloc[0]
                best_per_entry_lines.append(
                    f"  {et:10} → LTF {r.get('ltf_mode', '?')} TP{float(r.get('tp_R', 0)):.1f}R: "
                    f"PF {float(r.get('pf', 0)):.2f}, mean_r {float(r.get('mean_r', 0)):+.3f}, "
                    f"n={r.get('n_setups', 0)}"
                )
    best_per_entry_summary = "\n".join(best_per_entry_lines) if best_per_entry_lines else "  (no entries with setups)"

    # ── Wick adjustments summary ─────────────────────────────────────────────
    n_wick_adj = len(wick_adjustments) if wick_adjustments else 0
    if n_wick_adj == 0:
        wick_summary = "None (all swing points used as-is)"
    else:
        wick_summary = f"{n_wick_adj} adjustment(s) made (extreme wicks replaced with next-bar high/low)"

    # ── Distance to zone ─────────────────────────────────────────────────────
    zone_top    = float(fibo_zone.get("fib_786_zone_top", 0))
    zone_bottom = float(fibo_zone.get("fib_786_zone_bottom", 0))
    if current_price > zone_top:
        dist_pct = ((zone_top - current_price) / current_price) * 100
    elif current_price < zone_bottom:
        dist_pct = ((zone_bottom - current_price) / current_price) * 100
    else:
        dist_pct = 0.0

    # ── Best OB summary (bearish) ────────────────────────────────────────────
    if smart_obs:
        ob0 = smart_obs[0]
        best_ob_summary = (
            f"{ob0.get('tier', '?')} @ ${float(ob0.get('ob_low', 0)):.6f} "
            f"(vol×{float(ob0.get('volume_mult', 0)):.1f}, "
            f"status={ob0.get('status', '?')}, "
            f"upper_wick_ratio={float(ob0.get('wick_upper_ratio', 0)):.2f})"
        )
    else:
        best_ob_summary = "None"

    if fvgs:
        fvg0 = fvgs[0]
        best_fvg_summary = (
            f"{fvg0.get('status', '?')} "
            f"${float(fvg0.get('bottom', 0)):.6f}-${float(fvg0.get('top', 0)):.6f} "
            f"(fill {float(fvg0.get('fill_pct', 0)) * 100:.0f}%)"
        )
    else:
        best_fvg_summary = "None"

    # ── best_overall_summary ────────────────────────────────────────────────
    if best_overall:
        best_overall_summary = (
            f"{best_overall.get('entry_type', '?')} "
            f"LTF-{best_overall.get('ltf_mode', '?')} "
            f"TP{float(best_overall.get('tp_R', 0)):.1f}R: "
            f"PF {float(best_overall.get('pf', 0)):.2f}"
        )
    else:
        best_overall_summary = "N/A"

    # ── Fibo fields ──────────────────────────────────────────────────────────
    fib_786       = float(fibo_zone.get("fib_786", 0))
    atr_mult      = float(fibo_zone.get("atr_multiplier", 0.5))
    atr_used      = float(fibo_zone.get("atr_used", 0))
    tolerance     = atr_used * atr_mult
    leg_range     = float(fibo_zone.get("leg_range", 0))
    anchor_low    = float(fibo_zone.get("anchor_low", max(fib_786, 1e-9)))
    leg_range_pct = (leg_range / max(anchor_low, 1e-9)) * 100

    # ── Structure fields (note SHORT keys: leg_low_* not leg_high_*) ─────────
    current_leg = structure.get("current_leg") or {}

    return PROMPT_TEMPLATE_V2_SHORT.format(
        # Header
        symbol=symbol,
        mode=mode,
        htf_label=htf_label,
        ltf_label=ltf_label,
        btc_regime=btc_regime,
        fights_macro_note=" [SHORT vs BULL — fights macro]" if fights_macro else "",
        scan_timestamp=scan_timestamp,
        # Structure
        state=structure.get("state", "?"),
        choch_bar=structure.get("choch_bar", "-"),
        bos_bar=structure.get("bos_bar", "-"),
        overall_bearish_verified=structure.get("overall_bearish_verified", "unknown"),
        # Current leg (LH → LL for SHORT)
        leg_start_bar=current_leg.get("leg_start_bar", "-"),
        leg_start_price=float(current_leg.get("leg_start_price", 0)),
        leg_low_bar=current_leg.get("leg_low_bar", "-"),
        leg_low_price=float(current_leg.get("leg_low_price", 0)),
        leg_range=leg_range,
        leg_range_pct=leg_range_pct,
        # Fibo
        fib_786=fib_786,
        atr_multiplier=atr_mult,
        tolerance=tolerance,
        fib_786_zone_top=zone_top,
        fib_786_zone_bottom=zone_bottom,
        current_price=current_price,
        dist_to_zone_pct=dist_pct,
        # Zones
        n_smart_obs=len(smart_obs) if smart_obs else 0,
        best_ob_summary=best_ob_summary,
        n_fvgs=len(fvgs) if fvgs else 0,
        best_fvg_summary=best_fvg_summary,
        n_srs=len(sr_levels) if sr_levels else 0,
        # Wick + LTF
        wick_adjustments_summary=wick_summary,
        ltf_confirmation_status=ltf_confirmation,
        # Backtest
        top_5_variants_table=top_5_table,
        best_per_entry_summary=best_per_entry_summary,
        best_overall_summary=best_overall_summary,
    )


# ============================================================================
# Convenience wrapper — score_setup_short
# ============================================================================

def score_setup_short(
    result: Dict[str, Any],
    provider: str = "groq",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-call helper: takes a SHORT scanner result dict, builds the prompt,
    calls the LLM, and returns the parsed verdict.

    NOTE: This is NOT called by run_scan_short — too slow for batch scans.
    Call this only from the Deep Dive button or manual analyzer.
    """
    from datetime import datetime as _dt

    symbol            = result.get("symbol", "?")
    mode              = result.get("mode", "?")
    htf_tf            = str(result.get("htf_tf", "?"))
    ltf_tf            = str(result.get("ltf_tf", "?"))
    btc_regime        = str(result.get("btc_regime", "UNKNOWN"))
    fights_macro      = bool(result.get("fights_macro", False))
    structure         = result.get("structure", {})
    fibo_zone         = result.get("fibo_zone", {})
    zones             = result.get("zones", {})
    smart_obs         = zones.get("smart_obs", [])
    fvgs              = zones.get("fvgs", [])
    sr_levels         = zones.get("sr_levels", [])
    wick_adjustments  = result.get("wick_adjustments", [])
    ltf_confirmation  = str(result.get("ltf_confirmation", "NONE"))
    variant_grid      = result.get("variant_grid", pd.DataFrame())
    best_overall      = result.get("best_variant") or {}
    current_price     = float(result.get("current_price", 0.0))

    prompt = build_smc_short_prompt_v2(
        symbol=symbol,
        mode=mode,
        htf_label=htf_tf,
        ltf_label=ltf_tf,
        btc_regime=btc_regime,
        fights_macro=fights_macro,
        structure=structure,
        fibo_zone=fibo_zone,
        smart_obs=smart_obs,
        fvgs=fvgs,
        sr_levels=sr_levels,
        wick_adjustments=wick_adjustments,
        ltf_confirmation=ltf_confirmation,
        variant_grid=variant_grid,
        best_overall=best_overall,
        current_price=current_price,
        scan_timestamp=_dt.utcnow().isoformat(),
    )

    return get_verdict(prompt, provider=provider, api_key=api_key)
