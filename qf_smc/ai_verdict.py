"""
qf_smc/ai_verdict.py — AI verdict layer for SMC Long setups
============================================================
Builds prompt → calls LLM → parses JSON verdict.
Supports Groq (primary) and Anthropic (fallback).

Public API (v1.1 — preserved for backward compat):
  - build_smc_prompt(...)
  - get_verdict(prompt, provider, api_key, model, timeout)
  - score_setup(result)

Public API (v1.2 additions):
  - PROMPT_TEMPLATE_V2
  - build_smc_prompt_v2(...)
  - score_setup_v2(result, provider, api_key)
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st


# ============================================================================
# PROMPT TEMPLATE v1.1 (preserved — do not modify)
# ============================================================================

PROMPT_TEMPLATE = """You are an analyst for the QUANTFLOW SMC Long system. Your job is to assess a long-only setup on a crypto altcoin perpetual and output a structured JSON verdict.

SETUP CONTEXT:

Symbol: {symbol}
Mode: {mode} (HTF={htf_label}, LTF={ltf_label})
BTC Regime: {btc_regime}{fights_macro_note}

MARKET STRUCTURE:
State: {state}
CHoCH bar index: {choch_bar}
BOS bar index: {bos_bar}
Current leg: from price ${leg_start_price:.6f} to ${leg_high_price:.6f}

ENTRY ZONE:
Primary zone type: {primary_zone_type}
Tier (if Smart OB): {ob_tier}
FVG status (if FVG): {fvg_status}
Price within Fibo 0.786 zone: {in_fibo_786}
At S/R support (if SR): touches={sr_touches}

CURRENT PRICE: ${current_price:.6f}

LTF CONFIRMATION: {ltf_confirmation}

BACKTEST EVIDENCE:
Per-coin (this coin's history): WR {wr_pc:.1%} | PF {pf_pc:.2f} | mean R {mr_pc:+.3f} (n={n_pc})
Universe baseline (top alts):    WR {wr_un:.1%} | PF {pf_un:.2f} | mean R {mr_un:+.3f} (n={n_un})
Bayesian blended:                WR {wr_bl:.1%} | PF {pf_bl:.2f} | mean R {mr_bl:+.3f} (n_eff={n_bl})

RECENT-vs-EARLIER CHECK:
Earlier period ({earlier_label}): mean R {mr_e:+.3f} | PF {pf_e:.2f} (n={n_e})
Recent period ({recent_label}):   mean R {mr_r:+.3f} | PF {pf_r:.2f} (n={n_r})
Verdict: {recent_verdict}

TRADE PLAN:
Entry: ${entry:.6f}
SL: ${sl:.6f}  (risk = {risk_pct:.2f}%)
TP1 (2R): ${tp1:.6f}
TP2 (2.5R): ${tp2:.6f}
TP3 (3R): ${tp3:.6f}
R:R to TP2: {rr:.2f}

YOUR TASK:
Assess this setup. Output ONLY a JSON object (no markdown, no preamble, no postamble):

{{
  "verdict": "TAKE" | "WAIT" | "SKIP",
  "confidence": 0-100,
  "reasoning": "<2-3 sentences explaining your reasoning>",
  "key_risks": ["<risk 1>", "<risk 2>"]
}}

DECISION FRAMEWORK:
- TAKE: blended PF >= 1.3, recent verdict not WEAKER, LTF confirmed or pending, BTC macro not strongly against
- WAIT: LTF not confirmed yet, OR PF marginal (1.0-1.3), OR mixed signals
- SKIP: blended PF < 1.0, OR recent verdict WEAKER, OR setup quality low, OR BEAR macro + weak edge

Be honest. If evidence is thin, recommend WAIT or SKIP — don't force a TAKE."""


# ============================================================================
# PROMPT TEMPLATE v2 — SMC-context-aware (v1.2 new)
# ============================================================================

PROMPT_TEMPLATE_V2 = """You are an SMC (Smart Money Concepts) analyst for the QUANTFLOW SMC Long system. Assess this LONG setup using ICT/SMC methodology and output a structured JSON verdict.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
SMC METHODOLOGY CONTEXT (read this first):

The system uses TOP-DOWN ICT methodology:
1. HTF identifies trend via HL+HH pattern (higher lows + higher highs)
2. Fibo retracement anchored from MOST RECENT HL to MOST RECENT HH
3. Entry zones ONLY accepted inside Fibo 0.786 \u00b1 ATR-based tolerance
4. Entry types: Smart OB (best), FVG (good), Fibo 0.786 (acceptable), Classic S/R (weakest)
5. OB freshness: FRESH (never tapped) > TESTING (tap 1, still valid) > MITIGATED (rejected)
6. LIQUIDITY_SWEEP OB = lower wick > 50% body \u2192 swept stops then reversed \u2192 STRONG bullish signal
7. Wick adjustment: swing points with extreme wicks are replaced with next bar's close for accuracy

LTF entry modes:
  A. Limit, no confirmation \u2014 pre-place limit, accept wider SL, accept risk of being filled then SL'd
  B. LTF confirmed \u2014 wait for LTF CHoCH/strong reversal candle inside HTF zone, tighter SL

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
SETUP DETAILS:

Symbol: {symbol}
Mode: {mode} (HTF={htf_label}, LTF={ltf_label})
BTC Regime: {btc_regime}{fights_macro_note}
Scan Time: {scan_timestamp}

MARKET STRUCTURE:
  State: {state}
  CHoCH bar: {choch_bar} | BOS bar: {bos_bar}
  Overall bullish verified: {overall_bullish_verified}

CURRENT LEG (anchor for Fibo):
  HL (leg start): bar {leg_start_bar}, price ${leg_start_price:.6f}
  HH (leg high):  bar {leg_high_bar}, price ${leg_high_price:.6f}
  Leg range: ${leg_range:.6f} ({leg_range_pct:.2f}%)

FIBONACCI 0.786 ZONE:
  Level: ${fib_786:.6f}
  Tolerance: ATR\u00d7{atr_multiplier} = \u00b1${tolerance:.6f}
  Zone: [${fib_786_zone_bottom:.6f}, ${fib_786_zone_top:.6f}]
  Current price: ${current_price:.6f}
  Distance to zone: {dist_to_zone_pct:.2f}% (negative = already inside zone)

ENTRY ZONES IN FIBO 0.786 AREA:
  Smart OBs: {n_smart_obs} found
    Best OB: {best_ob_summary}
  FVGs: {n_fvgs} found
    Best FVG: {best_fvg_summary}
  S/R supports: {n_srs} found

WICK ADJUSTMENTS (transparency):
  {wick_adjustments_summary}

LTF CONFIRMATION:
  Current status: {ltf_confirmation_status}
  (CONFIRMED = LTF CHoCH inside zone | PENDING = inside zone, no signal yet | NONE = not in zone)

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
BACKTEST RESULTS (24 variants):

TOP 5 BY PROFIT FACTOR:
{top_5_variants_table}

BEST PER ENTRY TYPE:
{best_per_entry_summary}

DEEP-DIVE OVERALL BEST: {best_overall_summary}

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
YOUR TASK:

Assess this setup using SMC principles. Consider:
  - Is the HL+HH structure clean and respected?
  - Is the Fibo 0.786 zone well-defined or noisy (e.g. multiple wick adjustments)?
  - Are there high-quality fresh OBs/FVGs in the zone?
  - Does LIQUIDITY_SWEEP OB exist? (these are highest-conviction)
  - Is the macro (BTC regime) supportive or fighting?
  - Does backtest evidence corroborate the structure?
  - Which entry/management combo is highest expected value?

Output ONLY a JSON object (no markdown):
{{
  "verdict": "TAKE" | "WAIT" | "SKIP",
  "confidence": 0-100,
  "smc_quality_score": 0-100,
  "preferred_entry": "smart_ob" | "fvg" | "fibo_786" | "sr",
  "preferred_ltf_mode": "A" | "B",
  "preferred_tp_R": 2.0 | 2.5 | 3.0,
  "preferred_management": "Fixed" | "BE_at_1R" | "Trailing",
  "reasoning": "<3-5 sentences: explain structural read + backtest read + chosen variant>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "fights_macro_note": "<if BTC bear, address whether setup quality overrides macro>",
  "wick_concerns": "<if wick adjustments occurred or wick-heavy bars in zone, flag>"
}}

DECISION FRAMEWORK:
  - TAKE: SMC quality >= 70, best variant PF >= 1.4, recent verdict not WEAKER, BTC not strongly opposing
  - WAIT: structure good (>=60 quality) but LTF not confirmed, or PF marginal, or price not yet in zone
  - SKIP: SMC quality < 60, or best variant PF < 1.0, or recent verdict WEAKER, or wick concerns severe

Be honest. If evidence is thin or contradictory, recommend WAIT/SKIP."""


# ============================================================================
# MODE TIMEFRAME LABELS  (mirror MODE_CONFIG from scanner.py)
# ============================================================================

_MODE_TF_LABELS: Dict[str, Dict[str, str]] = {
    "SWING": {"htf_label": "1D",  "ltf_label": "4H"},
    "DAY":   {"htf_label": "4H",  "ltf_label": "15m"},
    "SCALP": {"htf_label": "1H",  "ltf_label": "5m"},
}


# ============================================================================
# PUBLIC API v1.1 — build_smc_prompt (preserved unchanged)
# ============================================================================

def build_smc_prompt(
    symbol: str,
    mode: str,
    structure: Dict[str, Any],
    zones: Dict[str, Any],
    current_zone_classification: Dict[str, Any],
    ltf_confirmation: str,
    backtest: Dict[str, Any],
    btc_regime: str,
    fights_macro: bool,
    trade_plan: Dict[str, Any],
) -> str:
    """
    Build a structured prompt for the LLM (v1.1 — preserved for backward compat).

    Args:
        symbol: e.g. "BTCUSDT"
        mode: "SWING" | "DAY" | "SCALP"
        structure: dict from classify_structure (has state, choch_bar, bos_bar, current_leg)
        zones: dict with smart_obs/fvgs/fibo/sr_levels lists
        current_zone_classification: dict from classify_current_price_in_zones
        ltf_confirmation: "CONFIRMED" | "PENDING" | "NONE"
        backtest: dict with per_coin/universe/blended/recent_check
        btc_regime: "BULL" | "CHOP" | "BEAR" | "UNKNOWN"
        fights_macro: bool — True if long setup vs BEAR macro
        trade_plan: dict with entry/sl/tp1/tp2/tp3/rr

    Returns:
        A string prompt (typically 1500-2500 chars) ready to send to LLM.
    """

    # ── Mode TF labels ────────────────────────────────────────────────────────
    tf_labels = _MODE_TF_LABELS.get(mode, {"htf_label": "?", "ltf_label": "?"})
    htf_label = tf_labels["htf_label"]
    ltf_label = tf_labels["ltf_label"]

    # ── fights_macro note ─────────────────────────────────────────────────────
    fights_macro_note = " ⚠ FIGHTS MACRO (long vs BEAR BTC)" if fights_macro else ""

    # ── Structure fields ──────────────────────────────────────────────────────
    state     = structure.get("state", "UNKNOWN")
    choch_bar = structure.get("choch_bar", "N/A")
    bos_bar   = structure.get("bos_bar", "N/A")
    current_leg = structure.get("current_leg") or {}
    leg_start_price = float(current_leg.get("leg_start_price", 0.0))
    leg_high_price  = float(current_leg.get("leg_high_price", 0.0))

    # ── Zone classification ───────────────────────────────────────────────────
    in_obs  = current_zone_classification.get("in_smart_ob", [])
    in_fvgs = current_zone_classification.get("in_fvg", [])
    in_fibo = current_zone_classification.get("in_fibo_786", False)
    in_srs  = current_zone_classification.get("at_sr_support", [])

    # Determine primary zone type label
    if in_obs:
        primary_zone_type = "Smart Order Block"
    elif in_fvgs:
        primary_zone_type = "Fair Value Gap"
    elif in_fibo:
        primary_zone_type = "Fibonacci 0.786"
    elif in_srs:
        primary_zone_type = "S/R Level"
    else:
        primary_zone_type = "None"

    # OB tier
    if in_obs:
        tiers = [ob.get("tier", "UNKNOWN") for ob in in_obs]
        ob_tier = "STRONG" if "STRONG" in tiers else tiers[0] if tiers else "N/A"
    else:
        ob_tier = "N/A"

    # FVG status
    if in_fvgs:
        statuses = [fvg.get("status", "UNKNOWN") for fvg in in_fvgs]
        fvg_status = statuses[0] if statuses else "N/A"
    else:
        fvg_status = "N/A"

    # SR touches
    sr_touches = in_srs[0].get("touches", "N/A") if in_srs else "N/A"

    # ── Current price ─────────────────────────────────────────────────────────
    entry_price = float(trade_plan.get("entry_price", 0.0))
    sl          = float(trade_plan.get("sl", 0.0))
    tp1_price   = float(trade_plan.get("tp1_price", 0.0))
    tp2_price   = float(trade_plan.get("tp2_price", 0.0))
    tp3_price   = float(trade_plan.get("tp3_price", 0.0))
    rr_to_tp2   = float(trade_plan.get("rr_to_tp2", 2.5))

    risk_pct = ((entry_price - sl) / entry_price * 100.0) if entry_price > 0 else 0.0
    current_price = entry_price  # best proxy available from trade_plan

    # ── Backtest fields ───────────────────────────────────────────────────────
    pc = backtest.get("per_coin", {})
    un = backtest.get("universe", {})
    bl = backtest.get("blended", {})
    rc = backtest.get("recent_check", {})

    wr_pc = float(pc.get("wr", 0.0))
    pf_pc = float(pc.get("pf", 0.0))
    mr_pc = float(pc.get("mean_r", 0.0))
    n_pc  = int(pc.get("n_setups", 0))

    wr_un = float(un.get("wr", 0.0))
    pf_un = float(un.get("pf", 0.0))
    mr_un = float(un.get("mean_r", 0.0))
    n_un  = int(un.get("n_setups", 0))

    wr_bl = float(bl.get("wr", 0.0))
    pf_bl = float(bl.get("pf", 0.0))
    mr_bl = float(bl.get("mean_r", 0.0))
    n_bl  = int(bl.get("n_effective", bl.get("n_setups", 0)))

    earlier_label  = rc.get("earlier_period_label", "N/A")
    recent_label   = rc.get("recent_period_label",  "N/A")
    recent_verdict = rc.get("verdict", "UNKNOWN")

    earlier_stats = rc.get("earlier_stats", {})
    recent_stats  = rc.get("recent_stats",  {})

    mr_e = float(earlier_stats.get("mean_r", 0.0))
    pf_e = float(earlier_stats.get("pf", 0.0))
    n_e  = int(earlier_stats.get("n", 0))

    mr_r = float(recent_stats.get("mean_r", 0.0))
    pf_r = float(recent_stats.get("pf", 0.0))
    n_r  = int(recent_stats.get("n", 0))

    # ── Render prompt ─────────────────────────────────────────────────────────
    return PROMPT_TEMPLATE.format(
        symbol=symbol,
        mode=mode,
        htf_label=htf_label,
        ltf_label=ltf_label,
        btc_regime=btc_regime,
        fights_macro_note=fights_macro_note,
        state=state,
        choch_bar=choch_bar,
        bos_bar=bos_bar,
        leg_start_price=leg_start_price,
        leg_high_price=leg_high_price,
        primary_zone_type=primary_zone_type,
        ob_tier=ob_tier,
        fvg_status=fvg_status,
        in_fibo_786=in_fibo,
        sr_touches=sr_touches,
        current_price=current_price,
        ltf_confirmation=ltf_confirmation,
        wr_pc=wr_pc, pf_pc=pf_pc, mr_pc=mr_pc, n_pc=n_pc,
        wr_un=wr_un, pf_un=pf_un, mr_un=mr_un, n_un=n_un,
        wr_bl=wr_bl, pf_bl=pf_bl, mr_bl=mr_bl, n_bl=n_bl,
        earlier_label=earlier_label, mr_e=mr_e, pf_e=pf_e, n_e=n_e,
        recent_label=recent_label,   mr_r=mr_r, pf_r=pf_r, n_r=n_r,
        recent_verdict=recent_verdict,
        entry=entry_price,
        sl=sl,
        risk_pct=risk_pct,
        tp1=tp1_price,
        tp2=tp2_price,
        tp3=tp3_price,
        rr=rr_to_tp2,
    )


# ============================================================================
# PUBLIC API v1.2 — build_smc_prompt_v2
# ============================================================================

def build_smc_prompt_v2(
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
    variant_grid: pd.DataFrame,    # 24-row or 72-row DataFrame from backtest
    best_overall: Dict,
    current_price: float,
    scan_timestamp: str,
) -> str:
    """
    Build the v2 SMC-aware prompt.

    Substitutes all variables into PROMPT_TEMPLATE_V2 from spec §2.8.

    Special formatting:
        - top_5_variants_table: from variant_grid.head(5) formatted as text table
        - best_per_entry_summary: best PF row per entry_type, formatted
        - wick_adjustments_summary: human-readable count + brief
        - dist_to_zone_pct: computed from current_price vs fibo_zone
    """

    # ── Format top 5 variants table ───────────────────────────────────────────
    top_5 = variant_grid.head(5) if isinstance(variant_grid, pd.DataFrame) and not variant_grid.empty else pd.DataFrame()
    if not top_5.empty:
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

    # ── Best per entry type ───────────────────────────────────────────────────
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

    # ── Wick adjustments summary ──────────────────────────────────────────────
    n_wick_adj = len(wick_adjustments) if wick_adjustments else 0
    if n_wick_adj == 0:
        wick_summary = "None (all swing points used as-is)"
    else:
        wick_summary = f"{n_wick_adj} adjustment(s) made (extreme wicks replaced with next-bar high/low)"

    # ── Distance to zone ─────────────────────────────────────────────────────
    zone_top = float(fibo_zone.get("fib_786_zone_top", 0))
    zone_bottom = float(fibo_zone.get("fib_786_zone_bottom", 0))
    if current_price > zone_top:
        dist_pct = ((zone_top - current_price) / current_price) * 100  # negative: above zone
    elif current_price < zone_bottom:
        dist_pct = ((zone_bottom - current_price) / current_price) * 100  # positive: below zone
    else:
        dist_pct = 0.0  # inside zone

    # ── Summaries for best OB / FVG ───────────────────────────────────────────
    if smart_obs:
        ob0 = smart_obs[0]
        best_ob_summary = (
            f"{ob0.get('tier', '?')} @ ${float(ob0.get('ob_high', 0)):.6f} "
            f"(vol\u00d7{float(ob0.get('volume_mult', 0)):.1f}, "
            f"status={ob0.get('status', '?')}, "
            f"wick_sweep={ob0.get('is_liquidity_sweep', False)})"
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

    # ── best_overall_summary ──────────────────────────────────────────────────
    if best_overall:
        best_overall_summary = (
            f"{best_overall.get('entry_type', '?')} "
            f"LTF-{best_overall.get('ltf_mode', '?')} "
            f"TP{float(best_overall.get('tp_R', 0)):.1f}R: "
            f"PF {float(best_overall.get('pf', 0)):.2f}"
        )
    else:
        best_overall_summary = "N/A"

    # ── Fibo fields ───────────────────────────────────────────────────────────
    fib_786     = float(fibo_zone.get("fib_786", 0))
    atr_mult    = float(fibo_zone.get("atr_multiplier", 0.5))
    atr_used    = float(fibo_zone.get("atr_used", 0))
    tolerance   = atr_used * atr_mult
    leg_range   = float(fibo_zone.get("leg_range", 0))
    anchor_low  = float(fibo_zone.get("anchor_low", max(fib_786, 1e-9)))
    leg_range_pct = (leg_range / max(anchor_low, 1e-9)) * 100

    # ── Structure fields ──────────────────────────────────────────────────────
    current_leg = structure.get("current_leg") or {}

    return PROMPT_TEMPLATE_V2.format(
        # Header
        symbol=symbol,
        mode=mode,
        htf_label=htf_label,
        ltf_label=ltf_label,
        btc_regime=btc_regime,
        fights_macro_note=" [LONG vs BEAR — fights macro]" if fights_macro else "",
        scan_timestamp=scan_timestamp,
        # Structure
        state=structure.get("state", "?"),
        choch_bar=structure.get("choch_bar", "-"),
        bos_bar=structure.get("bos_bar", "-"),
        overall_bullish_verified=structure.get("overall_bullish_verified", "unknown"),
        # Current leg
        leg_start_bar=current_leg.get("leg_start_bar", "-"),
        leg_start_price=float(current_leg.get("leg_start_price", 0)),
        leg_high_bar=current_leg.get("leg_high_bar", "-"),
        leg_high_price=float(current_leg.get("leg_high_price", 0)),
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
# INTERNAL — API key resolution
# ============================================================================

def _resolve_api_key(provider: str, explicit: Optional[str]) -> Optional[str]:
    """
    Priority order:
    1. explicit arg if provided
    2. st.session_state["{provider}_api_key"]
    3. environment variable {PROVIDER}_API_KEY
    """
    if explicit:
        return explicit
    key_name = f"{provider}_api_key"
    if key_name in st.session_state and st.session_state[key_name]:
        return st.session_state[key_name]
    env_name = f"{provider.upper()}_API_KEY"
    return os.environ.get(env_name)


# ============================================================================
# INTERNAL — LLM callers
# ============================================================================

def _call_groq(
    prompt: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    timeout: int = 20,
) -> str:
    """Returns raw response text from Groq."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_anthropic(
    prompt: str,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout: int = 20,
) -> str:
    """Returns raw response text from Anthropic."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


# ============================================================================
# INTERNAL — JSON parser (resilient)
# ============================================================================

def _parse_verdict_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Tolerantly parse the verdict JSON from the LLM response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    - JSON embedded in surrounding text (find the first {...} block)
    - Missing fields (returns None if essential fields missing)
    """
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        content_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                content_lines.append(line)
        text = "\n".join(content_lines).strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Find the first { ... } block
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return None
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    # Validate essential fields
    if "verdict" not in obj or obj["verdict"] not in {"TAKE", "WAIT", "SKIP"}:
        return None
    if "reasoning" not in obj:
        return None

    # Fill defaults for optional fields (v1.1)
    obj.setdefault("confidence", 50)
    obj.setdefault("key_risks", [])

    # Fill defaults for v1.2 new fields (only present in v2 responses)
    obj.setdefault("smc_quality_score", None)
    obj.setdefault("preferred_entry", None)
    obj.setdefault("preferred_ltf_mode", None)
    obj.setdefault("preferred_tp_R", None)
    obj.setdefault("preferred_management", None)
    obj.setdefault("fights_macro_note", "")
    obj.setdefault("wick_concerns", "")

    return obj


# ============================================================================
# PUBLIC API — get_verdict (v1.1 preserved unchanged)
# ============================================================================

def get_verdict(
    prompt: str,
    provider: str = "groq",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: int = 20,
) -> Dict[str, Any]:
    """
    Send prompt to LLM and parse the verdict response.

    Args:
        prompt: output of build_smc_prompt() or build_smc_prompt_v2()
        provider: "groq" | "anthropic" | "auto"
                  auto tries groq first, falls back to anthropic
        api_key: optional explicit key. If None, reads from:
                 - st.session_state.get("{provider}_api_key")
                 - env var {PROVIDER}_API_KEY as fallback
        model: optional model override
        timeout_seconds: HTTP timeout

    Returns:
        {
            "verdict":                "TAKE" | "WAIT" | "SKIP",
            "confidence":             int (0-100),
            "reasoning":              str,
            "key_risks":              List[str],
            "smc_quality_score":      int | None  (v1.2, from v2 prompt only)
            "preferred_entry":        str | None  (v1.2)
            "preferred_ltf_mode":     str | None  (v1.2)
            "preferred_tp_R":         float | None (v1.2)
            "preferred_management":   str | None  (v1.2)
            "fights_macro_note":      str         (v1.2)
            "wick_concerns":          str         (v1.2)
            "provider_used":          str,
            "model_used":             str,
            "ai_error":               Optional[str],
        }
    """
    _defaults = {
        "groq":      "llama-3.3-70b-versatile",
        "anthropic": "claude-haiku-4-5-20251001",
    }

    # ── Auto-fallback mode ────────────────────────────────────────────────────
    if provider == "auto":
        try:
            return get_verdict(prompt, "groq", api_key=None, model=model,
                               timeout_seconds=timeout_seconds)
        except Exception:
            pass
        try:
            return get_verdict(prompt, "anthropic", api_key=None, model=model,
                               timeout_seconds=timeout_seconds)
        except Exception as e:
            return {
                "verdict":       "WAIT",
                "confidence":    0,
                "reasoning":     "AI unavailable (both providers failed)",
                "key_risks":     [],
                "provider_used": "none",
                "model_used":    "none",
                "ai_error":      str(e),
            }

    # ── Single provider ───────────────────────────────────────────────────────
    resolved_key = _resolve_api_key(provider, api_key)
    if not resolved_key:
        return {
            "verdict":       "WAIT",
            "confidence":    0,
            "reasoning":     f"AI unavailable (no {provider} API key)",
            "key_risks":     [],
            "provider_used": "none",
            "model_used":    "none",
            "ai_error":      f"No API key for {provider}",
        }

    effective_model = model or _defaults.get(provider, "unknown")

    try:
        if provider == "groq":
            response_text = _call_groq(prompt, resolved_key, effective_model, timeout_seconds)
        elif provider == "anthropic":
            response_text = _call_anthropic(prompt, resolved_key, effective_model, timeout_seconds)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
    except Exception as e:
        return {
            "verdict":       "WAIT",
            "confidence":    0,
            "reasoning":     f"AI call failed: {str(e)[:100]}",
            "key_risks":     [],
            "provider_used": provider,
            "model_used":    effective_model,
            "ai_error":      str(e),
        }

    parsed = _parse_verdict_json(response_text)
    if parsed is None:
        return {
            "verdict":       "WAIT",
            "confidence":    0,
            "reasoning":     "AI response unparseable",
            "key_risks":     [],
            "provider_used": provider,
            "model_used":    effective_model,
            "ai_error":      f"Parse failure: {response_text[:200]}",
        }

    parsed["provider_used"] = provider
    parsed["model_used"]    = effective_model
    parsed["ai_error"]      = None
    return parsed


# ============================================================================
# PUBLIC API v1.1 — score_setup (preserved unchanged)
# ============================================================================

def score_setup(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper (v1.1): take a scan_one_symbol() result dict, build
    prompt, get verdict, return a dict ready to be merged into the result.

    Args:
        result: output of scan_one_symbol()

    Returns:
        {
            "ai_verdict":    "TAKE" | "WAIT" | "SKIP",
            "ai_confidence": int,
            "ai_reasoning":  str,
            "ai_key_risks":  List[str],
            "ai_provider":   str,
            "ai_error":      Optional[str],
        }
    """
    _error_out = lambda msg: {
        "ai_verdict":    "WAIT",
        "ai_confidence": 0,
        "ai_reasoning":  msg,
        "ai_key_risks":  [],
        "ai_provider":   "none",
        "ai_error":      msg,
    }

    try:
        prompt = build_smc_prompt(
            symbol=result.get("symbol", "UNKNOWN"),
            mode=result.get("mode", "DAY"),
            structure=result.get("structure", {}),
            zones=result.get("zones", {}),
            current_zone_classification=result.get("current_zone_classification", {}),
            ltf_confirmation=result.get("ltf_confirmation", "NONE"),
            backtest=result.get("backtest", {}),
            btc_regime=result.get("btc_regime", "UNKNOWN"),
            fights_macro=result.get("fights_macro", False),
            trade_plan=result.get("trade_plan", {}),
        )
    except Exception as e:
        return _error_out(f"Prompt build failed: {e}")

    verdict_dict = get_verdict(prompt, provider="auto")

    return {
        "ai_verdict":    verdict_dict.get("verdict",    "WAIT"),
        "ai_confidence": verdict_dict.get("confidence", 0),
        "ai_reasoning":  verdict_dict.get("reasoning",  ""),
        "ai_key_risks":  verdict_dict.get("key_risks",  []),
        "ai_provider":   verdict_dict.get("provider_used", "none"),
        "ai_error":      verdict_dict.get("ai_error"),
    }


# ============================================================================
# PUBLIC API v1.2 — score_setup_v2
# ============================================================================

def score_setup_v2(
    result: Dict[str, Any],
    provider: str = "groq",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    v2 wrapper: takes a scan result dict (with all v1.2 fields), builds the
    SMC-aware prompt, gets verdict, returns merged dict.

    Args:
        result: scan_one_symbol output (must include variant_grid, fibo_zone, etc.)
        provider: "groq" | "anthropic" | "auto"
        api_key: optional override

    Returns:
        Dict with keys:
            ai_verdict, ai_confidence, ai_reasoning, ai_key_risks,
            ai_smc_quality_score, ai_preferred_entry, ai_preferred_ltf_mode,
            ai_preferred_tp_R, ai_preferred_management, ai_wick_concerns,
            ai_provider_used, ai_error
    """
    _error_out = lambda msg: {
        "ai_verdict":              "WAIT",
        "ai_confidence":           0,
        "ai_reasoning":            msg,
        "ai_key_risks":            [],
        "ai_smc_quality_score":    None,
        "ai_preferred_entry":      None,
        "ai_preferred_ltf_mode":   None,
        "ai_preferred_tp_R":       None,
        "ai_preferred_management": None,
        "ai_wick_concerns":        "",
        "ai_provider_used":        "none",
        "ai_error":                msg,
    }

    # ── Resolve timeframe labels from mode ────────────────────────────────────
    mode = result.get("mode", "DAY")
    tf_labels = _MODE_TF_LABELS.get(mode, {"htf_label": "?", "ltf_label": "?"})

    # ── Pull all required fields from result ──────────────────────────────────
    structure     = result.get("structure", {})
    fibo_zone     = result.get("fibo_zone", {})
    zones         = result.get("zones", {})
    smart_obs     = zones.get("smart_obs", [])
    fvgs          = zones.get("fvgs", [])
    sr_levels     = zones.get("sr_levels", [])
    wick_adj      = result.get("wick_adjustments", [])
    ltf_conf      = result.get("ltf_confirmation", "NONE")
    variant_grid  = result.get("variant_grid", pd.DataFrame())
    best_variant  = result.get("best_variant", {}) or {}
    current_price = float(result.get("current_price", 0.0))
    btc_regime    = result.get("btc_regime", "UNKNOWN")
    fights_macro  = result.get("fights_macro", False)
    symbol        = result.get("symbol", "UNKNOWN")

    from datetime import datetime, timezone
    scan_timestamp = result.get(
        "scan_timestamp",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    # ── Build prompt ──────────────────────────────────────────────────────────
    try:
        prompt = build_smc_prompt_v2(
            symbol=symbol,
            mode=mode,
            htf_label=tf_labels["htf_label"],
            ltf_label=tf_labels["ltf_label"],
            btc_regime=btc_regime,
            fights_macro=fights_macro,
            structure=structure,
            fibo_zone=fibo_zone,
            smart_obs=smart_obs,
            fvgs=fvgs,
            sr_levels=sr_levels,
            wick_adjustments=wick_adj,
            ltf_confirmation=ltf_conf,
            variant_grid=variant_grid,
            best_overall=best_variant,
            current_price=current_price,
            scan_timestamp=scan_timestamp,
        )
    except Exception as e:
        return _error_out(f"Prompt v2 build failed: {e}")

    # ── Call LLM via existing get_verdict ─────────────────────────────────────
    verdict_dict = get_verdict(
        prompt,
        provider=provider,
        api_key=api_key,
        timeout_seconds=30,   # v2 prompt is longer, allow extra time
    )

    return {
        "ai_verdict":              verdict_dict.get("verdict",             "WAIT"),
        "ai_confidence":           verdict_dict.get("confidence",          0),
        "ai_reasoning":            verdict_dict.get("reasoning",           ""),
        "ai_key_risks":            verdict_dict.get("key_risks",           []),
        "ai_smc_quality_score":    verdict_dict.get("smc_quality_score",   None),
        "ai_preferred_entry":      verdict_dict.get("preferred_entry",     None),
        "ai_preferred_ltf_mode":   verdict_dict.get("preferred_ltf_mode",  None),
        "ai_preferred_tp_R":       verdict_dict.get("preferred_tp_R",      None),
        "ai_preferred_management": verdict_dict.get("preferred_management",None),
        "ai_wick_concerns":        verdict_dict.get("wick_concerns",       ""),
        "ai_provider_used":        verdict_dict.get("provider_used",       "none"),
        "ai_error":                verdict_dict.get("ai_error"),
    }
