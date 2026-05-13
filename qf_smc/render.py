"""
qf_smc/render.py — Detailed signal card UI + deep-dive handler
===============================================================
Replaces simple render_smc_signal_card() from app.py with a richer card
that exposes 24-variant backtest summary, OB/EMA tier badges, wick
adjustments, and a "Deep Dive" button to trigger full 72-variant
analysis + AI verdict.

Public API:
  - render_signal_card_detail_v2(result, allow_deep_dive=True)
  - render_deep_dive_results(result)
  - OB_TIER_DEFINITIONS   (dict)
  - EMA_TIER_DEFINITIONS  (dict)

Notes:
  - Deep-dive imports (backtest module) are lazy — imported inside the
    render_deep_dive_results() function only, never at module top level.
  - No circular imports: render.py does NOT import ai_verdict at module
    level (ai verdict is called inside backtest.deep_dive_backtest).
"""

from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd


# ============================================================================
# TIER DEFINITIONS  (used for badge tooltips)
# ============================================================================

OB_TIER_DEFINITIONS: Dict[str, str] = {
    "LIQUIDITY_SWEEP": (
        "Order Block candle has a LONG LOWER WICK (>50% of body), indicating "
        "price swept down to grab stop-loss liquidity then reversed. This is "
        "considered a high-conviction bullish signal in SMC theory — "
        "smart money 'engineered' the move by sweeping retail stops."
    ),
    "STRONG": (
        "Order Block with HIGH VOLUME (\u22652.0\u00d7 20-bar avg) AND aligned with a "
        "swing low AND followed by impulse that broke MULTIPLE swing highs. "
        "Highest baseline quality."
    ),
    "REGULAR": (
        "Order Block meeting minimum criteria: volume \u22651.2\u00d7 avg, body \u226530% "
        "of candle range. Standard quality."
    ),
}

EMA_TIER_DEFINITIONS: Dict[str, str] = {
    "STRONG": (
        "Price is ABOVE BOTH EMA50 AND EMA200 on the HTF \u2014 confirmed bullish "
        "regime. Highest probability environment for long setups."
    ),
    "MEDIUM": (
        "Price is above ONE of EMA50/EMA200 but not both. Mild bullish "
        "regime. Long setups acceptable but require stronger structure."
    ),
}


# ============================================================================
# HELPER — format variant DataFrame for display
# ============================================================================

def _format_variant_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready copy of a variant grid DataFrame."""
    display = df.copy()
    if "wr" in display.columns:
        display["wr"] = display["wr"].apply(lambda x: f"{float(x):.0%}")
    if "pf" in display.columns:
        display["pf"] = display["pf"].apply(
            lambda x: "\u221e" if float(x) == float("inf") else f"{float(x):.2f}"
        )
    if "mean_r" in display.columns:
        display["mean_r"] = display["mean_r"].apply(lambda x: f"{float(x):+.3f}")
    rename_map = {
        "entry_type": "Entry",
        "ltf_mode":   "LTF",
        "tp_R":       "TP\u00d7R",
        "wr":         "WR",
        "pf":         "PF",
        "mean_r":     "Mean R",
        "n_setups":   "n",
        "management": "Mgmt",
    }
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})
    return display


# ============================================================================
# MAIN DETAILED CARD RENDERER
# ============================================================================

def render_signal_card_detail_v2(
    result: Dict[str, Any],
    allow_deep_dive: bool = True,
) -> None:
    """
    Detailed v2 signal card — replaces v1.1 render_smc_signal_card().

    Layout:
        Header:     Symbol @ price  [OB tier badge]  [EMA tier badge]  [\u26a0\ufe0f FIGHTS MACRO if set]
        Column 1:   Structure & Fibo (HL bar, HH bar, current retrace %, Fibo 0.786 zone)
        Column 2:   Zones in Fibo Zone (Smart OBs, FVGs, S/R supports)
        Section 1:  Wick adjustments transparency (collapsible)
        Section 2:  24-variant backtest summary (top 5 sorted by PF, expandable to all 24)
        Section 3:  Recommended trade plan (from best variant)
        Section 4:  Tier tooltips (collapsible)
        Section 5:  Deep-Dive button \u2192 triggers 72-variant + AI verdict
    """
    symbol        = result.get("symbol", "?")
    current_price = float(result.get("current_price", 0.0))
    fights_macro  = result.get("fights_macro", False)
    ob_tier       = result.get("ob_tier", "")
    ema_tier      = result.get("ema_tier", "")

    # ── Build header badge string ─────────────────────────────────────────────
    badges: List[str] = []
    if ob_tier:
        badges.append(f"[OB: {ob_tier}]")
    if ema_tier:
        badges.append(f"[EMA: {ema_tier}]")
    if fights_macro:
        badges.append("\u26a0\ufe0f FIGHTS MACRO")
    badge_str = "  " + "  ".join(badges) if badges else ""

    with st.expander(
        f"**{symbol}** @ ${current_price:.6f}{badge_str}",
        expanded=False,
    ):
        # ─── Row 1: Structure & Zones ─────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**\U0001f4d0 Structure & Fibo**")
            structure   = result.get("structure", {})
            current_leg = structure.get("current_leg") or {}
            fibo        = result.get("fibo_zone", {})

            st.text(f"State:       {structure.get('state', '?')}")
            st.text(
                f"HL (anchor): bar {current_leg.get('leg_start_bar', '-')} @ "
                f"${float(current_leg.get('leg_start_price', 0)):.6f}"
            )
            st.text(
                f"HH (anchor): bar {current_leg.get('leg_high_bar', '-')} @ "
                f"${float(current_leg.get('leg_high_price', 0)):.6f}"
            )

            if fibo:
                fib_786  = float(fibo.get("fib_786",          0))
                zone_top = float(fibo.get("fib_786_zone_top",    0))
                zone_bot = float(fibo.get("fib_786_zone_bottom", 0))
                atr_mult = float(fibo.get("atr_multiplier",    0.5))
                st.text(f"Fibo 0.786:  ${fib_786:.6f}")
                st.text(f"Zone:        [${zone_bot:.6f}, ${zone_top:.6f}]")
                st.text(f"ATR mult:    \u00d7{atr_mult}")

                # Distance to zone
                if current_price > zone_top:
                    dist = ((zone_top - current_price) / current_price) * 100
                    st.text(f"Dist to zone:{dist:.2f}% (price above zone)")
                elif current_price < zone_bot:
                    dist = ((zone_bot - current_price) / current_price) * 100
                    st.text(f"Dist to zone:{dist:.2f}% (price below zone)")
                else:
                    st.success("\u2705 Price INSIDE Fibo 0.786 zone")

            # Wick adjustments inline count
            wick_adj = result.get("wick_adjustments", [])
            if wick_adj:
                st.text(f"Wick adj:    {len(wick_adj)} made")

        with col2:
            st.markdown("**\U0001f3af Zones in Fibo 0.786 Area**")
            zones     = result.get("zones", {})
            smart_obs = zones.get("smart_obs",  [])
            fvgs_list = zones.get("fvgs",       [])
            srs       = zones.get("sr_levels",  [])

            if smart_obs:
                st.markdown(f"**Smart OBs ({len(smart_obs)}):**")
                for ob in smart_obs[:3]:
                    sweep_icon = " \U0001f30a" if ob.get("is_liquidity_sweep") else ""
                    st.text(f"  {ob.get('tier', '?')}{sweep_icon} @ ${float(ob.get('ob_high', 0)):.6f}")
                    st.text(
                        f"    vol\u00d7{float(ob.get('volume_mult', 0)):.1f}, "
                        f"status={ob.get('status', '?')}"
                    )
            else:
                st.text("No Smart OBs in zone")

            if fvgs_list:
                st.markdown(f"**FVGs ({len(fvgs_list)}):**")
                for fvg in fvgs_list[:3]:
                    st.text(
                        f"  {fvg.get('status', '?')} "
                        f"[${float(fvg.get('bottom', 0)):.6f}, "
                        f"${float(fvg.get('top', 0)):.6f}]"
                    )
            else:
                st.text("No FVGs in zone")

            if srs:
                st.markdown(f"**S/R Supports ({len(srs)}):**")
                for sr in srs[:3]:
                    st.text(f"  ${float(sr.get('price', 0)):.6f} (touches={sr.get('touches', '?')})")
            else:
                st.text("No S/R supports in zone")

        # ─── Wick Adjustments Detail ─────────────────────────────────────────
        wick_adj = result.get("wick_adjustments", [])
        if wick_adj:
            with st.expander(
                f"\U0001f527 Wick Adjustments ({len(wick_adj)} made — click for detail)",
                expanded=False,
            ):
                st.caption(
                    "Swing points with extreme wicks (wick > 50% of total range) were "
                    "replaced with the next bar's high/low for more accurate Fibo anchoring."
                )
                for adj in wick_adj:
                    st.text(
                        f"  {adj.get('type', '?').upper()} adjusted: "
                        f"bar {adj.get('raw_idx', '?')} "
                        f"(${float(adj.get('raw_price', 0)):.6f}) "
                        f"\u2192 bar {adj.get('adj_idx', '?')} "
                        f"(${float(adj.get('adj_price', 0)):.6f})"
                    )

        # ─── 24-Variant Backtest Summary ─────────────────────────────────────
        st.markdown("**\U0001f4ca Backtest: 24 Entry Variants (sorted by PF)**")
        variant_grid = result.get("variant_grid")

        if isinstance(variant_grid, pd.DataFrame) and not variant_grid.empty:
            top_5 = variant_grid.head(5)
            display_cols = [c for c in
                ["entry_type", "ltf_mode", "tp_R", "wr", "pf", "mean_r", "n_setups"]
                if c in top_5.columns]
            st.dataframe(
                _format_variant_df(top_5[display_cols]),
                hide_index=True,
                use_container_width=True,
            )
            with st.expander("\U0001f4cb Show all 24 variants", expanded=False):
                display_all_cols = [c for c in
                    ["entry_type", "ltf_mode", "tp_R", "wr", "pf", "mean_r", "n_setups"]
                    if c in variant_grid.columns]
                st.dataframe(
                    _format_variant_df(variant_grid[display_all_cols]),
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.warning(
                "No variant grid data. Run a deep-dive scan or ensure backtest.py "
                "is at v1.2 (run_variant_grid returns a DataFrame)."
            )

        # ─── Recommended Trade Plan (best variant) ───────────────────────────
        best_variant = result.get("best_variant")
        if best_variant:
            st.markdown("**\U0001f3af Recommended Setup (Best Variant by PF)**")
            bv_cols = st.columns(3)
            with bv_cols[0]:
                st.text(f"Entry type: {best_variant.get('entry_type', '?')}")
                st.text(f"LTF mode:   {best_variant.get('ltf_mode', '?')}")
                st.text(f"TP\u00d7R:       {best_variant.get('tp_R', '?')}")
            with bv_cols[1]:
                pf_val = best_variant.get("pf", 0)
                pf_display = "\u221e" if pf_val == float("inf") else f"{float(pf_val):.2f}"
                st.metric("PF",     pf_display)
                st.metric("WR",     f"{float(best_variant.get('wr', 0)):.0%}")
            with bv_cols[2]:
                st.metric("Mean R", f"{float(best_variant.get('mean_r', 0)):+.3f}")
                st.metric("n setups", f"{best_variant.get('n_setups', 0)}")

        # ─── Tier Tooltips ───────────────────────────────────────────────────
        if ob_tier or ema_tier:
            with st.expander("\u2139\ufe0f What do the tier badges mean?", expanded=False):
                if ob_tier and ob_tier in OB_TIER_DEFINITIONS:
                    st.markdown(f"**OB: {ob_tier}**")
                    st.caption(OB_TIER_DEFINITIONS[ob_tier])
                if ema_tier and ema_tier in EMA_TIER_DEFINITIONS:
                    st.markdown(f"**EMA: {ema_tier}**")
                    st.caption(EMA_TIER_DEFINITIONS[ema_tier])
                # Show all tier definitions for reference
                if ob_tier not in OB_TIER_DEFINITIONS or ema_tier not in EMA_TIER_DEFINITIONS:
                    st.markdown("---")
                    st.markdown("**All OB tiers:**")
                    for tier, desc in OB_TIER_DEFINITIONS.items():
                        st.markdown(f"*{tier}*: {desc}")
                    st.markdown("**All EMA tiers:**")
                    for tier, desc in EMA_TIER_DEFINITIONS.items():
                        st.markdown(f"*{tier}*: {desc}")

        # ─── Deep-Dive Button ────────────────────────────────────────────────
        if allow_deep_dive:
            deep_dive_key = f"deep_dive_{symbol}"

            if st.button(
                "\U0001f50d Deep Dive: Full 72-variant + AI verdict",
                key=f"btn_{deep_dive_key}",
                help=(
                    "Run all 3 management modes (Fixed / BE-at-1R / Trailing) \u00d7 "
                    "24 variants, plus AI verdict. Takes ~30-60s per coin."
                ),
            ):
                st.session_state[deep_dive_key] = True

            # If deep-dive was triggered, render it
            if st.session_state.get(deep_dive_key):
                render_deep_dive_results(result)


# ============================================================================
# DEEP-DIVE RESULTS RENDERER
# ============================================================================

def render_deep_dive_results(result: Dict[str, Any]) -> None:
    """
    Triggers deep_dive_backtest and renders the detailed result.

    Lazy-imports backtest to avoid circular imports at module level.
    """
    # Lazy import — only executed when the user clicks Deep Dive
    from qf_smc.backtest import deep_dive_backtest  # noqa: PLC0415

    symbol = result.get("symbol", "?")

    with st.spinner(f"\U0001f50d Running deep-dive analysis for {symbol}... (~30-60s)"):
        ai_provider_raw = st.session_state.get("ai_provider", "Groq (Free)")
        ai_provider = ai_provider_raw.lower().split()[0]  # "groq" or "anthropic"
        ai_key_name = "groq_api_key" if ai_provider == "groq" else "anthropic_api_key"
        ai_key = st.session_state.get(ai_key_name, "")

        try:
            zones = result.get("zones", {})
            deep = deep_dive_backtest(
                symbol=symbol,
                df_htf=result["_df_htf_cache"],
                df_ltf=result.get("_df_ltf_cache"),
                structure=result.get("structure", {}),
                fibo_zone=result.get("fibo_zone", {}),
                smart_obs=zones.get("smart_obs", []),
                fvgs=zones.get("fvgs", []),
                sr_levels=zones.get("sr_levels", []),
                ai_provider=ai_provider if ai_key else None,
                ai_api_key=ai_key if ai_key else None,
            )
        except Exception as e:
            st.error(f"Deep-dive failed for {symbol}: {e}")
            return

    st.markdown("---")
    st.markdown(f"## \U0001f52c Deep Dive Results \u2014 {symbol}")

    # ─── AI Verdict ──────────────────────────────────────────────────────────
    ai = deep.get("ai_verdict") or {}
    if ai and "error" not in ai and ai.get("verdict") in {"TAKE", "WAIT", "SKIP"}:
        verdict    = ai.get("verdict", "?")
        confidence = ai.get("confidence", 0)
        emoji_map  = {"TAKE": "\U0001f7e2", "WAIT": "\U0001f7e1", "SKIP": "\U0001f534"}
        emoji      = emoji_map.get(verdict, "\u2753")

        st.markdown(
            f"### {emoji} AI Verdict: **{verdict}** "
            f"(confidence {confidence}/100)"
        )

        ai_cols = st.columns(2)
        with ai_cols[0]:
            smc_score = ai.get("smc_quality_score")
            st.markdown(
                f"**SMC Quality Score**: "
                f"{smc_score}/100" if smc_score is not None else "**SMC Quality Score**: N/A"
            )
            st.markdown(f"**Preferred Entry**: {ai.get('preferred_entry', '?')}")
            st.markdown(f"**Preferred LTF Mode**: {ai.get('preferred_ltf_mode', '?')}")
        with ai_cols[1]:
            st.markdown(f"**Preferred TP**: {ai.get('preferred_tp_R', '?')}R")
            st.markdown(f"**Preferred Management**: {ai.get('preferred_management', '?')}")

        st.markdown("**Reasoning:**")
        st.info(ai.get("reasoning", "(no reasoning provided)"))

        key_risks = ai.get("key_risks", [])
        if key_risks:
            st.markdown("**Key Risks:**")
            for risk in key_risks:
                st.warning(f"\u26a0\ufe0f {risk}")

        wick_concerns = ai.get("wick_concerns", "")
        if wick_concerns:
            st.warning(f"\U0001f527 Wick concerns: {wick_concerns}")

        fights_macro_note = ai.get("fights_macro_note", "")
        if fights_macro_note:
            st.info(f"\u26a0\ufe0f Macro note: {fights_macro_note}")

    elif ai.get("error") or ai.get("ai_error"):
        err = ai.get("error") or ai.get("ai_error", "unknown error")
        st.warning(f"AI verdict unavailable: {err}")
    else:
        st.info("AI verdict not available — no API key set or AI skipped.")

    # ─── Recommended Trade Plan ───────────────────────────────────────────────
    plan = deep.get("recommended_trade_plan") or {}
    if plan:
        st.markdown("### \U0001f4cb Recommended Trade Plan")
        plan_cols = st.columns(2)
        with plan_cols[0]:
            st.text(f"Entry type:  {plan.get('entry_type', '?')}")
            st.text(f"LTF mode:    {plan.get('ltf_mode', '?')}")
            st.text(f"Management:  {plan.get('management', '?')}")
            st.text(f"TP target:   {plan.get('tp_R', '?')}R")
        with plan_cols[1]:
            entry_p = float(plan.get("entry_price", 0))
            sl_p    = float(plan.get("sl",           0))
            risk_p  = float(plan.get("risk_pct",     0))
            tp1_p   = float(plan.get("tp1_price",    0))
            tp2_p   = float(plan.get("tp2_price",    0))
            tp3_p   = float(plan.get("tp3_price",    0))
            st.text(f"Entry price: ${entry_p:.6f}")
            st.text(f"Stop loss:   ${sl_p:.6f}  ({risk_p:.2f}% risk)")
            st.text(f"TP1 (2R):    ${tp1_p:.6f}")
            st.text(f"TP2 (2.5R):  ${tp2_p:.6f}")
            st.text(f"TP3 (3R):    ${tp3_p:.6f}")

    # ─── Best Per Entry Type ──────────────────────────────────────────────────
    bpe = deep.get("best_per_entry", [])
    if bpe:
        st.markdown("### \U0001f3c6 Best Per Entry Type")
        try:
            bpe_rows = []
            for entry_info in bpe:
                br = entry_info.get("best_row") or entry_info.get("best_variant") or {}
                pf_val = float(br.get("pf", 0))
                bpe_rows.append({
                    "Entry Type": entry_info.get("entry_type", "?"),
                    "LTF":        br.get("ltf_mode",  "?"),
                    "Mgmt":       br.get("management", "Fixed"),
                    "TP\u00d7R":  br.get("tp_R",      "?"),
                    "WR":         f"{float(br.get('wr', 0)):.0%}",
                    "PF":         "\u221e" if pf_val == float("inf") else f"{pf_val:.2f}",
                    "Mean R":     f"{float(br.get('mean_r', 0)):+.3f}",
                    "n":          br.get("n_setups", 0),
                })
            st.dataframe(pd.DataFrame(bpe_rows), hide_index=True, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render best-per-entry table: {e}")

    # ─── All 72 Variants ─────────────────────────────────────────────────────
    all_72 = deep.get("all_72_variants") or deep.get("all_variants")
    if isinstance(all_72, pd.DataFrame) and not all_72.empty:
        with st.expander(
            f"\U0001f4cb All {len(all_72)} variants (sorted by PF)",
            expanded=False,
        ):
            show_cols = [c for c in
                ["entry_type", "ltf_mode", "management", "tp_R", "wr", "pf", "mean_r", "n_setups"]
                if c in all_72.columns]
            st.dataframe(
                _format_variant_df(all_72[show_cols]),
                hide_index=True,
                use_container_width=True,
            )
