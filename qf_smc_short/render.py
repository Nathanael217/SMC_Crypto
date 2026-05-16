"""
qf_smc_short/render.py — Signal card UI for SHORT setups (Phase 2)
====================================================================
Full Phase 2 card: structure + zones + trade plan + 24-variant backtest
summary + Deep Dive button → 72-variant analysis + AI verdict.

Public API:
  - render_signal_card_short(result, allow_deep_dive=True)
  - render_deep_dive_results_short(result)
  - OB_TIER_DEFINITIONS_SHORT (dict)
  - EMA_TIER_DEFINITIONS_SHORT (dict)
"""

from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd


# ============================================================================
# TIER DEFINITIONS (SHORT-flavored)
# ============================================================================

OB_TIER_DEFINITIONS_SHORT: Dict[str, str] = {
    "LIQUIDITY_SWEEP": (
        "Order Block candle has a LONG UPPER WICK (>50% of body) — price "
        "swept UP to grab buy-stop liquidity above, then reversed sharply "
        "down. Smart money engineered the rally to trap longs before "
        "marking the asset down. Highest-conviction bearish OB."
    ),
    "STRONG": (
        "Bearish OB with HIGH VOLUME (≥2.0× 20-bar avg) AND aligned with a "
        "swing HIGH AND followed by impulse that broke MULTIPLE swing lows. "
        "Highest baseline quality for shorts."
    ),
    "REGULAR": (
        "Bearish OB meeting minimum criteria: volume ≥1.2× avg, body ≥30% "
        "of candle range. Standard quality."
    ),
}

EMA_TIER_DEFINITIONS_SHORT: Dict[str, str] = {
    "STRONG": (
        "Price is BELOW BOTH EMA50 AND EMA200 on the HTF — confirmed "
        "bearish regime. Highest probability environment for short setups."
    ),
    "MEDIUM": (
        "Price is below ONE of EMA50/EMA200 but not both. Mild bearish "
        "regime. Short setups acceptable but require stronger structure."
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
            lambda x: "∞" if float(x) == float("inf") else f"{float(x):.2f}"
        )
    if "mean_r" in display.columns:
        display["mean_r"] = display["mean_r"].apply(lambda x: f"{float(x):+.3f}")
    rename_map = {
        "entry_type": "Entry",
        "ltf_mode":   "LTF",
        "tp_R":       "TP×R",
        "wr":         "WR",
        "pf":         "PF",
        "mean_r":     "Mean R",
        "n_setups":   "n",
        "management": "Mgmt",
    }
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})
    return display


# ============================================================================
# MAIN SIGNAL CARD RENDERER
# ============================================================================

def render_signal_card_short(
    result: Dict[str, Any],
    allow_deep_dive: bool = True,
) -> None:
    """
    Render one SHORT signal card with backtest evidence + Deep Dive button.

    Layout (mirror of LONG render_signal_card_detail_v2):
        Header:  Symbol @ price  [OB tier]  [EMA tier]  [⚠️ FIGHTS MACRO]
        Col 1:   Structure (state, CHoCH↓ bar, BOS↓ bar, leg LH→LL) + Fibo
        Col 2:   Bearish Zones in Fibo Zone (OBs, FVGs, resistance)
        Sec 1:   Wick adjustments transparency (collapsible)
        Sec 2:   24-variant backtest summary (or "fast scan" notice)
        Sec 3:   Recommended trade plan (from best variant or scanner fallback)
        Sec 4:   Tier tooltips (collapsible)
        Sec 5:   Deep Dive button → triggers 72-variant + AI verdict
    Below expander: deep dive results (rendered at top indent so not clipped)
    """
    symbol        = result.get("symbol", "?")
    current_price = float(result.get("current_price", 0.0))
    fights_macro  = result.get("fights_macro", False)
    ob_tier       = result.get("ob_tier", "")
    ema_tier      = result.get("ema_tier", "")

    # ── Build header badges ──────────────────────────────────────────────────
    badges: List[str] = []
    if ob_tier:
        badges.append(f"[OB: {ob_tier}]")
    if ema_tier:
        badges.append(f"[EMA: {ema_tier}]")
    if fights_macro:
        badges.append("⚠️ FIGHTS MACRO")
    badge_str = "  " + "  ".join(badges) if badges else ""

    with st.expander(
        f"🔻 **{symbol}** @ ${current_price:.6f}{badge_str}",
        expanded=False,
    ):
        # ── Row 1: Structure / Fibo + Zones ──────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📐 Structure (bearish) & Fibo**")
            structure   = result.get("structure", {})
            current_leg = structure.get("current_leg") or {}
            fibo        = result.get("fibo_zone", {})

            st.text(f"State:       {structure.get('state', '?')}")
            st.text(
                f"LH (anchor): bar {current_leg.get('leg_start_bar', '-')} @ "
                f"${float(current_leg.get('leg_start_price', 0)):.6f}"
            )
            st.text(
                f"LL (anchor): bar {current_leg.get('leg_low_bar', '-')} @ "
                f"${float(current_leg.get('leg_low_price', 0)):.6f}"
            )

            if fibo:
                fib_786  = float(fibo.get("fib_786",            0))
                zone_top = float(fibo.get("fib_786_zone_top",    0))
                zone_bot = float(fibo.get("fib_786_zone_bottom", 0))
                atr_mult = float(fibo.get("atr_multiplier",      0.5))
                st.text(f"Fibo 0.786:  ${fib_786:.6f}")
                st.text(f"Zone:        [${zone_bot:.6f}, ${zone_top:.6f}]")
                st.text(f"ATR mult:    ×{atr_mult}")

                if current_price > zone_top:
                    dist = ((zone_top - current_price) / current_price) * 100
                    st.text(f"Dist to zone:{dist:.2f}% (price above zone)")
                elif current_price < zone_bot:
                    dist = ((zone_bot - current_price) / current_price) * 100
                    st.text(f"Dist to zone:{dist:.2f}% (price below zone)")
                else:
                    st.success("✅ Price INSIDE Fibo 0.786 zone")

            wick_adj = result.get("wick_adjustments", [])
            if wick_adj:
                st.text(f"Wick adj:    {len(wick_adj)} made")

            ltf = result.get("ltf_confirmation", "NONE")
            ltf_emoji = {"CONFIRMED": "✅", "PENDING": "🟡", "NONE": "⚪"}.get(ltf, "?")
            st.text(f"LTF:         {ltf_emoji} {ltf}")

        with col2:
            st.markdown("**🎯 Bearish Zones in Fibo Zone**")
            zones     = result.get("zones", {})
            smart_obs = zones.get("smart_obs",  [])
            fvgs_list = zones.get("fvgs",       [])
            srs       = zones.get("sr_levels",  [])

            if smart_obs:
                st.markdown(f"**Bearish OBs ({len(smart_obs)}):**")
                for ob in smart_obs[:3]:
                    sweep_icon = " 🌊" if ob.get("tier") == "LIQUIDITY_SWEEP" else ""
                    st.text(f"  {ob.get('tier', '?')}{sweep_icon} @ ${float(ob.get('ob_low', 0)):.6f}")
                    st.text(
                        f"    vol×{float(ob.get('volume_mult', 0)):.1f}, "
                        f"status={ob.get('status', '?')}"
                    )
            else:
                st.text("No Bearish OBs in zone")

            if fvgs_list:
                st.markdown(f"**Bearish FVGs ({len(fvgs_list)}):**")
                for fvg in fvgs_list[:3]:
                    st.text(
                        f"  {fvg.get('status', '?')} "
                        f"[${float(fvg.get('bottom', 0)):.6f}, "
                        f"${float(fvg.get('top', 0)):.6f}]"
                    )
            else:
                st.text("No Bearish FVGs in zone")

            if srs:
                st.markdown(f"**Resistance S/R ({len(srs)}):**")
                for sr in srs[:3]:
                    st.text(f"  ${float(sr.get('price', 0)):.6f} (touches={sr.get('touches', '?')})")
            else:
                st.text("No resistance levels in zone")

        # ── Wick Adjustments Detail ──────────────────────────────────────────
        wick_adj = result.get("wick_adjustments", [])
        if wick_adj:
            with st.expander(
                f"🔧 Wick Adjustments ({len(wick_adj)} made — click for detail)",
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
                        f"→ bar {adj.get('adj_idx', '?')} "
                        f"(${float(adj.get('adj_price', 0)):.6f})"
                    )

        # ── 24-Variant Backtest Summary (if grid present) ────────────────────
        variant_grid = result.get("variant_grid")
        _has_grid = isinstance(variant_grid, pd.DataFrame) and not variant_grid.empty

        if _has_grid:
            st.markdown("**📊 Backtest: 24 SHORT Entry Variants (sorted by PF)**")
            top_5 = variant_grid.head(5)
            display_cols = [c for c in
                ["entry_type", "ltf_mode", "tp_R", "wr", "pf", "mean_r", "n_setups"]
                if c in top_5.columns]
            st.dataframe(
                _format_variant_df(top_5[display_cols]),
                hide_index=True,
                use_container_width=True,
                height=220,
            )
            with st.expander("📋 Show all 24 variants", expanded=False):
                display_all_cols = [c for c in
                    ["entry_type", "ltf_mode", "tp_R", "wr", "pf", "mean_r", "n_setups"]
                    if c in variant_grid.columns]
                st.dataframe(
                    _format_variant_df(variant_grid[display_all_cols]),
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                )
        else:
            st.info(
                "⚡ **Fast scan** — structure + zone validated. "
                "Click **🔍 Deep Dive** below to run the full "
                "24/72-variant SHORT backtest for this coin."
            )

        # ── Recommended Setup (best variant) ─────────────────────────────────
        best_variant = result.get("best_variant")
        if best_variant:
            st.markdown("**🎯 Recommended Setup (Best Variant by PF)**")
            _pf_val = best_variant.get("pf", 0)
            _pf_disp = "∞" if _pf_val == float("inf") else f"{float(_pf_val):.2f}"
            st.text(
                f"  {best_variant.get('entry_type', '?')} / "
                f"LTF-{best_variant.get('ltf_mode', '?')} / "
                f"TP {best_variant.get('tp_R', '?')}R"
            )
            st.text(
                f"  PF {_pf_disp}  |  "
                f"WR {float(best_variant.get('wr', 0)):.0%}  |  "
                f"Mean R {float(best_variant.get('mean_r', 0)):+.3f}  |  "
                f"n={best_variant.get('n_setups', 0)}"
            )

        # ── Trade Plan (from scanner — always present) ───────────────────────
        tp = result.get("trade_plan", {})
        if tp:
            st.markdown("**📋 Trade Plan (SHORT)**")
            plan_cols = st.columns(2)
            with plan_cols[0]:
                st.text(f"Entry zone: {tp.get('entry_zone_type', '?')}")
                st.text(f"Entry:      ${float(tp.get('entry_price', 0)):.6f}")
                st.text(f"SL (above): ${float(tp.get('sl', 0)):.6f}")
            with plan_cols[1]:
                st.text(f"TP1 (2R):   ${float(tp.get('tp1_price', 0)):.6f}")
                st.text(f"TP2 (2.5R): ${float(tp.get('tp2_price', 0)):.6f}")
                st.text(f"TP3 (3R):   ${float(tp.get('tp3_price', 0)):.6f}")
                st.text(f"R:R to TP2: {float(tp.get('rr_to_tp2', 0)):.2f}")

        # ── Tier Tooltips ────────────────────────────────────────────────────
        if ob_tier or ema_tier:
            with st.expander("ℹ️ What do the tier badges mean?", expanded=False):
                if ob_tier and ob_tier in OB_TIER_DEFINITIONS_SHORT:
                    st.markdown(f"**OB: {ob_tier}**")
                    st.caption(OB_TIER_DEFINITIONS_SHORT[ob_tier])
                if ema_tier and ema_tier in EMA_TIER_DEFINITIONS_SHORT:
                    st.markdown(f"**EMA: {ema_tier}**")
                    st.caption(EMA_TIER_DEFINITIONS_SHORT[ema_tier])

        # ── Deep-Dive Button ─────────────────────────────────────────────────
        if allow_deep_dive:
            deep_dive_key = f"deep_dive_short_{symbol}"
            if st.button(
                "🔍 Deep Dive: Full 72-variant SHORT + AI verdict",
                key=f"btn_{deep_dive_key}",
                help=(
                    "Run all 3 management modes (Fixed / BE-at-1R / Trailing) × "
                    "24 variants, plus AI verdict. Takes ~30-60s per coin."
                ),
            ):
                st.session_state[deep_dive_key] = True

    # ── Deep-Dive Results (rendered OUTSIDE the expander to avoid clipping) ─
    if allow_deep_dive:
        deep_dive_key = f"deep_dive_short_{symbol}"
        if st.session_state.get(deep_dive_key):
            render_deep_dive_results_short(result)


# ============================================================================
# DEEP-DIVE RESULTS RENDERER
# ============================================================================

def render_deep_dive_results_short(result: Dict[str, Any]) -> None:
    """
    Triggers deep_dive_backtest_short and renders the detailed result.

    Lazy-imports backtest to avoid circular imports at module level.
    """
    # Lazy import — only executed when the user clicks Deep Dive
    from qf_smc_short.backtest import deep_dive_backtest_short

    symbol = result.get("symbol", "?")

    with st.spinner(f"🔍 Running SHORT deep-dive for {symbol}... (~30-60s)"):
        ai_provider_raw = st.session_state.get("ai_provider", "Groq (Free)")
        ai_provider = ai_provider_raw.lower().split()[0]   # "groq" or "anthropic"
        ai_key_name = "groq_api_key" if ai_provider == "groq" else "anthropic_api_key"
        ai_key = st.session_state.get(ai_key_name, "")

        try:
            zones = result.get("zones", {})
            deep = deep_dive_backtest_short(
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
            st.error(f"SHORT deep-dive failed for {symbol}: {e}")
            return

    st.markdown("---")
    st.markdown(f"## 🔬 SHORT Deep Dive Results — {symbol}")

    # ── AI Verdict ──────────────────────────────────────────────────────────
    ai = deep.get("ai_verdict") or {}
    if ai and "error" not in ai and ai.get("verdict") in {"TAKE", "WAIT", "SKIP"}:
        verdict    = ai.get("verdict", "?")
        confidence = ai.get("confidence", 0)
        emoji_map  = {"TAKE": "🟢", "WAIT": "🟡", "SKIP": "🔴"}
        emoji      = emoji_map.get(verdict, "❓")

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
                st.warning(f"⚠️ {risk}")

        wick_concerns = ai.get("wick_concerns", "")
        if wick_concerns:
            st.warning(f"🔧 Wick concerns: {wick_concerns}")

        fights_macro_note = ai.get("fights_macro_note", "")
        if fights_macro_note:
            st.info(f"⚠️ Macro note: {fights_macro_note}")

    elif ai.get("error") or ai.get("ai_error"):
        err = ai.get("error") or ai.get("ai_error", "unknown error")
        st.warning(f"AI verdict unavailable: {err}")
    else:
        st.info("AI verdict not available — no API key set or AI skipped.")

    # ── Recommended Trade Plan ──────────────────────────────────────────────
    plan = deep.get("recommended_trade_plan") or {}
    if plan:
        st.markdown("### 📋 Recommended SHORT Trade Plan")

        def _safe_float(val: Any, default: float = 0.0) -> float:
            if val is None:
                return default
            try:
                f = float(val)
                if f != f:  # NaN
                    return default
                return f
            except (TypeError, ValueError):
                return default

        if plan.get("entry_price") is None or plan.get("error"):
            st.warning(
                f"⚠️ No concrete trade plan available "
                f"(entry type **{plan.get('entry_type', '?')}** has no live zone right now). "
                f"{plan.get('error', '')}"
            )
            st.caption(
                "The backtest variant scored well historically, but the matching "
                "bearish OB/FVG/resistance zone is not currently present. Wait for "
                "the zone to form, or check the 'Best Per Entry Type' table below."
            )
        else:
            plan_cols = st.columns(2)
            with plan_cols[0]:
                st.text(f"Entry type:  {plan.get('entry_type', '?')}")
                st.text(f"LTF mode:    {plan.get('ltf_mode', '?')}")
                st.text(f"Management:  {plan.get('management', '?')}")
                st.text(f"TP target:   {plan.get('tp_R', '?')}R")
            with plan_cols[1]:
                entry_p = _safe_float(plan.get("entry_price"))
                sl_p    = _safe_float(plan.get("sl"))
                risk_p  = _safe_float(plan.get("risk_pct"))
                tp1_p   = _safe_float(plan.get("tp1_price"))
                tp2_p   = _safe_float(plan.get("tp2_price"))
                tp3_p   = _safe_float(plan.get("tp3_price"))
                st.text(f"Entry price: ${entry_p:.6f}")
                st.text(f"Stop loss:   ${sl_p:.6f}  ({risk_p:.2f}% risk)")
                st.text(f"TP1 (2R):    ${tp1_p:.6f}")
                st.text(f"TP2 (2.5R):  ${tp2_p:.6f}")
                st.text(f"TP3 (3R):    ${tp3_p:.6f}")

    # ── Best Per Entry Type ─────────────────────────────────────────────────
    bpe = deep.get("best_per_entry", [])
    if bpe:
        st.markdown("### 🏆 Best Per Entry Type (SHORT)")
        try:
            bpe_rows = []
            for entry_info in bpe:
                br = entry_info.get("best_row") or entry_info.get("best_variant") or {}
                pf_val = float(br.get("pf", 0))
                bpe_rows.append({
                    "Entry Type": entry_info.get("entry_type", "?"),
                    "LTF":        br.get("ltf_mode",  "?"),
                    "Mgmt":       br.get("management", "Fixed"),
                    "TP×R":       br.get("tp_R",      "?"),
                    "WR":         f"{float(br.get('wr', 0)):.0%}",
                    "PF":         "∞" if pf_val == float("inf") else f"{pf_val:.2f}",
                    "Mean R":     f"{float(br.get('mean_r', 0)):+.3f}",
                    "n":          br.get("n_setups", 0),
                })
            st.dataframe(
                pd.DataFrame(bpe_rows),
                hide_index=True,
                use_container_width=True,
                height=220,
            )
        except Exception as e:
            st.warning(f"Could not render best-per-entry table: {e}")

    # ── All 72 Variants ─────────────────────────────────────────────────────
    all_72 = deep.get("all_72_variants")
    if all_72 is None:
        all_72 = deep.get("all_variants")
    if isinstance(all_72, pd.DataFrame) and not all_72.empty:
        with st.expander(
            f"📋 All {len(all_72)} SHORT variants (sorted by PF)",
            expanded=False,
        ):
            show_cols = [c for c in
                ["entry_type", "ltf_mode", "management", "tp_R", "wr", "pf", "mean_r", "n_setups"]
                if c in all_72.columns]
            st.dataframe(
                _format_variant_df(all_72[show_cols]),
                hide_index=True,
                use_container_width=True,
                height=400,
            )
