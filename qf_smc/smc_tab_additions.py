# ============================================================================
# SMC Long Scanner — UI additions for app.py
#
# HOW TO INTEGRATE:
#   1. Add the imports block (marked CHANGE 1) after the `from qf_shared import`
#      block near the top of app.py.
#   2. Paste everything below the dashed line directly before `def main():`.
#   3. In main(), unpack the tabs as shown in CHANGE 2 and add `with tab_smc:`.
#
# CHANGE 1 — paste after `from qf_shared import (...)`:
# ─────────────────────────────────────────────────────────────────────────────
# try:
#     from qf_smc import run_scan, scan_one_symbol, MODE_CONFIG
#     from qf_smc.screener import (
#         fetch_screener_universe,
#         mode_full_scan,
#         mode_coinank_screener,
#         mode_custom_filter,
#         render_screener_table,
#     )
#     SMC_AVAILABLE = True
# except ImportError as e:
#     SMC_AVAILABLE = False
#     SMC_IMPORT_ERROR = str(e)
# ─────────────────────────────────────────────────────────────────────────────
#
# CHANGE 2 — replace the st.tabs() call in main() with:
#
#     tab_scanner, tab_manual, tab_pulse, tab_smc = st.tabs([
#         "🔭 Scanner — Momentum signals",
#         "🔍 Manual — Any coin, any candle",
#         "🫀 Pulse — On-chain intelligence",
#         "🎯 SMC Long Scanner",
#     ])
#     ...
#     with tab_smc:
#         render_smc_long_tab()
# ============================================================================


def render_smc_long_tab():
    """
    QUANTFLOW SMC Long Scanner — main tab UI.

    Layout:
        Top:  BTC regime banner
        Body: 3 sub-tabs (Pre-Filter, Mode & Scan, Results)
    """
    st.markdown("# 🎯 QUANTFLOW SMC Long Scanner")
    st.markdown("*Structure-based long-only altcoin scanner. Top-down ICT methodology.*")

    if not SMC_AVAILABLE:
        st.error(f"SMC module not available: {SMC_IMPORT_ERROR}")
        st.info("Make sure qf_smc/ package is in your project directory.")
        return

    # --- BTC Regime Banner (always shown at top) ---
    render_btc_regime_banner()

    # --- 3 sub-tabs ---
    sub_tabs = st.tabs(["1. Pre-Filter", "2. Mode & Scan", "3. Results"])

    with sub_tabs[0]:
        render_smc_prefilter_subtab()

    with sub_tabs[1]:
        render_smc_mode_scan_subtab()

    with sub_tabs[2]:
        render_smc_results_subtab()


def render_btc_regime_banner():
    """Render BTC regime status banner at top of SMC tab."""
    try:
        btc_regime = _scanner_btc_regime_for_combos()
    except Exception:
        btc_regime = "UNKNOWN"

    if btc_regime == "BULL":
        st.success("🟢 **BTC Regime: BULL** — Favorable macro for long setups.")
    elif btc_regime == "CHOP":
        st.warning("🟡 **BTC Regime: CHOP** — Mixed macro. Proceed with caution.")
    elif btc_regime == "BEAR":
        st.error("🔴 **BTC Regime: BEAR** — Long setups fight macro. Cards will be tagged ⚠️ FIGHTS MACRO.")
    else:
        st.info("⚪ **BTC Regime: UNKNOWN** — Could not determine. Verify manually.")

    # Store in session state for use by scanner
    st.session_state["smc_btc_regime"] = btc_regime


def render_smc_prefilter_subtab():
    """Sub-tab 1: 3-mode pre-filter screener."""
    st.markdown("### Pre-Filter: Select Candidate Coins")
    st.caption("Choose a filtering mode, then click Run to fetch and filter the universe.")

    mode = st.radio(
        "Filter Mode",
        options=["B. CoinAnk Screener", "A. Full Scan", "C. Custom Filter"],
        index=0,
        horizontal=True,
        help=(
            "- A. Full Scan: All USDT perpetuals (~340 coins). Slow but complete.\n"
            "- B. CoinAnk Screener: Top 25 ranked by composite score. Fast, default.\n"
            "- C. Custom Filter: User-adjustable. For specific hypotheses."
        ),
    )

    # Mode-specific controls
    params = {}
    if mode.startswith("A"):
        params["min_volume_usd"] = st.number_input(
            "Min 24h Volume USD", min_value=0, value=100_000, step=50_000
        )
    elif mode.startswith("B"):
        col1, col2 = st.columns(2)
        with col1:
            params["top_n"] = st.slider("Top N", min_value=5, max_value=50, value=25)
        with col2:
            params["preset"] = st.selectbox(
                "Preset Filter",
                options=["bullish_money_flow", "avoid_crowded_long", "all"],
                index=0,
            )
    else:  # C. Custom
        col1, col2 = st.columns(2)
        with col1:
            params["min_volume_24h_usd"] = st.number_input(
                "Min 24h Vol USD", value=1_000_000, step=100_000
            )
            min_oi = st.number_input("Min OI 4h % change", value=5.0, step=1.0)
            params["min_oi_4h_pct"] = min_oi if min_oi != 0 else None
            min_price = st.number_input("Min Price 24h %", value=0.0, step=0.5)
            params["min_price_24h_pct"] = min_price if min_price != 0 else None
        with col2:
            max_fund = st.number_input("Max Funding Rate %", value=0.05, step=0.01)
            params["max_funding_rate_pct"] = max_fund if max_fund > 0 else None
            max_ls = st.number_input("Max Top Trader L/S Ratio", value=3.0, step=0.5)
            params["max_top_trader_ls"] = max_ls if max_ls > 0 else None
            params["sort_by"] = st.selectbox(
                "Sort By",
                options=["composite", "oi_change", "vol_change", "price_change", "alphabetical"],
            )
            params["limit"] = st.number_input(
                "Limit Results", value=50, min_value=5, max_value=200
            )

    # Run button
    if st.button("Run Pre-Filter", type="primary"):
        with st.spinner("Fetching universe data..."):
            df_universe = fetch_screener_universe(top_n=400)

        if df_universe is None or df_universe.empty:
            st.error("Universe fetch returned no data. Check Binance API connectivity.")
            return

        # Apply selected mode
        with st.spinner("Filtering candidates..."):
            if mode.startswith("A"):
                symbols = mode_full_scan(
                    df_universe, min_volume_usd=params["min_volume_usd"]
                )
            elif mode.startswith("B"):
                symbols = mode_coinank_screener(
                    df_universe,
                    top_n=params["top_n"],
                    apply_preset=params["preset"],
                )
            else:
                # Strip the preset key (not used in custom mode) to avoid kwarg errors
                custom_params = {k: v for k, v in params.items() if k != "preset"}
                symbols = mode_custom_filter(df_universe, **custom_params)

        st.success(f"✅ Found {len(symbols)} candidate coins.")

        # Persist in session state for the other sub-tabs
        st.session_state["smc_universe_df"] = df_universe
        st.session_state["smc_candidates"] = symbols

        # Show filterable table
        render_screener_table(df_universe, symbols, mode[:1])

    # Show currently cached candidates even if the button wasn't just pressed
    if st.session_state.get("smc_candidates"):
        candidates = st.session_state["smc_candidates"]
        st.markdown(f"#### Currently selected: **{len(candidates)} coins**")
        st.code(
            ", ".join(candidates[:30]) + ("..." if len(candidates) > 30 else ""),
            language=None,
        )


def render_smc_mode_scan_subtab():
    """Sub-tab 2: mode selection + scan execution."""
    st.markdown("### Mode & Scan")

    if not st.session_state.get("smc_candidates"):
        st.warning("No candidates selected. Go to '1. Pre-Filter' first.")
        return

    candidates = st.session_state["smc_candidates"]
    st.info(f"Scanning **{len(candidates)}** candidate coins from Pre-Filter.")

    mode = st.selectbox(
        "Scan Mode",
        options=["DAY", "SWING", "SCALP"],
        index=0,
        help=(
            "- SWING: HTF=1D, LTF=4H. Hold 3-14 days.\n"
            "- DAY:   HTF=4H, LTF=15m. Hold 1-3 days. (default)\n"
            "- SCALP: HTF=1H, LTF=5m. Hold <24 hours."
        ),
    )
    mode_cfg = MODE_CONFIG[mode]
    st.caption(
        f"HTF: {mode_cfg['htf_label']} | "
        f"LTF: {mode_cfg['ltf_label']} | "
        f"Lookback: {mode_cfg['lookback']} bars"
    )

    if st.button("🚀 Run SMC Scan", type="primary"):
        btc_regime = st.session_state.get("smc_btc_regime", "UNKNOWN")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        def progress_callback(idx: int, total: int, symbol: str):
            progress_bar.progress(idx / total)
            progress_text.text(f"Scanning {idx}/{total}: {symbol}")

        with st.spinner("Running SMC scan..."):
            results = run_scan(
                mode=mode,
                symbols=candidates,
                btc_regime=btc_regime,
                progress_callback=progress_callback,
            )

        progress_bar.progress(1.0)
        progress_text.text(f"✅ Scan complete. Found {len(results)} setups.")

        st.session_state["smc_results"] = results
        st.session_state["smc_results_mode"] = mode

        if not results:
            st.warning("No setups found. Try a different mode or broaden pre-filter.")
        else:
            st.success(f"Found **{len(results)}** actionable setups. See '3. Results' tab.")


def render_smc_results_subtab():
    """Sub-tab 3: signal cards grouped by zone type."""
    st.markdown("### Scan Results")

    if not st.session_state.get("smc_results"):
        st.warning("No scan results yet. Go to '2. Mode & Scan' and run a scan.")
        return

    results = st.session_state["smc_results"]
    mode = st.session_state.get("smc_results_mode", "UNKNOWN")

    st.caption(f"Mode: **{mode}** | {len(results)} setups found")

    # Group by primary zone type
    _ZONE_ORDER = ["smart_ob", "fvg", "fibo_786", "sr"]
    groups: dict = {k: [] for k in _ZONE_ORDER}
    ungrouped = []
    for r in results:
        zone_type = r.get("primary_zone", {}).get("type", "")
        if zone_type in groups:
            groups[zone_type].append(r)
        else:
            ungrouped.append(r)

    icons = {"smart_ob": "🟦", "fvg": "🟩", "fibo_786": "🟧", "sr": "🟨"}
    labels = {
        "smart_ob": "Smart OB Setups",
        "fvg": "FVG Setups",
        "fibo_786": "Fibo 0.786 Setups",
        "sr": "Classic S/R Setups",
    }

    any_shown = False
    for zone_type in _ZONE_ORDER:
        items = groups[zone_type]
        if not items:
            continue
        any_shown = True
        st.markdown(f"## {icons[zone_type]} {labels[zone_type]} ({len(items)})")
        for r in items:
            render_smc_signal_card(r)

    if ungrouped:
        st.markdown(f"## ⬜ Other Setups ({len(ungrouped)})")
        for r in ungrouped:
            render_smc_signal_card(r)
        any_shown = True

    if not any_shown:
        st.info("All result entries had unrecognised zone types.")


def render_smc_signal_card(result: dict):
    """Render one signal card for a result dict."""
    symbol = result.get("symbol", "???")
    fights_macro = result.get("fights_macro", False)
    ema_tier = result.get("ema_tier", "")
    current_price = result.get("current_price", 0.0)

    badge = " ⚠️ **FIGHTS MACRO**" if fights_macro else ""
    tier_badge = f" `{ema_tier}`" if ema_tier else ""

    with st.expander(
        f"**{symbol}** @ ${current_price:.4f}{tier_badge}{badge}", expanded=False
    ):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Structure**")
            structure = result.get("structure", {})
            st.text(f"State:     {structure.get('state', '?')}")
            st.text(f"CHoCH bar: {structure.get('choch_bar', '-')}")
            st.text(f"BOS bar:   {structure.get('bos_bar', '-')}")

            st.markdown("**LTF Confirmation**")
            ltf = result.get("ltf_confirmation", "NONE")
            ltf_emoji = {"CONFIRMED": "✅", "PENDING": "🟡", "NONE": "⚪"}.get(ltf, "?")
            st.text(f"{ltf_emoji} {ltf}")

        with col2:
            st.markdown("**Trade Plan**")
            tp = result.get("trade_plan", {})
            st.text(f"Entry:      ${tp.get('entry_price', 0):.4f}")
            st.text(f"SL:         ${tp.get('sl', 0):.4f}")
            st.text(f"TP1 (2R):   ${tp.get('tp1_price', 0):.4f}")
            st.text(f"TP2 (2.5R): ${tp.get('tp2_price', 0):.4f}")
            st.text(f"TP3 (3R):   ${tp.get('tp3_price', 0):.4f}")
            st.text(f"R:R to TP2: {tp.get('rr_to_tp2', 0):.2f}")

        st.markdown("**Backtest Evidence**")
        bt = result.get("backtest", {})
        bt_cols = st.columns(3)
        with bt_cols[0]:
            pc = bt.get("per_coin", {})
            st.metric(
                "Per-Coin PF",
                f"{pc.get('pf', 0):.2f}",
                delta=f"n={pc.get('n_setups', 0)}",
            )
        with bt_cols[1]:
            un = bt.get("universe", {})
            st.metric(
                "Universe PF",
                f"{un.get('pf', 0):.2f}",
                delta=f"n={un.get('n_setups', 0)}",
            )
        with bt_cols[2]:
            bl = bt.get("blended", {})
            st.metric(
                "Blended PF",
                f"{bl.get('pf', 0):.2f}",
                delta=f"n_eff={bl.get('n_effective', 0)}",
            )

        # Recent check
        rc = bt.get("recent_check", {})
        verdict = rc.get("verdict", "")
        if verdict and verdict != "INSUFFICIENT_DATA":
            verdict_emoji = {"STRONGER": "📈", "STABLE": "➡️", "WEAKER": "📉"}.get(
                verdict, "?"
            )
            st.info(
                f"{verdict_emoji} **Recent Check**: "
                f"{rc.get('earlier_period_label', '?')} vs "
                f"{rc.get('recent_period_label', '?')} → **{verdict}**"
            )

        # AI Verdict (filled by Session 08; gracefully absent until then)
        ai_verdict = result.get("ai_verdict")
        if ai_verdict:
            verdict_fn = {"TAKE": st.success, "WAIT": st.warning, "SKIP": st.error}.get(
                ai_verdict, st.info
            )
            verdict_fn(
                f"**AI Verdict: {ai_verdict}** — {result.get('ai_reasoning', '')}"
            )
