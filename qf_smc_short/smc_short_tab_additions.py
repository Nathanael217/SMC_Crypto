# ============================================================================
# SMC SHORT Scanner — UI additions for app.py
# ============================================================================
#
# HOW TO INTEGRATE INTO YOUR app.py:
#
# STEP 1 — Add the imports block after the existing qf_smc import block
#          (around line 77 in your current app.py):
#
#     try:
#         from qf_smc_short import run_scan_short, scan_one_symbol_short, MODE_CONFIG as MODE_CONFIG_SHORT
#         from qf_smc_short.screener import mode_coinank_screener_short
#         from qf_smc_short.render import render_signal_card_short
#         SMC_SHORT_AVAILABLE = True
#         SMC_SHORT_IMPORT_ERROR = None
#     except ImportError as e:
#         SMC_SHORT_AVAILABLE = False
#         SMC_SHORT_IMPORT_ERROR = str(e)
#
# STEP 2 — Paste all the functions below before `def main():` in app.py.
#
# STEP 3 — Update the tabs unpack in main() to include the new SHORT tab:
#
#     tab_scanner, tab_manual, tab_pulse, tab_smc, tab_smc_short = st.tabs([
#         "🔭 Scanner — Momentum signals",
#         "🔍 Manual — Any coin, any candle",
#         "🫀 Pulse — On-chain intelligence",
#         "🎯 SMC Long Scanner",
#         "🔻 SMC Short Scanner",          # NEW
#     ])
#
#     ...existing tab blocks...
#
#     with tab_smc_short:                  # NEW
#         render_smc_short_tab()
#
# That's the entire integration. Existing tabs are untouched.
# ============================================================================


def render_smc_short_tab():
    """
    QUANTFLOW SMC SHORT Scanner — main tab UI.

    Mirror of render_smc_long_tab() with 3 sub-tabs (Pre-Filter, Mode & Scan,
    Results) — same UX, opposite direction.
    """
    st.markdown("# 🔻 QUANTFLOW SMC Short Scanner")
    st.markdown("*Structure-based short-only altcoin scanner. Top-down ICT methodology — bearish edition.*")

    if not SMC_SHORT_AVAILABLE:
        st.error(f"SMC Short module not available: {SMC_SHORT_IMPORT_ERROR}")
        st.info("Make sure the `qf_smc_short/` package is in your project directory next to `qf_smc/`.")
        return

    # ── BTC Regime Banner (flipped meaning for shorts) ───────────────────────
    render_btc_regime_banner_short()

    # ── 3 sub-tabs ───────────────────────────────────────────────────────────
    sub_tabs = st.tabs(["1. Pre-Filter", "2. Mode & Scan", "3. Results"])

    with sub_tabs[0]:
        render_smc_short_prefilter_subtab()

    with sub_tabs[1]:
        render_smc_short_mode_scan_subtab()

    with sub_tabs[2]:
        render_smc_short_results_subtab()


def render_btc_regime_banner_short():
    """BTC regime banner — meaning is inverted for shorts."""
    try:
        btc_regime = _scanner_btc_regime_for_combos()
    except Exception:
        btc_regime = "UNKNOWN"

    if btc_regime == "BEAR":
        st.success("🟢 **BTC Regime: BEAR** — Favorable macro for short setups.")
    elif btc_regime == "CHOP":
        st.warning("🟡 **BTC Regime: CHOP** — Mixed macro. Proceed with caution.")
    elif btc_regime == "BULL":
        st.error("🔴 **BTC Regime: BULL** — Short setups fight macro. Cards will be tagged ⚠️ FIGHTS MACRO.")
    else:
        st.info("⚪ **BTC Regime: UNKNOWN** — Could not determine. Verify manually.")

    st.session_state["smc_short_btc_regime"] = btc_regime


def render_smc_short_prefilter_subtab():
    """Sub-tab 1: pre-filter with bearish presets."""
    st.markdown("### Pre-Filter: Select Short Candidates")
    st.caption("Choose a filtering mode, then click Run to fetch and filter the universe.")

    mode = st.radio(
        "Filter Mode",
        options=["B. CoinAnk Short Screener", "A. Full Scan", "C. Custom Filter"],
        index=0,
        horizontal=True,
        help=(
            "- A. Full Scan: All USDT perpetuals (~340 coins). Slow but complete.\n"
            "- B. CoinAnk Short Screener: Top N ranked by bearish composite. Default.\n"
            "- C. Custom Filter: User-adjustable for specific bearish hypotheses."
        ),
        key="smc_short_prefilter_mode",
    )

    params = {}
    if mode.startswith("A"):
        params["min_volume_usd"] = st.number_input(
            "Min 24h Volume USD",
            min_value=0, value=100_000, step=50_000,
            key="smc_short_full_min_vol",
        )
    elif mode.startswith("B"):
        col1, col2 = st.columns(2)
        with col1:
            params["top_n"] = st.slider("Top N", 5, 50, 25, key="smc_short_top_n")
        with col2:
            params["preset"] = st.selectbox(
                "Preset Filter (bearish)",
                options=["bearish_money_flow", "avoid_crowded_short", "all"],
                index=0,
                key="smc_short_preset",
                help=(
                    "- bearish_money_flow: OI ↓ + Vol ↑ + funding > 0 (longs paying).\n"
                    "- avoid_crowded_short: funding > 0 + L/S > 1.5 (avoid packed shorts).\n"
                    "- all: no preset filter, pure composite ranking."
                ),
            )
    else:  # C. Custom
        col1, col2 = st.columns(2)
        with col1:
            params["min_volume_24h_usd"] = st.number_input(
                "Min 24h Vol USD", value=1_000_000, step=100_000,
                key="smc_short_cust_vol",
            )
            # For SHORT custom: user sets MAX OI 4h pct (looking for OI drops)
            max_oi = st.number_input(
                "Max OI 4h % change", value=-5.0, step=1.0,
                key="smc_short_cust_oi",
                help="OI dropping = bearish flow. Set negative.",
            )
            params["min_oi_4h_pct"] = max_oi if max_oi != 0 else None
            # For SHORT: max price 24h pct (looking for weakness)
            max_price = st.number_input(
                "Max Price 24h %", value=0.0, step=0.5,
                key="smc_short_cust_pct",
            )
            params["min_price_24h_pct"] = None   # we want price WEAK, not strong
            # If user wants strict weakness, they can use custom on max_price
        with col2:
            # For SHORT: max funding negative (avoid crowded shorts), but reuse
            # the LONG max_funding_rate_pct semantic — user can flip sign.
            max_fund = st.number_input(
                "Max Funding Rate %", value=0.10, step=0.01,
                key="smc_short_cust_fund",
                help="Positive funding = longs paying = short fuel.",
            )
            params["max_funding_rate_pct"] = max_fund if max_fund > 0 else None

            min_ls = st.number_input(
                "Min Top Trader L/S Ratio", value=1.5, step=0.1,
                key="smc_short_cust_ls",
                help="L/S > 1.5 = bullish positioning = short fuel.",
            )
            # The LONG custom uses MAX L/S; for SHORT we want MIN.
            # Workaround: we pass max_top_trader_ls = a very large number to skip
            # that filter, and apply min_ls manually via dataframe filter after.
            params["max_top_trader_ls"] = None
            params["_min_ls_short"] = min_ls if min_ls > 0 else None

            params["sort_by"] = st.selectbox(
                "Sort By",
                options=["composite", "oi_change", "vol_change", "price_change", "alphabetical"],
                key="smc_short_cust_sort",
            )
            params["limit"] = st.number_input(
                "Limit Results", value=50, min_value=5, max_value=200,
                key="smc_short_cust_limit",
            )

    # ── Run button ───────────────────────────────────────────────────────────
    if st.button("Run Pre-Filter (Short)", type="primary", key="smc_short_run_prefilter"):
        with st.spinner("Fetching universe data..."):
            df_universe = fetch_screener_universe(top_n=400)

        if df_universe is None or df_universe.empty:
            st.error("Universe fetch returned no data. Check Binance API connectivity.")
            return

        with st.spinner("Filtering bearish candidates..."):
            if mode.startswith("A"):
                symbols = mode_full_scan(df_universe, min_volume_usd=params["min_volume_usd"])
            elif mode.startswith("B"):
                symbols = mode_coinank_screener_short(
                    df_universe,
                    top_n=params["top_n"],
                    apply_preset=params["preset"],
                )
            else:
                # Custom — reuse LONG custom filter; then apply min_ls manually
                min_ls_short = params.pop("_min_ls_short", None)
                custom_params = {k: v for k, v in params.items() if not k.startswith("_")}
                symbols = mode_custom_filter(df_universe, **custom_params)
                if min_ls_short is not None and symbols:
                    df_filtered = df_universe[df_universe["symbol"].isin(symbols)]
                    df_filtered = df_filtered.loc[
                        df_filtered["top_trader_ls_ratio"].fillna(0.0) >= min_ls_short
                    ]
                    symbols = df_filtered["symbol"].tolist()

        st.success(f"✅ Found {len(symbols)} bearish candidate coins.")

        st.session_state["smc_short_universe_df"] = df_universe
        st.session_state["smc_short_candidates"]  = symbols

        render_screener_table(df_universe, symbols, mode[:1])

    # ── Show cached candidates ───────────────────────────────────────────────
    if st.session_state.get("smc_short_candidates"):
        candidates = st.session_state["smc_short_candidates"]
        st.markdown(f"#### Currently selected: **{len(candidates)} coins**")
        st.code(
            ", ".join(candidates[:30]) + ("..." if len(candidates) > 30 else ""),
            language=None,
        )


def render_smc_short_mode_scan_subtab():
    """Sub-tab 2: mode + scan execution for SHORT."""
    st.markdown("### Mode & Scan (Short)")

    if not st.session_state.get("smc_short_candidates"):
        st.warning("No candidates selected. Go to '1. Pre-Filter' first.")
        return

    candidates = st.session_state["smc_short_candidates"]
    st.info(f"Scanning **{len(candidates)}** candidate coins for SHORT setups.")

    mode = st.selectbox(
        "Scan Mode",
        options=["DAY", "SWING", "SCALP"],
        index=0,
        key="smc_short_scan_mode",
        help=(
            "- SWING: HTF=1D, LTF=4H. Hold 3-14 days.\n"
            "- DAY:   HTF=4H, LTF=15m. Hold 1-3 days. (default)\n"
            "- SCALP: HTF=1H, LTF=5m. Hold <24 hours."
        ),
    )
    mode_cfg = MODE_CONFIG_SHORT[mode]
    st.caption(
        f"HTF: {mode_cfg['htf_label']} | "
        f"LTF: {mode_cfg['ltf_label']} | "
        f"Lookback: {mode_cfg['lookback']} bars"
    )

    with st.expander("⚙️ Advanced Settings", expanded=False):
        atr_multiplier = st.slider(
            "Fibo 0.786 zone tolerance (×ATR)",
            min_value=0.3, max_value=2.0, value=0.5, step=0.1,
            key="smc_short_atr_mult",
            help=(
                "Width of Fibo 0.786 zone in multiples of ATR. "
                "Lower = stricter, fewer setups. Higher = wider, more setups."
            ),
        )
        st.session_state["smc_short_atr_multiplier"] = atr_multiplier

        # Phase 2: opt-in eager backtest
        run_bt_eager = st.checkbox(
            "Run 24-variant backtest during scan (slower)",
            value=False,
            key="smc_short_run_backtest",
            help=(
                "If checked, the scan runs the full 24-variant backtest grid for "
                "EVERY coin (much slower — minutes for large pre-filters). If "
                "unchecked (default), the scan is fast and you click Deep Dive on "
                "individual coins to run their backtest on demand."
            ),
        )
        st.session_state["smc_short_run_backtest_flag"] = run_bt_eager

    if st.button("🚀 Run SMC Short Scan", type="primary", key="smc_short_run_scan"):
        btc_regime = st.session_state.get("smc_short_btc_regime", "UNKNOWN")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        def progress_callback(idx: int, total: int, symbol: str):
            progress_bar.progress(idx / total)
            progress_text.text(f"Scanning {idx}/{total}: {symbol}")

        run_bt = st.session_state.get("smc_short_run_backtest_flag", False)
        spinner_msg = (
            "Running SMC SHORT scan with backtest (slower)..."
            if run_bt else "Running SMC SHORT scan..."
        )
        with st.spinner(spinner_msg):
            results = run_scan_short(
                mode=mode,
                symbols=candidates,
                btc_regime=btc_regime,
                atr_multiplier=st.session_state.get("smc_short_atr_multiplier", 0.5),
                run_backtest=run_bt,
                progress_callback=progress_callback,
            )

        progress_bar.progress(1.0)
        progress_text.text(f"✅ Scan complete. Found {len(results)} short setups.")

        st.session_state["smc_short_results"]      = results
        st.session_state["smc_short_results_mode"] = mode

        if not results:
            st.warning("No short setups found. Try a different mode or broaden the pre-filter.")
        else:
            st.success(f"Found **{len(results)}** actionable SHORT setups. See '3. Results' tab.")


def render_smc_short_results_subtab():
    """Sub-tab 3: signal cards grouped by zone type."""
    st.markdown("### Scan Results (Short)")

    if not st.session_state.get("smc_short_results"):
        st.warning("No scan results yet. Go to '2. Mode & Scan' and run a scan.")
        return

    results = st.session_state["smc_short_results"]
    mode    = st.session_state.get("smc_short_results_mode", "UNKNOWN")

    st.caption(f"Mode: **{mode}** | {len(results)} short setups found")

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

    icons = {"smart_ob": "🟥", "fvg": "🟧", "fibo_786": "🟪", "sr": "🟨"}
    labels = {
        "smart_ob": "Bearish OB Setups",
        "fvg":      "Bearish FVG Setups",
        "fibo_786": "Fibo 0.786 Setups",
        "sr":       "Resistance S/R Setups",
    }

    any_shown = False
    for zone_type in _ZONE_ORDER:
        items = groups[zone_type]
        if not items:
            continue
        any_shown = True
        st.markdown(f"## {icons[zone_type]} {labels[zone_type]} ({len(items)})")
        for r in items:
            render_signal_card_short(r)

    if ungrouped:
        st.markdown(f"## ⬜ Other Setups ({len(ungrouped)})")
        for r in ungrouped:
            render_signal_card_short(r)
        any_shown = True

    if not any_shown:
        st.info("All result entries had unrecognised zone types.")
