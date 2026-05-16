"""
qf_smc_short/screener.py — Pre-Filter Screener for SHORT setups
================================================================
Reuses the universe fetcher and rendering helpers from qf_smc/screener.py;
adds bearish-flavored presets and a SHORT-tuned composite score.

Modes:
  - A: Full Scan        — same as LONG (volume floor only)
  - B: CoinAnk Short    — composite score tuned for shorts, top N
  - C: Custom Filter    — same as LONG (params just inverted by user)

Bearish presets for Mode B:
  - bearish_money_flow:    OI down, vol up, funding > 0 (crowded longs)
  - avoid_crowded_short:   funding < 0 (don't add to packed shorts), L/S > 1.5
  - all:                   no preset filter, just composite ranking

SHORT composite score (inverts the LONG version):
  score = 0.30 * (-norm(oi_4h_pct))         # OI dropping = bearish flow
        + 0.25 * (-norm(price_24h_pct))     # price down = momentum
        + 0.25 * norm(volume_24h_pct_change) # vol up = participation
        + 0.15 * funding_rate_pos_signal     # positive funding = longs trapped
        + 0.05 * max(0, top_trader_ls_ratio - 2.0)  # heavy long bias = short fuel
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional

# Reuse all the heavy lifting — universe fetch, normalize helper, full scan,
# custom filter, and table renderer are direction-neutral.
from qf_smc.screener import (
    fetch_screener_universe,
    mode_full_scan,
    mode_custom_filter,
    render_screener_table,
    _normalize_series,
)


# ============================================================================
# MODE B — CoinAnk Short Screener (bearish composite)
# ============================================================================

def mode_coinank_screener_short(
    df_universe: pd.DataFrame,
    top_n: int = 25,
    apply_preset: str = "bearish_money_flow",
) -> List[str]:
    """
    Mode B (SHORT) — CoinAnk-style Screener tuned for bearish setups.

    Composite score (SHORT — inverted from LONG):
        score = 0.30 * (-norm(oi_4h_pct))            # OI ↓
              + 0.25 * (-norm(price_24h_pct))        # price ↓
              + 0.25 * norm(volume_24h_pct_change)   # vol ↑
              + 0.15 * max(0, funding_rate * 100)    # positive funding = long crowding
              + 0.05 * max(0, top_trader_ls_ratio - 2.0)  # over-bullish positioning
    """
    if df_universe is None or df_universe.empty:
        return []

    df = df_universe.copy()
    df = df.dropna(subset=["volume_24h", "price_24h_pct"])

    # ── Apply preset filter ──────────────────────────────────────────────────
    if apply_preset == "bearish_money_flow":
        # OI dropping + volume rising + funding positive (longs paying)
        mask = (
            (df["oi_4h_pct"].fillna(9999) < 0)
            & (df["volume_24h_pct_change"].fillna(-9999) > 0)
            & (df["funding_rate"].fillna(-9999) > 0.0001)
        )
        df = df.loc[mask]

    elif apply_preset == "below_emas":
        # EMA check deferred to caller — no filter applied here.
        pass

    elif apply_preset == "avoid_crowded_short":
        # Skip coins where shorts are already heavy (negative funding) and
        # require a long-biased L/S ratio (which is our short fuel)
        mask = (
            (df["funding_rate"].fillna(-9999) > -0.0001)
            & (df["top_trader_ls_ratio"].fillna(0.0) > 1.5)
        )
        df = df.loc[mask]

    elif apply_preset == "all":
        pass

    else:
        pass   # unknown preset → all

    if df.empty:
        return []

    # ── Composite SHORT score ────────────────────────────────────────────────
    n_oi4h  = _normalize_series(df["oi_4h_pct"].fillna(df["oi_4h_pct"].median()))
    n_p24h  = _normalize_series(df["price_24h_pct"].fillna(df["price_24h_pct"].median()))
    n_vol24 = _normalize_series(df["volume_24h_pct_change"].fillna(0.0))

    # Positive funding is a SHORT fuel signal (longs paying)
    funding_short_signal = df["funding_rate"].fillna(0.0).clip(lower=0.0) * 100

    # L/S ratio above 2.0 = over-bullish, more fuel for shorts
    ls_short_signal = (df["top_trader_ls_ratio"].fillna(0.0) - 2.0).clip(lower=0.0)

    score = (
        0.30 * (-n_oi4h)     # OI dropping is bearish
        + 0.25 * (-n_p24h)   # price falling is bearish momentum
        + 0.25 * n_vol24     # volume confirms either direction
        + 0.15 * funding_short_signal
        + 0.05 * ls_short_signal
    )

    df = df.copy()
    df["_score"] = score
    df = df.sort_values("_score", ascending=False)

    return df["symbol"].head(top_n).tolist()
