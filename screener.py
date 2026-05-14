"""
qf_smc/screener.py — 3-Mode Pre-Filter Screener
================================================
CoinAnk-style pre-filter for the SMC Long scanner per spec v1.1 §5.4.

Modes:
  - A: Full Scan (all USDT perpetuals)
  - B: CoinAnk Screener (composite score, top 25)
  - C: Custom Filter (user-tunable)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any

from qf_shared import (
    _scanner_get_universe_all,
    fetch_open_interest,
    fetch_funding_rate,
)


# ============================================================================
# CONSTANTS
# ============================================================================

_FAPI_BASE = "https://fapi.binance.com"
_REQUEST_TIMEOUT = 5  # seconds; keep tight so ThreadPool doesn't stall


# ============================================================================
# INTERNAL PER-SYMBOL HELPERS
# ============================================================================

def _safe_get(url: str, params: dict = None, timeout: int = _REQUEST_TIMEOUT) -> Optional[Any]:
    """
    GET url with params; return parsed JSON or None on any failure.
    Never raises — callers treat None as missing data.
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _pct_change_from_hist(hist: list, value_key: str) -> float:
    """
    Given an OI-hist list (oldest → newest, length >= 2),
    return (newest - oldest) / oldest * 100.
    Returns NaN if the list is too short or values are invalid.
    """
    try:
        if not hist or len(hist) < 2:
            return float("nan")
        oldest = float(hist[0][value_key])
        newest = float(hist[-1][value_key])
        if oldest == 0:
            return float("nan")
        return (newest - oldest) / abs(oldest) * 100.0
    except Exception:
        return float("nan")


def _fetch_oi_hist_pct(symbol: str, period: str) -> float:
    """
    Fetch /futures/data/openInterestHist for one period and return % change.
    period: "15m" | "1h" | "4h" | "1d"
    """
    data = _safe_get(
        f"{_FAPI_BASE}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": period, "limit": 2},
    )
    if not data or not isinstance(data, list):
        return float("nan")
    return _pct_change_from_hist(data, "sumOpenInterestValue")


def _fetch_top_trader_ls_ratio(symbol: str, period: str = "1h") -> float:
    """
    Fetch /futures/data/topLongShortAccountRatio for one symbol.
    Returns longShortRatio as float, or NaN on failure.
    """
    data = _safe_get(
        f"{_FAPI_BASE}/futures/data/topLongShortAccountRatio",
        params={"symbol": symbol, "period": period, "limit": 1},
    )
    try:
        if data and isinstance(data, list) and data:
            return float(data[0]["longShortRatio"])
    except Exception:
        pass
    return float("nan")


def _fetch_kline_pct_change(symbol: str, interval: str, lookback: int = 2) -> float:
    """
    Fetch the last `lookback` klines and return % change open-to-close
    from oldest-open to latest-close. Uses spot /api/v3/klines.
    """
    urls = [
        ("https://api.binance.com/api/v3/klines", True),
        ("https://data-api.binance.vision/api/v3/klines", False),
    ]
    for url, verify in urls:
        try:
            resp = requests.get(
                url,
                params={"symbol": symbol, "interval": interval, "limit": lookback},
                timeout=_REQUEST_TIMEOUT,
                verify=verify,
            )
            if resp.status_code != 200:
                continue
            klines = resp.json()
            if len(klines) < lookback:
                continue
            open_price  = float(klines[0][1])   # oldest open
            close_price = float(klines[-1][4])  # newest close
            if open_price == 0:
                continue
            return (close_price - open_price) / open_price * 100.0
        except Exception:
            continue
    return float("nan")


def _fetch_volume_pct_change(symbol: str, interval: str, lookback: int = 2) -> float:
    """
    Return % change of traded volume from the oldest bar to the newest bar.
    Positive = more volume recently.
    """
    urls = [
        ("https://api.binance.com/api/v3/klines", True),
        ("https://data-api.binance.vision/api/v3/klines", False),
    ]
    for url, verify in urls:
        try:
            resp = requests.get(
                url,
                params={"symbol": symbol, "interval": interval, "limit": lookback},
                timeout=_REQUEST_TIMEOUT,
                verify=verify,
            )
            if resp.status_code != 200:
                continue
            klines = resp.json()
            if len(klines) < lookback:
                continue
            old_vol = float(klines[0][5])
            new_vol = float(klines[-1][5])
            if old_vol == 0:
                continue
            return (new_vol - old_vol) / old_vol * 100.0
        except Exception:
            continue
    return float("nan")


def _fetch_one_symbol_data(symbol: str) -> dict:
    """
    Fetch all per-symbol data that can't be obtained from batch endpoints:
      - OI changes (15m / 1h / 4h / 24h)
      - top-trader L/S ratio
      - price change for 15m, 7d
      - volume change 1h
    Returns a dict keyed by column name.  Missing values are NaN — never raises.
    """
    out: dict = {"symbol": symbol}

    # ── OI % changes ────────────────────────────────────────────────────────
    out["oi_15m_pct"] = _fetch_oi_hist_pct(symbol, "15m")
    out["oi_1h_pct"]  = _fetch_oi_hist_pct(symbol, "1h")
    out["oi_4h_pct"]  = _fetch_oi_hist_pct(symbol, "4h")
    out["oi_24h_pct"] = _fetch_oi_hist_pct(symbol, "1d")

    # ── Top-trader L/S ratio ─────────────────────────────────────────────────
    out["top_trader_ls_ratio"] = _fetch_top_trader_ls_ratio(symbol, period="1h")

    # ── Price % change 15m (2 × 15m klines) ──────────────────────────────────
    out["price_15m_pct"] = _fetch_kline_pct_change(symbol, "15m", lookback=2)

    # ── Price % change 7d (2 × 1w klines from spot) ─────────────────────────
    # Using 7 × 1d bars for a more reliable 7d pct.
    out["price_7d_pct"] = _fetch_kline_pct_change(symbol, "1d", lookback=8)

    # ── Volume % change 1h (2 × 1h klines) ──────────────────────────────────
    out["volume_1h_pct_change"] = _fetch_volume_pct_change(symbol, "1h", lookback=2)

    return out


# ============================================================================
# BATCH ENDPOINT HELPERS
# ============================================================================

def _fetch_24h_tickers() -> Dict[str, dict]:
    """
    Fetch all Binance spot 24h tickers in one call.
    Returns {symbol: ticker_dict}.  Falls back to mirror on failure.
    """
    urls = [
        ("https://api.binance.com/api/v3/ticker/24hr", True),
        ("https://data-api.binance.vision/api/v3/ticker/24hr", False),
    ]
    for url, verify in urls:
        try:
            resp = requests.get(url, timeout=15, verify=verify)
            if resp.status_code == 200:
                tickers = resp.json()
                return {t["symbol"]: t for t in tickers if "symbol" in t}
        except Exception:
            continue
    return {}


def _fetch_premium_index_all() -> Dict[str, dict]:
    """
    Fetch /fapi/v1/premiumIndex for ALL symbols (no symbol param).
    Returns {symbol: {"markPrice", "lastFundingRate", ...}}.
    """
    data = _safe_get(f"{_FAPI_BASE}/fapi/v1/premiumIndex", timeout=10)
    if not data or not isinstance(data, list):
        return {}
    return {item["symbol"]: item for item in data if "symbol" in item}


# ============================================================================
# MAIN UNIVERSE FETCH
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)  # 5-min cache
def fetch_screener_universe(
    min_volume_usd: float = 0,
    top_n: int = 400,
) -> pd.DataFrame:
    """
    Fetch all data needed by all three screener modes in one batch.

    Returns DataFrame with columns:
      symbol, price, market_cap, volume_24h,
      price_15m_pct, price_1h_pct, price_4h_pct, price_24h_pct, price_7d_pct,
      volume_1h_pct_change, volume_24h_pct_change,
      oi_15m_pct, oi_1h_pct, oi_4h_pct, oi_24h_pct,
      funding_rate, top_trader_ls_ratio, oi_vol_ratio
    """

    # ── 1. Universe list ─────────────────────────────────────────────────────
    universe_raw = _scanner_get_universe_all()   # [{symbol, volume_24h, price}]
    if not universe_raw:
        return pd.DataFrame()

    # Apply volume gate + top-N cap
    if min_volume_usd > 0:
        universe_raw = [u for u in universe_raw if u["volume_24h"] >= min_volume_usd]
    universe_raw = universe_raw[:top_n]
    symbols = [u["symbol"] for u in universe_raw]

    # ── 2. Batch: 24h spot tickers (price, volume, pct changes) ─────────────
    tickers_map = _fetch_24h_tickers()

    # ── 3. Batch: premiumIndex (mark price + funding rate for futures) ───────
    premium_map = _fetch_premium_index_all()

    # ── 4. Per-symbol threaded fetches ───────────────────────────────────────
    per_sym_results: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures_map = {ex.submit(_fetch_one_symbol_data, sym): sym for sym in symbols}
        for fut in as_completed(futures_map):
            sym = futures_map[fut]
            try:
                per_sym_results[sym] = fut.result()
            except Exception:
                per_sym_results[sym] = {"symbol": sym}

    # ── 5. Assemble DataFrame ────────────────────────────────────────────────
    rows = []
    for u in universe_raw:
        sym = u["symbol"]
        ticker  = tickers_map.get(sym, {})
        premium = premium_map.get(sym, {})
        per_sym = per_sym_results.get(sym, {})

        # ── Price ─────────────────────────────────────────────────────────────
        try:
            price = float(ticker.get("lastPrice") or u.get("price") or 0)
        except Exception:
            price = float(u.get("price", 0))

        # ── Volume ────────────────────────────────────────────────────────────
        try:
            volume_24h = float(ticker.get("quoteVolume") or u.get("volume_24h") or 0)
        except Exception:
            volume_24h = float(u.get("volume_24h", 0))

        # ── 24h price % change (from ticker.priceChangePercent) ───────────────
        try:
            price_24h_pct = float(ticker.get("priceChangePercent", 0))
        except Exception:
            price_24h_pct = float("nan")

        # ── 1h price % change (2-bar 1h kline — cheap, comes from per_sym)
        # We compute price_1h_pct from a 1h kline fetch; add it to per_sym pipeline.
        # For now derive from per_sym if available, else NaN.
        try:
            price_1h_pct = float(per_sym.get("price_1h_pct", float("nan")))
        except Exception:
            price_1h_pct = float("nan")

        # ── 4h price % change
        try:
            price_4h_pct = float(per_sym.get("price_4h_pct", float("nan")))
        except Exception:
            price_4h_pct = float("nan")

        # ── Funding rate: prefer premiumIndex (free batch), fall back to NaN ─
        try:
            funding_rate = float(premium.get("lastFundingRate", float("nan")))
        except Exception:
            funding_rate = float("nan")

        # ── OI / volume ratio ─────────────────────────────────────────────────
        # We don't have an absolute OI value cheaply in batch, so we skip this
        # until the threaded fetch provides it.  Set to NaN; could be enriched
        # by calling fetch_open_interest per symbol — but that doubles the calls.
        # For Mode B/C, oi_vol_ratio is informational and not part of the score.
        oi_vol_ratio = float("nan")

        # ── 24h volume % change — approximate from ticker's quoteVolume vs
        # previous 24h. Binance doesn't expose this directly; use per_sym fallback.
        try:
            vol_24h_pct_change = float(per_sym.get("volume_24h_pct_change", float("nan")))
        except Exception:
            vol_24h_pct_change = float("nan")

        rows.append({
            "symbol":               sym,
            "price":                price,
            "market_cap":           0.0,          # CoinGecko batch would be a separate call; omitted for speed
            "volume_24h":           volume_24h,
            "price_15m_pct":        per_sym.get("price_15m_pct", float("nan")),
            "price_1h_pct":         price_1h_pct,
            "price_4h_pct":         price_4h_pct,
            "price_24h_pct":        price_24h_pct,
            "price_7d_pct":         per_sym.get("price_7d_pct", float("nan")),
            "volume_1h_pct_change": per_sym.get("volume_1h_pct_change", float("nan")),
            "volume_24h_pct_change":vol_24h_pct_change,
            "oi_15m_pct":           per_sym.get("oi_15m_pct", float("nan")),
            "oi_1h_pct":            per_sym.get("oi_1h_pct",  float("nan")),
            "oi_4h_pct":            per_sym.get("oi_4h_pct",  float("nan")),
            "oi_24h_pct":           per_sym.get("oi_24h_pct", float("nan")),
            "funding_rate":         funding_rate,
            "top_trader_ls_ratio":  per_sym.get("top_trader_ls_ratio", float("nan")),
            "oi_vol_ratio":         oi_vol_ratio,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure all required columns are present as floats (guard against partial rows)
    _float_cols = [
        "price", "market_cap", "volume_24h",
        "price_15m_pct", "price_1h_pct", "price_4h_pct", "price_24h_pct", "price_7d_pct",
        "volume_1h_pct_change", "volume_24h_pct_change",
        "oi_15m_pct", "oi_1h_pct", "oi_4h_pct", "oi_24h_pct",
        "funding_rate", "top_trader_ls_ratio", "oi_vol_ratio",
    ]
    for col in _float_cols:
        if col not in df.columns:
            df[col] = float("nan")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


# ============================================================================
# INTERNAL: per-symbol 1h/4h klines (added to threaded worker)
# We extend _fetch_one_symbol_data implicitly by patching the result dict.
# To keep the public API clean, we override the per-symbol worker to also
# fetch 1h and 4h price pct changes.
# ============================================================================

# Monkey-patch: replace _fetch_one_symbol_data with an extended version that
# also returns price_1h_pct, price_4h_pct, and volume_24h_pct_change.

def _fetch_one_symbol_data(symbol: str) -> dict:  # noqa: F811 — intentional override
    """
    Full per-symbol data fetch (runs inside ThreadPoolExecutor).
    Gathers OI hist, top-trader ratio, kline-based price/volume changes.
    Never raises — missing data becomes NaN.
    """
    out: dict = {"symbol": symbol}

    # ── OI % changes (4 periods) ─────────────────────────────────────────────
    out["oi_15m_pct"] = _fetch_oi_hist_pct(symbol, "15m")
    out["oi_1h_pct"]  = _fetch_oi_hist_pct(symbol, "1h")
    out["oi_4h_pct"]  = _fetch_oi_hist_pct(symbol, "4h")
    out["oi_24h_pct"] = _fetch_oi_hist_pct(symbol, "1d")

    # ── Top-trader L/S ratio ─────────────────────────────────────────────────
    out["top_trader_ls_ratio"] = _fetch_top_trader_ls_ratio(symbol, period="1h")

    # ── Price % changes from klines ──────────────────────────────────────────
    out["price_15m_pct"] = _fetch_kline_pct_change(symbol, "15m", lookback=2)
    out["price_1h_pct"]  = _fetch_kline_pct_change(symbol, "1h",  lookback=2)
    out["price_4h_pct"]  = _fetch_kline_pct_change(symbol, "4h",  lookback=2)
    out["price_7d_pct"]  = _fetch_kline_pct_change(symbol, "1d",  lookback=8)

    # ── Volume % changes ─────────────────────────────────────────────────────
    out["volume_1h_pct_change"]  = _fetch_volume_pct_change(symbol, "1h", lookback=2)
    # 24h volume change: compare yesterday's 24h bar vs the bar before that.
    out["volume_24h_pct_change"] = _fetch_volume_pct_change(symbol, "1d", lookback=2)

    return out


# ============================================================================
# THREE SCREENER MODES
# ============================================================================

def mode_full_scan(
    df_universe: pd.DataFrame,
    min_volume_usd: float = 100_000,
) -> List[str]:
    """
    Mode A — Full Scan.

    Returns ALL symbols with vol >= min_volume_usd, sorted alphabetically.
    No ranking, no filtering on OI/funding.
    """
    if df_universe is None or df_universe.empty:
        return []

    mask = df_universe["volume_24h"] >= min_volume_usd
    filtered = df_universe.loc[mask, "symbol"].dropna().tolist()
    return sorted(filtered)


# ---------------------------------------------------------------------------

def _normalize_series(s: pd.Series) -> pd.Series:
    """
    Robust z-score normalization: (x - median) / std.
    Returns NaN where std == 0 or all-NaN.
    """
    std = s.std(skipna=True)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.median()) / std


def mode_coinank_screener(
    df_universe: pd.DataFrame,
    top_n: int = 25,
    apply_preset: str = "bullish_money_flow",
) -> List[str]:
    """
    Mode B — CoinAnk-style Screener.

    Applies a preset filter, ranks by composite score, returns top N symbols.

    Composite score:
        score = 0.30 * norm(oi_4h_pct)
              + 0.25 * norm(price_24h_pct)
              + 0.25 * norm(volume_24h_pct_change)
              - 0.15 * abs(funding_rate) * 100
              - 0.05 * max(0, top_trader_ls_ratio - 3)
    """
    if df_universe is None or df_universe.empty:
        return []

    df = df_universe.copy()

    # ── Drop rows missing the two most critical columns ───────────────────────
    df = df.dropna(subset=["volume_24h", "price_24h_pct"])

    # ── Apply preset filter ──────────────────────────────────────────────────
    if apply_preset == "bullish_money_flow":
        mask = (
            (df["oi_4h_pct"].fillna(-9999) > 0)
            & (df["volume_24h_pct_change"].fillna(-9999) > 0)
            & (df["funding_rate"].fillna(9999) < 0.0005)
        )
        df = df.loc[mask]

    elif apply_preset == "above_emas":
        # EMA check deferred to caller — no filter applied here per spec.
        pass

    elif apply_preset == "avoid_crowded_long":
        mask = (
            (df["funding_rate"].fillna(9999) < 0.0003)
            & (df["top_trader_ls_ratio"].fillna(9999) < 2.5)
        )
        df = df.loc[mask]

    elif apply_preset == "all":
        pass  # no filter

    else:
        # Unknown preset — treat as "all"
        pass

    if df.empty:
        return []

    # ── Composite score ──────────────────────────────────────────────────────
    n_oi4h  = _normalize_series(df["oi_4h_pct"].fillna(df["oi_4h_pct"].median()))
    n_p24h  = _normalize_series(df["price_24h_pct"].fillna(df["price_24h_pct"].median()))
    n_vol24 = _normalize_series(df["volume_24h_pct_change"].fillna(0.0))

    funding_penalty   = df["funding_rate"].abs().fillna(0.0) * 100
    ls_ratio_penalty  = (df["top_trader_ls_ratio"].fillna(3.0) - 3.0).clip(lower=0.0)

    score = (
        0.30 * n_oi4h
        + 0.25 * n_p24h
        + 0.25 * n_vol24
        - 0.15 * funding_penalty
        - 0.05 * ls_ratio_penalty
    )

    df = df.copy()
    df["_score"] = score
    df = df.sort_values("_score", ascending=False)

    return df["symbol"].head(top_n).tolist()


# ---------------------------------------------------------------------------

def mode_custom_filter(
    df_universe: pd.DataFrame,
    min_volume_24h_usd: float = 1_000_000,
    min_oi_4h_pct: Optional[float] = 5.0,
    max_funding_rate_pct: Optional[float] = 0.05,
    min_price_24h_pct: Optional[float] = 0.0,
    max_top_trader_ls: Optional[float] = 3.0,
    sort_by: str = "composite",
    limit: int = 50,
) -> List[str]:
    """
    Mode C — Custom Filter.

    Any parameter set to None is IGNORED (no filter applied for that param).

    sort_by: "composite" | "oi_change" | "vol_change" | "price_change" | "alphabetical"
    """
    if df_universe is None or df_universe.empty:
        return []

    df = df_universe.copy()

    # ── Hard volume minimum (always applied, never None) ──────────────────────
    df = df.loc[df["volume_24h"] >= min_volume_24h_usd]

    # ── Optional filters — NaN rows treated as failing each filter ────────────
    if min_oi_4h_pct is not None:
        df = df.loc[df["oi_4h_pct"].fillna(-9999) >= min_oi_4h_pct]

    if max_funding_rate_pct is not None:
        # max_funding_rate_pct is in percent (e.g. 0.05 → 0.05%)
        # funding_rate column is fractional (0.0005 = 0.05%)
        threshold_frac = max_funding_rate_pct / 100.0
        df = df.loc[df["funding_rate"].fillna(9999) <= threshold_frac]

    if min_price_24h_pct is not None:
        df = df.loc[df["price_24h_pct"].fillna(-9999) >= min_price_24h_pct]

    if max_top_trader_ls is not None:
        df = df.loc[df["top_trader_ls_ratio"].fillna(9999) <= max_top_trader_ls]

    if df.empty:
        return []

    # ── Sort ─────────────────────────────────────────────────────────────────
    if sort_by == "composite":
        # Reuse the composite scoring logic (no preset filter)
        n_oi4h  = _normalize_series(df["oi_4h_pct"].fillna(0.0))
        n_p24h  = _normalize_series(df["price_24h_pct"].fillna(0.0))
        n_vol24 = _normalize_series(df["volume_24h_pct_change"].fillna(0.0))
        funding_penalty  = df["funding_rate"].abs().fillna(0.0) * 100
        ls_penalty       = (df["top_trader_ls_ratio"].fillna(3.0) - 3.0).clip(lower=0.0)
        df = df.copy()
        df["_sort_key"] = (
            0.30 * n_oi4h
            + 0.25 * n_p24h
            + 0.25 * n_vol24
            - 0.15 * funding_penalty
            - 0.05 * ls_penalty
        )
        df = df.sort_values("_sort_key", ascending=False)

    elif sort_by == "oi_change":
        df = df.sort_values("oi_4h_pct", ascending=False, na_position="last")

    elif sort_by == "vol_change":
        df = df.sort_values("volume_24h_pct_change", ascending=False, na_position="last")

    elif sort_by == "price_change":
        df = df.sort_values("price_24h_pct", ascending=False, na_position="last")

    elif sort_by == "alphabetical":
        df = df.sort_values("symbol", ascending=True)

    else:
        # Unknown sort_by — fall back to alphabetical
        df = df.sort_values("symbol", ascending=True)

    return df["symbol"].head(limit).tolist()


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def render_screener_table(
    df_universe: pd.DataFrame,
    selected_symbols: List[str],
    mode: str,
) -> None:
    """
    Render screener results as a color-coded Streamlit dataframe.

    Args:
        df_universe:      full DataFrame from fetch_screener_universe()
        selected_symbols: symbols to display
        mode:             "FULL" | "COINANK" | "CUSTOM"
    """
    if df_universe is None or df_universe.empty or not selected_symbols:
        st.info("No symbols to display.")
        return

    # Filter to selected rows, preserve symbol order from selected_symbols
    sym_order = {s: i for i, s in enumerate(selected_symbols)}
    df = df_universe.loc[df_universe["symbol"].isin(selected_symbols)].copy()
    df["_order"] = df["symbol"].map(sym_order)
    df = df.sort_values("_order").drop(columns=["_order"])

    # ── Column selection per mode ─────────────────────────────────────────────
    _base_cols  = ["symbol", "price", "volume_24h", "price_24h_pct"]
    _oi_cols    = ["oi_1h_pct", "oi_4h_pct", "oi_24h_pct"]
    _extra_full = ["price_1h_pct", "price_4h_pct", "price_7d_pct", "volume_1h_pct_change"]
    _score_cols = ["funding_rate", "top_trader_ls_ratio"]

    if mode == "FULL":
        show_cols = _base_cols + _extra_full + _oi_cols
    elif mode == "COINANK":
        show_cols = _base_cols + _oi_cols + _score_cols + ["volume_24h_pct_change"]
    else:  # CUSTOM
        show_cols = _base_cols + _oi_cols + _score_cols + ["volume_24h_pct_change", "price_1h_pct", "price_4h_pct"]

    # Keep only columns that actually exist in df
    show_cols = [c for c in show_cols if c in df.columns]
    df_display = df[show_cols].copy()

    # ── Select All checkbox ───────────────────────────────────────────────────
    select_all = st.checkbox(
        f"Select all {len(df_display)} symbols",
        value=False,
        key=f"screener_select_all_{mode}",
    )
    if select_all:
        st.write("All symbols selected (pass `selected_symbols` list to scanner).")

    # ── Format numbers ────────────────────────────────────────────────────────
    def _fmt_pct(val):
        if pd.isna(val):
            return "—"
        return f"{val:+.2f}%"

    def _fmt_large(val):
        if pd.isna(val):
            return "—"
        if val >= 1e9:
            return f"${val/1e9:.2f}B"
        if val >= 1e6:
            return f"${val/1e6:.1f}M"
        if val >= 1e3:
            return f"${val/1e3:.0f}K"
        return f"${val:.2f}"

    def _fmt_funding(val):
        if pd.isna(val):
            return "—"
        return f"{val*100:.4f}%"

    pct_cols     = [c for c in show_cols if "pct" in c]
    funding_col  = "funding_rate" if "funding_rate" in show_cols else None
    volume_col   = "volume_24h"   if "volume_24h"   in show_cols else None
    price_col    = "price"        if "price"        in show_cols else None

    for col in pct_cols:
        df_display[col] = df_display[col].apply(_fmt_pct)
    if funding_col:
        df_display[funding_col] = df_display[funding_col].apply(_fmt_funding)
    if volume_col:
        df_display[volume_col] = df_display[volume_col].apply(_fmt_large)
    if price_col:
        df_display[price_col] = df_display[price_col].apply(
            lambda v: f"{v:,.4f}" if pd.notna(v) else "—"
        )
    if "top_trader_ls_ratio" in df_display.columns:
        df_display["top_trader_ls_ratio"] = df_display["top_trader_ls_ratio"].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
        )

    # ── Color-code: green positive %, red negative % ──────────────────────────
    def _color_pct_str(val: str):
        if isinstance(val, str) and val.startswith("+"):
            return "color: #22c55e"   # green-500
        if isinstance(val, str) and val.startswith("-"):
            return "color: #ef4444"   # red-500
        return ""

    style = df_display.style
    for col in pct_cols:
        if col in df_display.columns:
            try:
                style = style.map(_color_pct_str, subset=[col])
            except AttributeError:
                # pandas < 2.1.0 fallback
                style = style.applymap(_color_pct_str, subset=[col])

    st.dataframe(style, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------

def render_screener_controls() -> Dict[str, Any]:
    """
    Render sidebar/inline controls for the screener.

    Returns:
        {
            "mode":        "FULL" | "COINANK" | "CUSTOM",
            "params":      {...mode-specific params...},
            "run_clicked": bool,
        }
    """
    st.subheader("🔍 Screener Controls")

    mode_label = st.radio(
        "Screener Mode",
        options=["FULL", "COINANK", "CUSTOM"],
        horizontal=True,
        help=(
            "FULL: all USDT perps above volume gate | "
            "COINANK: composite score top-N | "
            "CUSTOM: user-tunable filters"
        ),
    )

    params: Dict[str, Any] = {}

    # ── Mode A ────────────────────────────────────────────────────────────────
    if mode_label == "FULL":
        params["min_volume_usd"] = st.number_input(
            "Min 24h volume (USD)",
            min_value=0,
            max_value=100_000_000,
            value=100_000,
            step=50_000,
            format="%d",
            key="full_min_vol",
        )

    # ── Mode B ────────────────────────────────────────────────────────────────
    elif mode_label == "COINANK":
        params["top_n"] = st.slider(
            "Top N symbols",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key="coinank_top_n",
        )
        params["apply_preset"] = st.selectbox(
            "Preset filter",
            options=["bullish_money_flow", "above_emas", "avoid_crowded_long", "all"],
            index=0,
            key="coinank_preset",
        )

    # ── Mode C ────────────────────────────────────────────────────────────────
    elif mode_label == "CUSTOM":
        col1, col2 = st.columns(2)
        with col1:
            params["min_volume_24h_usd"] = st.number_input(
                "Min 24h volume (USD)",
                min_value=0,
                max_value=500_000_000,
                value=1_000_000,
                step=100_000,
                format="%d",
                key="custom_min_vol",
            )
            _oi_enable = st.checkbox("Filter: min OI 4h change %", value=True, key="custom_oi_en")
            params["min_oi_4h_pct"] = (
                st.slider("Min OI 4h %", -50.0, 100.0, 5.0, 0.5, key="custom_oi_val")
                if _oi_enable else None
            )
            _fund_enable = st.checkbox("Filter: max funding rate", value=True, key="custom_fund_en")
            params["max_funding_rate_pct"] = (
                st.slider("Max funding rate (%)", 0.0, 0.5, 0.05, 0.01, key="custom_fund_val")
                if _fund_enable else None
            )

        with col2:
            _price_enable = st.checkbox("Filter: min 24h price change %", value=False, key="custom_price_en")
            params["min_price_24h_pct"] = (
                st.slider("Min price 24h %", -20.0, 50.0, 0.0, 0.5, key="custom_price_val")
                if _price_enable else None
            )
            _ls_enable = st.checkbox("Filter: max L/S ratio", value=True, key="custom_ls_en")
            params["max_top_trader_ls"] = (
                st.slider("Max L/S ratio", 0.5, 10.0, 3.0, 0.1, key="custom_ls_val")
                if _ls_enable else None
            )
            params["sort_by"] = st.selectbox(
                "Sort by",
                options=["composite", "oi_change", "vol_change", "price_change", "alphabetical"],
                index=0,
                key="custom_sort",
            )
            params["limit"] = st.slider(
                "Max results",
                min_value=5,
                max_value=200,
                value=50,
                step=5,
                key="custom_limit",
            )

    # ── Run button ────────────────────────────────────────────────────────────
    run_clicked = st.button("▶ Run Screener", type="primary", use_container_width=True)

    return {
        "mode":        mode_label,
        "params":      params,
        "run_clicked": run_clicked,
    }
