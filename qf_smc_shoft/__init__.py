"""
qf_smc_short — QUANTFLOW Smart Money Concepts SHORT Scanner
============================================================

Mirror of qf_smc/ for SHORT direction setups. Same architecture, same UX,
same backtest evidence pattern — every comparison flipped:

    LONG (qf_smc)             SHORT (qf_smc_short)
    ─────────────────         ──────────────────────
    LH/LL → CHoCH↑ → BOS↑     HH/HL → CHoCH↓ → BOS↓
    Uptrend confirmed         Downtrend confirmed
    Bullish OB                Bearish OB (up-close → bearish impulse)
    Bullish FVG               Bearish FVG (high[n] < low[n-2])
    Fibo of UP-leg            Fibo of DOWN-leg
    Support S/R               Resistance S/R
    EMA above                 EMA below
    Fights macro in BEAR      Fights macro in BULL
    profit = exit - entry     profit = entry - exit
    SL hits when low <= sl    SL hits when high >= sl
    TP hits when high >= tp   TP hits when low <= tp

Phase 2 modules:
    structure.py    — bearish CHoCH/BOS classifier
    zones.py        — bearish OB/FVG, Fibo down-leg, resistance S/R
    scanner.py      — orchestrator (with opt-in 24-variant backtest)
    screener.py     — bearish presets
    backtest.py     — SHORT P&L simulator + 24/72-variant grids
    ai_verdict.py   — SHORT-flavored SMC prompt template
    render.py       — signal card UI + deep-dive results

Standalone — does NOT modify qf_smc/. The two packages coexist.
"""

__version__ = "2.0.0"

# ── Re-export the main public API for app.py ────────────────────────────────
from qf_smc_short.scanner import (
    MODE_CONFIG,
    run_scan_short,
    scan_one_symbol_short,
)
