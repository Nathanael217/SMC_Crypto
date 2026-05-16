"""
Sanity test for SHORT backtest math.

Validates that:
1. Entry/SL/TP geometry is correctly flipped (SL above, TP below for shorts)
2. Win condition: bar_low <= tp (price drops to target)
3. Loss condition: bar_high >= sl (price rallies to stop)
4. R-multiple on timeout: (entry - final_price) / risk (positive when price drops)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Simulate the core trade logic for one variant
def simulate_short_trade(entry, sl, tp, future_bars):
    """
    future_bars: list of (high, low, close) tuples representing forward bars.
    Returns r_net.
    """
    risk = sl - entry
    assert risk > 0, f"SL ({sl}) must be above entry ({entry}) for SHORT"

    for high, low, close in future_bars:
        # Loss: price rallies up to SL
        if high >= sl:
            return -1.0
        # Win: price drops down to TP
        if low <= tp:
            return (entry - tp) / risk   # = positive multiple of R
    # Timeout
    return (entry - future_bars[-1][2]) / risk


def test_win_case():
    """Short at $100, SL $103, TP $94 (2R). Price drops to $93."""
    entry, sl, tp = 100.0, 103.0, 94.0
    bars = [(101, 99, 100), (100, 96, 97), (98, 93, 94)]   # price drops cleanly
    r = simulate_short_trade(entry, sl, tp, bars)
    expected = (100 - 94) / 3.0
    assert abs(r - expected) < 1e-9, f"Expected {expected}, got {r}"
    print(f"  WIN case: r={r:.4f}  ✓")


def test_loss_case():
    """Short at $100, SL $103, TP $94. Price rallies to $105."""
    entry, sl, tp = 100.0, 103.0, 94.0
    bars = [(102, 99, 101), (105, 100, 104)]   # price rallies, hits SL
    r = simulate_short_trade(entry, sl, tp, bars)
    assert r == -1.0, f"Expected -1.0, got {r}"
    print(f"  LOSS case: r={r}  ✓")


def test_timeout_profit_case():
    """Short at $100, SL $103, TP $94. Price drifts to $98 (still profitable, no TP)."""
    entry, sl, tp = 100.0, 103.0, 94.0
    bars = [(101, 97, 98), (99, 97, 98), (99, 97, 98)]   # drifts down, no hit
    r = simulate_short_trade(entry, sl, tp, bars)
    expected = (100 - 98) / 3.0
    assert abs(r - expected) < 1e-9, f"Expected {expected}, got {r}"
    print(f"  TIMEOUT-profit case: r={r:.4f}  ✓ (positive — price drifted down)")


def test_timeout_loss_case():
    """Short at $100, SL $103, TP $94. Price drifts to $101 (small loss)."""
    entry, sl, tp = 100.0, 103.0, 94.0
    bars = [(101, 99, 100), (102, 100, 101), (101, 100, 101)]
    r = simulate_short_trade(entry, sl, tp, bars)
    expected = (100 - 101) / 3.0   # = -0.333
    assert abs(r - expected) < 1e-9, f"Expected {expected}, got {r}"
    assert r < 0, f"Expected negative r, got {r}"
    print(f"  TIMEOUT-loss case: r={r:.4f}  ✓ (negative — price drifted up)")


def test_geometry_assertion():
    """Verify SL must be above entry for SHORT."""
    try:
        simulate_short_trade(entry=100, sl=98, tp=94, future_bars=[(101, 99, 100)])
        print("  FAIL: should have raised on SL below entry")
        sys.exit(1)
    except AssertionError as e:
        print(f"  Geometry guard: caught ({e})  ✓")


print("Running SHORT backtest math sanity tests...")
print()
test_win_case()
test_loss_case()
test_timeout_profit_case()
test_timeout_loss_case()
test_geometry_assertion()
print()
print("All sanity tests passed. SHORT P&L math is correctly inverted.")
