"""Market structure -- BOS and CHoCH -- by the five rules at the top of
strategy.py: swings are confirmed fractals, a break is a CLOSE beyond the
last confirmed swing, the label depends on the previous break's direction,
and a level breaks only once."""
import numpy as np

import backtest as bt
from strategy import (
    detect_order_blocks,
    detect_structure_breaks,
    find_fractals,
    ohlcv_to_df,
    structure_state,
)

STEP = 900_000


def candles(closes):
    """A price path as 15m candles: each opens at the previous close; an up
    candle gets a slightly longer upper wick and a down candle a longer
    lower one, so turning points are unambiguous swing highs / lows."""
    rows, prev = [], closes[0]
    for i, c in enumerate(closes):
        o = prev
        h, l = (c + 0.2, o - 0.1) if c >= o else (o + 0.1, c - 0.2)
        rows.append([i * STEP, o, h, l, c, 1.0])
        prev = c
    return rows


def df_of(rows):
    return ohlcv_to_df(rows)


# Up to 103, back to 101, up to 105, back to 103, up to 107, then down
# through the last higher low to 99, and back up through the last lower high.
ZIGZAG = [100, 101, 102, 103, 102, 101, 102, 103, 104, 105, 104, 103, 104, 105, 106, 107, 106, 105, 104, 103,
          102, 101, 102, 103, 102, 101, 100, 99, 100, 101, 102, 103, 104, 105, 104, 104.5]


def brief(breaks):
    return [(b["idx"], b["direction"], b["kind"]) for b in breaks]


def test_rule_1_swings_are_fractals():
    fr = find_fractals(df_of(candles(ZIGZAG)), n=2)
    assert list(fr.index[fr["swing_high"]]) == [3, 9, 15, 23, 33]
    assert list(fr.index[fr["swing_low"]]) == [5, 11, 21, 27]


def test_rules_3_and_4_bos_continues_choch_turns():
    breaks = detect_structure_breaks(df_of(candles(ZIGZAG)), n=2)
    assert brief(breaks) == [
        (8, "bullish", "BOS"),      # first break: nothing to compare with
        (14, "bullish", "BOS"),     # same direction again: continuation
        (20, "bearish", "CHoCH"),   # first close below the last higher low
        (26, "bearish", "BOS"),     # and the next one down is a BOS again
        (32, "bullish", "CHoCH"),   # first close above the last lower high
    ]
    # each break names the swing it broke
    assert [(b["swing_idx"], round(b["level"], 1)) for b in breaks] == [
        (3, 103.2), (9, 105.2), (11, 102.8), (21, 100.8), (23, 103.2)]


def test_rule_2_a_wick_through_the_level_is_not_a_break():
    rows = candles(ZIGZAG[:9])
    rows[7][2] = 104.0                      # candle 7 spikes above the 103.2 swing high...
    assert rows[7][4] < 103.2               # ...but closes back below it
    breaks = detect_structure_breaks(df_of(rows), n=2)
    assert brief(breaks) == [(8, "bullish", "BOS")]      # only the candle that CLOSED above
    assert breaks[0]["level"] == 103.2


def test_rule_5_one_break_per_swing():
    breaks = detect_structure_breaks(df_of(candles(ZIGZAG)), n=2)
    # candles 8..15 all close above the 103.2 swing high; it is broken once
    assert sum(1 for b in breaks if b["swing_idx"] == 3) == 1
    assert len({b["swing_idx"] for b in breaks}) == len(breaks)


def test_structure_state_uses_completed_candles_only():
    df = df_of(candles(ZIGZAG))
    state = structure_state(df, n=2)
    assert (state["direction"], state["kind"], state["idx"]) == ("bullish", "CHoCH", 32)
    # with candle 32 still forming, its close doesn't count yet
    state = structure_state(df_of(candles(ZIGZAG[:33])), n=2)
    assert (state["direction"], state["kind"], state["idx"]) == ("bearish", "BOS", 26)
    assert structure_state(df_of(candles(ZIGZAG[:6])), n=2) is None


def test_order_blocks_carry_the_break_they_caused():
    obs = detect_order_blocks(df_of(candles(ZIGZAG)), n=2)
    assert [(o["type"], o["structure"], o["formed_idx"], o["bos_idx"], o["swing_idx"]) for o in obs] == [
        ("bullish", "BOS", 5, 8, 3),
        ("bullish", "BOS", 11, 14, 9),
        ("bearish", "CHoCH", 15, 20, 11),
        ("bearish", "BOS", 23, 26, 21),
        ("bullish", "CHoCH", 27, 32, 23),
    ]


# ---- rule 1, the part that used to go wrong ---------------------------------

# Swing high at 103, pullback, then ONE candle closes above it at 104 and
# price turns straight back down -- so the breaking candle (7) is itself a
# swing high once the two candles after it exist.
BREAK_THEN_PULLBACK = [100, 101, 102, 103, 102, 101, 102, 104, 103, 102, 101.5]


def test_a_break_stays_when_the_breaking_candle_turns_out_to_be_a_swing():
    for length in (9, 10, 11):
        df = df_of(candles(BREAK_THEN_PULLBACK[:length]))
        assert brief(detect_structure_breaks(df, n=2)) == [(7, "bullish", "BOS")], length
        assert [(o["formed_idx"], o["bos_idx"]) for o in detect_order_blocks(df, n=2)] == [(5, 7)], length


def test_before_the_fix_that_order_block_vanished_one_candle_later():
    """confirmed_swings=False is the old behaviour, kept for the backtest
    comparison: the break is seen while candle 7's swing isn't known yet
    and is gone as soon as it is."""
    seen = df_of(candles(BREAK_THEN_PULLBACK[:9]))
    gone = df_of(candles(BREAK_THEN_PULLBACK[:10]))
    assert [(o["formed_idx"], o["bos_idx"]) for o in detect_order_blocks(seen, n=2, confirmed_swings=False)] == [(5, 7)]
    assert detect_order_blocks(gone, n=2, confirmed_swings=False) == []
    assert detect_structure_breaks(gone, n=2, confirmed_swings=False) == []


def _random_candles(seed, length=600):
    end = 1_790_000_000_000 - 1_790_000_000_000 % STEP
    data = bt.synthetic_history(["AAA/USDT:USDT"], end - 8 * 86_400_000, end, seed=seed, exec_tf="15m",
                                htf_timeframes=["1h"], htf_lookback=10)
    return data["AAA/USDT:USDT"]["15m"][-length:].tolist()


def _key(b):
    return (b["idx"], b["direction"], b["kind"], b["level"], b["swing_idx"])


def test_structure_never_repaints():
    """What is known after candle k closes is still true later: the breaks
    found on the first k candles (the last one still forming) are exactly
    the breaks the full history shows up to there."""
    for seed in (5, 6):
        rows = _random_candles(seed)
        full = [_key(b) for b in detect_structure_breaks(df_of(rows), n=2)]
        assert len(full) > 40
        old_differs = 0
        for k in range(30, len(rows), 7):
            part = df_of(rows[:k])
            then = [_key(b) for b in detect_structure_breaks(part, n=2) if b["idx"] < k - 1]
            assert then == [b for b in full if b[0] < k - 1], (seed, k)
            old_then = {(b["idx"], b["direction"]) for b in detect_structure_breaks(part, n=2, confirmed_swings=False)
                        if b["idx"] < k - 1}
            old_full = {(b["idx"], b["direction"]) for b in detect_structure_breaks(df_of(rows), n=2, confirmed_swings=False)
                        if b["idx"] < k - 1}
            old_differs += old_then != old_full
        assert old_differs > 0          # the behaviour before did repaint


def test_order_blocks_never_vanish_between_runs():
    """Run after run on a sliding window, as the bot fetches candles: an
    order block whose break a completed candle made is found again at the
    next run (until its candles slide out of the window)."""
    rows = np.array(_random_candles(7, length=700))
    window = 300
    previous, checked = None, 0
    for end in range(window, len(rows)):
        win = rows[end - window:end + 1].copy()
        win[-1, 2:5] = win[-1, 1]                      # the newest candle has only just opened
        obs = detect_order_blocks(df_of(win.tolist()), n=2)
        now = {(win[o["bos_idx"], 0], o["type"]) for o in obs if o["bos_idx"] < len(win) - 1}
        if previous is not None:
            for ts, kind in previous:
                if ts > win[60, 0]:                    # well inside the window
                    assert (ts, kind) in now
                    checked += 1
        previous = now
    assert checked > 5000


# ---- strong swings: only order blocks at a swing that caused a BOS/CHoCH ----

# Swing high at 104, a two-candle pause (103.5, 103.9) that never makes a
# swing low, then a close above 104. Structure breaks -- but not from a swing.
BREAK_FROM_A_PAUSE = [100, 101, 102, 103, 104, 103.5, 103.9, 105, 106, 105.8, 105.9, 106.2]


def test_a_strong_swing_is_a_swing_that_caused_a_break():
    """Every order block of the zigzag starts at a swing low/high (rule 1)
    from which price went on to close beyond the last swing: its origin is
    exactly one of the fractals."""
    df = df_of(candles(ZIGZAG))
    fr = find_fractals(df, n=2)
    obs = detect_order_blocks(df, n=2)
    assert [(o["origin_idx"], o["strong_swing"]) for o in obs] == [(5, True), (11, True), (15, True), (23, True), (27, True)]
    for o in obs:
        assert fr["swing_low" if o["type"] == "bullish" else "swing_high"].iloc[o["origin_idx"]]


def test_an_order_block_from_a_pause_that_never_formed_a_swing_does_not_count():
    df = df_of(candles(BREAK_FROM_A_PAUSE))
    assert brief(detect_structure_breaks(df, n=2)) == [(7, "bullish", "BOS")]       # the break itself is real
    assert detect_order_blocks(df, n=2) == []                                        # ...but there is no strong swing under it
    weak = detect_order_blocks(df, n=2, strong_swings_only=False)                   # REQUIRE_STRONG_SWING=false: as before
    assert [(o["formed_idx"], o["bos_idx"], o["origin_idx"], o["strong_swing"]) for o in weak] == [(5, 7, 5, False)]


def test_a_strong_swing_needs_its_candles_complete_and_then_stays():
    # low at candle 5, and candle 6 already closes above the swing high
    closes = [100, 101, 102, 103, 102, 101, 104, 105, 104.5, 104.8]
    assert detect_order_blocks(df_of(candles(closes[:8])), n=2) == []      # candle 7, the 2nd after the low, is still forming
    for length in (9, 10):
        obs = detect_order_blocks(df_of(candles(closes[:length])), n=2)
        assert [(o["formed_idx"], o["bos_idx"], o["origin_idx"]) for o in obs] == [(5, 6, 5)], length
