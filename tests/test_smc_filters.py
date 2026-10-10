"""The smart-money context of a setup (liquidity sweep, displacement, fair
value gap, premium/discount, BOS vs CHoCH, hour) and the filters built on
it: off by default, and only ever removing setups."""
import inspect

import numpy as np
import pytest

import bot
import config
import strategy
from strategy import (
    _atr,
    _displacement_atr,
    _has_fair_value_gap,
    _liquidity_taken,
    add_smc_context,
    apply_entry_filters,
    describe_smc,
    detect_order_blocks,
    hour_allowed,
    limit_setups,
    parse_hour_windows,
    smc_filter_reason,
)
from test_structure import candles, df_of

SMC_ENV = ("OB_STRUCTURE", "REQUIRE_LIQUIDITY_SWEEP", "MIN_DISPLACEMENT_ATR", "REQUIRE_FVG",
           "PREMIUM_DISCOUNT_FILTER", "DEALING_RANGE_CANDLES", "ENTRY_HOURS_UTC", "STRUCTURE_CONFIRMED_SWINGS",
           "REQUIRE_STRONG_SWING")

# Swing low at 101 (low 100.8), rally to 105, pullback to 101, rally through
# 105. What the pullback does at the old low decides the liquidity tag.
PULLBACK_TO_THE_OLD_LOW = [100, 101, 102, 103, 102, 101, 102, 103, 104, 105, 104, 103, 102, 101,
                           102, 103, 104, 105, 106, 107, 106.5]


def one_setup(rows):
    df = df_of(rows)
    setups = limit_setups(df, detect_order_blocks(df, n=2), max_setups=2)
    assert len(setups) == 1
    return add_smc_context(df, setups, n=2)[0], df


# ---------------------------------------------------------------- the tags

def test_liquidity_sweep_wick_through_an_old_low_that_closes_back():
    rows = candles(PULLBACK_TO_THE_OLD_LOW)
    rows[13][3] = 100.5                       # wick below the 100.8 swing low, close (101) above it
    setup, _ = one_setup(rows)
    assert setup["order_block"]["formed_idx"] == 13 and setup["order_block"]["bos_idx"] == 18
    assert setup["smc"]["liquidity"] == "sweep" and setup["smc"]["sweep"] is True
    assert setup["smc"]["structure"] == "BOS"                  # no close below the low: structure never turned


def test_liquidity_run_when_a_candle_closes_beyond_the_old_low():
    closes = list(PULLBACK_TO_THE_OLD_LOW)
    closes[13] = 100.6                        # CLOSES below the 100.8 swing low
    setup, _ = one_setup(candles(closes))
    assert setup["smc"]["liquidity"] == "run" and setup["smc"]["sweep"] is False
    assert setup["smc"]["structure"] == "CHoCH"                # that close was a bearish break, so this one turns it back


def test_no_liquidity_taken_when_the_pullback_stays_above_the_old_low():
    closes = [100, 101, 102, 103, 102, 101, 102, 103, 104, 105, 104, 103, 102, 103, 104, 105, 106, 107, 106.5]
    setup, _ = one_setup(candles(closes))
    assert setup["smc"]["liquidity"] == "none" and setup["smc"]["sweep"] is False


def test_an_old_low_already_traded_through_is_not_liquidity():
    #        idx: 0    1    2    3    4    5    6    7    8    9    10
    lows = np.array([10., 8.0, 10., 10., 7.0, 10., 10., 9.0, 7.5, 9.0, 10.])
    highs = lows + 2
    closes = lows + 1
    swing_lows = np.zeros(11, dtype=bool)
    swing_lows[[1, 4]] = True
    swing_highs = np.zeros(11, dtype=bool)
    ob = {"type": "bullish", "swing_idx": 6, "bos_idx": 10}
    # the move's low is 7.5: above the 7.0 low (not reached), and the 8.0
    # low was already traded through by candle 4 before the move began
    assert _liquidity_taken(highs, lows, closes, swing_highs, swing_lows, ob) == "none"
    lows[4] = 8.5                              # now nothing traded below 8.0 before the move
    assert _liquidity_taken(highs, lows, closes, swing_highs, swing_lows, ob) == "sweep"
    closes[8] = 7.8                            # ...and if the move also closed below it
    assert _liquidity_taken(highs, lows, closes, swing_highs, swing_lows, ob) == "run"


def test_liquidity_mirror_image_for_a_bearish_order_block():
    highs = np.array([10., 12., 10., 10., 10., 11., 12.5, 11., 9.0])
    lows = highs - 2
    closes = highs - 1
    swing_highs = np.zeros(9, dtype=bool)
    swing_highs[1] = True
    ob = {"type": "bearish", "swing_idx": 4, "bos_idx": 8}
    none = np.zeros(9, dtype=bool)
    assert _liquidity_taken(highs, lows, closes, swing_highs, none, ob) == "sweep"   # 12.5 wicks above 12, closes 11.5
    closes[6] = 12.2
    assert _liquidity_taken(highs, lows, closes, swing_highs, none, ob) == "run"
    highs[6] = 11.9
    assert _liquidity_taken(highs, lows, closes, swing_highs, none, ob) == "none"


def test_fair_value_gap_needs_its_third_candle_closed():
    #                 0    1    2(OB) 3    4(BOS) 5    6
    highs = np.array([10., 10., 10.0, 11.5, 12.5, 13., 13.])
    lows = np.array([9.0, 9.0, 9.0, 9.8, 10.5, 12., 12.])
    ob = {"type": "bullish", "formed_idx": 2, "bos_idx": 4}
    # candles 2-3-4: candle 4's low (10.5) is above candle 2's high (10.0)
    assert _has_fair_value_gap(highs, lows, ob, last=6) is True
    assert _has_fair_value_gap(highs, lows, ob, last=5) is True
    assert _has_fair_value_gap(highs, lows, ob, last=4) is False     # candle 4 is still forming
    lows[4] = 9.9                                                     # 2-3-4 overlap now...
    assert _has_fair_value_gap(highs, lows, ob, last=5) is False      # ...3-4-5 isn't complete yet
    assert _has_fair_value_gap(highs, lows, ob, last=6) is True       # candle 5's low (12) > candle 3's high (11.5)
    bear = {"type": "bearish", "formed_idx": 2, "bos_idx": 4}
    assert _has_fair_value_gap(-lows, -highs, bear, last=6) is True   # mirrored prices


def test_displacement_is_the_biggest_body_in_average_true_ranges():
    n = 20
    opens = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)                    # every candle: range 1.0, no gaps -> ATR 1.0
    assert _atr(highs, lows, closes, end_idx=16) == pytest.approx(1.0)
    opens[17], closes[17], highs[17] = 100.0, 100.8, 100.9
    opens[18], closes[18], highs[18], lows[18] = 100.8, 103.3, 103.4, 100.7      # body 2.5
    ob = {"formed_idx": 16, "bos_idx": 18}
    assert _displacement_atr(opens, highs, lows, closes, ob) == pytest.approx(2.5)
    assert _displacement_atr(opens, highs, lows, closes, {"formed_idx": 0, "bos_idx": 2}) is None   # no ATR yet


def test_range_position_and_hour():
    rows = candles(PULLBACK_TO_THE_OLD_LOW)
    setup, df = one_setup(rows)
    lo, hi = df["low"].min(), df["high"].max()
    assert setup["smc"]["range_pos"] == pytest.approx((setup["entry"] - lo) / (hi - lo))
    assert 0 < setup["smc"]["range_pos"] < 0.5                  # the entry sits in the lower half: discount
    assert setup["smc"]["hour_utc"] == 5                        # candle 20 x 15 minutes = 05:00 UTC
    # a shorter dealing range moves the position
    short = add_smc_context(df, [dict(setup)], n=2, dealing_range_candles=4)[0]["smc"]["range_pos"]
    assert short != pytest.approx(setup["smc"]["range_pos"])
    # epoch-millisecond timestamps (what backtest.py feeds in) give the same hour
    raw = df.copy()
    raw["ts"] = [r[0] for r in rows]
    assert add_smc_context(raw, [dict(setup)], n=2)[0]["smc"]["hour_utc"] == 5


def test_describe_smc():
    text = describe_smc({"structure": "CHoCH", "liquidity": "sweep", "sweep": True, "fvg": True,
                         "displacement_atr": 2.14, "range_pos": 0.34, "hour_utc": 13})
    assert text == ("CHoCH, liquidity swept (wick only), fair value gap yes, displacement 2.1 ATR, "
                    "entry at 34% of the range (discount)")
    assert describe_smc(None) == "n/a"


# -------------------------------------------------------------- the filters

def setup(side="LONG", **smc):
    base = {"structure": "BOS", "liquidity": "none", "sweep": False, "displacement_atr": 1.2,
            "fvg": False, "range_pos": 0.3, "hour_utc": 9}
    base.update(smc)
    return {"side": "buy" if side == "LONG" else "sell", "position_side": side, "entry": 100.0,
            "stop_loss": 99.0 if side == "LONG" else 101.0, "take_profit": 0.0, "smc": base}


def keeps(setups, **filters):
    kept, dropped = apply_entry_filters(setups, **filters)
    assert len(kept) + len(dropped) == len(setups)
    return [setups.index(s) for s in kept]


def test_smart_money_filters_are_off_by_default(monkeypatch):
    for key in SMC_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = config.load_config()
    assert cfg.structure_confirmed_swings is True and cfg.require_strong_swing is True
    settings = config.smc_filter_settings(cfg)
    assert settings == {"ob_structure": "any", "require_sweep": False, "min_displacement_atr": 0.0,
                        "require_fvg": False, "premium_discount": False, "entry_hours": []}
    # the settings are exactly the arguments the filter takes
    assert set(settings) == set(inspect.signature(smc_filter_reason).parameters) - {"setup"}
    worst = [setup(structure="CHoCH", displacement_atr=None, range_pos=0.99, hour_utc=3),
             {"position_side": "LONG", "entry": 100.0, "stop_loss": 99.0}]      # not even tagged
    kept, dropped = apply_entry_filters(worst, **settings)
    assert kept == worst and dropped == []


def test_empty_github_variables_mean_off(monkeypatch):
    for key in SMC_ENV:
        monkeypatch.setenv(key, "")           # `${{ vars.X }}` for a variable that was never created
    cfg = config.load_config()
    assert config.smc_filter_settings(cfg)["entry_hours"] == [] and cfg.ob_structure == "any"
    assert cfg.require_liquidity_sweep is False and cfg.require_fvg is False and cfg.premium_discount_filter is False
    assert cfg.min_displacement_atr == 0.0 and cfg.dealing_range_candles == 0
    assert cfg.structure_confirmed_swings is True


def test_bad_values_fall_back_to_off(monkeypatch):
    monkeypatch.setenv("OB_STRUCTURE", "reversal")
    monkeypatch.setenv("MIN_DISPLACEMENT_ATR", "-2")
    monkeypatch.setenv("DEALING_RANGE_CANDLES", "-5")
    monkeypatch.setenv("ENTRY_HOURS_UTC", "morning")
    cfg = config.load_config()
    assert (cfg.ob_structure, cfg.min_displacement_atr, cfg.dealing_range_candles, cfg.entry_hours_utc) == ("any", 0.0, 0, [])
    monkeypatch.setenv("OB_STRUCTURE", " ChoCH ")
    monkeypatch.setenv("ENTRY_HOURS_UTC", "7-10, 13-16")
    cfg = config.load_config()
    assert cfg.ob_structure == "choch" and cfg.entry_hours_utc == [(7, 10), (13, 16)]


def test_each_filter_removes_exactly_what_it_says():
    bos, choch = setup(structure="BOS"), setup(structure="CHoCH")
    assert keeps([bos, choch], ob_structure="any") == [0, 1]
    assert keeps([bos, choch], ob_structure="bos") == [0]
    assert keeps([bos, choch], ob_structure="choch") == [1]

    swept, ran, untouched = setup(liquidity="sweep", sweep=True), setup(liquidity="run"), setup()
    assert keeps([swept, ran, untouched], require_sweep=True) == [0]

    weak, strong, unknown = setup(displacement_atr=1.49), setup(displacement_atr=1.5), setup(displacement_atr=None)
    assert keeps([weak, strong, unknown], min_displacement_atr=1.5) == [1]

    assert keeps([setup(fvg=True), setup(fvg=False)], require_fvg=True) == [0]

    reason = apply_entry_filters([setup(displacement_atr=0.8)], min_displacement_atr=1.5)[1][0][1]
    assert "0.80 ATR" in reason and "MIN_DISPLACEMENT_ATR=1.5" in reason


def test_premium_discount_filter():
    longs = [setup("LONG", range_pos=p) for p in (0.1, 0.5, 0.51, None)]
    shorts = [setup("SHORT", range_pos=p) for p in (0.9, 0.5, 0.49, None)]
    assert keeps(longs, premium_discount=True) == [0, 1]       # a LONG must be in the lower half
    assert keeps(shorts, premium_discount=True) == [0, 1]      # a SHORT in the upper half
    assert keeps(longs + shorts, premium_discount=False) == list(range(8))


def test_entry_hours():
    assert parse_hour_windows("") == [] and parse_hour_windows(None) == []
    assert parse_hour_windows(" 7-10 ,13-16,") == [(7, 10), (13, 16)]
    for bad in ("7", "7-7", "25-3", "a-b", "7-10;13-16"):
        with pytest.raises(ValueError):
            parse_hour_windows(bad)
    assert [h for h in range(24) if hour_allowed(h, [(7, 10)])] == [7, 8, 9]            # end excluded
    assert [h for h in range(24) if hour_allowed(h, [(22, 2)])] == [0, 1, 22, 23]       # wraps midnight
    assert [h for h in range(24) if hour_allowed(h, [(20, 24)])] == [20, 21, 22, 23]
    assert all(hour_allowed(h, []) for h in range(24))
    at = [setup(hour_utc=h) for h in (6, 7, 9, 10)]
    assert keeps(at, entry_hours=[(7, 10)]) == [1, 2]
    assert "ENTRY_HOURS_UTC=7-10" in apply_entry_filters(at, entry_hours=[(7, 10)])[1][0][1]


def test_a_filter_that_is_on_drops_a_setup_it_cannot_judge():
    untagged = {"position_side": "LONG", "entry": 100.0, "stop_loss": 99.0}
    kept, dropped = apply_entry_filters([untagged], require_fvg=True)
    assert kept == [] and "no smart-money context" in dropped[0][1]


# ------------------------------------------------------ through the bot code

class Candles:
    def __init__(self, rows):
        self.rows = rows

    def fetch_ohlcv(self, symbol, timeframe=None, limit=None):
        return self.rows


def test_find_setups_tags_and_filters(monkeypatch, capsys):
    """bot._find_setups on real candles: the setup comes back tagged, and
    each smart-money setting keeps or removes it as the tags say."""
    for key in SMC_ENV + ("MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TREND_FILTER_ENABLED", "false")
    rows = candles(PULLBACK_TO_THE_OLD_LOW)
    rows[13][3] = 100.5                       # the wick sweep from the first test

    def find(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        out = bot._find_setups(Candles(rows), config.load_config(), "BTC/USDT:USDT", {})[0]
        for k in env:
            monkeypatch.delenv(k)
        return out

    found = find()
    assert len(found) == 1
    smc = found[0]["smc"]
    assert (smc["structure"], smc["liquidity"], smc["fvg"], smc["hour_utc"]) == ("BOS", "sweep", True, 5)
    assert smc["range_pos"] < 0.5 and 0.5 < smc["displacement_atr"] < 1.0

    assert len(find(REQUIRE_LIQUIDITY_SWEEP="true", REQUIRE_FVG="true", PREMIUM_DISCOUNT_FILTER="true",
                    OB_STRUCTURE="bos", MIN_DISPLACEMENT_ATR="0.5", ENTRY_HOURS_UTC="5-6")) == 1
    assert find(OB_STRUCTURE="choch") == []
    assert find(MIN_DISPLACEMENT_ATR="1.5") == []
    assert find(ENTRY_HOURS_UTC="7-10") == []
    out = capsys.readouterr().out
    assert out.count("filtered out") == 3 and "OB_STRUCTURE=choch" in out
    # DEALING_RANGE_CANDLES reaches the tag: against only the last 3 candles
    # (106.4 and up) the 102.1 entry is below the whole range
    assert find(DEALING_RANGE_CANDLES="3")[0]["smc"]["range_pos"] < 0

    # STRUCTURE_CONFIRMED_SWINGS reaches the detection
    modes = []

    def spy(df, n, confirmed_swings, strong_swings_only):
        modes.append((confirmed_swings, strong_swings_only))
        return strategy.detect_order_blocks(df, n=n, confirmed_swings=confirmed_swings,
                                            strong_swings_only=strong_swings_only)

    monkeypatch.setattr(bot, "detect_order_blocks", spy)
    find()
    find(STRUCTURE_CONFIRMED_SWINGS="false")
    find(REQUIRE_STRONG_SWING="false")
    assert modes == [(True, True), (False, True), (True, False)]


def test_a_full_bot_run_journals_the_tags_and_obeys_the_filters(make_harness):
    """bot.run_once() in DRY_RUN on candles with one order block: the limit
    order is placed with its smart-money context in the journal; with a
    filter the setup fails, the next run cancels it again."""
    h = make_harness(DRY_RUN="true", SYMBOLS="SOL/USDT:USDT")
    rows = candles(PULLBACK_TO_THE_OLD_LOW)
    rows[13][3] = 100.5
    flat = h.ex.fetch_ohlcv

    def fetch_ohlcv(symbol, timeframe="15m", since=None, limit=None, params=None):
        if timeframe == h.cfg.timeframe:
            return [list(r) for r in rows]
        return flat(symbol, timeframe, since, limit, params)       # flat higher timeframes: no bias

    h.ex.fetch_ohlcv = fetch_ohlcv
    h.ex.price["SOL/USDT:USDT"] = rows[-1][4]
    risk = h.run()
    orders = risk.pending_orders("SOL/USDT:USDT")
    assert len(orders) == 1 and orders[0]["entry_price"] == pytest.approx(102.1)
    placed = [r for r in h.journal("DRY_RUN") if r["event"] == "LIMIT_PLACED"]
    assert len(placed) == 1
    assert "BOS confirmed" in placed[0]["trigger"] and "| stop beyond the last swing |" in placed[0]["trigger"]
    assert ("| SMC: BOS, liquidity swept (wick only), fair value gap yes, displacement 0.8 ATR, "
            "entry at 30% of the range (discount)") in placed[0]["trigger"]

    h.mp.setenv("MIN_DISPLACEMENT_ATR", "1.5")
    risk = h.run()
    assert risk.pending_orders("SOL/USDT:USDT") == []
    assert [r["event"] for r in h.journal("DRY_RUN")] == ["LIMIT_PLACED", "LIMIT_CANCELLED"]

    h.mp.setenv("MIN_DISPLACEMENT_ATR", "0.5")
    h.mp.setenv("REQUIRE_LIQUIDITY_SWEEP", "true")
    assert len(h.run().pending_orders("SOL/USDT:USDT")) == 1
