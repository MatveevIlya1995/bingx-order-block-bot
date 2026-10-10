"""Where the stop-loss goes (STOP_PLACEMENT): beyond the last swing by
default, beyond the order block candle as before, or beyond the swing
before that."""
import pytest

import bot
import config
from strategy import detect_order_blocks, limit_setups, stop_prices
from test_smc_filters import PULLBACK_TO_THE_OLD_LOW, Candles
from test_structure import ZIGZAG, candles, df_of


def hammer_rows():
    """The pullback bottoms with a down candle (13, low 100.8 -- the order
    block), then the NEXT candle dips to 100.3 before closing up: the swing
    low is 100.3, half a point below the order block."""
    rows = candles(PULLBACK_TO_THE_OLD_LOW)
    rows[14][3] = 100.3
    return rows


def the_setup(rows, **kw):
    df = df_of(rows)
    setups = limit_setups(df, detect_order_blocks(df, n=2), max_setups=2, **kw)
    assert len(setups) == 1
    return setups[0]


def test_default_stop_is_below_the_last_swing_low():
    s = the_setup(hammer_rows())
    ob = s["order_block"]
    assert (ob["formed_idx"], ob["origin_idx"], ob["bottom"], ob["swing_price"]) == (13, 14, 100.8, 100.3)
    assert s["position_side"] == "LONG" and s["entry"] == pytest.approx(102.1)
    assert s["stop_placement"] == "swing"
    assert s["stop_loss"] == pytest.approx(100.3 * 0.999)             # 0.1% below the swing low
    assert s["stop_loss"] < ob["bottom"] * 0.999                      # wider than the order-block stop
    assert s["take_profit"] == pytest.approx(s["entry"] + 2 * (s["entry"] - s["stop_loss"]))
    assert s["stops"] == pytest.approx({"order_block": 100.8 * 0.999, "swing": 100.3 * 0.999,
                                        "previous_swing": 100.3 * 0.999})   # no earlier swing low below 100.3


def test_order_block_placement_is_the_stop_as_before():
    s = the_setup(hammer_rows(), stop_placement="order_block")
    assert s["stop_loss"] == pytest.approx(100.8 * 0.999) and s["stop_placement"] == "order_block"


def test_same_stop_when_the_order_block_candle_made_the_swing_low():
    rows = candles(PULLBACK_TO_THE_OLD_LOW)            # candle 13 is both the order block and the low
    a, b = the_setup(rows), the_setup(rows, stop_placement="order_block")
    assert a["order_block"]["formed_idx"] == a["order_block"]["origin_idx"] == 13
    assert a["stop_loss"] == b["stop_loss"] == pytest.approx(100.8 * 0.999)


def test_previous_swing_is_the_earlier_swing_further_out():
    obs = detect_order_blocks(df_of(candles(ZIGZAG)), n=2)
    second = obs[1]                                     # bullish, at the 102.8 swing low; the swing low before it is 100.8
    assert (second["type"], second["swing_price"], second["prev_swing_price"]) == ("bullish", 102.8, 100.8)
    assert stop_prices(second) == pytest.approx({"order_block": 102.8 * 0.999, "swing": 102.8 * 0.999,
                                                 "previous_swing": 100.8 * 0.999})
    first = obs[0]                                      # nothing earlier below its swing low: falls back to the swing
    assert first["prev_swing_price"] is None
    assert stop_prices(first)["previous_swing"] == stop_prices(first)["swing"]


def test_short_stops_sit_above():
    obs = detect_order_blocks(df_of(candles(ZIGZAG)), n=2)
    short = [o for o in obs if o["type"] == "bearish"][0]          # at the 107.2 swing high
    assert short["swing_price"] == 107.2
    stops = stop_prices(short)
    assert stops["swing"] == pytest.approx(107.2 * 1.001) and stops["swing"] > short["top"]
    # a swing high above the order block's own high widens the stop upward
    wider = dict(short, swing_price=108.0, prev_swing_price=109.5)
    assert stop_prices(wider) == pytest.approx({"order_block": 107.2 * 1.001, "swing": 108.0 * 1.001,
                                                "previous_swing": 109.5 * 1.001})
    # ...and a "swing" inside the zone never tightens it
    inside = dict(short, swing_price=106.5, prev_swing_price=None)
    assert stop_prices(inside)["swing"] == stop_prices(inside)["order_block"]


def test_setting(monkeypatch, capsys):
    monkeypatch.delenv("STOP_PLACEMENT", raising=False)
    assert config.load_config().stop_placement == "swing"
    monkeypatch.setenv("STOP_PLACEMENT", "")                        # an empty GitHub variable
    assert config.load_config().stop_placement == "swing"
    for value in ("order_block", " Previous_Swing ", "swing"):
        monkeypatch.setenv("STOP_PLACEMENT", value)
        assert config.load_config().stop_placement == value.strip().lower()
    monkeypatch.setenv("STOP_PLACEMENT", "atr")
    assert config.load_config().stop_placement == "swing" and "STOP_PLACEMENT='atr'" in capsys.readouterr().out


def test_the_bot_uses_the_setting(monkeypatch):
    for key in ("STOP_PLACEMENT", "MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TREND_FILTER_ENABLED", "false")
    client = Candles(hammer_rows())

    def stop(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return bot._find_setups(client, config.load_config(), "BTC/USDT:USDT", {})[0][0]["stop_loss"]

    assert stop() == pytest.approx(100.3 * 0.999)
    assert stop(STOP_PLACEMENT="order_block") == pytest.approx(100.8 * 0.999)
    # the minimum-stop filter measures the stop that will actually be used:
    # 102.1 -> 100.6992 is 1.37%, 102.1 -> 100.1997 is 1.86%
    monkeypatch.setenv("MIN_STOP_DISTANCE_PCT", "1.5")
    monkeypatch.setenv("STOP_PLACEMENT", "order_block")
    assert bot._find_setups(client, config.load_config(), "BTC/USDT:USDT", {})[0] == []
    monkeypatch.setenv("STOP_PLACEMENT", "swing")
    assert len(bot._find_setups(client, config.load_config(), "BTC/USDT:USDT", {})[0]) == 1
