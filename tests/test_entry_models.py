"""The smart-money entry models (ENTRY_MODEL): order block, fair value gap,
optimal trade entry, breaker and liquidity sweep reversal. All are zones
with the same shape, found from the same market structure."""
import numpy as np
import pytest

import bot
import config
from strategy import ENTRY_MODELS, detect_order_blocks, detect_zones, limit_setups, stop_prices
from test_smc_filters import PULLBACK_TO_THE_OLD_LOW, Candles
from test_structure import ZIGZAG, candles, df_of

# The zigzag of test_structure, plus three candles so its last break is complete.
PATH = ZIGZAG + [105, 104.6, 104.9]


def zone(model, i, rows=None):
    return detect_zones(df_of(rows or candles(PATH)), n=2)[model][i]


def brief(z):
    return (z["type"], z["structure"], z["formed_idx"], z["bos_idx"], round(z["bottom"], 3), round(z["top"], 3))


def test_every_model_reads_the_same_breaks():
    zones = detect_zones(df_of(candles(PATH)), n=2)
    assert tuple(zones) == ENTRY_MODELS
    for model in ("order_block", "fvg", "ote", "breaker"):
        assert [(z["type"], z["structure"]) for z in zones[model]] == [
            ("bullish", "BOS"), ("bullish", "BOS"), ("bearish", "CHoCH"), ("bearish", "BOS"), ("bullish", "CHoCH")], model
        assert all(z["model"] == model and z["top"] > z["bottom"] for z in zones[model])
    assert zones["order_block"] == detect_order_blocks(df_of(candles(PATH)), n=2)
    assert detect_zones(df_of(candles(PATH)), n=2, models=("fvg",)) == {"fvg": zones["fvg"]}


def test_first_break_in_every_model():
    """The first break: swing high 103.2 (candle 3) is broken by candle 8,
    out of the swing low 100.8 (candle 5); the next swing high is 105.2."""
    assert brief(zone("order_block", 0)) == ("bullish", "BOS", 5, 8, 100.8, 102.1)      # the last down candle at the low
    assert brief(zone("fvg", 0)) == ("bullish", "BOS", 8, 9, 103.2, 103.9)             # candle 9's low above candle 7's high
    assert brief(zone("breaker", 0)) == ("bullish", "BOS", 3, 8, 101.9, 103.2)         # the candle that made the broken high
    ote = zone("ote", 0)                                                                # 62%-79% back from 105.2 to 100.8
    assert (ote["formed_idx"], ote["bos_idx"]) == (9, 11)
    assert ote["top"] == pytest.approx(105.2 - 0.62 * 4.4) and ote["bottom"] == pytest.approx(105.2 - 0.79 * 4.4)
    # every one of them belongs to the same swing low: the stop goes below 100.8
    for model in ("order_block", "fvg", "ote", "breaker"):
        z = zone(model, 0)
        assert z["swing_price"] == 100.8 and z["origin_idx"] == 5
        assert stop_prices(z)["swing"] == pytest.approx(100.8 * 0.999)


def test_bearish_zones_mirror():
    # third break: swing low 102.8 (candle 11) broken by candle 20, out of the swing high 107.2 (candle 15)
    assert brief(zone("fvg", 2)) == ("bearish", "CHoCH", 20, 21, 102.1, 102.8)
    assert brief(zone("breaker", 2)) == ("bearish", "CHoCH", 11, 20, 102.8, 104.1)
    ote = zone("ote", 2)                                    # first swing low after the break: 100.8 (candle 21)
    assert ote["bottom"] == pytest.approx(100.8 + 0.62 * 6.4) and ote["top"] == pytest.approx(100.8 + 0.79 * 6.4)
    for model in ("fvg", "ote", "breaker"):
        z = zone(model, 2)
        assert z["swing_price"] == 107.2 and stop_prices(z)["swing"] == pytest.approx(107.2 * 1.001)
    setups = limit_setups(df_of(candles(PATH)), [dict(zone("breaker", 2), mitigated=False)], max_setups=1)
    assert setups[0]["position_side"] == "SHORT" and setups[0]["entry"] == 102.8      # the broken low, retested from below


def test_gap_and_retracement_wait_until_they_are_known():
    # the fair value gap needs the candle after the break closed: break at 8, so 11 candles (0..10, 10 forming)
    assert detect_zones(df_of(candles(PATH[:10])), n=2)["fvg"] == []
    assert [brief(z)[2:4] for z in detect_zones(df_of(candles(PATH[:11])), n=2)["fvg"]] == [(8, 9)]
    # the retracement needs the swing high after the break (candle 9) confirmed by two closed candles
    assert detect_zones(df_of(candles(PATH[:12])), n=2)["ote"] == []
    assert [brief(z)[2:4] for z in detect_zones(df_of(candles(PATH[:13])), n=2)["ote"]] == [(9, 11)]


def test_zones_never_vanish_between_runs():
    import backtest as bt
    end = 1_790_000_000_000 - 1_790_000_000_000 % 900_000
    data = bt.synthetic_history(["AAA/USDT:USDT"], end - 8 * 86_400_000, end, seed=9, exec_tf="15m",
                                htf_timeframes=["1h"], htf_lookback=10)
    rows = data["AAA/USDT:USDT"]["15m"][-560:]
    previous, checked = None, {m: 0 for m in ENTRY_MODELS}
    for stop in range(300, len(rows)):
        win = rows[stop - 300:stop + 1].copy()
        win[-1, 2:5] = win[-1, 1]                          # the newest candle has only just opened
        zones = detect_zones(df_of(win.tolist()), n=2)
        now = {m: {(win[z["bos_idx"], 0], win[z["formed_idx"], 0], z["type"], round(z["top"], 9), round(z["bottom"], 9))
                   for z in zs if z["bos_idx"] < 300} for m, zs in zones.items()}
        if previous is not None:
            for m in ENTRY_MODELS:
                for key in previous[m]:
                    if key[1] > win[70, 0]:                 # well inside the window
                        assert key in now[m], (m, key)
                        checked[m] += 1
        previous = now
    assert all(count > 300 for count in checked.values()), checked


def test_liquidity_sweep_zone():
    rows = candles(PULLBACK_TO_THE_OLD_LOW)
    rows[13][3] = 100.5                       # wick below the 100.8 swing low, close (101) back above it
    sweeps = detect_zones(df_of(rows), n=2)["sweep"]
    assert [brief(z) for z in sweeps] == [("bullish", "sweep", 13, 13, 100.5, 100.8)]
    z = sweeps[0]
    assert z["swing_price"] == 100.5 and z["mitigated"] is False
    setup = limit_setups(df_of(rows), sweeps, max_setups=2)[0]
    assert setup["position_side"] == "LONG" and setup["entry"] == 100.8                  # limit at the swept low
    assert setup["stop_loss"] == pytest.approx(100.5 * 0.999)                            # stop under the wick
    # a candle that CLOSES below the low is a break, not a sweep
    closes = list(PULLBACK_TO_THE_OLD_LOW)
    closes[13] = 100.6
    assert detect_zones(df_of(candles(closes)), n=2)["sweep"] == []
    # and no wick, no sweep
    assert detect_zones(df_of(candles(PULLBACK_TO_THE_OLD_LOW)), n=2)["sweep"] == []


def test_a_zone_price_has_visited_and_left_is_used_up():
    zones = detect_zones(df_of(candles(PATH)), n=2)
    # the first gap (103.2-103.9) was traded back into by candle 11 and left: used up
    assert zones["fvg"][0]["mitigated"] is True
    # the last breaker (101.9-103.2, from the break at candle 32) has not been revisited
    assert zones["breaker"][-1]["mitigated"] is False
    assert [s["entry"] for s in limit_setups(df_of(candles(PATH)), zones["breaker"], max_setups=2)] == [103.2]


def test_setting(monkeypatch, capsys):
    monkeypatch.delenv("ENTRY_MODEL", raising=False)
    assert config.load_config().entry_model == "order_block"
    monkeypatch.setenv("ENTRY_MODEL", "")
    assert config.load_config().entry_model == "order_block"
    for model in ENTRY_MODELS:
        monkeypatch.setenv("ENTRY_MODEL", f" {model.upper()} ")
        assert config.load_config().entry_model == model
    monkeypatch.setenv("ENTRY_MODEL", "supply_demand")
    assert config.load_config().entry_model == "order_block" and "ENTRY_MODEL='supply_demand'" in capsys.readouterr().out


def test_the_bot_rests_its_orders_in_the_chosen_model(monkeypatch):
    for key in ("ENTRY_MODEL", "STOP_PLACEMENT", "MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TREND_FILTER_ENABLED", "false")
    rows = candles(PULLBACK_TO_THE_OLD_LOW)
    rows[13][3] = 100.5
    entries = {}
    for model in ENTRY_MODELS:
        monkeypatch.setenv("ENTRY_MODEL", model)
        setups = bot._find_setups(Candles(rows), config.load_config(), "BTC/USDT:USDT", {})[0]
        entries[model] = [(s["position_side"], round(s["entry"], 3), s["order_block"]["model"]) for s in setups]
    assert entries["order_block"] == [("LONG", 102.1, "order_block")]
    assert entries["sweep"] == [("LONG", 100.8, "sweep")]
    assert entries["breaker"] == [("LONG", 105.2, "breaker")]         # the broken 105.2 high, awaiting its retest
    assert all(m == model for model, found in entries.items() for _, _, m in found)
    df = df_of(rows)
    zones = detect_zones(df, n=2)
    for model in ("fvg", "ote"):
        want = [("LONG", round(s["entry"], 3), model) for s in limit_setups(df, zones[model], max_setups=2)]
        assert entries[model] == want


def test_swing_detection_equals_the_plain_definition():
    """find_fractals is vectorised; check it against the definition spelled out."""
    from strategy import find_fractals
    rng = np.random.default_rng(4)
    for n in (1, 2, 3, 5):
        for _ in range(20):
            length = int(rng.integers(1, 80))
            highs = np.round(rng.normal(100, 1, length), 1)          # rounded: plenty of ties
            lows = highs - np.round(rng.random(length), 1)
            rows = [[i * 60_000, lows[i], highs[i], lows[i], highs[i], 1.0] for i in range(length)]
            fr = find_fractals(df_of(rows), n=n)
            for i in range(length):
                inside = n <= i < length - n
                wh, wl = highs[max(i - n, 0):i + n + 1], lows[max(i - n, 0):i + n + 1]
                assert fr["swing_high"].iloc[i] == (inside and highs[i] == wh.max() and int(np.argmax(wh)) == n)
                assert fr["swing_low"].iloc[i] == (inside and lows[i] == wl.min() and int(np.argmin(wl)) == n)
