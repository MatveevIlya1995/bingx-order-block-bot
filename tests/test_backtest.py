"""
The backtest must (1) see exactly the setups the live bot's own code finds on
the same history, and (2) account for fills and fees correctly.
"""
import numpy as np
import pytest

import backtest as bt
import bot
import config

SYMS = ["AAA/USDT:USDT", "BBB/USDT:USDT"]
DAY = 86_400_000


CLEAN_ENV = ("SYMBOLS", "MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT",
             "MAX_CONCURRENT_POSITIONS", "RISK_PCT_PER_TRADE", "LEVERAGE", "WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT",
             "STRUCTURE_CONFIRMED_SWINGS", "REQUIRE_STRONG_SWING", "OB_STRUCTURE", "REQUIRE_LIQUIDITY_SWEEP",
             "MIN_DISPLACEMENT_ATR", "REQUIRE_FVG", "PREMIUM_DISCOUNT_FILTER", "DEALING_RANGE_CANDLES",
             "ENTRY_HOURS_UTC", "TIMEFRAME", "HTF_TIMEFRAMES", "HTF_PRIMARY_TIMEFRAMES", "STOP_PLACEMENT", "ENTRY_MODEL")
ACCOUNT_ENV = {"MAX_CONCURRENT_POSITIONS": "5", "RISK_PCT_PER_TRADE": "2", "LEVERAGE": "10",
               "WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT": "1"}
# The cascade the bot ran before the timeframes moved down a level. The
# simulation tests below were written against it (15m steps, weekly warm-up),
# and it stays a supported configuration.
WEEKLY_DAILY_15M = {"TIMEFRAME": "15m", "HTF_TIMEFRAMES": "1w,1d,4h,1h", "HTF_PRIMARY_TIMEFRAMES": "1w,1d"}


def build_world(days, seed, **env):
    import os
    for key in CLEAN_ENV:
        os.environ.pop(key, None)
    os.environ.update({**ACCOUNT_ENV, **env})
    cfg = config.load_config()
    end = 1_790_000_000_000 - 1_790_000_000_000 % bt.TF_MS["15m"]
    start = end - days * DAY
    data = bt.synthetic_history(SYMS, start, end, seed=seed, exec_tf="5m",
                                htf_timeframes=cfg.htf_timeframes, htf_lookback=cfg.trend_candle_limit + 5)
    signals = {s: bt.compute_signals((cfg, s, data[s], start, end)) for s in SYMS}
    for key in list(ACCOUNT_ENV) + list(env):
        os.environ.pop(key, None)
    return cfg, data, signals, start, end


@pytest.fixture(scope="module")
def world():
    return build_world(12, 11, **WEEKLY_DAILY_15M)


@pytest.fixture(scope="module")
def world_5m():
    """The default cascade: daily + 4h direction, 1h + 15m timing, 5m entries."""
    return build_world(5, 13)


class ClientAt:
    """What ExchangeClient.fetch_ohlcv would have returned at time t_run:
    the last `limit` candles, the newest one still forming."""

    def __init__(self, cfg, data, t_run, px):
        self.cfg, self.data, self.t, self.px = cfg, data, t_run, px

    def fetch_ohlcv(self, symbol, timeframe=None, limit=None):
        tf = timeframe or self.cfg.timeframe
        limit = limit or self.cfg.candle_limit
        step = bt.TF_MS[tf]
        c = self.data[symbol][tf]
        done = c[c[:, 0] + step <= self.t]
        start = done[-1, 0] + step
        base = self.data[symbol][self.cfg.timeframe]
        seg = base[(base[:, 0] >= start) & (base[:, 0] < self.t)]
        o = seg[0, 1] if len(seg) else self.px
        h = max(self.px, seg[:, 2].max()) if len(seg) else self.px
        l = min(self.px, seg[:, 3].min()) if len(seg) else self.px
        rows = np.vstack([done[-(limit - 1):], [start, o, h, l, self.px, 0.0]])
        return rows.tolist()


def test_signals_equal_what_the_live_bot_code_finds(world):
    assert (world[0].timeframe, world[0].primary_htf_timeframes) == ("15m", ["1w", "1d"])
    check_signals(world, min_checked=300, min_setups=60)


def test_signals_equal_on_the_default_timeframes(world_5m):
    cfg = world_5m[0]
    assert (cfg.timeframe, cfg.htf_timeframes, cfg.primary_htf_timeframes) == ("5m", ["1d", "4h", "1h", "15m"], ["1d", "4h"])
    assert config.override_levels(cfg) == ("1d", "4h")
    overrides = check_signals(world_5m, min_checked=300, min_setups=60)
    assert overrides > 0          # the room override works between daily and 4h too


def check_signals(world, min_checked, min_setups):
    cfg, data, signals, start, end = world
    rng = np.random.default_rng(3)
    checked = with_setups = overrides = conflicts = widened = 0
    tags = set()
    for symbol in SYMS:
        sig = signals[symbol]
        picks = set(rng.choice(len(sig["steps"]), size=140, replace=False).tolist())
        picks |= set(list(sig["signals"])[:40])          # make sure steps with setups are covered
        for i in sorted(picks):
            t_run, px = sig["steps"][i], sig["px"][i]
            client = ClientAt(cfg, data, t_run, px)
            setups, detail, is_override, df = bot._find_setups(client, cfg, symbol, {})
            live = sorted((s["position_side"], s["entry"], s["stop_loss"]) for s in setups)
            got = sig["signals"].get(i)
            mine = sorted((s[0], s[1], s[2]) for s in got[2]) if got else []
            assert mine == pytest.approx(live), (symbol, i)
            if got:
                with_setups += 1
                assert got[0] == is_override
                overrides += is_override
                # TP2 level too
                for s in setups:
                    tp2, _ = bot._find_tp2_level(df, s, cfg)
                    raw = [r for r in got[2] if r[1] == pytest.approx(s["entry"])][0]
                    assert raw[6] == (pytest.approx(tp2) if tp2 is not None else None)
                # the smart-money tags the filters read
                for s in setups:
                    raw = [r for r in got[2] if r[1] == pytest.approx(s["entry"])][0]
                    assert set(raw[7]) == set(s["smc"]) == {"structure", "liquidity", "sweep", "displacement_atr",
                                                            "fvg", "range_pos", "hour_utc"}
                    for key, value in s["smc"].items():
                        assert raw[7][key] == (pytest.approx(value) if isinstance(value, float) else value), key
                    tags.add((raw[7]["structure"], raw[7]["liquidity"], raw[7]["fvg"]))
                    # every stop placement, with the TP2 that goes with it
                    assert raw[2] == pytest.approx(s["stops"][cfg.stop_placement])
                    for placement, stop in s["stops"].items():
                        alt_tp2, _ = bot._find_tp2_level(df, {**s, "stop_loss": stop}, cfg)
                        assert raw[8][placement][0] == pytest.approx(stop)
                        assert raw[8][placement][1] == (pytest.approx(alt_tp2) if alt_tp2 is not None else None)
                    widened += raw[8]["swing"][0] != raw[8]["order_block"][0]
                # and the 4h/1h trends the timing filter uses
                _, _, trends = bot.analyze_higher_timeframes(client, cfg, symbol)
                assert got[1] == tuple(trends[tf] for tf in cfg.htf_timeframes if tf not in cfg.primary_htf_timeframes)
            conflicts += "disagree" in detail or setups == [] and "NOT overridden" in detail
            checked += 1
    assert checked >= min_checked and with_setups >= min_setups
    # the tags vary across the sample (both kinds of break, every liquidity outcome, gap and no gap)
    assert {t[0] for t in tags} == {"BOS", "CHoCH"} and {t[1] for t in tags} == {"sweep", "run", "none"}
    assert {t[2] for t in tags} == {True, False}
    assert widened > 10            # the swing stop differs from the order-block stop often enough to matter
    return overrides


def test_htf_trend_equals_detect_trend(world):
    cfg, data, signals, start, end = world
    from strategy import detect_trend, ohlcv_to_df
    symbol = SYMS[0]
    seen = set()
    for i in range(0, len(signals[symbol]["steps"]), 37):
        t_run, px = signals[symbol]["steps"][i], signals[symbol]["px"][i]
        client = ClientAt(cfg, data, t_run, px)
        for tf in cfg.htf_timeframes:
            view = bt.HtfView(tf, data[symbol][tf], cfg.trend_candle_limit, cfg.trend_ema_period)
            expected = detect_trend(ohlcv_to_df(client.fetch_ohlcv(symbol, timeframe=tf, limit=cfg.trend_candle_limit)),
                                    ema_period=cfg.trend_ema_period)
            assert view.trend(view.completed_count(t_run), px) == expected
            seen.add(expected)
    assert len(seen) >= 2


# ---------------------------------------------------------------- simulation

def make_sim(world, **kw):
    cfg = world[0]
    costs = bt.Costs(0.02, 0.05, 0.0)
    return bt.Sim(cfg, costs, 1000.0, **kw)


def long_signal(entry=100.0, stop=99.5, tp2=101.0, timing=("bullish", "bullish")):
    return (False, timing, [("LONG", entry, stop, "bullish", entry, stop / 0.999, tp2)])


S = "AAA/USDT:USDT"


def test_sim_full_trade_accounting(world):
    sim = make_sim(world)
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])
    order = sim.pending[S][0]
    qty = order["qty"]
    assert qty == pytest.approx(50 * 10 / 100.0)            # $50 margin at the 10x ceiling
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0]], 2)     # trades through the limit: maker fill at 100.0
    t = sim.positions[S]
    assert t["entry"] == 100.0 and t["tp1"] == pytest.approx(100.5) and t["tp2"] == 101.0
    sim.run(2, {S: 100.2}, {}, [S])                         # this run puts TP1/TP2 on the exchange
    assert t["tp1_resting"] and t["tp2_resting"]
    sim.walk(S, [[3, 100.2, 100.55, 100.1, 100.4, 0]], 4)   # TP1 fills at 100.5
    assert t["tp1_done"] and t["qty"] == pytest.approx(qty * 0.5) and t["stop"] == 99.5
    sim.run(4, {S: 100.4}, {}, [S])                         # stop to breakeven
    assert t["stop"] == 100.0
    sim.walk(S, [[5, 100.4, 101.05, 100.3, 101.0, 0]], 6)   # TP2 fills at 101.0
    sim.run(6, {S: 101.0}, {}, [S])                         # trailing: 101.0 - 0.5
    assert t["stop"] == pytest.approx(100.5) and t["qty"] == pytest.approx(qty * 0.2)
    sim.walk(S, [[7, 101.0, 101.1, 100.4, 100.6, 0]], 8)    # runner stopped at 100.5
    assert S not in sim.positions and len(sim.trades) == 1
    fees = qty * 100.0 * 0.0002 + (qty * 0.5 * 100.5 + qty * 0.3 * 101.0 + qty * 0.2 * 100.5) * 0.0005
    gross = qty * 0.5 * 0.5 + qty * 0.3 * 1.0 + qty * 0.2 * 0.5
    assert sim.trades[0]["pnl"] == pytest.approx(gross - fees)
    assert sim.cash == pytest.approx(1000.0 + gross - fees)
    assert sim.trades[0]["fees"] == pytest.approx(fees)


def test_sim_stop_comes_first_when_a_candle_touches_both(world):
    sim = make_sim(world)
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0]], 2)
    sim.run(2, {S: 100.2}, {}, [S])
    sim.walk(S, [[3, 100.2, 100.6, 99.4, 99.6, 0]], 4)      # touches TP1 (100.5) AND the stop (99.5)
    assert S not in sim.positions
    assert [e[1] for e in sim.trades[0]["exits"]] == ["stop"]
    assert sim.trades[0]["pnl"] < 0


def test_sim_no_take_profit_before_the_first_run_after_the_fill(world):
    sim = make_sim(world)
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])
    # fills and runs past TP1 before the bot has run again: nothing rests yet
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0], [2, 100.2, 100.8, 100.1, 100.3, 0]], 3)
    t = sim.positions[S]
    assert not t["tp1_done"]
    sim.run(3, {S: 100.3}, {}, [S])
    assert t["tp1_resting"]                                 # 100.3 < TP1, so it can rest now


def test_sim_run_mode_closes_at_the_run_price(world):
    sim = make_sim(world, exit_mode="run")
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0]], 2)
    sim.run(2, {S: 100.2}, {}, [S])
    sim.walk(S, [[3, 100.2, 100.9, 100.1, 100.3, 0]], 4)    # touches TP1 between runs...
    t = sim.positions[S]
    assert not t["tp1_done"]                                # ...and the old behaviour misses it
    sim.run(4, {S: 100.3}, {}, [S])
    assert not t["tp1_done"]
    sim.run(5, {S: 100.7}, {}, [S])                         # a run that sees price beyond TP1
    assert t["tp1_done"] and t["exits"][0][3] == 100.7 and t["stop"] == 100.0


def test_sim_filters(world):
    for kw, expect in [({}, 1), ({"min_stop_pct": 0.6}, 0), ({"min_stop_pct": 0.5}, 1),
                       ({"timing_mode": "aligned"}, 0), ({"timing_mode": "not_against"}, 1),
                       ({"entry_symbols": ["BBB"]}, 0), ({"entry_symbols": ["aaa"]}, 1)]:
        sim = make_sim(world, **kw)
        sim.run(0, {S: 100.6}, {S: long_signal(timing=("bullish", "neutral"))}, [S])
        assert len(sim.pending.get(S, [])) == expect, kw


def smc_signal(**smc):
    tags = {"structure": "BOS", "liquidity": "none", "sweep": False, "displacement_atr": 1.2,
            "fvg": False, "range_pos": 0.3, "hour_utc": 9}
    tags.update(smc)
    sig = long_signal()
    return (sig[0], sig[1], [sig[2][0] + (tags,)])


def test_sim_smart_money_filters(world):
    """The simulation applies the smart-money filters through the bot's own
    filter function, on the tags the signal pass stored."""
    cases = [
        ({}, {}, 1),
        ({"ob_structure": "choch"}, {}, 0), ({"ob_structure": "choch"}, {"structure": "CHoCH"}, 1),
        ({"require_sweep": True}, {"liquidity": "run"}, 0), ({"require_sweep": True}, {"liquidity": "sweep", "sweep": True}, 1),
        ({"min_displacement_atr": 1.5}, {}, 0), ({"min_displacement_atr": 1.0}, {}, 1),
        ({"require_fvg": True}, {}, 0), ({"require_fvg": True}, {"fvg": True}, 1),
        ({"premium_discount": True}, {"range_pos": 0.7}, 0), ({"premium_discount": True}, {}, 1),
        ({"entry_hours": [(12, 16)]}, {}, 0), ({"entry_hours": [(7, 11)]}, {}, 1),
    ]
    for filters, tags, expect in cases:
        sim = make_sim(world, **filters)
        sim.run(0, {S: 100.6}, {S: smc_signal(**tags)}, [S])
        assert len(sim.pending.get(S, [])) == expect, (filters, tags)
    # the tags travel with the order into the trade, for the report's breakdowns
    sim = make_sim(world)
    sim.run(0, {S: 100.6}, {S: smc_signal(structure="CHoCH", liquidity="sweep", sweep=True)}, [S])
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0]], 2)
    trade = sim.positions[S]
    assert bt.structure_tag(trade) == "CHoCH" and bt.liquidity_tag(trade) == bt.LIQUIDITY_NAMES["sweep"]
    assert bt.range_half(trade) == bt.RANGE_RIGHT and bt.displacement_bucket(trade) == "1-1.5 ATR"
    # a resting order whose setup stops passing a filter is cancelled, as in the bot
    sim = make_sim(world, require_fvg=True)
    sim.run(0, {S: 100.6}, {S: smc_signal(fvg=True)}, [S])
    assert len(sim.pending[S]) == 1
    sim.run(1, {S: 100.6}, {S: smc_signal(fvg=False)}, [S])
    assert S not in sim.pending
    # signals recorded before the tags existed (7 fields) still simulate
    sim = make_sim(world)
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0]], 2)
    assert bt.structure_tag(sim.positions[S]) == "n/a" and bt.liquidity_tag(sim.positions[S]) == "n/a"


def test_every_entry_model_equals_what_the_bot_finds(world_5m):
    """One signal pass holds every entry model's setups; each equals what
    the bot's own code finds with ENTRY_MODEL set to that model."""
    from dataclasses import replace
    from strategy import ENTRY_MODELS
    cfg, data, signals, start, end = world_5m
    symbol = SYMS[0]
    sig = signals[symbol]
    steps = sorted(sig["signals"])[::max(1, len(sig["signals"]) // 60)]
    found = {m: 0 for m in ENTRY_MODELS}
    for i in steps:
        got = sig["signals"][i]
        assert got[2] == got[3].get(cfg.entry_model, [])
        client = ClientAt(cfg, data, sig["steps"][i], sig["px"][i])
        for model in ENTRY_MODELS:
            setups = bot._find_setups(client, replace(cfg, entry_model=model), symbol, {})[0]
            live = sorted((s["position_side"], s["entry"], s["stop_loss"]) for s in setups)
            mine = sorted((s[0], s[1], s[2]) for s in got[3].get(model, []))
            assert mine == pytest.approx(live), (i, model)
            found[model] += len(live)
    assert all(count > 10 for count in found.values()), found


def test_sim_entry_model(world):
    """The simulation trades the model its variant names."""
    ob = long_signal()[2][0]                                                  # entry 100
    gap = ("LONG", 100.4, 99.9, "bullish", 100.4, 100.2, 101.4)
    signal = (False, ("bullish", "bullish"), [ob], {"order_block": [ob], "fvg": [gap]})
    for model, entry in ((None, 100.0), ("order_block", 100.0), ("fvg", 100.4)):
        sim = make_sim(world, entry_model=model)
        sim.run(0, {S: 100.6}, {S: signal}, [S])
        assert [o["entry"] for o in sim.pending[S]] == [entry], model
    sim = make_sim(world, entry_model="sweep")                               # a model with no setup at this run
    sim.run(0, {S: 100.6}, {S: signal}, [S])
    assert S not in sim.pending
    # signals recorded before there were several models (three fields) still simulate
    sim = make_sim(world, entry_model="fvg")
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])
    assert [o["entry"] for o in sim.pending[S]] == [100.0]


def test_sim_stop_placement(world):
    """One signal carries the stop under every placement; the simulation
    uses the one its variant asks for, with the matching TP2 and size."""
    sig = long_signal()                                   # entry 100, stop 99.5, TP2 101
    stops = {"order_block": (99.8, 100.9), "swing": (99.5, 101.0), "previous_swing": (99.0, 102.5)}
    signal = (sig[0], sig[1], [sig[2][0] + (None, stops)])
    for placement, (stop, tp2) in list(stops.items()) + [(None, (99.5, 101.0))]:
        sim = make_sim(world, stop_placement=placement)
        sim.run(0, {S: 100.6}, {S: signal}, [S])
        order = sim.pending[S][0]
        assert (order["stop"], order["tp2"]) == (stop, tp2), placement
        sim.walk(S, [[1, 100.6, 100.7, 99.95, 100.2, 0]], 2)
        trade = sim.positions[S]
        assert trade["stop"] == stop and trade["tp2"] == tp2
        assert trade["tp1"] == pytest.approx(100.0 + (100.0 - stop))      # TP1 mirrors the stop
        assert trade["setup_stop_pct"] == pytest.approx(100.0 - stop)
    # the minimum-stop filter sees the placement's stop
    tight = make_sim(world, stop_placement="order_block", min_stop_pct=0.3)
    tight.run(0, {S: 100.6}, {S: signal}, [S])
    assert S not in tight.pending
    wide = make_sim(world, stop_placement="previous_swing", min_stop_pct=0.3)
    wide.run(0, {S: 100.6}, {S: signal}, [S])
    assert len(wide.pending[S]) == 1


def test_sim_targets_in_multiples_of_the_risk(world):
    """tp1_r / tp2_r move the take-profits: TP1 at 2R (half off, stop to
    breakeven at the next run), TP2 at 3R, whatever TP2 the signal carried."""
    sim = make_sim(world, tp1_r=2.0, tp2_r=3.0)
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])           # entry 100, stop 99.5: risk 0.5
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0]], 2)
    t = sim.positions[S]
    qty = t["qty"]
    assert (t["tp1"], t["tp2"]) == (pytest.approx(101.0), pytest.approx(101.5))
    sim.run(2, {S: 100.2}, {}, [S])
    sim.walk(S, [[3, 100.2, 100.9, 100.1, 100.8, 0]], 4)     # 1R (100.5) is no longer a target
    assert not t["tp1_done"] and t["qty"] == qty
    sim.walk(S, [[5, 100.8, 101.1, 100.7, 101.0, 0]], 6)     # TP1 at 101.0
    assert t["tp1_done"] and t["qty"] == pytest.approx(qty * 0.5)
    sim.run(6, {S: 101.0}, {}, [S])
    assert t["stop"] == 100.0
    sim.walk(S, [[7, 101.0, 101.6, 100.9, 101.5, 0]], 8)     # TP2 at 101.5
    assert t["tp2_done"] and t["qty"] == pytest.approx(qty * 0.2)
    # only TP1 moved: TP2 stays the signal's (101.0) unless it is no longer beyond TP1
    sim = make_sim(world, tp1_r=1.5)
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0]], 2)
    t = sim.positions[S]
    assert t["tp1"] == pytest.approx(100.75) and t["tp2"] == 101.0
    sim = make_sim(world, tp1_r=2.0)                          # TP1 101.0 reaches the signal's TP2: fallback beyond it
    sim.run(0, {S: 100.6}, {S: long_signal()}, [S])
    sim.walk(S, [[1, 100.6, 100.7, 99.9, 100.2, 0]], 2)
    t = sim.positions[S]
    assert t["tp1"] == pytest.approx(101.0) and t["tp2"] == pytest.approx(101.25)     # 2R + 0.5R
    assert bt.exit_kind({"exits": [(0, "stop", 1, 1)]}) == "stopped out before TP1"
    assert bt.exit_kind({"exits": [(0, "tp1", 1, 1), (1, "tp2", 1, 1), (2, "stop", 1, 1)]}) == "TP1 + TP2, runner trailed"


def test_swing_confirmation_changes_the_signals(world):
    """STRUCTURE_CONFIRMED_SWINGS reaches the signal pass: the two settings
    do not find the same setups."""
    from dataclasses import replace
    cfg, data, signals, start, end = world
    short_end = start + 3 * DAY
    on = bt.compute_signals((cfg, SYMS[0], data[SYMS[0]], start, short_end))
    off = bt.compute_signals((replace(cfg, structure_confirmed_swings=False), SYMS[0], data[SYMS[0]], start, short_end))
    assert on["steps"] == off["steps"]
    assert {i: [s[:3] for s in v[2]] for i, v in on["signals"].items()} != \
           {i: [s[:3] for s in v[2]] for i, v in off["signals"].items()}


def test_sim_slots_zones_and_one_position_per_symbol(world):
    cfg = world[0]
    sim = make_sim(world)
    symbols = [f"S{i}/USDT:USDT" for i in range(7)]
    sim.run(0, {s: 100.6 for s in symbols}, {s: long_signal() for s in symbols}, symbols)
    assert len(sim.pending) == cfg.max_concurrent_positions == 5
    s0 = symbols[0]
    sim.walk(s0, [[1, 100.6, 100.7, 99.4, 99.45, 0]], 2)    # fills and stops out in one candle
    assert s0 not in sim.positions and len(sim.trades) == 1
    sim.run(2, {s: 100.6 for s in symbols}, {s: long_signal() for s in symbols}, symbols)
    assert s0 not in sim.pending                            # that zone was traded: no second order
    assert len(sim.pending) == 5                            # the freed slot went to another symbol


def test_sim_immediate_fill_inside_the_zone_pays_taker(world):
    sim = make_sim(world)
    sim.run(0, {S: 99.8}, {S: long_signal()}, [S])           # price already inside the zone
    t = sim.positions[S]
    assert t["entry"] == 99.8 and t["taker_entry"]
    assert t["fees"] == pytest.approx(t["qty"] * 99.8 * 0.0005)


def test_whole_period_accounting_identity(world):
    cfg, data, signals, start, end = world
    costs = bt.Costs(0.02, 0.05, 0.02)
    exec_data = {s: data[s]["5m"] for s in SYMS}
    for variant in ({}, {"exit_mode": "run"}, {"min_stop_pct": 0.3, "timing_mode": "not_against"}):
        sim = bt.simulate(cfg, costs, 230.0, signals, exec_data, SYMS, **variant)
        assert len(sim.trades) > 10
        open_fees = sum(t["fees"] for t in sim.positions.values())
        booked_open = sum(t["pnl"] for t in sim.positions.values())
        assert sim.cash == pytest.approx(230.0 + sum(t["pnl"] for t in sim.trades) + booked_open)
        for t in sim.trades:
            assert sum(e[2] for e in t["exits"]) == pytest.approx(t["initial_qty"])
            assert t["risk_usd"] <= 230.0 * 0.03 * 1.5       # never sized past the risk cap (equity moves a little)
        assert len({(t["symbol"], t["entry_time"]) for t in sim.trades}) == len(sim.trades)
        assert open_fees >= 0
    filtered = bt.simulate(cfg, costs, 230.0, signals, exec_data, SYMS, min_stop_pct=0.3)
    assert all(t["setup_stop_pct"] >= 0.3 for t in filtered.trades)


def test_period_shrinks_to_the_history_bingx_actually_has(world):
    cfg, data, signals, start, end = world
    step = bt.TF_MS["15m"]
    warm = max(cfg.candle_limit * step, bt.TF_MS["1w"]) + step

    # everything available: 5m is used, the period is untouched
    syms, begin, exec_tf, notes = bt.usable_period(cfg, data, SYMS, start, "auto")
    assert (syms, begin, exec_tf, notes) == (SYMS, start, "5m", [])

    # 5m history starts late: auto falls back to the entry timeframe for the whole period
    cut = {s: dict(data[s]) for s in SYMS}
    for s in SYMS:
        cut[s]["5m"] = data[s]["5m"][data[s]["5m"][:, 0] >= start + 5 * DAY]
    syms, begin, exec_tf, notes = bt.usable_period(cfg, cut, SYMS, start, "auto")
    assert begin == start and exec_tf == "15m" and len(notes) == 1
    # ...and an explicit 5m shortens the period instead
    syms, begin, exec_tf, notes = bt.usable_period(cfg, cut, SYMS, start, "5m")
    assert exec_tf == "5m" and begin >= start + 5 * DAY

    # entry history starts late: the test starts after the warm-up
    for s in SYMS:
        cut[s]["15m"] = data[s]["15m"][data[s]["15m"][:, 0] >= start - DAY]
    syms, begin, exec_tf, notes = bt.usable_period(cfg, cut, SYMS, start, "auto")
    assert begin == start - DAY + warm and begin % step == 0 and syms == SYMS
    sig = bt.compute_signals((cfg, SYMS[0], cut[SYMS[0]], begin, end))
    assert sig["steps"][0] == begin and len(sig["signals"]) > 5

    # one symbol with a much shorter history is left out rather than shrinking everyone
    three = SYMS + ["CCC/USDT:USDT"]
    more = dict(data)
    more["CCC/USDT:USDT"] = {tf: a[a[:, 0] >= end - 2 * DAY] for tf, a in data[SYMS[0]].items()}
    syms, begin, exec_tf, notes = bt.usable_period(cfg, more, three, start, "auto")
    assert syms == SYMS and begin == start and any("CCC left out" in n for n in notes)
