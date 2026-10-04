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


@pytest.fixture(scope="module")
def world():
    import os
    for key in ("SYMBOLS", "MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT",
                "MAX_CONCURRENT_POSITIONS", "RISK_PCT_PER_TRADE", "LEVERAGE", "WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT"):
        os.environ.pop(key, None)
    os.environ.update({"MAX_CONCURRENT_POSITIONS": "5", "RISK_PCT_PER_TRADE": "2", "LEVERAGE": "10",
                       "WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT": "1"})
    cfg = config.load_config()
    end = 1_790_000_000_000 - 1_790_000_000_000 % bt.TF_MS["15m"]
    start = end - 12 * DAY
    data = bt.synthetic_history(SYMS, start, end, seed=11, exec_tf="5m",
                                htf_timeframes=cfg.htf_timeframes, htf_lookback=cfg.trend_candle_limit + 5)
    signals = {s: bt.compute_signals((cfg, s, data[s], start, end)) for s in SYMS}
    yield cfg, data, signals, start, end
    for key in ("MAX_CONCURRENT_POSITIONS", "RISK_PCT_PER_TRADE", "LEVERAGE", "WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT"):
        os.environ.pop(key, None)


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
    cfg, data, signals, start, end = world
    rng = np.random.default_rng(3)
    checked = with_setups = overrides = conflicts = 0
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
                # and the 4h/1h trends the timing filter uses
                _, _, trends = bot.analyze_higher_timeframes(client, cfg, symbol)
                assert got[1] == tuple(trends[tf] for tf in cfg.htf_timeframes if tf not in cfg.primary_htf_timeframes)
            conflicts += "disagree" in detail or setups == [] and "NOT overridden" in detail
            checked += 1
    assert checked >= 300 and with_setups >= 60


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
