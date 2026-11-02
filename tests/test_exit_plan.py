"""TP2_MIN_R_MULTIPLE: TP2 never sits closer to the entry than that many R.
Off by default; one rule (strategy.apply_tp2_floor) for the bot and the backtest."""
import pytest

import bot
import config
from strategy import apply_tp2_floor

SOL = "SOL/USDT:USDT"
ENTRY, STOP = 120.991, 120.60927          # 1R = 0.38173
SWING = 121.583                           # the harness's next swing: 1.55R away


def test_floor_rule():
    # LONG, 1R = 1.0: a swing 1.4R away moves to 2R, one 2.5R away stays
    assert apply_tp2_floor(100.0, 1.0, True, 101.4, 2.0) == (102.0, True)
    assert apply_tp2_floor(100.0, 1.0, True, 102.5, 2.0) == (102.5, False)
    assert apply_tp2_floor(100.0, 1.0, True, 102.0, 2.0) == (102.0, False)
    # SHORT, mirrored
    assert apply_tp2_floor(100.0, 1.0, False, 98.6, 2.0) == (98.0, True)
    assert apply_tp2_floor(100.0, 1.0, False, 97.5, 2.0) == (97.5, False)
    # off, and nothing to move
    assert apply_tp2_floor(100.0, 1.0, True, 101.4, 0.0) == (101.4, False)
    assert apply_tp2_floor(100.0, 1.0, True, None, 2.0) == (None, False)


def test_off_by_default_and_never_negative(monkeypatch):
    monkeypatch.delenv("TP2_MIN_R_MULTIPLE", raising=False)
    assert config.load_config().tp2_min_r_multiple == 0.0
    monkeypatch.setenv("TP2_MIN_R_MULTIPLE", "")          # an Actions Variable that was never created
    assert config.load_config().tp2_min_r_multiple == 0.0
    monkeypatch.setenv("TP2_MIN_R_MULTIPLE", "-1")
    assert config.load_config().tp2_min_r_multiple == 0.0


@pytest.mark.parametrize("side, entry, stop, swing, floored", [
    ("LONG", 100.0, 99.0, 101.4, 102.0),
    ("SHORT", 100.0, 101.0, 98.6, 98.0),
])
def test_bot_picks_tp2_with_the_minimum(monkeypatch, side, entry, stop, swing, floored):
    monkeypatch.setattr(bot, "next_swing_beyond", lambda df, direction, beyond, n=2: swing)
    setup = {"position_side": side, "entry": entry, "stop_loss": stop}
    monkeypatch.delenv("TP2_MIN_R_MULTIPLE", raising=False)
    price, source = bot._find_tp2_level(None, setup, config.load_config())
    assert price == swing and "next swing" in source and "(1.40R)" in source
    monkeypatch.setenv("TP2_MIN_R_MULTIPLE", "2")
    price, source = bot._find_tp2_level(None, setup, config.load_config())
    assert price == pytest.approx(floored)
    assert source.startswith("2R minimum") and "1.40R" in source
    # a swing already further away than the minimum is used as it is
    monkeypatch.setattr(bot, "next_swing_beyond", lambda df, direction, beyond, n=2: entry + (3 if side == "LONG" else -3))
    price, source = bot._find_tp2_level(None, setup, config.load_config())
    assert abs(price - entry) == pytest.approx(3.0) and "next swing" in source and "(3.00R)" in source


def test_live_trade_default_keeps_the_swing(make_harness):
    h = make_harness()
    h.open_entry()
    h.run()
    assert h.trade()["tp2_price"] == SWING
    assert ("TAKE_PROFIT_MARKET", SWING, 0.22) in h.exit_orders_open()


def test_live_trade_with_the_minimum_rests_tp2_at_2r(make_harness):
    h = make_harness(TP2_MIN_R_MULTIPLE="2")
    h.open_entry()                                  # its swing is only 1.55R away
    h.run()
    t = h.trade()
    want = ENTRY + 2 * (ENTRY - STOP)               # 121.75446
    assert t["tp2_price"] == pytest.approx(want)
    assert t["tp2_source"].startswith("2R minimum")
    tps = [o for o in h.exit_orders_open() if o[0] == "TAKE_PROFIT_MARKET"]
    assert len(tps) == 2 and tps[1][1] == pytest.approx(want, abs=0.001) and tps[1][2] == 0.22
    # TP1 still at 1R with half the position; price reaching the old swing no longer takes TP2
    h.ex.set_price(SOL, SWING + 0.01)
    h.run()
    t = h.trade()
    assert t["partial_taken"] and not t["tp2_taken"] and t["qty"] == pytest.approx(0.37)
    h.ex.set_price(SOL, want + 0.01)
    h.run()
    t = h.trade()
    assert t["tp2_taken"] and t["qty"] == pytest.approx(0.15)


def test_dry_run_trade_with_the_minimum(make_harness):
    h = make_harness(TP2_MIN_R_MULTIPLE="2", DRY_RUN="true")
    risk = h.risk()
    risk.track_dry_run_trade({
        "trade_id": "t1", "entry_time_utc": "2026-10-11T00:00:00+00:00", "mode": "DRY_RUN", "symbol": SOL,
        "side": "BUY", "position_side": "LONG", "qty": 0.74, "entry_price": ENTRY, "stop_loss": STOP,
        "take_profit": ENTRY + 2 * (ENTRY - STOP), "trigger": "t",
        "tp2_price": SWING, "tp2_source": "next swing high (1.55R)",
    })
    t = h.risk().state["open_dry_run_trades"][0]
    assert t["tp2_price"] == pytest.approx(ENTRY + 2 * (ENTRY - STOP))
    assert t["tp2_source"] == "2R minimum (was next swing high (1.55R))"
    assert t["partial_tp_price"] == pytest.approx(ENTRY + (ENTRY - STOP))
