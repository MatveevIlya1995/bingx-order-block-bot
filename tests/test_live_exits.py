"""
LIVE exit management, run end to end: the real bot.run_once(), RiskManager
and ExchangeClient on top of tests/fake_bingx.py.

Every scenario using the `h` fixture runs twice -- against a fake exchange
that rejects a triggered close order bigger than the position, and against
one that clamps it -- because that BingX behaviour could not be verified.
"""
import csv
import json

import ccxt
import pytest

SOL = "SOL/USDT:USDT"
BTC = "BTC/USDT:USDT"

ENTRY, STOP = 120.991, 120.60927          # 1R = 0.38173
TP1, TP2 = 121.37273, 121.583
STOP_TICK, TP1_TICK = 120.609, 121.373    # what BingX stores after rounding to the price tick


def realized_matches_account(h):
    """The bot's realized P/L for the day must equal what the fake account
    really made: every fill's price P/L minus every fee, entry fee included."""
    state = h.risk().state
    assert state["realized_pnl_today"] == pytest.approx(h.ex.account_pnl(), abs=1e-6)
    rows = [r for r in h.journal() if r["event"] in ("PARTIAL_EXIT", "EXIT")]
    assert sum(float(r["pnl"]) for r in rows) == pytest.approx(h.ex.account_pnl(), abs=1e-3)
    fees = sum(float(r["fee"]) for r in h.journal() if r["fee"] not in ("", None))
    assert fees == pytest.approx(h.ex.fees, abs=1e-4)


def test_arming_puts_exits_on_the_exchange(h):
    h.open_entry()
    assert h.exit_orders_open() == [("STOP_MARKET", STOP_TICK, 0.74)]   # attached by BingX

    h.run()

    assert h.exit_orders_open() == [
        ("STOP_MARKET", STOP_TICK, 0.15),
        ("STOP_MARKET", STOP_TICK, 0.22),
        ("STOP_MARKET", STOP_TICK, 0.37),
        ("TAKE_PROFIT_MARKET", TP1_TICK, 0.37),
        ("TAKE_PROFIT_MARKET", TP2, 0.22),
    ]
    eo = h.trade()["exit_orders"]
    assert eo["mode"] == "resting" and len(eo["stop_ids"]) == 3
    assert eo["tp1_id"] and eo["tp2_id"] and eo["tp1_qty"] == 0.37 and eo["tp2_qty"] == 0.22
    entry_row = [r for r in h.journal() if r["event"] == "ENTRY"][0]
    assert float(entry_row["fee"]) == pytest.approx(0.74 * ENTRY * 0.0002, abs=1e-6)

    # A second run must not place anything again.
    before = len(h.ex.orders)
    h.run()
    assert len(h.ex.orders) == before


def test_full_lifecycle_tp1_tp2_trail_stop(h):
    h.open_entry()
    h.run()

    # TP1 fills on the exchange between two runs -- no bot involved.
    h.ex.set_price(SOL, 121.40)
    assert h.ex.position_qty(SOL) == pytest.approx(0.37)
    h.run()
    t = h.trade()
    assert t["partial_taken"] and t["qty"] == pytest.approx(0.37) and t["stop_loss"] == ENTRY
    assert h.exit_orders_open() == [
        ("STOP_MARKET", ENTRY, 0.15),
        ("STOP_MARKET", ENTRY, 0.22),
        ("TAKE_PROFIT_MARKET", TP2, 0.22),
    ]
    row = [r for r in h.journal() if r["event"] == "PARTIAL_EXIT"][-1]
    assert float(row["qty"]) == 0.37 and float(row["price"]) == 121.40      # the REAL fill, not the TP level
    assert float(row["fee"]) == pytest.approx(0.37 * 121.40 * 0.0005, abs=1e-6)
    entry_fee = 0.74 * ENTRY * 0.0002
    expected = (121.40 - ENTRY) * 0.37 - 0.37 * 121.40 * 0.0005 - entry_fee * 0.5
    assert float(row["pnl"]) == pytest.approx(expected, abs=1e-4)
    assert "resting order on BingX" in row["trigger"] and "fill price from BingX" in row["trigger"]

    # TP2 fills between runs; the same run then starts trailing the runner.
    h.ex.set_price(SOL, 121.60)
    h.run()
    t = h.trade()
    assert t["tp2_taken"] and t["qty"] == pytest.approx(0.15)
    assert h.exit_orders_open() == [("STOP_MARKET", round(121.60 - 0.38173, 3), 0.15)]

    h.ex.set_price(SOL, 122.50)
    h.run()
    assert h.exit_orders_open() == [("STOP_MARKET", round(122.50 - 0.38173, 3), 0.15)]

    # The trailing stop is hit between runs.
    h.ex.set_price(SOL, 122.00)
    assert h.ex.position_qty(SOL) == 0
    h.run()
    assert h.trade() is None
    assert h.exit_orders_open() == []
    events = [r["event"] for r in h.journal() if r["event"] in ("ENTRY", "PARTIAL_EXIT", "EXIT")]
    assert events == ["ENTRY", "PARTIAL_EXIT", "PARTIAL_EXIT", "EXIT"]
    exit_row = [r for r in h.journal() if r["event"] == "EXIT"][0]
    assert float(exit_row["qty"]) == 0.15 and float(exit_row["price"]) == 122.00
    realized_matches_account(h)


@pytest.mark.parametrize("trigger_order", ["oldest", "newest", "largest"])
def test_tp1_fills_then_price_crashes_through_the_stop_before_the_next_run(h, trigger_order):
    """The case per-part stops exist for: after TP1 fills, the stops still on
    the exchange add up to MORE than the position. Whatever order BingX fires
    them in, and whether it rejects or clamps the oversized ones, the position
    must end up flat."""
    h.ex.trigger_order = trigger_order
    h.open_entry()
    h.run()
    h.ex.set_price(SOL, 121.40)          # TP1 fills
    h.ex.set_price(SOL, 120.55)          # ...and price falls through the original stop
    assert h.ex.position_qty(SOL) == 0   # nothing stranded without a stop
    h.run()
    assert h.trade() is None and h.exit_orders_open() == []
    rows = [r for r in h.journal() if r["event"] in ("PARTIAL_EXIT", "EXIT")]
    assert [r["event"] for r in rows] == ["PARTIAL_EXIT", "EXIT"]
    assert float(rows[1]["qty"]) == pytest.approx(0.37) and float(rows[1]["price"]) == 120.55
    realized_matches_account(h)


@pytest.mark.parametrize("trigger_order", ["oldest", "newest", "largest"])
def test_tp1_and_tp2_fill_then_stop_before_the_next_run(h, trigger_order):
    h.ex.trigger_order = trigger_order
    h.open_entry()
    h.run()
    h.ex.set_price(SOL, 121.40)
    h.ex.set_price(SOL, 121.60)
    assert h.ex.position_qty(SOL) == pytest.approx(0.15)
    h.ex.set_price(SOL, 120.55)
    assert h.ex.position_qty(SOL) == 0
    h.run()
    assert h.trade() is None and h.exit_orders_open() == []
    rows = [r for r in h.journal() if r["event"] in ("PARTIAL_EXIT", "EXIT")]
    assert [r["event"] for r in rows] == ["PARTIAL_EXIT", "PARTIAL_EXIT", "EXIT"]
    realized_matches_account(h)


def test_a_single_full_size_stop_would_strand_the_position(make_harness):
    """Why the stop is split: with ONE full-size stop and a resting TP1, an
    exchange that rejects oversized closes leaves half the position open
    with no stop at all."""
    h = make_harness(oversize="reject")
    h.open_entry()                                        # one 0.74 stop, attached by BingX
    h.client.place_take_profit_order(SOL, "sell", 0.37, TP1, "LONG")
    h.ex.set_price(SOL, 121.40)
    h.ex.set_price(SOL, 120.55)
    assert h.ex.position_qty(SOL) == pytest.approx(0.37)  # stranded
    assert h.exit_orders_open() == []                     # and unprotected


def test_stopped_out_before_the_first_run_sees_the_fill(h):
    h.open_entry()
    h.ex.set_price(SOL, 120.55)          # attached stop closes everything
    h.run()
    assert h.trade() is None and h.exit_orders_open() == []
    rows = h.journal()
    assert [r["event"] for r in rows if r["event"] in ("ENTRY", "EXIT")] == ["ENTRY", "EXIT"]
    assert float(rows[-1]["price"]) == 120.55
    realized_matches_account(h)


def test_tp1_cannot_rest_falls_back_to_market_close_on_the_run(h):
    h.open_entry()
    h.ex.set_price(SOL, 121.45)          # already past TP1 when the bot first runs
    h.run()
    t = h.trade()
    assert t["partial_taken"] and t["qty"] == pytest.approx(0.37)
    assert t["exit_orders"]["tp2_id"]    # TP2 still rests
    row = [r for r in h.journal() if r["event"] == "PARTIAL_EXIT"][0]
    assert "market close on the scheduled run" in row["trigger"]
    assert float(row["price"]) == 121.45 and "fill price from BingX" in row["trigger"]
    assert h.exit_orders_open() == [
        ("STOP_MARKET", ENTRY, 0.15),
        ("STOP_MARKET", ENTRY, 0.22),
        ("TAKE_PROFIT_MARKET", TP2, 0.22),
    ]
    h.ex.set_price(SOL, 121.0)
    h.ex.set_price(SOL, 120.9)           # breakeven stops
    h.run()
    assert h.trade() is None and h.exit_orders_open() == []
    realized_matches_account(h)


def test_stops_cannot_be_split_falls_back_to_the_old_behaviour(h):
    h.open_entry()
    h.ex.fail_next.append(("STOP_MARKET", ccxt.ExchangeError("bingx says no")))
    h.run()
    eo = h.trade()["exit_orders"]
    assert eo["mode"] == "legacy" and "tp1_id" not in eo
    assert h.exit_orders_open() == [("STOP_MARKET", STOP_TICK, 0.74)]   # untouched original

    h.ex.set_price(SOL, 121.40)          # nothing rests, so nothing happens on the exchange
    assert h.ex.position_qty(SOL) == pytest.approx(0.74)
    h.run()                              # the run closes TP1 at market, as before
    t = h.trade()
    assert t["partial_taken"] and t["qty"] == pytest.approx(0.37)
    # ...and the superseded full-size stop is really gone now
    assert h.exit_orders_open() == [("STOP_MARKET", ENTRY, 0.15), ("STOP_MARKET", ENTRY, 0.22)]

    h.ex.set_price(SOL, 121.60)
    h.run()
    assert h.trade()["tp2_taken"] and h.trade()["qty"] == pytest.approx(0.15)
    h.ex.set_price(SOL, 121.0)
    h.run()
    assert h.trade() is None and h.exit_orders_open() == []
    realized_matches_account(h)


def test_breakeven_stop_refused_keeps_old_stops_and_retries(h):
    h.open_entry()
    h.run()
    h.ex.set_price(SOL, 121.40)          # TP1 fills
    h.ex.set_price(SOL, 120.80)          # back below the entry before the next run
    h.run()
    t = h.trade()
    assert t["partial_taken"] and t["breakeven_pending"] and t["stop_loss"] == STOP
    # still fully protected at the original level
    assert [o for o in h.exit_orders_open() if o[0] == "STOP_MARKET"] == [
        ("STOP_MARKET", STOP_TICK, 0.15), ("STOP_MARKET", STOP_TICK, 0.22), ("STOP_MARKET", STOP_TICK, 0.37)]

    h.ex.set_price(SOL, 121.10)
    h.run()
    t = h.trade()
    assert "breakeven_pending" not in t and t["stop_loss"] == ENTRY
    assert [o for o in h.exit_orders_open() if o[0] == "STOP_MARKET"] == [
        ("STOP_MARKET", ENTRY, 0.15), ("STOP_MARKET", ENTRY, 0.22)]


def test_breakeven_pending_then_original_stop_hits(h):
    h.open_entry()
    h.run()
    h.ex.set_price(SOL, 121.40)
    h.ex.set_price(SOL, 120.80)
    h.run()
    h.ex.set_price(SOL, 120.55)
    assert h.ex.position_qty(SOL) == 0
    h.run()
    assert h.trade() is None and h.exit_orders_open() == []
    realized_matches_account(h)


def test_unreadable_tp_order_never_causes_a_second_close(h):
    h.open_entry()
    h.run()
    tp1_id = h.trade()["exit_orders"]["tp1_id"]
    h.ex.fail_fetch_order.add(tp1_id)

    # Order unreadable and NOT filled: the bot must just wait.
    h.ex.set_price(SOL, 121.20)
    h.run()
    assert not h.trade()["partial_taken"]
    assert h.ex.position_qty(SOL) == pytest.approx(0.74)

    # Order unreadable and filled: the shrunken position gives it away.
    h.ex.set_price(SOL, 121.40)
    h.run()
    t = h.trade()
    assert t["partial_taken"] and t["qty"] == pytest.approx(0.37)
    assert h.ex.position_qty(SOL) == pytest.approx(0.37)       # no extra market close
    market_orders = [c for c in h.ex.calls if c[0] == "create_order" and c[2].upper() == "MARKET"]
    assert market_orders == []
    row = [r for r in h.journal() if r["event"] == "PARTIAL_EXIT"][0]
    assert "ESTIMATED" in row["trigger"]


def test_tp_order_cancelled_by_hand_switches_to_the_fallback(h):
    h.open_entry()
    h.run()
    h.ex.cancel_order(h.trade()["exit_orders"]["tp1_id"], SOL)
    h.ex.set_price(SOL, 121.40)
    assert h.ex.position_qty(SOL) == pytest.approx(0.74)
    h.run()
    t = h.trade()
    assert t["partial_taken"] and t["qty"] == pytest.approx(0.37)
    row = [r for r in h.journal() if r["event"] == "PARTIAL_EXIT"][0]
    assert "market close on the scheduled run" in row["trigger"]


def test_arming_adopts_orders_when_the_state_update_was_lost(h):
    h.open_entry()
    state_path = h.tmp / "state.json"
    before_run = state_path.read_text()
    h.run()
    n_orders = len(h.ex.orders)
    armed = h.exit_orders_open()

    state_path.write_text(before_run)     # the run's state never made it back to the repo
    h.run()
    assert len(h.ex.orders) == n_orders   # nothing placed twice
    assert h.exit_orders_open() == armed
    eo = h.trade()["exit_orders"]
    assert eo["mode"] == "resting" and len(eo["stop_ids"]) == 3 and eo["tp1_id"] and eo["tp2_id"]


def test_trades_open_before_the_deploy_are_picked_up(h):
    """The live state at deploy time: one trade past TP1 (with the stops the
    previous version left behind), one fresh trade."""
    ex = h.ex
    ex.positions[(BTC, "LONG")] = {"qty": 0.0005, "entry": 85104.8}
    ex.positions[(SOL, "LONG")] = {"qty": 0.74, "entry": ENTRY}
    ex.price[BTC], ex.price[SOL] = 85250.0, 121.10
    ex._new_order(BTC, "STOP_MARKET", "SELL", "LONG", 0.001, stop_price=84909.1)    # original, never cancelled
    ex._new_order(BTC, "STOP_MARKET", "SELL", "LONG", 0.0005, stop_price=85104.8)   # breakeven stop
    ex._new_order(SOL, "STOP_MARKET", "SELL", "LONG", 0.74, stop_price=STOP_TICK)
    state = {
        "date": None, "day_start_equity": None, "realized_pnl_today": 0.0,
        "open_trades": [
            {"trade_id": "a8fea25da9c4", "entry_time_utc": "2026-10-04T12:54:10.781000+00:00",
             "mode": "LIVE", "symbol": BTC, "side": "BUY", "position_side": "LONG", "qty": 0.0005,
             "entry_price": 85104.8, "stop_loss": 85104.8, "take_profit": 85496.18819999999,
             "trigger": "t", "tp2_price": 85481.7, "tp2_source": "next swing high (1.93R)",
             "initial_qty": 0.001, "risk_per_unit": 195.69409999999334,
             "partial_tp_price": 85300.4941, "partial_taken": True, "tp2_taken": False},
            {"trade_id": "9566255165e5", "entry_time_utc": "2026-10-04T20:00:09.739000+00:00",
             "mode": "LIVE", "symbol": SOL, "side": "BUY", "position_side": "LONG", "qty": 0.74,
             "entry_price": ENTRY, "stop_loss": STOP, "take_profit": 121.75445999999998,
             "trigger": "t", "tp2_price": TP2, "tp2_source": "next swing high (1.55R)",
             "initial_qty": 0.74, "risk_per_unit": 0.38172999999999035,
             "partial_tp_price": TP1, "partial_taken": False, "tp2_taken": False},
        ],
        "open_dry_run_trades": [], "pending_orders": [], "traded_zones": {},
    }
    (h.tmp / "state.json").write_text(json.dumps(state))

    h.run()

    assert h.exit_orders_open(BTC) == [
        ("STOP_MARKET", 84909.1, 0.001),          # pre-deploy leftover, cleared when the trade closes
        ("STOP_MARKET", 85104.8, 0.0002),
        ("STOP_MARKET", 85104.8, 0.0003),
        ("TAKE_PROFIT_MARKET", 85481.7, 0.0003),
    ]
    assert h.exit_orders_open(SOL) == [
        ("STOP_MARKET", STOP_TICK, 0.15), ("STOP_MARKET", STOP_TICK, 0.22), ("STOP_MARKET", STOP_TICK, 0.37),
        ("TAKE_PROFIT_MARKET", TP1_TICK, 0.37), ("TAKE_PROFIT_MARKET", TP2, 0.22),
    ]

    ex.set_price(BTC, 85490.0)             # BTC TP2 fills
    h.run()
    t = h.trade(BTC)
    assert t["tp2_taken"] and t["qty"] == pytest.approx(0.0002)
    row = [r for r in h.journal() if r["event"] == "PARTIAL_EXIT"][0]
    assert "entry fee estimated" in row["trigger"]      # this trade predates fee tracking

    ex.set_price(BTC, 85200.0)             # trailing stop (85490 - 195.69 = 85294.3) hit
    h.run()
    assert h.trade(BTC) is None
    assert h.exit_orders_open(BTC) == []   # the old leftover stop is gone too


def test_resting_tp_orders_switched_off_is_the_old_behaviour(make_harness):
    h = make_harness(RESTING_TP_ORDERS="false")
    h.open_entry()
    h.run()
    assert "exit_orders" not in h.trade()
    assert h.exit_orders_open() == [("STOP_MARKET", STOP_TICK, 0.74)]
    h.ex.set_price(SOL, 121.40)
    h.run()
    assert h.trade()["partial_taken"]
    assert h.exit_orders_open() == [("STOP_MARKET", ENTRY, 0.15), ("STOP_MARKET", ENTRY, 0.22)]
    realized = h.risk().state["realized_pnl_today"]
    assert realized == pytest.approx(h.ex.account_pnl() + 0.74 * ENTRY * 0.0002 * 0.5, abs=1e-6)


def test_short_trade_lifecycle(h):
    entry, stop = 121.000, 121.400       # 1R = 0.4 -> TP1 120.6
    h.ex.price[SOL] = 120.90             # market below the sell limit, so it rests first
    h.open_entry(entry=entry, stop=stop, qty=0.74, tp2=120.300, side="sell")
    h.run()
    assert h.exit_orders_open() == [
        ("STOP_MARKET", 121.4, 0.15), ("STOP_MARKET", 121.4, 0.22), ("STOP_MARKET", 121.4, 0.37),
        ("TAKE_PROFIT_MARKET", 120.3, 0.22), ("TAKE_PROFIT_MARKET", 120.6, 0.37),
    ]
    h.ex.set_price(SOL, 120.55)
    h.run()
    assert h.trade()["partial_taken"] and h.trade()["stop_loss"] == entry
    h.ex.set_price(SOL, 120.25)
    h.run()
    assert h.trade()["tp2_taken"]
    assert h.exit_orders_open() == [("STOP_MARKET", 120.65, 0.15)]      # 120.25 + 0.4 trail
    h.ex.set_price(SOL, 120.70)
    h.run()
    assert h.trade() is None and h.exit_orders_open() == []
    realized_matches_account(h)


def test_leftover_sweep_is_skipped_while_an_entry_order_rests_on_the_symbol(h):
    h.open_entry()
    h.run()
    risk = h.risk()
    risk.track_pending_order({"order_id": "424242", "trade_id": "x", "mode": "LIVE", "symbol": SOL,
                              "side": "buy", "position_side": "LONG", "qty": 0.1, "entry_price": 119.0,
                              "stop_loss": 118.5, "zone_key": "z"})
    stray = h.ex._new_order(SOL, "STOP_MARKET", "SELL", "LONG", 0.1, stop_price=118.5)
    h.ex.positions.pop((SOL, "LONG"))
    risk = h.risk()
    closed = risk.reconcile_closed_trades([], h.client)
    assert len(closed) == 1
    assert h.ex.orders[stray["orderId"]]["status"] == "NEW"      # not swept
    # the closed trade's own orders are cancelled regardless
    assert [o for o in h.ex.open_orders(SOL) if o["orderId"] != stray["orderId"]] == []


def test_exit_order_kind_reads_the_raw_bingx_type(h):
    from exchange_client import ExchangeClient
    h.client.create_limit_order_with_stop(
        symbol=SOL, side="buy", amount=0.5, price=119.0, stop_loss_price=118.5, position_side="LONG")
    stop = h.ex._new_order(SOL, "STOP_MARKET", "SELL", "LONG", 0.5, stop_price=118.5)
    tp = h.ex._new_order(SOL, "TAKE_PROFIT_MARKET", "SELL", "LONG", 0.5, stop_price=125.0)
    by_id = {o["id"]: o for o in h.ex.fetch_open_orders(SOL)}
    limit_entry = [o for o in by_id.values() if o["info"]["type"] == "LIMIT"][0]
    assert limit_entry["stopLossPrice"]                           # ccxt shows the ATTACHED stop here...
    assert ExchangeClient._exit_order_kind(limit_entry) is None   # ...and it is still not a stop order
    assert by_id[str(stop["orderId"])]["type"] == "market"        # ccxt's unified type is useless
    assert ExchangeClient._exit_order_kind(by_id[str(stop["orderId"])]) == "stop"
    assert ExchangeClient._exit_order_kind(by_id[str(tp["orderId"])]) == "tp"
    kinds = sorted(o["kind"] for o in h.client.open_exit_orders(SOL, "LONG", "sell"))
    assert kinds == ["stop", "tp"]


def test_order_fill_refuses_to_guess(h):
    h.open_entry()
    h.run()
    tp1_id = h.trade()["exit_orders"]["tp1_id"]
    assert h.client.order_fill(tp1_id, SOL)["status"] == "open"
    h.ex.orders[int(tp1_id)]["status"] = "SOMETHING_NEW"
    with pytest.raises(RuntimeError):
        h.client.order_fill(tp1_id, SOL)
    h.ex.orders[int(tp1_id)]["status"] = "FAILED"
    assert h.client.order_fill(tp1_id, SOL)["status"] == "dead"


def test_place_stop_orders_is_all_or_nothing(h):
    h.open_entry()
    before = h.exit_orders_open()
    h.ex.fail_next.append(("STOP_MARKET", ccxt.ExchangeError("first ok, second fails")))
    # make the SECOND create fail: consume the injected failure only on call 2
    original = h.ex.create_order
    n = {"calls": 0}

    def flaky(symbol, type, side, amount, price=None, params=None):
        n["calls"] += 1
        if n["calls"] != 2:
            saved, h.ex.fail_next = h.ex.fail_next, []
            try:
                return original(symbol, type, side, amount, price, params)
            finally:
                h.ex.fail_next = saved
        return original(symbol, type, side, amount, price, params)

    h.ex.create_order = flaky
    with pytest.raises(ccxt.ExchangeError):
        h.client.place_stop_orders(SOL, "sell", [0.37, 0.22, 0.15], STOP, "LONG")
    assert h.exit_orders_open() == before      # the one that got through was cancelled again


def test_journal_header_upgrade_keeps_every_row(make_harness):
    h = make_harness()
    old_fields = ["event", "trade_id", "timestamp_utc", "mode", "symbol", "side", "position_side",
                  "qty", "price", "stop_loss", "take_profit", "equity", "trigger", "pnl", "duration_minutes"]
    rows = [
        ["ENTRY", "aaa", "2026-10-01T00:00:00+00:00", "LIVE", SOL, "BUY", "LONG", "0.74", "120.991",
         "120.60927", "121.75446", "230.1", 'zone 120.73-120.991 | HTF: 1w*=bullish(2 OB), "quoted", commas', "", ""],
        ["EXIT", "aaa", "2026-10-01T01:00:00+00:00", "LIVE", SOL, "BUY", "LONG", "0.74", "120.6",
         "120.60927", "121.75446", "", "stop", "-0.2893", "60.0"],
    ]
    path = h.tmp / "trades.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(old_fields)
        w.writerows(rows)
    original = path.read_bytes()

    risk = h.risk()
    risk._write_journal_row({"event": "ENTRY", "trade_id": "bbb", "mode": "LIVE", "symbol": SOL, "fee": 0.0179})

    with open(path, newline="") as f:
        got = list(csv.reader(f))
    assert got[0] == old_fields + ["fee"]
    assert got[1] == rows[0] + [""] and got[2] == rows[1] + [""]
    assert got[3][0] == "ENTRY" and got[3][-1] == "0.0179" and len(got[3]) == 16
    assert all(len(r) == 16 for r in got)
    # byte-for-byte: old rows are untouched apart from the added empty cell
    upgraded = path.read_bytes().split(b"\r\n")
    for old_line, new_line in zip(original.split(b"\r\n")[1:3], upgraded[1:3]):
        assert new_line == old_line + b","
    assert not (h.tmp / "trades.csv.tmp").exists()

    # idempotent
    h.risk()._write_journal_row({"event": "ENTRY", "trade_id": "ccc", "mode": "LIVE", "symbol": SOL})
    with open(path, newline="") as f:
        assert len(list(csv.reader(f))) == 5


def test_journal_with_unknown_header_is_left_alone(make_harness):
    h = make_harness()
    path = h.tmp / "trades.csv"
    path.write_bytes(b"something,else\r\n1,2\r\n")
    h.risk()._write_journal_row({"event": "ENTRY", "trade_id": "bbb", "mode": "LIVE", "symbol": SOL})
    assert path.read_bytes().startswith(b"something,else\r\n1,2\r\n")


def test_circuit_breaker_counts_realized_losses_in_live(make_harness):
    h = make_harness()                       # MAX_DAILY_LOSS_PCT default 3%
    risk = h.risk()
    assert risk.circuit_breaker_tripped(230.0) is False       # sets day start = 230
    risk._record_realized_pnl(-5.0)                            # -2.2%
    assert risk.circuit_breaker_tripped(230.0) is False
    risk._record_realized_pnl(-2.0)                            # -3.04% realized
    assert risk.circuit_breaker_tripped(231.0) is True         # ...even though equity is up
    assert risk.circuit_breaker_tripped(222.0) is True         # equity-based reading still works

    risk._save_state()
    assert h.risk().circuit_breaker_tripped(231.0) is True     # same day: still tripped after a reload


def test_pre_deploy_runner_keeps_trailing(h):
    ex = h.ex
    ex.positions[(SOL, "LONG")] = {"qty": 0.15, "entry": ENTRY}
    ex.price[SOL] = 122.0
    ex._new_order(SOL, "STOP_MARKET", "SELL", "LONG", 0.74, stop_price=STOP_TICK)   # leftovers of the
    ex._new_order(SOL, "STOP_MARKET", "SELL", "LONG", 0.37, stop_price=ENTRY)       # old never-cancel bug
    ex._new_order(SOL, "STOP_MARKET", "SELL", "LONG", 0.15, stop_price=121.3)
    state = {
        "open_trades": [
            {"trade_id": "r1", "entry_time_utc": "2026-10-04T20:00:09+00:00", "mode": "LIVE",
             "symbol": SOL, "side": "BUY", "position_side": "LONG", "qty": 0.15, "entry_price": ENTRY,
             "stop_loss": 121.3, "take_profit": 121.75, "trigger": "t", "tp2_price": TP2,
             "tp2_source": "s", "initial_qty": 0.74, "risk_per_unit": 0.38173,
             "partial_tp_price": TP1, "partial_taken": True, "tp2_taken": True, "trail_distance": 0.38173},
        ],
    }
    (h.tmp / "state.json").write_text(json.dumps(state))
    h.run()
    assert h.trade()["stop_loss"] == pytest.approx(122.0 - 0.38173)
    assert h.exit_orders_open() == [("STOP_MARKET", round(122.0 - 0.38173, 3), 0.15)]


def test_tp1_and_tp2_both_fill_between_two_runs(h):
    h.open_entry()
    h.run()
    h.ex.set_price(SOL, 121.40)
    h.ex.set_price(SOL, 121.60)
    h.run()                                # one run books both and starts trailing
    t = h.trade()
    assert t["partial_taken"] and t["tp2_taken"] and t["qty"] == pytest.approx(0.15)
    assert h.exit_orders_open() == [("STOP_MARKET", round(121.60 - 0.38173, 3), 0.15)]
    rows = [r for r in h.journal() if r["event"] == "PARTIAL_EXIT"]
    assert [float(r["price"]) for r in rows] == [121.40, 121.60]
    h.ex.set_price(SOL, 121.0)
    h.run()
    realized_matches_account(h)


def test_tp_fractions_covering_the_whole_position(make_harness):
    h = make_harness(PARTIAL_TP_FRACTION="0.5", TP2_FRACTION="0.5")
    h.open_entry()
    h.run()
    assert h.exit_orders_open() == [
        ("STOP_MARKET", STOP_TICK, 0.37), ("STOP_MARKET", STOP_TICK, 0.37),
        ("TAKE_PROFIT_MARKET", TP1_TICK, 0.37), ("TAKE_PROFIT_MARKET", TP2, 0.37),
    ]
    h.ex.set_price(SOL, 121.40)
    h.ex.set_price(SOL, 121.60)
    assert h.ex.position_qty(SOL) == 0
    h.run()
    assert h.trade() is None and h.exit_orders_open() == []
    rows = [r["event"] for r in h.journal() if r["event"] in ("PARTIAL_EXIT", "EXIT")]
    assert rows == ["PARTIAL_EXIT", "PARTIAL_EXIT"]          # no zero-size EXIT row
    realized_matches_account(h)


def test_tp2_disabled_trails_right_after_tp1(make_harness):
    h = make_harness(TP2_FRACTION="0")
    h.open_entry()
    h.run()
    assert h.exit_orders_open() == [
        ("STOP_MARKET", STOP_TICK, 0.37), ("STOP_MARKET", STOP_TICK, 0.37),
        ("TAKE_PROFIT_MARKET", TP1_TICK, 0.37),
    ]
    h.ex.set_price(SOL, 121.40)
    h.run()
    t = h.trade()
    assert t["partial_taken"] and t["tp2_taken"] and t["trail_distance"] == pytest.approx(0.38173)
    assert h.exit_orders_open() == [("STOP_MARKET", round(121.40 - 0.38173, 3), 0.37)]


def test_no_candles_still_arms_and_books_resting_fills(h):
    h.open_entry()
    original = h.ex.fetch_ohlcv

    def no_candles(symbol, timeframe="15m", since=None, limit=None, params=None):
        if symbol == SOL and timeframe == "15m":
            raise ccxt.NetworkError("klines timed out")
        return original(symbol, timeframe, since, limit, params)

    h.ex.fetch_ohlcv = no_candles
    h.run()
    assert h.trade()["exit_orders"]["mode"] == "resting"
    h.ex.set_price(SOL, 121.40)
    h.run()
    assert h.trade()["partial_taken"]


def test_dry_run_journal_gets_the_fee_column(make_harness):
    h = make_harness(DRY_RUN="true")
    risk = h.risk()
    risk.track_dry_run_trade({
        "trade_id": "d1", "entry_time_utc": "2026-10-04T20:00:09+00:00", "mode": "DRY_RUN",
        "symbol": SOL, "side": "BUY", "position_side": "LONG", "qty": 0.74, "entry_price": ENTRY,
        "stop_loss": STOP, "take_profit": 121.75, "trigger": "t", "tp2_price": TP2, "tp2_source": "s",
    })
    risk.reconcile_dry_run_exits(SOL, high=121.40, low=121.0)
    risk.reconcile_dry_run_exits(SOL, high=121.0, low=120.9)
    rows = h.journal(mode="DRY_RUN")
    assert [r["event"] for r in rows] == ["PARTIAL_EXIT", "EXIT"]
    assert float(rows[0]["fee"]) == pytest.approx(0.37 * (ENTRY + TP1) * 0.0005, abs=1e-6)
    assert float(rows[1]["fee"]) > 0


def test_position_too_small_to_split_three_ways(h):
    """0.0002 BTC with a 0.0001 step: TP1 = 0.0001, TP2 (0.00006) is below the
    step, so the plan is TP1 + runner and TP2 is simply skipped."""
    h.ex.price[BTC] = 85500.0
    h.open_entry(symbol=BTC, entry=85104.8, stop=84909.1, qty=0.0002, tp2=85481.7)
    h.run()
    assert h.exit_orders_open(BTC) == [
        ("STOP_MARKET", 84909.1, 0.0001), ("STOP_MARKET", 84909.1, 0.0001),
        ("TAKE_PROFIT_MARKET", 85300.5, 0.0001),
    ]
    h.ex.set_price(BTC, 85310.0)
    h.run()
    t = h.trade(BTC)
    assert t["partial_taken"] and t["qty"] == pytest.approx(0.0001)
    h.ex.set_price(BTC, 85500.0)           # beyond TP2: nothing to sell there, trailing starts
    h.run()
    t = h.trade(BTC)
    assert t["tp2_taken"] and t["qty"] == pytest.approx(0.0001)
    assert h.exit_orders_open(BTC) == [("STOP_MARKET", round(85500.0 - (85104.8 - 84909.1), 1), 0.0001)]
    h.ex.set_price(BTC, 85200.0)
    h.run()
    assert h.trade(BTC) is None and h.exit_orders_open(BTC) == []
    realized_matches_account(h)


def test_one_failing_trade_does_not_stop_the_others(h, monkeypatch):
    h.ex.price[BTC] = 85500.0
    h.open_entry(symbol=BTC, entry=85104.8, stop=84909.1, qty=0.001, tp2=85481.7)
    h.open_entry()
    import bot as bot_module
    real = bot_module.RiskManager.arm_live_exit_orders

    def arm(self, symbol, exchange_client):
        if symbol == BTC:
            raise RuntimeError("unexpected")
        return real(self, symbol, exchange_client)

    monkeypatch.setattr(bot_module.RiskManager, "arm_live_exit_orders", arm)
    h.run()                                                    # must not raise
    assert h.trade(SOL)["exit_orders"]["mode"] == "resting"   # SOL (sorted after BTC) still managed
    assert "exit_orders" not in h.trade(BTC)

    monkeypatch.setattr(bot_module.RiskManager, "arm_live_exit_orders", real)
    h.run()
    assert h.trade(BTC)["exit_orders"]["mode"] == "resting"   # picked up on the next run


def test_settling_one_closed_trade_failing_does_not_lose_the_other(h, monkeypatch):
    h.ex.price[BTC] = 85500.0
    h.open_entry(symbol=BTC, entry=85104.8, stop=84909.1, qty=0.001, tp2=85481.7)
    h.open_entry()
    h.run()
    h.ex.set_price(BTC, 84900.0)
    h.ex.set_price(SOL, 120.55)
    from risk_manager import RiskManager
    real = RiskManager._settle_closed_trade

    def settle(self, trade, exchange_client):
        if trade["symbol"] == BTC:
            raise RuntimeError("unexpected")
        return real(self, trade, exchange_client)

    monkeypatch.setattr(RiskManager, "_settle_closed_trade", settle)
    h.run()
    assert h.trade(BTC) is None and h.trade(SOL) is None
    assert [r["symbol"] for r in h.journal() if r["event"] == "EXIT"] == [SOL]
    assert h.exit_orders_open(SOL) == []
