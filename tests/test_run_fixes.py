"""Fixes from the 2026-11-09 review:
1. BingX 100410 ("retry later") on candle requests is retried, not skipped.
2. Every MAX_CONCURRENT_POSITIONS slot can be filled (equal share of the cap).
3. DRY_RUN replays every closed candle since the last check, never fills on
   a move made before the order existed, and fills only one order per symbol.
"""
import ccxt
import pandas as pd
import pytest

import bot
import config
import exchange_client
from exchange_client import ExchangeClient
from risk_manager import RiskManager

SOL = "SOL/USDT:USDT"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for k in ("MAX_CONCURRENT_POSITIONS", "MARGIN_PER_TRADE_USD", "LEVERAGE",
              "RISK_PCT_PER_TRADE", "MAX_MARGIN_UTILIZATION_PCT", "DRY_RUN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("TRADES_CSV_PATH", str(tmp_path / "trades.csv"))
    monkeypatch.setenv("MAX_CONCURRENT_POSITIONS", "5")
    monkeypatch.setenv("LEVERAGE", "10")
    monkeypatch.setenv("RISK_PCT_PER_TRADE", "2")
    return config.load_config()


# --- 1. rate limit -----------------------------------------------------------

class _Flaky:
    def __init__(self, failures, exc):
        self.failures, self.exc, self.calls = failures, exc, 0

    def fetch_ohlcv(self, symbol, timeframe=None, limit=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return [[0, 1, 1, 1, 1, 1]]


def _client(cfg, ex):
    c = ExchangeClient.__new__(ExchangeClient)
    c.cfg, c.exchange = cfg, ex
    return c


def test_100410_is_retried(cfg, monkeypatch):
    monkeypatch.setattr(exchange_client.time, "sleep", lambda s: None)
    ex = _Flaky(2, ccxt.ExchangeError('bingx {"code":100410,"msg":"retry later"}'))
    assert _client(cfg, ex).fetch_ohlcv(SOL) == [[0, 1, 1, 1, 1, 1]]
    assert ex.calls == 3


def test_retries_give_up_and_other_errors_are_not_retried(cfg, monkeypatch):
    monkeypatch.setattr(exchange_client.time, "sleep", lambda s: None)
    ex = _Flaky(99, ccxt.RateLimitExceeded("slow down"))
    with pytest.raises(ccxt.RateLimitExceeded):
        _client(cfg, ex).fetch_ohlcv(SOL)
    assert ex.calls == exchange_client.RATE_LIMIT_RETRIES + 1
    ex = _Flaky(1, ccxt.BadSymbol("nope"))
    with pytest.raises(ccxt.BadSymbol):
        _client(cfg, ex).fetch_ohlcv(SOL)
    assert ex.calls == 1


# --- 2. every slot can be used ------------------------------------------------

def test_fifth_slot_still_gets_a_real_size(cfg):
    risk = RiskManager(cfg)
    equity = 157.0
    cap = equity * cfg.max_leverage_hard_cap * cfg.max_margin_utilization_pct
    used = 0.0
    for _ in range(cfg.max_concurrent_positions):
        # tight stop: the risk target alone would ask for the 10x ceiling
        qty, _lev = risk.position_size(equity, 1.0, 0.997, existing_notional=used)
        assert qty * 1.0 >= cfg.min_notional_usdt
        used += qty
    assert used <= cap + 1e-9


# --- 3. DRY_RUN fills and exits --------------------------------------------------

def _df(rows, start="2026-11-09 10:00"):
    ts = pd.date_range(start, periods=len(rows), freq="5min")
    return pd.DataFrame(
        [{"ts": t, "open": o, "high": h, "low": l, "close": c, "volume": 1.0}
         for t, (o, h, l, c) in zip(ts, rows)]
    )


def _order(entry, placed="2026-11-09T10:02:00+00:00", side="SHORT"):
    return {"entry_price": entry, "position_side": side, "placed_time_utc": placed}


def test_no_fill_from_a_move_made_before_the_order_existed():
    # The 10:00 candle spiked to 105 (before the 10:02 order); nothing since.
    df = _df([(100, 105, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
    assert bot._dry_run_fill(_order(104.0), df) is None
    # A later candle trading through it does fill, at the limit.
    df = _df([(100, 105, 99, 100), (100, 104.5, 99, 100), (100, 101, 99, 100)])
    price, ms = bot._dry_run_fill(_order(104.0), df)
    assert price == 104.0 and ms == bot._ts_ms(df["ts"].iloc[1])


def _short_trade(cfg, risk, entry=100.0, stop=101.0, entry_time="2026-11-09T10:05:00+00:00"):
    trade = {
        "trade_id": "t1", "entry_time_utc": entry_time, "mode": "DRY_RUN", "symbol": SOL,
        "side": "SELL", "position_side": "SHORT", "qty": 1.0, "entry_price": entry,
        "stop_loss": stop, "take_profit": entry - 2, "trigger": "test",
        "tp2_price": entry - 2, "tp2_source": "test", "entry_fee": None,
        "exits_checked_through_ms": bot._ts_ms(pd.Timestamp(entry_time).tz_convert(None)) - 1,
    }
    risk.track_dry_run_trade(trade)


def test_stop_hit_in_a_middle_candle_is_not_missed(cfg):
    cfg.dry_run = True
    risk = RiskManager(cfg)
    _short_trade(cfg, risk)
    # candles: 10:05 quiet, 10:10 spikes through the 101 stop, 10:15 quiet,
    # 10:20 still forming. The old code only looked at 10:20.
    df = _df([(100, 100.5, 99.8, 100), (100, 101.3, 99.9, 100.2),
              (100.2, 100.4, 99.9, 100), (100, 100.2, 99.9, 100)], start="2026-11-09 10:05")
    closed = []
    for row in df.iloc[:-1].itertuples(index=False):
        closed += risk.reconcile_dry_run_exits(SOL, row.high, row.low, candle_ms=bot._ts_ms(row.ts))
    assert len(closed) == 1 and risk.state["open_dry_run_trades"] == []


def test_a_candle_is_never_replayed_twice(cfg):
    cfg.dry_run = True
    risk = RiskManager(cfg)
    _short_trade(cfg, risk)
    df = _df([(100, 100.5, 98.95, 99)], start="2026-11-09 10:05")   # TP1 (99) hit
    row = next(df.itertuples(index=False))
    risk.reconcile_dry_run_exits(SOL, row.high, row.low, candle_ms=bot._ts_ms(row.ts))
    t = risk.state["open_dry_run_trades"][0]
    assert t.get("partial_taken")
    # Same candle again (next run): its high (100.5) is above the new
    # breakeven stop (100) but must not stop the trade out a second time.
    risk.reconcile_dry_run_exits(SOL, row.high, row.low, candle_ms=bot._ts_ms(row.ts))
    assert len(risk.state["open_dry_run_trades"]) == 1


def test_only_one_order_per_symbol_fills(cfg, monkeypatch):
    cfg.dry_run = True
    risk = RiskManager(cfg)
    for oid, entry in (("a", 103.5), ("b", 103.0)):
        risk.state.setdefault("pending_orders", []).append({
            "order_id": oid, "trade_id": oid, "mode": "DRY_RUN", "symbol": SOL,
            "side": "sell", "position_side": "SHORT", "qty": 1.0, "entry_price": entry,
            "stop_loss": entry + 0.5, "take_profit": entry - 0.5, "tp2_price": entry - 1,
            "tp2_source": "test", "trigger": "test", "ob": {"type": "bearish", "bottom": entry, "top": entry + 0.2},
            "placed_time_utc": "2026-11-09T10:02:00+00:00",
        })
    # One candle runs up through both limits.
    df = _df([(100, 100.5, 99.5, 100), (100, 104, 99.8, 103.8), (103.8, 104, 103.5, 103.6)])
    monkeypatch.setattr(bot, "_fetch_df", lambda client, symbol, cache: df)
    monkeypatch.setattr(bot, "_cancel_pending", lambda client, r, o, why: r.remove_pending_order(o["order_id"]))
    bot._process_pending_fills(None, risk, cfg, {})
    open_trades = risk.state["open_dry_run_trades"]
    assert [t["trade_id"] for t in open_trades] == ["b"]   # 103.0 is reached first going up
    assert risk.state.get("pending_orders") == []
