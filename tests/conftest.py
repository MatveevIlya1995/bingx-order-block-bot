import datetime
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bot  # noqa: E402
import config  # noqa: E402
from exchange_client import ExchangeClient  # noqa: E402
from fake_bingx import FakeBingX  # noqa: E402
from risk_manager import RiskManager  # noqa: E402

BOT_ENV = [
    "SYMBOLS", "DRY_RUN", "RISK_PCT_PER_TRADE", "LEVERAGE", "MAX_CONCURRENT_POSITIONS",
    "RESTING_TP_ORDERS", "MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT",
    "STATE_PATH", "TRADES_CSV_PATH", "BINGX_API_KEY", "BINGX_API_SECRET",
    "PARTIAL_TP_FRACTION", "TP2_FRACTION", "MAX_DAILY_LOSS_PCT",
]


class Harness:
    """One fake BingX account + the real bot code on top of it."""

    def __init__(self, tmp_path, monkeypatch, oversize="reject", **env):
        self.tmp = tmp_path
        self.mp = monkeypatch
        for key in BOT_ENV:
            monkeypatch.delenv(key, raising=False)
        base = {
            "SYMBOLS": "BTC/USDT:USDT,SOL/USDT:USDT",
            "DRY_RUN": "false",
            "RISK_PCT_PER_TRADE": "2",
            "LEVERAGE": "10",
            "MAX_CONCURRENT_POSITIONS": "5",
            "STATE_PATH": str(tmp_path / "state.json"),
            "TRADES_CSV_PATH": str(tmp_path / "trades.csv"),
            "BINGX_API_KEY": "test-key",
            "BINGX_API_SECRET": "test-secret",
        }
        base.update(env)
        for k, v in base.items():
            monkeypatch.setenv(k, v)
        self.ex = FakeBingX(oversize=oversize)
        self.ex.price["BTC/USDT:USDT"] = 85500.0
        self.ex.price["SOL/USDT:USDT"] = 121.5
        self.cfg = config.load_config()
        self.client = ExchangeClient.__new__(ExchangeClient)
        self.client.cfg = self.cfg
        self.client.exchange = self.ex
        monkeypatch.setattr(bot, "ExchangeClient", lambda cfg: self._client_for(cfg))

    def _client_for(self, cfg):
        self.client.cfg = cfg
        return self.client

    def risk(self):
        return RiskManager(config.load_config())

    def run(self):
        bot.run_once()
        return self.risk()

    def open_entry(self, symbol="SOL/USDT:USDT", entry=120.991, stop=120.60927, qty=0.74,
                   tp2=121.583, side="buy"):
        """Rests a LIVE limit entry exactly like bot._sync_symbol_orders does,
        tracks it in state, then lets the market trade through it."""
        ps = "LONG" if side == "buy" else "SHORT"
        direction = 1 if side == "buy" else -1
        risk = self.risk()
        placed = self.client.create_limit_order_with_stop(
            symbol=symbol, side=side, amount=qty, price=entry, stop_loss_price=stop, position_side=ps,
        )
        ob = {"type": "bullish" if side == "buy" else "bearish",
              "top": max(entry, stop), "bottom": min(entry, stop)}
        risk.track_pending_order({
            "order_id": str(placed["id"]), "trade_id": risk.new_trade_id(), "mode": "LIVE",
            "symbol": symbol, "side": side, "position_side": ps, "qty": qty, "leverage": 10,
            "entry_price": entry, "stop_loss": stop,
            "take_profit": entry + direction * abs(entry - stop) * 2,
            "tp2_price": tp2, "tp2_source": "next swing (test)", "equity": 230.0,
            "zone_key": risk._zone_key(symbol, ob), "ob": ob,
            "placed_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "trigger": "test order block limit",
        })
        self.ex.set_price(symbol, entry)   # fills the limit, BingX attaches the stop
        assert self.ex.position_qty(symbol, ps) == pytest.approx(qty)
        return str(placed["id"])

    def trade(self, symbol="SOL/USDT:USDT"):
        for t in self.risk().state["open_trades"]:
            if t["symbol"] == symbol:
                return t
        return None

    def journal(self, mode="LIVE"):
        import csv
        path = self.tmp / "trades.csv"
        if not path.exists():
            return []
        with open(path, newline="") as f:
            return [r for r in csv.DictReader(f) if r["mode"] == mode]

    def exit_orders_open(self, symbol="SOL/USDT:USDT"):
        return sorted(
            (o["type"], float(o["stopPrice"]), float(o["origQty"]))
            for o in self.ex.open_orders(symbol) if o["type"] != "LIMIT"
        )


@pytest.fixture(params=["reject", "clamp"])
def h(request, tmp_path, monkeypatch):
    """Every scenario runs twice: once with a fake exchange that REJECTS a
    triggered close bigger than the position, once with one that clamps it."""
    return Harness(tmp_path, monkeypatch, oversize=request.param)


@pytest.fixture
def make_harness(tmp_path, monkeypatch):
    def _make(**kw):
        return Harness(tmp_path, monkeypatch, **kw)
    return _make
