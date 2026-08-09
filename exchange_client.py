"""
Thin wrapper around ccxt's BingX integration for USDT-margined perpetual
futures (swap markets). ccxt is BingX's own recommended/integrated client
library, so this avoids hand-rolling HMAC request signing for a bot that
will be moving real money.

Requires: pip install ccxt

IMPORTANT: BingX supports both "one-way" and "hedge" position modes, which
changes whether `positionSide` must be "BOTH" or "LONG"/"SHORT", and some
order-param details can shift as BingX/ccxt update their APIs. Before
running this live, place one small test trade manually (or via a Python
REPL) and confirm `create_order` behaves as expected for your account's
position mode -- see the README for a step-by-step check.
"""
import datetime
import ccxt


class ExchangeClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.exchange = ccxt.bingx(
            {
                "apiKey": cfg.api_key,
                "secret": cfg.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )

    def fetch_ohlcv(self):
        return self.exchange.fetch_ohlcv(
            self.cfg.symbol, timeframe=self.cfg.timeframe, limit=self.cfg.candle_limit
        )

    def fetch_balance_usdt(self) -> float:
        bal = self.exchange.fetch_balance()
        usdt = bal.get("USDT", {})
        free = usdt.get("free")
        if free:
            return float(free)
        total = bal.get("total", {}).get("USDT")
        return float(total) if total else 0.0

    def fetch_open_positions(self):
        positions = self.exchange.fetch_positions([self.cfg.symbol])
        return [p for p in positions if p.get("contracts") and float(p["contracts"]) > 0]

    def set_leverage(self, leverage: int):
        try:
            self.exchange.set_leverage(leverage, self.cfg.symbol)
        except Exception as e:
            print(f"[exchange] could not set leverage: {e}")

    def amount_to_precision(self, amount: float) -> float:
        return float(self.exchange.amount_to_precision(self.cfg.symbol, amount))

    def create_market_order_with_stops(
        self, side: str, amount: float, stop_loss_price: float, take_profit_price: float, position_side: str
    ):
        """Submits a market entry, then a reduce-only stop-loss and
        take-profit. Every live entry MUST go through this method (or an
        equivalent that always attaches a stop) -- never place a bare
        entry with no protective stop on a leveraged position."""
        amount = self.amount_to_precision(amount)
        entry_params = {"positionSide": position_side}
        order = self.exchange.create_order(self.cfg.symbol, "market", side, amount, None, entry_params)

        opposite = "sell" if side == "buy" else "buy"
        sl_params = {"positionSide": position_side, "stopPrice": stop_loss_price, "reduceOnly": True}
        tp_params = {"positionSide": position_side, "stopPrice": take_profit_price, "reduceOnly": True}

        self.exchange.create_order(self.cfg.symbol, "STOP_MARKET", opposite, amount, None, sl_params)
        self.exchange.create_order(self.cfg.symbol, "TAKE_PROFIT_MARKET", opposite, amount, None, tp_params)
        return order

    def fetch_trade_exit(self, symbol: str, position_side: str, since_iso: str):
        """Looks up the fill(s) that closed a position opened at/after
        since_iso, so the trade journal can record exit price/time and
        realized P/L. Returns
        {"price": float, "timestamp": datetime, "realized_pnl": float|None}
        or None if no matching closing fill can be found (e.g. an API
        gap) -- callers should treat that as "closed, but confirm the
        exact fill on BingX directly."

        NOTE: like the rest of this bot's BingX wiring, the exact shape of
        a BingX fill via ccxt (whether `info` carries a realized-pnl
        field, and under what key) could not be verified against live
        BingX data from this environment -- confirm against a real closed
        trade before trusting the `realized_pnl` field blindly. The P/L
        fallback computed from entry vs. exit price is always returned as
        a backup and does not depend on that field, though it won't
        include exchange fees or funding.
        """
        since_ms = int(datetime.datetime.fromisoformat(since_iso).timestamp() * 1000)
        try:
            trades = self.exchange.fetch_my_trades(symbol, since=since_ms)
        except Exception as e:
            print(f"[exchange] could not fetch closing trades for {symbol}: {e}")
            return None

        if not trades:
            return None

        closing = [
            t for t in trades
            if t.get("reduceOnly") or (t.get("info", {}) or {}).get("reduceOnly")
        ]
        candidates = closing or trades
        last = max(candidates, key=lambda t: t.get("timestamp") or 0)
        info = last.get("info", {}) or {}

        realized_pnl = None
        for key in ("realizedPnl", "realisedPnl", "profit"):
            if info.get(key) not in (None, ""):
                try:
                    realized_pnl = float(info[key])
                except (TypeError, ValueError):
                    pass
                break

        timestamp = None
        if last.get("timestamp"):
            timestamp = datetime.datetime.fromtimestamp(
                last["timestamp"] / 1000, tz=datetime.timezone.utc
            )

        return {
            "price": float(last["price"]) if last.get("price") is not None else None,
            "timestamp": timestamp,
            "realized_pnl": realized_pnl,
        }
