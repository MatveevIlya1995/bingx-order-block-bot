"""
Thin wrapper around ccxt's BingX integration for USDT-margined perpetual
futures (swap markets). ccxt is BingX's own recommended/integrated client
library, so this avoids hand-rolling HMAC request signing for a bot that
will be moving real money.

Requires: pip install ccxt

Every method that touches the market takes `symbol` as a parameter (rather
than reading a single `cfg.symbol`) so the bot can watch a whole list of
symbols in one run -- see `cfg.symbols` in config.py.

IMPORTANT: BingX supports both "one-way" and "hedge" position modes, which
changes whether `positionSide` must be "BOTH" or "LONG"/"SHORT", and some
order-param details can shift as BingX/ccxt update their APIs. Before
running this live, place one small test trade manually (or via a Python
REPL) and confirm `create_order` behaves as expected for your account's
position mode -- see the README for a step-by-step check.
"""
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

    def fetch_ohlcv(self, symbol: str, timeframe: str = None, limit: int = None):
        """Fetches OHLCV candles. Defaults to the strategy timeframe/limit
        from config, but callers (e.g. the higher-timeframe trend filter)
        can override both to pull a different timeframe."""
        return self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe or self.cfg.timeframe,
            limit=limit or self.cfg.candle_limit,
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
        """Open positions across every symbol the bot is configured to watch
        (cfg.symbols) -- used both to count how many are open and to skip
        re-signaling on a symbol that already has one."""
        positions = self.exchange.fetch_positions(self.cfg.symbols)
        return [p for p in positions if p.get("contracts") and float(p["contracts"]) > 0]

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self.exchange.set_leverage(leverage, symbol)
        except Exception as e:
            print(f"[exchange] {symbol}: could not set leverage: {e}")

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return float(self.exchange.amount_to_precision(symbol, amount))

    def create_market_order_with_stops(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_loss_price: float,
        take_profit_price: float,
        position_side: str,
    ):
        """Submits a market entry, then a reduce-only stop-loss and
        take-profit. Every live entry MUST go through this method (or an
        equivalent that always attaches a stop) -- never place a bare
        entry with no protective stop on a leveraged position."""
        amount = self.amount_to_precision(symbol, amount)
        entry_params = {"positionSide": position_side}
        order = self.exchange.create_order(symbol, "market", side, amount, None, entry_params)

        opposite = "sell" if side == "buy" else "buy"
        sl_params = {"positionSide": position_side, "stopPrice": stop_loss_price, "reduceOnly": True}
        tp_params = {"positionSide": position_side, "stopPrice": take_profit_price, "reduceOnly": True}

        self.exchange.create_order(symbol, "STOP_MARKET", opposite, amount, None, sl_params)
        self.exchange.create_order(symbol, "TAKE_PROFIT_MARKET", opposite, amount, None, tp_params)
        return order
