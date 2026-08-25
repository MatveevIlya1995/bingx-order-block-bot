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

    def fetch_trade_exit(self, symbol: str, position_side: str, since_iso: str):
        """Looks up the fill that closed a previously-tracked position, so
        risk_manager.reconcile_closed_trades() can log an accurate EXIT
        row instead of falling back to an approximate "now" exit with
        unknown P/L.

        Returns {"timestamp": datetime, "price": float, "realized_pnl":
        float or None} for the most recent reduce-only fill on `symbol`
        since `since_iso`, or None if nothing could be found.

        NOTE: like the rest of this file, the exact `info` field names
        BingX returns for a fill's reduceOnly flag / positionSide /
        realized PnL could not be verified against live BingX data from
        the sandbox this bot was written in -- confirm these once a real
        trade has gone through (see the README's Step 3), and adjust the
        lookups below if BingX/ccxt name them differently.
        """
        since_ms = int(datetime.datetime.fromisoformat(since_iso).timestamp() * 1000)
        try:
            trades = self.exchange.fetch_my_trades(symbol, since=since_ms)
        except Exception as e:
            print(f"[exchange] {symbol}: could not fetch trade history for exit reconciliation: {e}")
            return None
        if not trades:
            return None

        def _is_reduce_only(t):
            info = t.get("info", {}) or {}
            return str(info.get("reduceOnly", "")).strip().lower() == "true"

        closing = [t for t in trades if _is_reduce_only(t)]
        if position_side:
            same_side = [
                t for t in closing
                if str((t.get("info", {}) or {}).get("positionSide", "")).upper() == position_side.upper()
            ]
            if same_side:
                closing = same_side
        if not closing:
            # Couldn't identify a reduce-only fill (e.g. unexpected info
            # field names) -- fall back to the most recent fill of any
            # kind rather than reporting nothing.
            closing = trades

        closing = sorted(closing, key=lambda t: t.get("timestamp") or 0)
        last = closing[-1]
        ts = last.get("timestamp")
        if ts is None:
            return None

        realized_pnl = None
        info = last.get("info", {}) or {}
        for key in ("realizedPnl", "realisedPnl", "profit"):
            if key in info:
                try:
                    realized_pnl = float(info[key])
                except (TypeError, ValueError):
                    realized_pnl = None
                break

        return {
            "timestamp": datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc),
            "price": float(last["price"]) if last.get("price") is not None else None,
            "realized_pnl": realized_pnl,
        }
