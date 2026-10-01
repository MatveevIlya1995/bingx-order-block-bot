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
        """Total USDT account equity -- NOT the free/available balance.
        Resting limit orders and open positions reserve margin, which
        shrinks "free" without any money being lost; using it made the
        daily-loss circuit breaker trip on the bot's own orders and
        undersized every trade. Falls back to free only if BingX/ccxt
        doesn't report a total."""
        bal = self.exchange.fetch_balance()
        usdt = bal.get("USDT", {}) or {}
        total = usdt.get("total")
        if total is None:
            total = (bal.get("total", {}) or {}).get("USDT")
        if total:
            return float(total)
        free = usdt.get("free")
        return float(free) if free else 0.0

    def fetch_open_positions(self):
        """Open positions across every symbol the bot is configured to watch
        (cfg.symbols) -- used both to count how many are open and to skip
        re-signaling on a symbol that already has one."""
        positions = self.exchange.fetch_positions(self.cfg.symbols)
        return [p for p in positions if p.get("contracts") and float(p["contracts"]) > 0]

    def set_leverage(self, symbol: str, leverage: int, position_side: str = None):
        """ccxt's BingX integration requires the position side for leverage
        (LONG/SHORT in hedge mode, BOTH in one-way mode) and rejects the
        call without it -- so leverage was never actually being set and
        positions ran at whatever leverage the symbol already had on BingX."""
        side = (position_side or "BOTH").upper()
        try:
            self.exchange.set_leverage(leverage, symbol, {"side": side})
        except Exception as e:
            print(f"[exchange] {symbol}: could not set {side} leverage to {leverage}x: {e}")

    @staticmethod
    def _close_params(position_side: str) -> dict:
        """Order params for anything that reduces/closes a position.
        BingX hedge mode (positionSide LONG/SHORT) rejects the reduceOnly
        field -- positionSide alone marks the order as a close there.
        One-way mode (positionSide BOTH) needs reduceOnly instead."""
        ps = (position_side or "BOTH").upper()
        if ps in ("LONG", "SHORT"):
            return {"positionSide": ps}
        return {"positionSide": "BOTH", "reduceOnly": True}

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
        sl_params = {**self._close_params(position_side), "stopPrice": stop_loss_price}
        tp_params = {**self._close_params(position_side), "stopPrice": take_profit_price}

        try:
            self.exchange.create_order(symbol, "STOP_MARKET", opposite, amount, None, sl_params)
        except Exception as e:
            # Never leave a position without a stop: flatten it right away.
            print(f"[exchange] {symbol}: STOP-LOSS placement FAILED ({e}) -- closing the new position at market")
            try:
                self.exchange.create_order(
                    symbol, "market", opposite, amount, None, self._close_params(position_side)
                )
            except Exception as e2:
                print(f"[exchange] {symbol}: EMERGENCY CLOSE ALSO FAILED ({e2}) -- CLOSE THIS POSITION MANUALLY ON BINGX NOW")
            raise
        if take_profit_price is not None:
            try:
                self.exchange.create_order(symbol, "TAKE_PROFIT_MARKET", opposite, amount, None, tp_params)
            except Exception as e:
                print(f"[exchange] {symbol}: take-profit order failed ({e}); stop-loss is in place, trade continues")
        return order

    def close_partial_position(self, symbol: str, side: str, amount: float, position_side: str):
        """Reduce-only market order closing PART of an open position --
        used for the partial take-profit (see risk_manager.py). `side` is
        the CLOSING side (opposite of the original entry side), same
        convention as the SL/TP orders in create_market_order_with_stops().

        UNVERIFIED against live BingX data, like the rest of this file --
        confirm the resulting position size matches expectations the
        first time this fires on a real account (see the README)."""
        amount = self.amount_to_precision(symbol, amount)
        params = self._close_params(position_side)
        return self.exchange.create_order(symbol, "market", side, amount, None, params)

    def replace_stop_order(
        self, symbol: str, side: str, amount: float, new_stop_price: float, position_side: str,
    ):
        """Moves a position's stop-loss: places the NEW stop first, then
        cancels the old stop order(s) for that position side (the
        take-profit is left alone). Placing first means the position is
        never without a stop -- if the new order fails, the old one stays."""
        amount = self.amount_to_precision(symbol, amount)
        try:
            old_orders = self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            print(f"[exchange] {symbol}: could not fetch open orders to replace stop: {e}")
            old_orders = []
        params = {**self._close_params(position_side), "stopPrice": new_stop_price}
        new_order = self.exchange.create_order(symbol, "STOP_MARKET", side, amount, None, params)
        new_id = (new_order or {}).get("id")
        for o in old_orders:
            if new_id and o.get("id") == new_id:
                continue
            o_type = (o.get("type") or "").upper()
            o_info = o.get("info", {}) or {}
            o_side = str(o_info.get("positionSide", "")).upper()
            if "STOP" in o_type and "TAKE_PROFIT" not in o_type and (
                not position_side or o_side == position_side.upper()
            ):
                try:
                    self.exchange.cancel_order(o["id"], symbol)
                except Exception as e:
                    print(f"[exchange] {symbol}: could not cancel old stop order {o.get('id')}: {e}")
        return new_order

    def price_to_precision(self, symbol: str, price: float) -> float:
        return float(self.exchange.price_to_precision(symbol, price))

    def create_limit_order_with_stop(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        stop_loss_price: float,
        position_side: str,
    ):
        """Places a resting LIMIT entry with its stop-loss ATTACHED to the
        order, so BingX itself creates the stop the moment the order fills
        -- the fill can happen at any time between two bot runs, and the
        position must never sit there without a stop. Every limit entry
        MUST go through this method.

        No take-profit is attached: TP1/TP2/trailing are managed by
        RiskManager once the bot sees the fill (see bot.py).

        If price is already past the limit price when this is placed (e.g.
        it's sitting inside the order block right now), BingX fills it
        straight away at the best available price -- which is then inside
        the zone, since the limit sits at the zone's near edge.

        UNVERIFIED against live BingX data, like the rest of this file:
        confirm on the first real fill that BingX shows the attached stop
        on the position."""
        amount = self.amount_to_precision(symbol, amount)
        price = self.price_to_precision(symbol, price)
        stop = self.price_to_precision(symbol, stop_loss_price)
        params = {
            "positionSide": position_side,
            "stopLoss": {
                "triggerPrice": stop,
                "type": "STOP_MARKET",
                "workingType": "MARK_PRICE",
            },
        }
        return self.exchange.create_order(symbol, "limit", side, amount, price, params)

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        return self.exchange.fetch_order(order_id, symbol)

    def cancel_order(self, order_id: str, symbol: str):
        return self.exchange.cancel_order(order_id, symbol)

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
