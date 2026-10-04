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

    @staticmethod
    def _usdt_equity_from_info(info):
        """Finds the USDT account's 'equity' field anywhere in BingX's raw
        balance payload (its shape differs between API versions: a dict
        under data.balance, or a list of per-asset dicts)."""
        stack = [info]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if "equity" in node and str(node.get("asset", "USDT")).upper() == "USDT":
                    try:
                        return float(node["equity"])
                    except (TypeError, ValueError):
                        pass
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return None

    def fetch_balance_usdt(self) -> float:
        """Total USDT account equity (wallet balance + unrealized P/L), read
        from BingX's own 'equity' field.

        NOT the free/available balance, and not ccxt's computed 'total'
        either: both shrink when margin is frozen by resting limit orders
        (ccxt's total = available + position margin, which leaves frozen
        order margin out). Using them made the bot read its own resting
        orders as an ~80% loss, trip the daily-loss breaker and cancel
        everything every other run."""
        bal = self.exchange.fetch_balance()
        equity = self._usdt_equity_from_info(bal.get("info"))
        if equity is not None and equity > 0:
            return equity
        print("[exchange] WARNING: no 'equity' field in BingX balance response -- falling back to ccxt total/free.")
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
        the fallback for a take-profit that has no resting order on the
        exchange (see risk_manager.reconcile_live_exits). `side` is
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
            if new_id and str(o.get("id")) == str(new_id):
                continue
            # NOTE: this used to test ccxt's unified o["type"], but ccxt
            # reports BingX STOP_MARKET orders as plain "market", so no old
            # stop ever matched and superseded stops were left on the
            # exchange. _exit_order_kind() reads BingX's raw type instead.
            if self._exit_order_kind(o) == "stop" and self._is_close_order_for(o, side, position_side):
                try:
                    self.exchange.cancel_order(o["id"], symbol)
                except Exception as e:
                    print(f"[exchange] {symbol}: could not cancel old stop order {o.get('id')}: {e}")
        return new_order

    # ------------------------------------------------------------------
    # exit orders resting on the exchange: tranche stops + take-profits
    # ------------------------------------------------------------------

    _STOP_TYPES = ("STOP_MARKET", "STOP")
    _TAKE_PROFIT_TYPES = ("TAKE_PROFIT_MARKET", "TAKE_PROFIT")

    @classmethod
    def _exit_order_kind(cls, order: dict):
        """"stop", "tp" or None for an order as returned by ccxt.

        Decided ONLY from BingX's raw order type (order["info"]["type"]).
        ccxt's unified order["type"] can't be used: it reports STOP_MARKET
        and TAKE_PROFIT_MARKET alike as "market". Its stopLossPrice /
        takeProfitPrice can't be used either: a resting LIMIT *entry* with
        an attached stop-loss also carries a stopLossPrice, and must never
        be mistaken for a stop order. Anything that is not plainly a
        stop / take-profit order type returns None and is left alone."""
        info = order.get("info") or {}
        raw = str(info.get("type") or info.get("orderType") or "").upper()
        if raw in cls._TAKE_PROFIT_TYPES:
            return "tp"
        if raw in cls._STOP_TYPES:
            return "stop"
        return None

    @staticmethod
    def _is_close_order_for(order: dict, close_side: str, position_side: str) -> bool:
        """True if `order` sits on the closing side of `position_side`.
        A field BingX didn't send back is not held against the order; a
        field that is present and different rules it out."""
        info = order.get("info") or {}
        o_side = str(order.get("side") or info.get("side") or "").lower()
        if close_side and o_side and o_side != close_side.lower():
            return False
        o_ps = str(info.get("positionSide") or "").upper()
        ps = (position_side or "").upper()
        if ps in ("LONG", "SHORT") and o_ps and o_ps != ps:
            return False
        return True

    @staticmethod
    def _order_id(order) -> str:
        oid = (order or {}).get("id")
        if oid is None or str(oid) in ("", "None"):
            raise RuntimeError("BingX accepted the order but returned no order id")
        return str(oid)

    def place_stop_orders(
        self, symbol: str, side: str, amounts: list, stop_price: float, position_side: str,
    ) -> list:
        """Places one reduce-only STOP_MARKET per entry in `amounts`, all at
        `stop_price`, and returns their order ids. Same order shape as
        replace_stop_order() (the one stop call verified on the live
        account), just several of them.

        All-or-nothing: if any of them fails, the ones already placed by
        this call are cancelled again and the error is re-raised, so the
        caller's previous stops stay the only stops."""
        placed = []
        try:
            for amount in amounts:
                amount = self.amount_to_precision(symbol, round(amount, 8))
                if amount <= 0:
                    continue
                params = {**self._close_params(position_side), "stopPrice": stop_price}
                order = self.exchange.create_order(symbol, "STOP_MARKET", side, amount, None, params)
                placed.append(self._order_id(order))
        except Exception:
            self.cancel_orders(symbol, placed)
            raise
        if not placed:
            raise RuntimeError("no stop order was placed (every amount rounded to zero)")
        return placed

    def place_take_profit_order(
        self, symbol: str, side: str, amount: float, trigger_price: float, position_side: str,
    ) -> str:
        """Places a resting reduce-only TAKE_PROFIT_MARKET order: BingX
        itself closes `amount` at market the moment price reaches
        `trigger_price`, whether or not the bot happens to be running.
        Returns the order id.

        A trigger order, not a limit order, on purpose: a resting limit
        close reserves that part of the position on BingX, and the
        stop-loss could then be refused for "not enough position" when it
        triggers. Trigger orders reserve nothing until they fire.

        `side` is the CLOSING side, as for the stop orders. Raises if BingX
        rejects it (e.g. price is already past the trigger) -- the caller
        then falls back to the market close on the scheduled run."""
        amount = self.amount_to_precision(symbol, round(amount, 8))
        if amount <= 0:
            raise RuntimeError("take-profit amount rounds to zero")
        params = {**self._close_params(position_side), "stopPrice": trigger_price}
        order = self.exchange.create_order(symbol, "TAKE_PROFIT_MARKET", side, amount, None, params)
        return self._order_id(order)

    def cancel_orders(self, symbol: str, order_ids: list) -> list:
        """Best-effort cancel of several orders. Returns the ids that could
        NOT be cancelled (normally because they already triggered, filled
        or were cancelled -- nothing to do about those)."""
        failed = []
        for oid in order_ids or []:
            if not oid:
                continue
            try:
                self.exchange.cancel_order(oid, symbol)
            except Exception as e:
                failed.append(oid)
                print(f"[exchange] {symbol}: could not cancel order {oid}: {e}")
        return failed

    def open_exit_orders(self, symbol: str, position_side: str, close_side: str) -> list:
        """Stop / take-profit orders currently open on the closing side of
        one position: [{"id", "kind": "stop"|"tp", "price", "amount"}].
        Raises if the open orders can't be fetched."""
        out = []
        for o in self.exchange.fetch_open_orders(symbol):
            kind = self._exit_order_kind(o)
            if kind is None or not self._is_close_order_for(o, close_side, position_side):
                continue
            info = o.get("info") or {}
            price = None
            for candidate in (info.get("stopPrice"), o.get("stopLossPrice"),
                              o.get("takeProfitPrice"), o.get("triggerPrice")):
                try:
                    if candidate not in (None, "") and float(candidate) > 0:
                        price = float(candidate)
                        break
                except (TypeError, ValueError):
                    continue
            try:
                amount = float(o.get("amount")) if o.get("amount") is not None else None
            except (TypeError, ValueError):
                amount = None
            out.append({"id": str(o.get("id")), "kind": kind, "price": price, "amount": amount})
        return out

    def cancel_exit_orders_for_side(self, symbol: str, position_side: str, close_side: str) -> int:
        """Cancels every stop / take-profit order still open on the closing
        side of a position that no longer exists, so nothing left over from
        a finished trade can fire against the next position on that symbol.
        Returns how many were cancelled. Only call this when the position
        is gone AND the bot has no resting entry order on the symbol."""
        try:
            leftovers = self.open_exit_orders(symbol, position_side, close_side)
        except Exception as e:
            print(f"[exchange] {symbol}: could not list leftover exit orders: {e}")
            return 0
        ids = [o["id"] for o in leftovers]
        failed = self.cancel_orders(symbol, ids)
        return len(ids) - len(failed)

    def order_fill(self, order_id: str, symbol: str) -> dict:
        """What actually happened to one order:

            {"status": "open" | "filled" | "dead",
             "filled": qty filled, "price": average fill price or None,
             "fee": fee paid in USDT (positive) or None, "timestamp": datetime or None}

        "dead" = cancelled / expired / rejected / failed with nothing
        filled. price and fee come from BingX's own order record (avgPrice,
        commission); None means BingX didn't report them (a commission of
        exactly 0 on a filled order is read as "not reported"). Raises if the
        order can't be fetched or its state can't be recognised -- callers
        must treat that as "unknown", never as "not filled"."""
        o = self.exchange.fetch_order(order_id, symbol)
        status = str(o.get("status") or "").lower()
        try:
            filled = float(o.get("filled") or 0.0)
        except (TypeError, ValueError):
            filled = 0.0
        price = None
        try:
            if o.get("average") is not None and float(o["average"]) > 0:
                price = float(o["average"])
        except (TypeError, ValueError):
            price = None
        fee = None
        fee_info = o.get("fee") or {}
        try:
            if fee_info.get("cost") is not None and float(fee_info["cost"]) != 0:
                fee = abs(float(fee_info["cost"]))
        except (TypeError, ValueError):
            fee = None
        ts = o.get("lastTradeTimestamp") or o.get("lastUpdateTimestamp") or o.get("timestamp")
        when = (
            datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc) if ts else None
        )
        if status == "open":
            state = "open"
        elif status in ("canceled", "cancelled", "expired", "rejected"):
            state = "filled" if filled > 0 else "dead"
        elif status == "closed" and filled > 0:
            state = "filled"
        else:
            # Anything else (no status, or "closed" with nothing filled) is
            # not safe to read as either "still resting" or "gone".
            raise RuntimeError(f"unrecognised state for order {order_id}: status={status!r} filled={filled}")
        if state != "filled":
            price, fee = None, None
        return {"status": state, "filled": filled, "price": price, "fee": fee, "timestamp": when}

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
        float or None, "amount": float or None, "fee": float or None} for
        the most recent reduce-only fill on `symbol` since `since_iso`, or
        None if nothing could be found.

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

        fee = None
        try:
            cost = (last.get("fee") or {}).get("cost")
            if cost is not None and float(cost) != 0:
                fee = abs(float(cost))
        except (TypeError, ValueError):
            fee = None
        try:
            amount = float(last["amount"]) if last.get("amount") is not None else None
        except (TypeError, ValueError):
            amount = None

        return {
            "timestamp": datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc),
            "price": float(last["price"]) if last.get("price") is not None else None,
            "realized_pnl": realized_pnl,
            "amount": amount,
            "fee": fee,
        }
