"""
A small in-memory stand-in for ccxt's BingX client, for tests.

It keeps positions and orders, moves a price, fills resting limit entries,
fires stop / take-profit trigger orders, charges fees, and answers the
handful of ccxt calls exchange_client.py makes. Orders are stored in
BingX's raw shape and returned through ccxt's real bingx.parse_order(), so
the code under test sees exactly the unified fields ccxt produces (for
example: a STOP_MARKET order comes back with type "market").

What BingX does when a triggered close order is bigger than the position
could not be checked from the environment this was written in, so the fake
supports both possibilities and the tests run against each:

    oversize="reject" -- the triggered order fails, nothing is closed
    oversize="clamp"  -- the order closes whatever is left of the position
"""
import itertools
import json

import ccxt

TAKER = 0.0005
MAKER = 0.0002

STEPS = {  # symbol -> (amount step, price tick)
    "BTC/USDT:USDT": (0.0001, 0.1),
    "SOL/USDT:USDT": (0.01, 0.001),
    "ETH/USDT:USDT": (0.01, 0.01),
}


def _market_id(symbol):
    return symbol.split(":")[0].replace("/", "-")


class OrderRejected(Exception):
    pass


class FakeBingX:
    def __init__(self, oversize="reject", equity=230.0):
        assert oversize in ("reject", "clamp")
        self.oversize = oversize
        self.trigger_order = "oldest"   # or "newest" / "largest": which trigger order fires first
        self.equity0 = equity
        self.realized = 0.0          # gross price P/L of everything closed
        self.fees = 0.0              # every fee charged
        self.price = {}              # symbol -> last price
        self.positions = {}          # (symbol, positionSide) -> {"qty", "entry"}
        self.orders = {}             # id -> raw BingX order dict (+ "_symbol")
        self.fills = []              # raw fills
        self.clock = 1_790_000_000_000
        self._ids = itertools.count(1000)
        self.fail_next = []          # [(order type, exception)] injected placement failures
        self.fail_fetch_order = set()  # order ids whose fetch_order raises
        self.fail_open_orders = False
        self.calls = []              # (method, args) log
        self._parser = ccxt.bingx({"options": {"defaultType": "swap"}})
        self._parser.markets = {}
        self._parser.markets_by_id = {}
        for symbol, (amount_step, tick) in STEPS.items():
            base = symbol.split("/")[0]
            m = {
                "id": _market_id(symbol), "symbol": symbol, "base": base, "quote": "USDT",
                "settle": "USDT", "spot": False, "swap": True, "contract": True,
                "inverse": False, "linear": True, "type": "swap",
                "precision": {"amount": amount_step, "price": tick}, "limits": {},
            }
            self._parser.markets[symbol] = m
            self._parser.markets_by_id[m["id"]] = [m]

    # ------------------------------------------------------------------ helpers
    def _tick(self):
        self.clock += 1000
        return self.clock

    def _unified(self, raw):
        clean = {k: v for k, v in raw.items() if not k.startswith("_")}
        return self._parser.parse_order(clean)

    def amount_to_precision(self, symbol, amount):
        step = STEPS[symbol][0]
        n = int(round(amount / step, 9) + 1e-9)  # truncate, like ccxt
        if n <= 0:
            raise ccxt.InvalidOrder(f"amount {amount} below the minimum step {step}")
        return f"{n * step:.8f}".rstrip("0").rstrip(".")

    def price_to_precision(self, symbol, price):
        tick = STEPS[symbol][1]
        return f"{round(round(price / tick) * tick, 8):.8f}".rstrip("0").rstrip(".")

    def _pos(self, symbol, ps):
        return self.positions.get((symbol, ps))

    def position_qty(self, symbol, ps="LONG"):
        p = self._pos(symbol, ps)
        return p["qty"] if p else 0.0

    def open_orders(self, symbol=None):
        return [o for o in self.orders.values()
                if o["status"] in ("NEW", "PENDING") and (symbol is None or o["_symbol"] == symbol)]

    def account_pnl(self):
        """What the account actually made: closed price P/L minus all fees."""
        return self.realized - self.fees

    # ------------------------------------------------------------------ matching
    def _fill_open(self, symbol, ps, qty, price, fee_rate, order):
        pos = self.positions.setdefault((symbol, ps), {"qty": 0.0, "entry": 0.0})
        total = pos["qty"] + qty
        pos["entry"] = (pos["entry"] * pos["qty"] + price * qty) / total
        pos["qty"] = round(total, 8)
        fee = qty * price * fee_rate
        self.fees += fee
        self._record_fill(symbol, order, qty, price, fee)

    def _fill_close(self, symbol, ps, qty, price, order):
        pos = self._pos(symbol, ps)
        direction = 1 if ps == "LONG" else -1
        self.realized += (price - pos["entry"]) * direction * qty
        fee = qty * price * TAKER
        self.fees += fee
        pos["qty"] = round(pos["qty"] - qty, 8)
        if pos["qty"] <= 0:
            del self.positions[(symbol, ps)]
        self._record_fill(symbol, order, qty, price, fee)

    def _record_fill(self, symbol, order, qty, price, fee):
        now = self._tick()
        order["executedQty"] = f"{qty:.8f}"
        order["avgPrice"] = f"{price:.8f}"
        order["commission"] = f"{-fee:.8f}"
        order["status"] = "FILLED"
        order["updateTime"] = now
        self.fills.append({
            "symbol": symbol, "timestamp": now, "price": price, "amount": qty,
            "side": order["side"].lower(), "fee": {"cost": fee, "currency": "USDT"},
            "order": str(order["orderId"]),
            "info": {"positionSide": order["positionSide"], "orderId": str(order["orderId"])},
        })

    def _try_close(self, symbol, order):
        """Execute a (triggered or plain) market close order."""
        ps = order["positionSide"]
        qty = float(order["origQty"])
        pos = self._pos(symbol, ps)
        have = pos["qty"] if pos else 0.0
        if have <= 0 or (qty > have + 1e-12 and self.oversize == "reject"):
            order["status"] = "FAILED"
            order["updateTime"] = self._tick()
            return False
        self._fill_close(symbol, ps, min(qty, have), self.price[symbol], order)
        return True

    def set_price(self, symbol, price):
        """Move the market to `price` and process everything that reacts."""
        self.price[symbol] = price
        # 1. resting limit entries
        for o in list(self.open_orders(symbol)):
            if o["type"] != "LIMIT":
                continue
            limit = float(o["price"])
            is_buy = o["side"] == "BUY"
            if (is_buy and price <= limit) or (not is_buy and price >= limit):
                qty = float(o["origQty"])
                self._fill_open(symbol, o["positionSide"], qty, limit, MAKER, o)
                attached = o.get("_stopLoss")
                if attached:
                    self._new_order(symbol, "STOP_MARKET", "SELL" if is_buy else "BUY",
                                    o["positionSide"], qty, stop_price=attached["stopPrice"])
        # 2. trigger orders, in the order self.trigger_order says
        ordered = sorted(self.open_orders(symbol), key=lambda x: x["orderId"],
                         reverse=(self.trigger_order == "newest"))
        if self.trigger_order == "largest":
            ordered = sorted(ordered, key=lambda x: -float(x["origQty"]))
        for o in ordered:
            if o["type"] not in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                continue
            trig = float(o["stopPrice"])
            closing_long = o["positionSide"] == "LONG"
            if o["type"] == "STOP_MARKET":
                hit = price <= trig if closing_long else price >= trig
            else:
                hit = price >= trig if closing_long else price <= trig
            if hit:
                self._try_close(symbol, o)

    def _new_order(self, symbol, type_, side, ps, qty, price=None, stop_price=None):
        oid = next(self._ids)
        now = self._tick()
        raw = {
            "symbol": _market_id(symbol), "orderId": oid, "side": side, "positionSide": ps,
            "type": type_, "origQty": f"{qty:.8f}", "price": f"{price or 0}",
            "executedQty": "0", "avgPrice": "0", "cumQuote": "0",
            "stopPrice": f"{stop_price}" if stop_price is not None else "",
            "profit": "0.0000", "commission": "0.000000", "status": "NEW",
            "time": now, "updateTime": now, "workingType": "MARK_PRICE",
            "_symbol": symbol,
        }
        self.orders[oid] = raw
        return raw

    # ------------------------------------------------------------------ ccxt API
    def create_order(self, symbol, type, side, amount, price=None, params=None):
        params = dict(params or {})
        self.calls.append(("create_order", symbol, type, side, amount, price, params))
        type_ = type.upper()
        for i, (t, exc) in enumerate(self.fail_next):
            if t == type_:
                del self.fail_next[i]
                raise exc
        side_ = side.upper()
        ps = params.get("positionSide", "BOTH")
        px = self.price[symbol]
        if type_ == "LIMIT":
            raw = self._new_order(symbol, "LIMIT", side_, ps, amount, price=price)
            sl = params.get("stopLoss")
            if sl:
                raw["_stopLoss"] = {"stopPrice": float(sl["triggerPrice"])}
                raw["stopLoss"] = json.dumps({"stopPrice": float(sl["triggerPrice"]),
                                              "type": "STOP_MARKET", "quantity": amount})
            self.set_price(symbol, px)
            return self._unified(raw)
        if type_ == "MARKET":
            raw = self._new_order(symbol, "MARKET", side_, ps, amount)
            opening = (ps == "LONG" and side_ == "BUY") or (ps == "SHORT" and side_ == "SELL")
            if opening:
                self._fill_open(symbol, ps, amount, px, TAKER, raw)
            else:
                pos = self._pos(symbol, ps)
                if not pos or amount > pos["qty"] + 1e-12:
                    raw["status"] = "FAILED"
                    raise ccxt.InvalidOrder("bingx {\"code\":101205,\"msg\":\"No position to close\"}")
                self._fill_close(symbol, ps, amount, px, raw)
            return self._unified(raw)
        if type_ in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            trig = float(params["stopPrice"])
            trig = float(self.price_to_precision(symbol, trig))
            closing_long = ps == "LONG"
            if type_ == "STOP_MARKET":
                bad = trig >= px if closing_long else trig <= px
            else:
                bad = trig <= px if closing_long else trig >= px
            if bad:
                raise ccxt.InvalidOrder(
                    "bingx {\"code\":80016,\"msg\":\"trigger price is on the wrong side of the market\"}")
            raw = self._new_order(symbol, type_, side_, ps, amount, stop_price=trig)
            return self._unified(raw)
        raise NotImplementedError(type_)

    def cancel_order(self, id, symbol=None, params=None):
        self.calls.append(("cancel_order", str(id), symbol))
        raw = self.orders.get(int(id))
        if raw is None or raw["status"] not in ("NEW", "PENDING"):
            raise ccxt.OrderNotFound(f"bingx order {id} not found or not open")
        raw["status"] = "CANCELLED"
        raw["updateTime"] = self._tick()
        return self._unified(raw)

    def fetch_order(self, id, symbol=None, params=None):
        self.calls.append(("fetch_order", str(id), symbol))
        if str(id) in self.fail_fetch_order:
            raise ccxt.NetworkError("bingx fetch_order timed out")
        raw = self.orders.get(int(id))
        if raw is None:
            raise ccxt.OrderNotFound(f"bingx order {id} does not exist")
        return self._unified(raw)

    def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
        self.calls.append(("fetch_open_orders", symbol))
        if self.fail_open_orders:
            raise ccxt.NetworkError("bingx fetch_open_orders timed out")
        return [self._unified(o) for o in self.open_orders(symbol)]

    def fetch_positions(self, symbols=None, params=None):
        out = []
        for (symbol, ps), p in self.positions.items():
            if symbols and symbol not in symbols:
                continue
            out.append({"symbol": symbol, "side": ps.lower(), "contracts": p["qty"],
                        "entryPrice": p["entry"], "info": {"positionSide": ps}})
        return out

    def fetch_my_trades(self, symbol=None, since=None, limit=None, params=None):
        return [f for f in self.fills if f["symbol"] == symbol and (since is None or f["timestamp"] >= since)]

    def fetch_balance(self, params=None):
        equity = self.equity0 + self.account_pnl()
        return {"info": {"data": {"balance": {"asset": "USDT", "equity": f"{equity:.4f}"}}},
                "USDT": {"total": equity, "free": equity}}

    def fetch_ohlcv(self, symbol, timeframe="15m", since=None, limit=None, params=None):
        """Flat candles at the current price: no order blocks, no trend."""
        px = self.price[symbol]
        step = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}[timeframe] * 1000
        n = limit or 300
        end = self.clock - self.clock % step
        return [[end - (n - 1 - i) * step, px, px, px, px, 1.0] for i in range(n)]

    def set_leverage(self, leverage, symbol=None, params=None):
        self.calls.append(("set_leverage", leverage, symbol, params))
        return {}
