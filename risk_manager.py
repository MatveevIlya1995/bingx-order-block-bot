"""
Risk management: position sizing, leverage cap, a daily-loss circuit
breaker, and a CSV trade journal that tracks both trade entries and exits
(so duration and realized P/L per trade can be reported later).

The circuit breaker compares CURRENT account equity (fetched fresh from
BingX every cycle) against the equity recorded at the start of the UTC
day. This is deliberately simple and exchange-agnostic: it doesn't try to
parse individual closed-trade fills for the circuit breaker itself -- it
just asks "is my balance down too much vs. this morning?" every time it
runs, which also catches large *unrealized* drawdowns on an open
position, not only realized losses. State (including which trades are
still open) persists to a small JSON file so it survives across separate
scheduled-task invocations, which each run as a fresh process.
"""
import csv
import json
import os
import uuid
import datetime

TRADE_JOURNAL_FIELDS = [
    "event", "trade_id", "timestamp_utc", "mode", "symbol", "side",
    "position_side", "qty", "price", "stop_loss", "take_profit",
    "equity", "trigger", "pnl", "duration_minutes",
]

# How many days to remember a DRY_RUN signal's (symbol, trigger) key for,
# so the same not-yet-invalidated order block doesn't get logged again
# every cycle it keeps getting retested. Comfortably longer than
# CANDLE_LIMIT * TIMEFRAME (default ~300 * 15m =~ 3.1 days), so a zone
# still inside the strategy's lookback window won't fall out of dedup
# memory before it can actually re-trigger legitimately.
DRY_RUN_SIGNAL_MEMORY_DAYS = 7

class RiskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.cfg.state_path):
            with open(self.cfg.state_path) as f:
                data = json.load(f)
        else:
            data = {}
        data.setdefault("date", None)
        data.setdefault("day_start_equity", None)
        data.setdefault("realized_pnl_today", 0.0)
        data.setdefault("open_trades", [])
        data.setdefault("open_dry_run_trades", [])
        data.setdefault("dry_run_signaled", {})
        data.setdefault("traded_zones", {})
        data.setdefault("last_seen_equity", None)
        return data

    def _save_state(self):
        os.makedirs(os.path.dirname(self.cfg.state_path) or ".", exist_ok=True)
        with open(self.cfg.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _roll_day_if_needed(self, equity: float):
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if self.state.get("date") != today:
            self.state["date"] = today
            self.state["day_start_equity"] = equity
            self.state["realized_pnl_today"] = 0.0
            self._save_state()

    def circuit_breaker_tripped(self, equity: float) -> bool:
        """True if account equity has dropped too much vs. the start of
        the UTC day. Halts NEW entries; does not touch existing
        positions/stops (those are managed by the exchange directly)."""
        self._roll_day_if_needed(equity)
        start = self.state.get("day_start_equity") or equity
        if start <= 0:
            return False
        loss_pct = (start - equity) / start
        return loss_pct >= self.cfg.max_daily_loss_pct

    def check_equity_jump(self, equity: float, threshold_pct: float = 0.15) -> str | None:
        """Flags an equity change between consecutive cycles too large to
        be explained by this bot's own activity (it never sizes a single
        position anywhere near threshold_pct of equity). Surfaces things
        like external transfers, funding events, or reading the wrong
        wallet/sub-account balance instead of letting them pass through
        silently into the circuit breaker and the trade journal."""
        prev = self.state.get("last_seen_equity")
        self.state["last_seen_equity"] = equity
        self._save_state()
        if prev is None or prev <= 0:
            return None
        change_pct = abs(equity - prev) / prev
        if change_pct >= threshold_pct:
            return (
                f"equity moved {change_pct:.1%} since last cycle "
                f"(${prev:.2f} -> ${equity:.2f}) with no matching trade in this "
                f"run -- check the BingX account directly."
            )
        return None

    def _write_journal_row(self, row: dict):
        path = self.cfg.trades_csv_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        file_exists = os.path.exists(path)
        full_row = {field: row.get(field, "") for field in TRADE_JOURNAL_FIELDS}
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_JOURNAL_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(full_row)

    def log_trade(self, row: dict):
        """Appends one ENTRY row to the CSV trade journal (data/trades.csv
        by default), creating the header if the file is new. This is a
        lightweight local record of what the bot *attempted* -- BingX's
        own order/position history is always the authoritative source
        for actual fills and realized P&L."""
        self._write_journal_row({**row, "event": "ENTRY"})

    def _zone_key(self, symbol: str, ob: dict) -> str:
        """Stable identity for an order-block zone -- deliberately
        excludes the formed/BOS candle timestamps, which can shift across
        cycles as new candles arrive and the swing/BOS labelling gets
        re-evaluated. Keying on the full trigger string (as before) let
        the same physical zone get "rediscovered" under a new label and
        traded again; price bounds are rounded to avoid float-formatting
        drift between cycles."""
        return f"{symbol}|{ob['type']}|{round(ob['bottom'], 6)}-{round(ob['top'], 6)}"

    def zone_already_traded(self, symbol: str, ob: dict) -> bool:
        """True if a trade (DRY_RUN or LIVE) was already taken in this
        order-block zone, so the bot only ever takes one trade per zone
        no matter how many times it gets retested or re-detected under a
        different formed/BOS labelling across cycles."""
        key = self._zone_key(symbol, ob)
        return key in self.state.get("traded_zones", {})

    def remember_zone_trade(self, symbol: str, ob: dict):
        self._prune_old_zone_trades()
        key = self._zone_key(symbol, ob)
        self.state.setdefault("traded_zones", {})[key] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        self._save_state()

    def _prune_old_zone_trades(self):
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=DRY_RUN_SIGNAL_MEMORY_DAYS
        )
        traded = self.state.get("traded_zones", {})
        pruned = {}
        for key, seen_iso in traded.items():
            try:
                seen = datetime.datetime.fromisoformat(seen_iso)
            except (TypeError, ValueError):
                continue
            if seen > cutoff:
                pruned[key] = seen_iso
        self.state["traded_zones"] = pruned

    def track_open_trade(self, trade: dict):
        """Remembers a just-submitted LIVE entry so a later cycle can
        detect when it closes and log the exit. `trade` must include:
        trade_id, symbol, position_side, side, qty, entry_price,
        stop_loss, take_profit, trigger, entry_time_utc."""
        self.state.setdefault("open_trades", []).append(trade)
        self._save_state()

    def reconcile_closed_trades(self, open_positions: list, exchange_client) -> list:
        """Compares tracked open trades against the exchange's current
        open positions. Any tracked trade no longer open is treated as
        closed: looks up the closing fill via the exchange client, logs
        an EXIT row (with duration and realized P/L when it can be
        determined), and updates today's realized P/L running total.
        Returns the list of trades that were closed this cycle.

        Must be called every cycle (bot.py does this right after fetching
        open_positions, before the circuit-breaker check) -- otherwise
        LIVE trades never get an EXIT row, duration, or realized P/L
        recorded, and closed trades never leave `open_trades`.

        NOTE: matching a tracked trade to a live position by
        (symbol, position_side) assumes at most one open position per
        side per symbol, which holds for this bot's default config
        (MAX_CONCURRENT_POSITIONS=1) but is a simplification if that's
        raised. Like the rest of this bot's BingX wiring, the exact field
        names on a `fetch_positions()` result could not be verified
        against live BingX data from this environment -- confirm once
        you have a real trade go through.
        """
        still_open_keys = set()
        for p in open_positions:
            side = (p.get("side") or p.get("positionSide") or "").upper()
            still_open_keys.add((p.get("symbol"), side))

        remaining = []
        closed = []
        for trade in self.state.get("open_trades", []):
            key = (trade["symbol"], trade["position_side"].upper())
            if key in still_open_keys:
                remaining.append(trade)
            else:
                closed.append(trade)

        self.state["open_trades"] = remaining
        self._save_state()

        for trade in closed:
            exit_info = exchange_client.fetch_trade_exit(
                symbol=trade["symbol"],
                position_side=trade["position_side"],
                since_iso=trade["entry_time_utc"],
            )
            entry_time = datetime.datetime.fromisoformat(trade["entry_time_utc"])

            if exit_info and exit_info.get("timestamp"):
                exit_time = exit_info["timestamp"]
                exit_price = exit_info.get("price")
                pnl = exit_info.get("realized_pnl")
                if pnl is None and exit_price is not None:
                    direction = 1 if trade["position_side"].upper() == "LONG" else -1
                    pnl = (exit_price - trade["entry_price"]) * direction * trade["qty"]
            else:
                # Couldn't confirm the exact fill from BingX (e.g. an API
                # gap) -- still record that the position closed, using
                # "now" as an approximate exit time. Check BingX directly
                # for the exact fill/P&L on this trade.
                exit_time = datetime.datetime.now(datetime.timezone.utc)
                exit_price = None
                pnl = None

            duration_minutes = round((exit_time - entry_time).total_seconds() / 60, 1)

            if pnl is not None:
                self.state["realized_pnl_today"] = self.state.get("realized_pnl_today", 0.0) + pnl
                self._save_state()

            self._write_journal_row({
                "event": "EXIT",
                "trade_id": trade["trade_id"],
                "timestamp_utc": exit_time.isoformat(),
                "mode": trade.get("mode", "LIVE"),
                "symbol": trade["symbol"],
                "side": trade["side"],
                "position_side": trade["position_side"],
                "qty": trade["qty"],
                "price": exit_price if exit_price is not None else "",
                "stop_loss": trade.get("stop_loss", ""),
                "take_profit": trade.get("take_profit", ""),
                "trigger": trade.get("trigger", ""),
                "pnl": round(pnl, 4) if pnl is not None else "",
                "duration_minutes": duration_minutes,
            })

        return closed

    def track_dry_run_trade(self, trade: dict):
        """Remembers a just-logged DRY_RUN paper trade so a later cycle can
        check it against fresh candles and log a simulated exit once its
        stop-loss or take-profit is touched. `trade` must include the
        same fields as track_open_trade(): trade_id, symbol,
        position_side, side, qty, entry_price, stop_loss, take_profit,
        trigger, entry_time_utc."""
        self.state.setdefault("open_dry_run_trades", []).append(trade)
        self._save_state()

    def reconcile_dry_run_exits(self, symbol: str, high: float, low: float) -> list:
        """Checks every open DRY_RUN paper trade on `symbol` against the
        high/low of the most recently fetched candle. If the candle's
        range crosses the trade's stop-loss or take-profit, closes it at
        that level, computes realized P/L, and logs a simulated EXIT row
        the same way reconcile_closed_trades() does for real LIVE fills.

        Must be called every cycle for every symbol right after fetching
        its candle -- this is the only mechanism that ever resolves a
        DRY_RUN paper trade; without it, DRY_RUN entries stay open
        forever and no real win rate can ever be measured.

        If a single candle's range covers BOTH the stop-loss and the
        take-profit (an unusually volatile bar), the stop-loss is assumed
        to have been hit first -- the standard conservative assumption
        when the exact intra-candle tick order isn't known, so this never
        overstates paper performance.
        """
        open_trades = self.state.get("open_dry_run_trades", [])
        if not open_trades:
            return []

        remaining = []
        closed = []
        for trade in open_trades:
            if trade["symbol"] != symbol:
                remaining.append(trade)
                continue

            is_long = trade["position_side"].upper() == "LONG"
            sl = trade["stop_loss"]
            tp = trade["take_profit"]
            hit_sl = (low <= sl) if is_long else (high >= sl)
            hit_tp = (high >= tp) if is_long else (low <= tp)

            if not (hit_sl or hit_tp):
                remaining.append(trade)
                continue

            exit_price = sl if hit_sl else tp
            direction = 1 if is_long else -1
            pnl = (exit_price - trade["entry_price"]) * direction * trade["qty"]
            exit_time = datetime.datetime.now(datetime.timezone.utc)
            entry_time = datetime.datetime.fromisoformat(trade["entry_time_utc"])
            duration_minutes = round((exit_time - entry_time).total_seconds() / 60, 1)

            self._write_journal_row({
                "event": "EXIT",
                "trade_id": trade["trade_id"],
                "timestamp_utc": exit_time.isoformat(),
                "mode": "DRY_RUN",
                "symbol": trade["symbol"],
                "side": trade["side"],
                "position_side": trade["position_side"],
                "qty": trade["qty"],
                "price": round(exit_price, 6),
                "stop_loss": sl,
                "take_profit": tp,
                "trigger": "stop_loss hit" if hit_sl else "take_profit hit",
                "pnl": round(pnl, 4),
                "duration_minutes": duration_minutes,
            })
            closed.append(trade)

        if closed:
            self.state["open_dry_run_trades"] = remaining
            self._save_state()
        return closed

    def position_size(self, equity: float, entry: float, stop_loss: float) -> float:
        """Size the position so a stop-out loses roughly risk_pct_per_trade
        of equity. Caller is still responsible for rounding to the
        exchange's amount precision/step size before submitting."""
        risk_amount = equity * self.cfg.risk_pct_per_trade
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return 0.0
        return risk_amount / stop_distance

    def safe_leverage(self) -> int:
        return min(self.cfg.leverage, self.cfg.max_leverage_hard_cap)

    @staticmethod
    def new_trade_id() -> str:
        return uuid.uuid4().hex[:12]
