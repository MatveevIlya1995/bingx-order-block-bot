"""
Risk management: position sizing (with a margin/notional cap), leverage
cap, a daily-loss circuit breaker, partial take-profit + trailing-stop
management for open trades, and a CSV trade journal that tracks entries,
partial exits, and full exits (so duration and realized P/L per trade can
be reported later).

The circuit breaker compares CURRENT account equity (fetched fresh from
BingX every cycle) against the equity recorded at the start of the UTC
day. This is deliberately simple and exchange-agnostic: it doesn't try to
parse individual closed-trade fills for the circuit breaker itself -- it
just asks "is my balance down too much vs. this morning?" every time it
runs, which also catches large *unrealized* drawdowns on an open
position, not only realized losses. State (including which trades are
still open) persists to a small JSON file so it survives across separate
scheduled-task invocations, which each run as a fresh process.

Partial take-profit + trailing stop (see config.py's PARTIAL_TP_* /
TRAIL_R_MULTIPLE): once a trade reaches partial_tp_r_multiple times its
original risk (1R), PARTIAL_EXIT closes partial_tp_fraction of the
position and the remainder's stop moves to breakeven; from there the
remainder is a trailing-stop runner with no fixed take-profit cap, so a
strong trend can pay out beyond the fixed 2R the trade would otherwise be
capped at. This is simulated candle-by-candle for DRY_RUN trades (see
reconcile_dry_run_exits) and, for LIVE trades, actually places the
partial-close and replacement-stop orders on the exchange (see
reconcile_live_exits and exchange_client.py) -- like the rest of this
bot's BingX order wiring, the live path has not been verified against a
real BingX account; validate it with a small real trade before trusting
it (see the README).
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
        self._migrate_legacy_trades()

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

    def _migrate_legacy_trades(self):
        """Back-fills the partial-take-profit/trailing-stop bookkeeping
        fields (initial_qty, risk_per_unit, partial_tp_price,
        partial_taken -- see _arm_partial_and_trail()) onto any trade
        that was opened by an OLDER version of this bot, before those
        fields existed, and is still sitting in state.json from a prior
        run. Without this, reconcile_dry_run_exits()/reconcile_live_exits()
        crash with a KeyError the first cycle they see one of these older
        trades, since they assume every open trade has already been
        through _arm_partial_and_trail(). Safe to run every time the bot
        starts: it only touches trades missing the new fields, and is a
        no-op once every trade has been migrated once."""
        migrated = False
        for key in ("open_trades", "open_dry_run_trades"):
            trades = self.state.get(key, [])
            for i, trade in enumerate(trades):
                if "partial_taken" not in trade:
                    trades[i] = self._arm_partial_and_trail(trade)
                    migrated = True
        if migrated:
            print(
                "[risk] migrated one or more pre-existing open trade(s), opened before "
                "partial-take-profit/trailing-stop support was added, to the new schema."
            )
            self._save_state()

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

    # --- occupied symbols (LIVE + DRY_RUN), used so paper trading obeys
    # the same "one position per symbol" / max_concurrent_positions rules
    # a real account would -- previously only real exchange positions
    # counted, so DRY_RUN could (and did) stack unlimited paper trades on
    # the same symbol at once, understating how selective LIVE trading
    # would actually be. ---
    def occupied_symbols(self) -> set:
        live = {t["symbol"] for t in self.state.get("open_trades", [])}
        paper = {t["symbol"] for t in self.state.get("open_dry_run_trades", [])}
        return live | paper

    def total_open_count(self) -> int:
        return len(self.state.get("open_trades", [])) + len(self.state.get("open_dry_run_trades", []))

    def total_open_notional(self, mark_prices: dict | None = None) -> float:
        """Approximate combined notional of every currently-tracked open
        trade (LIVE + DRY_RUN), used to cap how much MORE margin a new
        trade is allowed to use (see position_size()). Uses each trade's
        entry price as a stand-in for current mark price unless
        mark_prices (a {symbol: price} dict) supplies something fresher --
        an approximation, not an exact live margin read."""
        total = 0.0
        for trade in self.state.get("open_trades", []) + self.state.get("open_dry_run_trades", []):
            price = (mark_prices or {}).get(trade["symbol"], trade.get("entry_price", 0.0))
            total += trade.get("qty", 0.0) * price
        return total

    def track_open_trade(self, trade: dict):
        """Remembers a just-submitted LIVE entry so a later cycle can
        detect when it closes and log the exit. `trade` must include:
        trade_id, symbol, position_side, side, qty, entry_price,
        stop_loss, take_profit, trigger, entry_time_utc."""
        trade = self._arm_partial_and_trail(trade)
        self.state.setdefault("open_trades", []).append(trade)
        self._save_state()

    def _arm_partial_and_trail(self, trade: dict) -> dict:
        """Attaches the partial-take-profit / trailing-stop bookkeeping
        fields to a freshly-opened trade record (used for both LIVE and
        DRY_RUN trades so they're managed identically)."""
        is_long = trade["position_side"].upper() == "LONG"
        risk_per_unit = abs(trade["entry_price"] - trade["stop_loss"])
        partial_tp_price = (
            trade["entry_price"] + risk_per_unit * self.cfg.partial_tp_r_multiple
            if is_long else
            trade["entry_price"] - risk_per_unit * self.cfg.partial_tp_r_multiple
        )
        return {
            **trade,
            "initial_qty": trade["qty"],
            "risk_per_unit": risk_per_unit,
            "partial_tp_price": partial_tp_price,
            "partial_taken": False,
        }

    def reconcile_closed_trades(self, open_positions: list, exchange_client) -> list:
        """Compares tracked open LIVE trades against the exchange's
        current open positions. Any tracked trade no longer open at all
        is treated as fully closed (its stop or the trailing stop was
        hit, or it was closed manually): looks up the closing fill via
        the exchange client, logs an EXIT row (with duration and realized
        P/L when it can be determined), and updates today's realized P/L
        running total. Returns the list of trades that were closed this
        cycle.

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

    def reconcile_live_exits(self, symbol: str, high: float, low: float, exchange_client) -> list:
        """LIVE counterpart of reconcile_dry_run_exits(): checks every
        tracked open LIVE trade on `symbol` against the latest candle's
        high/low for a partial-take-profit trigger, and actually executes
        it on the exchange (a reduce-only partial close, then replacing
        the stop order at breakeven -- see exchange_client.py). A full
        stop-out or trailing-stop exit is NOT decided here: it's detected
        the normal way, by reconcile_closed_trades() noticing the position
        is no longer open on the exchange (the exchange's own stop order
        is what actually closes it). This method only ever *adds* the
        partial-exit order and moves the stop; it never assumes a fill
        happened without the exchange confirming it.

        UNVERIFIED against live BingX data, like the rest of this bot's
        order-submission code -- validate with a small real trade before
        relying on it (see the README).
        """
        open_trades = self.state.get("open_trades", [])
        touched = [t for t in open_trades if t["symbol"] == symbol and not t.get("partial_taken")]
        if not touched:
            return []

        acted = []
        for trade in touched:
            is_long = trade["position_side"].upper() == "LONG"
            partial_level = trade["partial_tp_price"]
            hit_partial = (high >= partial_level) if is_long else (low <= partial_level)
            if not hit_partial:
                continue

            partial_qty = round(trade["qty"] * self.cfg.partial_tp_fraction, 8)
            remaining_qty = round(trade["qty"] - partial_qty, 8)
            if partial_qty <= 0 or remaining_qty <= 0:
                continue

            close_side = "sell" if trade["side"].upper() == "BUY" else "buy"
            try:
                exchange_client.close_partial_position(
                    symbol=symbol, side=close_side, amount=partial_qty,
                    position_side=trade["position_side"],
                )
                exchange_client.replace_stop_order(
                    symbol=symbol, side=close_side, amount=remaining_qty,
                    new_stop_price=trade["entry_price"], position_side=trade["position_side"],
                )
            except Exception as e:
                print(f"[risk] {symbol}: partial take-profit order failed, leaving trade untouched: {e}")
                continue

            now = datetime.datetime.now(datetime.timezone.utc)
            entry_time = datetime.datetime.fromisoformat(trade["entry_time_utc"])
            self._write_journal_row({
                "event": "PARTIAL_EXIT",
                "trade_id": trade["trade_id"],
                "timestamp_utc": now.isoformat(),
                "mode": "LIVE",
                "symbol": symbol,
                "side": trade["side"],
                "position_side": trade["position_side"],
                "qty": partial_qty,
                "price": round(partial_level, 6),
                "stop_loss": trade["entry_price"],
                "take_profit": trade.get("take_profit", ""),
                "trigger": (
                    f"partial take-profit ({self.cfg.partial_tp_fraction:.0%} of position) at "
                    f"{self.cfg.partial_tp_r_multiple:g}R -- stop moved to breakeven, remainder trailing"
                ),
                "pnl": "",  # realized P/L for this slice comes from reconcile_closed_trades/exchange
                "duration_minutes": round((now - entry_time).total_seconds() / 60, 1),
            })

            trade["qty"] = remaining_qty
            trade["stop_loss"] = trade["entry_price"]
            trade["partial_taken"] = True
            trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple
            acted.append(trade)

        if acted:
            self._save_state()
        return acted

    def trail_live_stops(self, symbol: str, high: float, low: float, exchange_client):
        """For LIVE trades already past their partial take-profit (i.e.
        trailing runners), ratchets the exchange's stop order forward as
        price makes new favorable extremes -- mirrors the DRY_RUN trailing
        simulation in reconcile_dry_run_exits(). Never moves the stop
        backward. UNVERIFIED against live BingX data -- see the README."""
        open_trades = self.state.get("open_trades", [])
        trailing = [t for t in open_trades if t["symbol"] == symbol and t.get("partial_taken")]
        moved = False
        for trade in trailing:
            is_long = trade["position_side"].upper() == "LONG"
            if is_long:
                candidate = high - trade["trail_distance"]
                new_stop = max(trade["stop_loss"], candidate)
            else:
                candidate = low + trade["trail_distance"]
                new_stop = min(trade["stop_loss"], candidate)
            if new_stop == trade["stop_loss"]:
                continue
            close_side = "sell" if trade["side"].upper() == "BUY" else "buy"
            try:
                exchange_client.replace_stop_order(
                    symbol=symbol, side=close_side, amount=trade["qty"],
                    new_stop_price=new_stop, position_side=trade["position_side"],
                )
            except Exception as e:
                print(f"[risk] {symbol}: could not move trailing stop, leaving it where it was: {e}")
                continue
            trade["stop_loss"] = new_stop
            moved = True
        if moved:
            self._save_state()

    def track_dry_run_trade(self, trade: dict):
        """Remembers a just-logged DRY_RUN paper trade so a later cycle can
        check it against fresh candles and log a simulated exit once its
        stop-loss, partial take-profit, or trailing stop is touched.
        `trade` must include: trade_id, symbol, position_side, side, qty,
        entry_price, stop_loss, take_profit, trigger, entry_time_utc."""
        trade = self._arm_partial_and_trail(trade)
        self.state.setdefault("open_dry_run_trades", []).append(trade)
        self._save_state()

    def _estimate_round_trip_fee(self, qty: float, entry_price: float, exit_price: float) -> float:
        """Rough taker-fee estimate for one round trip (entry + this
        exit), applied to DRY_RUN P/L only -- see config.py's
        dry_run_fee_pct_per_side for why. LIVE P/L is unaffected: it
        comes from BingX's own realized-PnL field, which already reflects
        whatever BingX actually charged."""
        return (qty * entry_price + qty * exit_price) * self.cfg.dry_run_fee_pct_per_side

    def _close_dry_run_trade(self, trade: dict, exit_price: float, qty: float, reason: str):
        is_long = trade["position_side"].upper() == "LONG"
        direction = 1 if is_long else -1
        gross_pnl = (exit_price - trade["entry_price"]) * direction * qty
        fee = self._estimate_round_trip_fee(qty, trade["entry_price"], exit_price)
        pnl = gross_pnl - fee
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
            "qty": qty,
            "price": round(exit_price, 6),
            "stop_loss": trade["stop_loss"],
            "take_profit": trade.get("take_profit", ""),
            "trigger": reason,
            "pnl": round(pnl, 4),
            "duration_minutes": duration_minutes,
        })

    def _take_dry_run_partial(self, trade: dict, symbol: str):
        partial_qty = round(trade["qty"] * self.cfg.partial_tp_fraction, 8)
        remaining_qty = round(trade["qty"] - partial_qty, 8)
        exit_price = trade["partial_tp_price"]
        is_long = trade["position_side"].upper() == "LONG"
        direction = 1 if is_long else -1
        gross_pnl = (exit_price - trade["entry_price"]) * direction * partial_qty
        fee = self._estimate_round_trip_fee(partial_qty, trade["entry_price"], exit_price)
        pnl = gross_pnl - fee
        now = datetime.datetime.now(datetime.timezone.utc)
        entry_time = datetime.datetime.fromisoformat(trade["entry_time_utc"])
        self._write_journal_row({
            "event": "PARTIAL_EXIT",
            "trade_id": trade["trade_id"],
            "timestamp_utc": now.isoformat(),
            "mode": "DRY_RUN",
            "symbol": symbol,
            "side": trade["side"],
            "position_side": trade["position_side"],
            "qty": partial_qty,
            "price": round(exit_price, 6),
            "stop_loss": trade["entry_price"],
            "take_profit": trade.get("take_profit", ""),
            "trigger": (
                f"partial take-profit ({self.cfg.partial_tp_fraction:.0%} of position) at "
                f"{self.cfg.partial_tp_r_multiple:g}R -- stop moved to breakeven, remainder trailing"
            ),
            "pnl": round(pnl, 4),
            "duration_minutes": round((now - entry_time).total_seconds() / 60, 1),
        })
        trade["qty"] = remaining_qty
        trade["stop_loss"] = trade["entry_price"]
        trade["partial_taken"] = True
        trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple

    def reconcile_dry_run_exits(self, symbol: str, high: float, low: float) -> list:
        """Checks every open DRY_RUN paper trade on `symbol` against the
        high/low of the most recently fetched candle:

        - Before the partial take-profit: a stop-loss hit closes the
          whole trade; a partial-take-profit hit closes
          partial_tp_fraction of it, moves the stop to breakeven for the
          rest, and arms trailing for the remainder (only one of these
          two is applied per candle -- if the candle's range covers the
          stop AND the partial level, the stop is assumed to have been
          hit first, the same conservative assumption used for
          simultaneous SL/TP hits, so this never overstates performance).
        - After the partial take-profit: the trade is a trailing runner.
          Each candle first checks the CURRENT trailing stop (so a stop
          that was only just moved can't retroactively claim a hit from
          this same candle's favorable extreme); only if that isn't hit
          does the stop then ratchet forward using this candle's high (for
          a LONG) or low (for a SHORT), for the next candle to check.

        Must be called every cycle for every symbol right after fetching
        its candle -- this is the only mechanism that ever resolves a
        DRY_RUN paper trade; without it, DRY_RUN entries stay open
        forever and no real win rate can ever be measured.
        """
        open_trades = self.state.get("open_dry_run_trades", [])
        if not open_trades:
            return []

        remaining = []
        closed = []
        changed = False
        for trade in open_trades:
            if trade["symbol"] != symbol:
                remaining.append(trade)
                continue

            is_long = trade["position_side"].upper() == "LONG"

            if not trade.get("partial_taken"):
                sl = trade["stop_loss"]
                hit_sl = (low <= sl) if is_long else (high >= sl)
                if hit_sl:
                    self._close_dry_run_trade(
                        trade, exit_price=sl, qty=trade["qty"], reason="stop_loss hit"
                    )
                    closed.append(trade)
                    changed = True
                    continue

                partial_level = trade["partial_tp_price"]
                hit_partial = (high >= partial_level) if is_long else (low <= partial_level)
                if hit_partial:
                    self._take_dry_run_partial(trade, symbol)
                    changed = True
                remaining.append(trade)
                continue

            # Trailing runner: check the stop as it stood BEFORE this
            # candle first, then ratchet it forward for next time.
            sl = trade["stop_loss"]
            hit_sl = (low <= sl) if is_long else (high >= sl)
            if hit_sl:
                self._close_dry_run_trade(
                    trade, exit_price=sl, qty=trade["qty"],
                    reason=f"trailing stop hit ({self.cfg.trail_r_multiple:g}R trail)",
                )
                closed.append(trade)
                changed = True
                continue

            if is_long:
                candidate = high - trade["trail_distance"]
                new_stop = max(sl, candidate)
            else:
                candidate = low + trade["trail_distance"]
                new_stop = min(sl, candidate)
            if new_stop != sl:
                trade["stop_loss"] = new_stop
                changed = True
            remaining.append(trade)

        if changed:
            self.state["open_dry_run_trades"] = remaining
            self._save_state()
        return closed

    def position_size(
        self, equity: float, entry: float, stop_loss: float,
        leverage: int | None = None, existing_notional: float = 0.0,
    ) -> float:
        """Size the position so a stop-out loses roughly risk_pct_per_trade
        of equity, THEN cap that size so the position's notional (plus
        whatever's already open) doesn't require more margin than
        max_margin_utilization_pct of the account's theoretical capacity
        at the given leverage. A tight stop-loss on its own can otherwise
        make risk_amount / stop_distance far larger than the account can
        actually margin -- this second cap is what keeps that in check.
        Caller is still responsible for rounding to the exchange's amount
        precision/step size before submitting."""
        risk_amount = equity * self.cfg.risk_pct_per_trade
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return 0.0
        qty = risk_amount / stop_distance

        lev = leverage if leverage else self.safe_leverage()
        max_total_notional = equity * lev * self.cfg.max_margin_utilization_pct
        available_notional = max(max_total_notional - existing_notional, 0.0)
        if entry > 0:
            qty = min(qty, available_notional / entry)
        return max(qty, 0.0)

    def safe_leverage(self) -> int:
        return min(self.cfg.leverage, self.cfg.max_leverage_hard_cap)

    @staticmethod
    def new_trade_id() -> str:
        return uuid.uuid4().hex[:12]
