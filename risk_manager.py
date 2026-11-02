"""
Risk management: fixed-margin position sizing with per-trade leverage
solved from the stop-loss distance (see position_size()), a margin/
notional cap as a backstop on total exposure, a daily-loss circuit breaker, partial take-profit + trailing-stop
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
reconcile_dry_run_exits). For LIVE trades TP1 and TP2 rest on BingX as
reduce-only take-profit orders next to one stop order per part of the
position (see arm_live_exit_orders), each run journals the fills BingX
reports -- real price, real fee, P/L net of fees -- and feeds them into
today's realized P/L (see reconcile_live_exits / _book_live_exit). A
take-profit that could not be placed falls back to a market close on the
scheduled run.
"""
import csv
import json
import math
import os
import uuid
import datetime

from strategy import apply_tp2_floor

TRADE_JOURNAL_FIELDS = [
    "event", "trade_id", "timestamp_utc", "mode", "symbol", "side",
    "position_side", "qty", "price", "stop_loss", "take_profit",
    "equity", "trigger", "pnl", "duration_minutes", "fee",
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
        data.setdefault("pending_orders", [])
        data.setdefault("last_seen_equity", None)
        return data

    def _migrate_legacy_trades(self):
        """Bring open trades saved by older versions of the bot up to the
        current exit-plan schema, so a deploy never strands an open trade.

        - Trades opened before partial-take-profit support existed (no
          "partial_taken" key) get the full TP1/TP2/trailing plan armed.
        - Trades opened before TP2 support existed (no "tp2_taken" key):
          if their TP1 already fired they were already trailing under the
          old rules, so they keep trailing (tp2_taken=True); otherwise they
          get a TP2 at the TP2_FALLBACK_R_MULTIPLE fallback, since the
          candles they were opened on are no longer available to find the
          swing level they would have used.
        """
        migrated = False
        for key in ("open_trades", "open_dry_run_trades"):
            trades = self.state.get(key, [])
            for i, trade in enumerate(trades):
                if "partial_taken" not in trade:
                    trades[i] = self._arm_partial_and_trail(trade)
                    migrated = True
                elif "tp2_taken" not in trade:
                    if trade.get("partial_taken"):
                        trade["tp2_taken"] = True
                    else:
                        trade["tp2_price"] = self._fallback_tp2_price(trade)
                        trade["tp2_source"] = (
                            f"{self._tp2_fallback_r():g}R fallback (trade opened before TP2 support)"
                        )
                        trade["tp2_taken"] = False
                    migrated = True
        if migrated:
            print(
                "[risk] migrated one or more pre-existing open trade(s) to the "
                "TP1 / TP2 / trailing-stop exit schema."
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
            # Only reset realized P/L if it wasn't already recorded for
            # today -- exits reconciled earlier this cycle (before the
            # circuit-breaker check) already belong to the new day.
            if self.state.get("realized_pnl_date") != today:
                self.state["realized_pnl_date"] = today
                self.state["realized_pnl_today"] = 0.0
            self._save_state()

    def _record_realized_pnl(self, pnl: float):
        """Adds a closed (or partially closed) trade's P/L -- LIVE or
        DRY_RUN -- to today's realized total, starting a fresh total if
        the stored one belongs to a previous UTC day."""
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if self.state.get("realized_pnl_date") != today:
            self.state["realized_pnl_date"] = today
            self.state["realized_pnl_today"] = 0.0
        self.state["realized_pnl_today"] = self.state.get("realized_pnl_today", 0.0) + pnl

    def circuit_breaker_tripped(self, equity: float) -> bool:
        """True if account equity has dropped too much vs. the start of
        the UTC day, or today's realized P/L is down that much. Halts NEW
        entries; does not touch existing positions/stops (those are
        managed by the exchange directly)."""
        self._roll_day_if_needed(equity)
        start = self.state.get("day_start_equity") or equity
        if start <= 0:
            return False
        loss_pct = (start - equity) / start
        # Today's realized P/L is checked as well, in both modes, and the
        # worse of the two readings decides:
        # - DRY_RUN: paper trades never move the real account balance, so
        #   the equity comparison above can't see them at all.
        # - LIVE: every exit -- partial take-profits included -- is booked
        #   at its real fill price net of fees (_book_live_exit), so a day
        #   of realized losses halts new entries even while unrealized
        #   gains on open positions (or a deposit) hold the equity up.
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        realized = (
            self.state.get("realized_pnl_today", 0.0)
            if self.state.get("realized_pnl_date") == today else 0.0
        )
        loss_pct = max(loss_pct, -realized / start)
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

    def _upgrade_journal_header(self, path: str):
        """One-time upgrade of a journal written before a column was added
        (currently: "fee"): rewrites the file with the current header and
        pads every existing row with empty cells for the new column(s), so
        the CSV stays rectangular. Rows are otherwise copied unchanged. The
        new file is written next to the old one and swapped in only when
        complete."""
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows or rows[0] == TRADE_JOURNAL_FIELDS:
            return
        old_header = rows[0]
        if old_header != TRADE_JOURNAL_FIELDS[:len(old_header)]:
            # Not a header this bot wrote -- leave the file alone.
            print(f"[risk] WARNING: {path} has an unexpected header; not upgrading it.")
            return
        width = len(TRADE_JOURNAL_FIELDS)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TRADE_JOURNAL_FIELDS)
            for r in rows[1:]:
                if r:
                    writer.writerow(r + [""] * (width - len(r)))
        os.replace(tmp, path)
        print(f"[risk] upgraded {path} to the current journal columns ({', '.join(TRADE_JOURNAL_FIELDS[len(old_header):])} added).")

    def _write_journal_row(self, row: dict):
        path = self.cfg.trades_csv_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        file_exists = os.path.exists(path)
        if file_exists and not getattr(self, "_journal_header_checked", False):
            self._upgrade_journal_header(path)
        self._journal_header_checked = True
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

    # --- resting limit orders: one per unmitigated order block, waiting
    # for price to come back into the zone. Tracked in state so later runs
    # can detect fills (and turn them into open trades), and cancel orders
    # whose order block got mitigated/invalidated without filling. ---
    def pending_orders(self, symbol: str | None = None) -> list:
        orders = self.state.get("pending_orders", [])
        if symbol is None:
            return list(orders)
        return [o for o in orders if o["symbol"] == symbol]

    def pending_symbols(self) -> set:
        return {o["symbol"] for o in self.state.get("pending_orders", [])}

    def track_pending_order(self, order: dict):
        self.state.setdefault("pending_orders", []).append(order)
        self._save_state()

    def remove_pending_order(self, order_id: str):
        self.state["pending_orders"] = [
            o for o in self.state.get("pending_orders", []) if o["order_id"] != order_id
        ]
        self._save_state()

    def pending_notional(self, exclude_symbol: str | None = None) -> float:
        """Notional reserved by resting limit orders, for the shared margin
        cap. Only one order per symbol can ever turn into a position (the
        others are cancelled on the first fill), so each symbol reserves
        its single largest order, not the sum."""
        per_symbol = {}
        for o in self.state.get("pending_orders", []):
            if o["symbol"] == exclude_symbol:
                continue
            notional = o.get("qty", 0.0) * o.get("entry_price", 0.0)
            per_symbol[o["symbol"]] = max(per_symbol.get(o["symbol"], 0.0), notional)
        return sum(per_symbol.values())

    def log_order_event(self, event: str, order: dict, reason: str = ""):
        """Journals a limit-order lifecycle event (LIMIT_PLACED /
        LIMIT_CANCELLED) next to the ENTRY/EXIT rows, so the CSV shows
        which order blocks got an order and why unfilled ones were pulled."""
        self._write_journal_row({
            "event": event,
            "trade_id": order.get("trade_id", ""),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": order.get("mode", ""),
            "symbol": order["symbol"],
            "side": order.get("side", ""),
            "position_side": order.get("position_side", ""),
            "qty": order.get("qty", ""),
            "price": order.get("entry_price", ""),
            "stop_loss": order.get("stop_loss", ""),
            "take_profit": order.get("take_profit", ""),
            "trigger": (order.get("trigger", "") + (f" | {reason}" if reason else "")),
        })

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

    def _tp2_fallback_r(self) -> float:
        """R-multiple used for TP2 when no swing level beyond TP1 exists.
        Always kept strictly beyond TP1 so TP2 can never sit before it."""
        return max(self.cfg.tp2_fallback_r_multiple, self.cfg.partial_tp_r_multiple + 0.5)

    def _fallback_tp2_price(self, trade: dict) -> float:
        is_long = trade["position_side"].upper() == "LONG"
        direction = 1 if is_long else -1
        risk_per_unit = trade.get("risk_per_unit") or abs(trade["entry_price"] - trade["stop_loss"])
        return trade["entry_price"] + direction * risk_per_unit * self._tp2_fallback_r()

    def _tp_fractions(self) -> tuple:
        """(TP1 fraction, TP2 fraction) of the ORIGINAL position size. TP2
        is clamped so TP1 + TP2 never exceeds 100% of the position."""
        f1 = min(max(self.cfg.partial_tp_fraction, 0.0), 1.0)
        f2 = min(max(self.cfg.tp2_fraction, 0.0), 1.0 - f1)
        return f1, f2

    @staticmethod
    def _trailing_armed(trade: dict) -> bool:
        """The trailing stop only runs once BOTH take-profits are done.
        (Trades from before TP2 support have no "tp2_taken" key and were
        already trailing after TP1, so they count as armed.)"""
        return bool(trade.get("partial_taken")) and bool(trade.get("tp2_taken", True))

    def _arm_partial_and_trail(self, trade: dict) -> dict:
        """Attach the three-stage exit plan to a newly opened trade.

        TP1    -- the stop-loss mirrored to the other side of the entry
                  (PARTIAL_TP_R_MULTIPLE x the initial risk, 1R by default:
                  a stop $5 below entry means TP1 $5 above it). Closes
                  PARTIAL_TP_FRACTION of the original position (50%) and
                  moves the stop for the rest to breakeven.
        TP2    -- the next swing high (LONG) / swing low (SHORT) beyond TP1
                  on the entry timeframe, found by bot.py and passed in as
                  trade["tp2_price"]; TP2_FALLBACK_R_MULTIPLE x risk (2R)
                  when there is no such swing; never closer to the entry
                  than TP2_MIN_R_MULTIPLE x risk (0 = no minimum). Closes
                  TP2_FRACTION of the original position (30%). The stop
                  stays at breakeven between TP1 and TP2.
        Runner -- whatever is left (20%) trails TRAIL_R_MULTIPLE x risk
                  behind the best price reached, with no fixed cap, and
                  only starts trailing after TP2.
        """
        is_long = trade["position_side"].upper() == "LONG"
        direction = 1 if is_long else -1
        risk_per_unit = abs(trade["entry_price"] - trade["stop_loss"])
        partial_tp_price = trade["entry_price"] + direction * risk_per_unit * self.cfg.partial_tp_r_multiple

        tp2_price = trade.get("tp2_price")
        tp2_source = trade.get("tp2_source")
        beyond_tp1 = tp2_price is not None and (
            tp2_price > partial_tp_price if is_long else tp2_price < partial_tp_price
        )
        if not beyond_tp1:
            tp2_price = trade["entry_price"] + direction * risk_per_unit * self._tp2_fallback_r()
            tp2_source = f"{self._tp2_fallback_r():g}R fallback (no swing beyond TP1)"
        # TP2_MIN_R_MULTIPLE: a TP2 closer than that many R moves out to it.
        # bot._find_tp2_level already applied this to the limit price; here it
        # is applied to the price the entry really filled at.
        min_r = self.cfg.tp2_min_r_multiple
        tp2_price, moved = apply_tp2_floor(trade["entry_price"], risk_per_unit, is_long, tp2_price, min_r)
        if moved and not str(tp2_source or "").startswith(f"{min_r:g}R minimum"):
            tp2_source = f"{min_r:g}R minimum (was {tp2_source})"

        return {
            **trade,
            "initial_qty": trade["qty"],
            "risk_per_unit": risk_per_unit,
            "partial_tp_price": partial_tp_price,
            "partial_taken": False,
            "tp2_price": tp2_price,
            "tp2_source": tp2_source,
            "tp2_taken": False,
        }

    def reconcile_closed_trades(self, open_positions: list, exchange_client) -> list:
        """Compares tracked open LIVE trades against the exchange's
        current open positions. Any tracked trade no longer open at all
        is treated as fully closed (its stop or the trailing stop was
        hit, or it was closed manually) and settled -- see
        _settle_closed_trade(): take-profit orders that filled since the
        last run get their PARTIAL_EXIT rows, the closing fill gets an
        EXIT row (with duration and realized P/L when it can be
        determined), today's realized P/L running total is updated, and
        whatever exit orders the trade still has on the exchange are
        cancelled. Returns the list of trades that were closed this cycle.

        Must be called every cycle (bot.py does this right after fetching
        open_positions, before the circuit-breaker check) -- otherwise
        LIVE trades never get an EXIT row, duration, or realized P/L
        recorded, and closed trades never leave `open_trades`.

        NOTE: matching a tracked trade to a live position by
        (symbol, position_side) assumes at most one open position per
        side per symbol, which this bot guarantees (one position per
        symbol).
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
            try:
                self._settle_closed_trade(trade, exchange_client)
            except Exception as e:
                # Never let one trade's bookkeeping stop the others from
                # being settled (they are already out of open_trades).
                print(f"[risk] {trade['symbol']}: could not fully settle the closed trade {trade.get('trade_id')} ({e!r}) -- check its journal rows and leftover orders on BingX.")
            self._save_state()

        return closed

    # ------------------------------------------------------------------
    # LIVE exits: orders resting on the exchange, real fills, real fees
    # ------------------------------------------------------------------

    @staticmethod
    def _close_side(trade: dict) -> str:
        return "sell" if trade["side"].upper() == "BUY" else "buy"

    def _entry_fee_share(self, trade: dict, qty: float) -> tuple:
        """(the part of the ENTRY fee that belongs to `qty`, estimated?).
        Uses the fee BingX reported on the entry fill (trade["entry_fee"],
        stored by bot._promote_fill) split pro rata over the original
        size; falls back to the configured fee rate when that is unknown
        (trades opened before fees were recorded)."""
        initial = trade.get("initial_qty") or trade.get("qty") or 0.0
        entry_fee = trade.get("entry_fee")
        if entry_fee is not None and initial > 0:
            return entry_fee * qty / initial, False
        return qty * trade["entry_price"] * self.cfg.dry_run_fee_pct_per_side, True

    def _book_live_exit(self, trade: dict, event: str, qty: float, price: float, exit_fee,
                        price_is_real: bool, when, trigger: str, stop_after):
        """Journals one LIVE exit (PARTIAL_EXIT or EXIT) and adds its P/L
        to today's realized total.

        pnl = (exit price - entry price) x qty in the trade's direction,
              minus the exit fee, minus this part's share of the entry fee.

        `price` and `exit_fee` are what BingX reported for the fill when
        available. A missing fee is estimated from the configured rate; a
        missing price means the caller passed its best estimate
        (price_is_real=False). Either way the journal row says so."""
        direction = 1 if trade["position_side"].upper() == "LONG" else -1
        gross = (price - trade["entry_price"]) * direction * qty
        exit_fee_estimated = exit_fee is None
        if exit_fee_estimated:
            exit_fee = qty * price * self.cfg.dry_run_fee_pct_per_side
        entry_share, entry_estimated = self._entry_fee_share(trade, qty)
        pnl = gross - exit_fee - entry_share
        self._record_realized_pnl(pnl)

        notes = []
        if not price_is_real:
            notes.append("fill price ESTIMATED (BingX order details unavailable)")
        else:
            notes.append("fill price from BingX")
        notes.append("exit fee estimated" if exit_fee_estimated else "exit fee from BingX")
        notes.append(
            f"pnl is net of the exit fee and a {entry_share:.4f} USDT entry-fee share"
            + (" (entry fee estimated)" if entry_estimated else "")
        )
        now = when or datetime.datetime.now(datetime.timezone.utc)
        entry_time = datetime.datetime.fromisoformat(trade["entry_time_utc"])
        self._write_journal_row({
            "event": event,
            "trade_id": trade["trade_id"],
            "timestamp_utc": now.isoformat(),
            "mode": trade.get("mode", "LIVE"),
            "symbol": trade["symbol"],
            "side": trade["side"],
            "position_side": trade["position_side"],
            "qty": qty,
            "price": round(price, 8),
            "stop_loss": stop_after,
            "take_profit": trade.get("take_profit", ""),
            "trigger": f"{trigger} | " + "; ".join(notes),
            "pnl": round(pnl, 4),
            "duration_minutes": round((now - entry_time).total_seconds() / 60, 1),
            "fee": round(exit_fee, 6),
        })
        return pnl

    def _settle_closed_trade(self, trade: dict, exchange_client):
        """A LIVE trade whose position is gone from the exchange: journal
        everything that happened since the last run, then clean up."""
        symbol = trade["symbol"]
        side = self._close_side(trade)
        leftovers = []
        try:
            self._journal_closed_trade(trade, exchange_client, leftovers)
        finally:
            # Whatever happened above: nothing of this trade may stay on
            # the exchange.
            exchange_client.cancel_orders(symbol, leftovers)
            if not self.pending_orders(symbol):
                # Also clears stops left behind by older versions of the
                # bot. Skipped while the symbol has a resting entry order,
                # so that order's own attached stop can never be touched.
                cancelled = exchange_client.cancel_exit_orders_for_side(symbol, trade["position_side"], side)
                if cancelled:
                    print(f"[risk] {symbol}: cancelled {cancelled} leftover stop/take-profit order(s) after the trade closed.")

    def _journal_closed_trade(self, trade: dict, exchange_client, leftovers: list):
        """Journal rows for a closed LIVE trade. Order ids that may still be
        open on the exchange are appended to `leftovers` for the caller to
        cancel."""
        symbol = trade["symbol"]
        eo = trade.get("exit_orders") or {}

        # 1. take-profits that filled since the last run
        for stage in (1, 2):
            order_id = eo.get(f"tp{stage}_id")
            taken_key = "partial_taken" if stage == 1 else "tp2_taken"
            if not order_id or trade.get(taken_key):
                continue
            try:
                info = exchange_client.order_fill(order_id, symbol)
            except Exception as e:
                print(f"[risk] {symbol}: could not read resting TP{stage} order {order_id} while closing the trade ({e}) -- its fill, if any, is NOT in the journal; check BingX.")
                leftovers.append(order_id)
                continue
            if info["status"] != "filled":
                if info["status"] == "open":
                    leftovers.append(order_id)
                continue
            tp_price = trade["partial_tp_price"] if stage == 1 else trade["tp2_price"]
            qty = min(info["filled"], trade["qty"])
            text = self._tp1_trigger_text(trade) if stage == 1 else self._tp2_trigger_text(trade)
            self._book_live_exit(
                trade, "PARTIAL_EXIT", qty, info["price"] or tp_price, info["fee"],
                info["price"] is not None, info["timestamp"],
                text + " (resting order on BingX)", trade["stop_loss"],
            )
            trade["qty"] = max(round(trade["qty"] - qty, 8), 0.0)
            trade[taken_key] = True

        # 2. the stop(s) that closed what was left
        filled_qty, notional, fee_total, fee_known, price_known, last_ts = 0.0, 0.0, 0.0, True, True, None
        for order_id in eo.get("stop_ids") or []:
            try:
                info = exchange_client.order_fill(order_id, symbol)
            except Exception:
                leftovers.append(order_id)
                continue
            if info["status"] == "open":
                leftovers.append(order_id)
                continue
            if info["status"] != "filled":
                continue
            px = info["price"]
            if px is None:
                px, price_known = trade["stop_loss"], False
            filled_qty += info["filled"]
            notional += info["filled"] * px
            if info["fee"] is None:
                fee_known = False
            else:
                fee_total += info["fee"]
            if info["timestamp"] and (last_ts is None or info["timestamp"] > last_ts):
                last_ts = info["timestamp"]

        qty = trade["qty"]
        entry_time = datetime.datetime.fromisoformat(trade["entry_time_utc"])
        if qty <= 0:
            # The take-profits closed the whole position: nothing is left
            # for an EXIT row.
            pass
        elif filled_qty > 0 and abs(filled_qty - qty) <= 0.02 * qty:
            # The trade's own stop orders account for the close.
            self._book_live_exit(
                trade, "EXIT", filled_qty, notional / filled_qty,
                fee_total if fee_known else None, price_known, last_ts,
                trade.get("trigger", ""), trade.get("stop_loss", ""),
            )
        else:
            # Stop order ids unknown (trade opened before they were tracked,
            # or closed by hand) -- look the closing fill up in the trade
            # history instead, as before.
            exit_info = exchange_client.fetch_trade_exit(
                symbol=symbol,
                position_side=trade["position_side"],
                since_iso=trade["entry_time_utc"],
            )
            if exit_info and exit_info.get("timestamp") and exit_info.get("price") is not None:
                # The lookup returns the LAST closing fill. Its fee only
                # covers the whole remainder if that one fill closed it.
                amount = exit_info.get("amount")
                whole = amount is not None and abs(amount - qty) <= 0.02 * qty
                self._book_live_exit(
                    trade, "EXIT", qty, exit_info["price"],
                    exit_info.get("fee") if whole else None,
                    True, exit_info["timestamp"], trade.get("trigger", ""),
                    trade.get("stop_loss", ""),
                )
            else:
                # Couldn't confirm the exact fill from BingX (e.g. an API
                # gap) -- still record that the position closed, using
                # "now" as an approximate exit time. Check BingX directly
                # for the exact fill/P&L on this trade.
                exit_time = (exit_info or {}).get("timestamp") or datetime.datetime.now(datetime.timezone.utc)
                self._write_journal_row({
                    "event": "EXIT",
                    "trade_id": trade["trade_id"],
                    "timestamp_utc": exit_time.isoformat(),
                    "mode": trade.get("mode", "LIVE"),
                    "symbol": symbol,
                    "side": trade["side"],
                    "position_side": trade["position_side"],
                    "qty": qty,
                    "price": "",
                    "stop_loss": trade.get("stop_loss", ""),
                    "take_profit": trade.get("take_profit", ""),
                    "trigger": trade.get("trigger", "") + " | closing fill not found on BingX -- pnl unknown",
                    "pnl": "",
                    "duration_minutes": round((exit_time - entry_time).total_seconds() / 60, 1),
                })

    def _journal_partial(self, trade: dict, mode: str, qty: float, price: float,
                         stop_after: float, trigger: str, pnl="", fee=""):
        now = datetime.datetime.now(datetime.timezone.utc)
        entry_time = datetime.datetime.fromisoformat(trade["entry_time_utc"])
        self._write_journal_row({
            "event": "PARTIAL_EXIT",
            "trade_id": trade["trade_id"],
            "timestamp_utc": now.isoformat(),
            "mode": mode,
            "symbol": trade["symbol"],
            "side": trade["side"],
            "position_side": trade["position_side"],
            "qty": qty,
            "price": round(price, 6),
            "stop_loss": stop_after,
            "take_profit": trade.get("take_profit", ""),
            "trigger": trigger,
            "pnl": pnl if pnl == "" else round(pnl, 4),
            "duration_minutes": round((now - entry_time).total_seconds() / 60, 1),
            "fee": fee if fee == "" else round(fee, 6),
        })

    def _tp1_trigger_text(self, trade: dict) -> str:
        f1, f2 = self._tp_fractions()
        if f2 > 0:
            nxt = f"waiting for TP2 at {trade['tp2_price']:.6g} ({trade.get('tp2_source') or 'swing level'})"
        else:
            nxt = "remainder trailing (TP2 disabled)"
        return (
            f"TP1: {f1:.0%} of position closed at {self.cfg.partial_tp_r_multiple:g}R "
            f"(stop-loss mirrored) -- stop moved to breakeven, {nxt}"
        )

    def _tp2_trigger_text(self, trade: dict) -> str:
        f1, f2 = self._tp_fractions()
        runner = max(0.0, 1.0 - f1 - f2)
        return (
            f"TP2: {f2:.0%} of position closed at {trade.get('tp2_source') or 'swing level'} -- "
            f"remaining {runner:.0%} trailing {self.cfg.trail_r_multiple:g}R behind best price"
        )

    @staticmethod
    def _exchange_qty(exchange_client, symbol: str, qty: float) -> float:
        """Rounds a partial-close size to the exchange's amount step BEFORE
        the bot works out what's left. Otherwise the exchange rounds each
        order down on its own (0.00285 BTC -> 0.0028) while the bot's
        bookkeeping keeps the unrounded number, the next stop is sized for
        less than what is really left, and a dust-sized position survives
        every exit -- keeping the symbol's slot occupied.

        A size below the exchange's smallest step comes back as 0: that
        part can't be traded on its own and simply stays with the rest."""
        try:
            return float(exchange_client.amount_to_precision(symbol, qty))
        except Exception as e:
            if type(e).__name__ == "InvalidOrder" or "minimum amount precision" in str(e):
                return 0.0
            return qty

    def _tranche_plan(self, trade: dict, exchange_client) -> list:
        """The parts the still-open position is made of, in exit order:
        [("tp1", qty), ("tp2", qty), ("runner", qty)], leaving out parts
        already closed or too small to exist. The quantities always add up
        to exactly trade["qty"].

        Every part gets its OWN stop order (see arm_live_exit_orders), and
        the take-profit parts also get a resting take-profit order."""
        f1, f2 = self._tp_fractions()
        symbol = trade["symbol"]
        initial = trade.get("initial_qty", trade["qty"])
        left = trade["qty"]
        plan = []
        if not trade.get("partial_taken"):
            q1 = self._exchange_qty(exchange_client, symbol, round(min(initial * f1, left), 8))
            # TP1 never closes the whole position (same rule as before).
            if q1 > 0 and round(left - q1, 8) > 0:
                plan.append(("tp1", q1))
                left = round(left - q1, 8)
        if not trade.get("tp2_taken", False) and f2 > 0:
            q2 = self._exchange_qty(exchange_client, symbol, round(min(initial * f2, left), 8))
            if q2 > 0:
                plan.append(("tp2", q2))
                left = round(left - q2, 8)
        if left > 0:
            plan.append(("runner", left))
        return plan

    @staticmethod
    def _match_orders(existing: list, kind: str, price: float, amounts: list):
        """Finds, among open exit orders, one order per entry in `amounts`
        of the given kind at `price`. Returns their ids, or None unless
        every amount has a match. Lets arming adopt orders it already
        placed (e.g. when a previous run placed them but its state update
        was lost) instead of placing them twice."""
        pool = [o for o in existing if o["kind"] == kind and o.get("price") and o.get("amount")]
        ids = []
        for amount in amounts:
            hit = None
            for o in pool:
                if abs(o["price"] - price) <= 5e-4 * price and abs(o["amount"] - amount) <= 1e-3 * amount:
                    hit = o
                    break
            if hit is None:
                return None
            pool.remove(hit)
            ids.append(hit["id"])
        return ids

    def arm_live_exit_orders(self, symbol: str, exchange_client):
        """LIVE: puts a newly filled trade's exits on the exchange.

        Called every cycle; acts once per trade, on the first run that
        sees it (trade has no "exit_orders" yet -- this includes trades
        that were already open when this feature was deployed):

        1. One stop order per part of the position (TP1 part, TP2 part,
           runner), all at the trade's stop price, replacing the single
           full-size stop BingX attached to the entry. Reason: a resting
           take-profit can fill between two runs, and until the next run
           the stops must still add up to what is left. With one stop per
           part they always do -- whichever parts already closed, the
           stops for the remaining parts are each no bigger than the
           position.
        2. TP1 and TP2 as resting reduce-only TAKE_PROFIT_MARKET orders.

        Fallbacks (the behaviour before this feature):
        - the stops can't be placed -> nothing changes for this trade:
          the original full-size stop stays, no take-profit is placed, and
          TP1/TP2 are closed at market by the scheduled run as before.
        - one take-profit can't be placed (e.g. price is already past it)
          -> only that take-profit falls back to the market close on the
          run; the other one still rests.
        """
        if not self.cfg.resting_tp_orders:
            return
        for trade in self.state.get("open_trades", []):
            if trade["symbol"] != symbol or "exit_orders" in trade:
                continue
            if self._trailing_armed(trade):
                # Already a trailing runner: nothing to place, the trailing
                # logic owns its stop from the next move on.
                trade["exit_orders"] = {"mode": "legacy", "stop_ids": []}
                self._save_state()
                continue

            side = self._close_side(trade)
            ps = trade["position_side"]
            plan = self._tranche_plan(trade, exchange_client)
            amounts = [q for _, q in plan]
            stop_price = trade["stop_loss"]
            try:
                existing = exchange_client.open_exit_orders(symbol, ps, side)
            except Exception as e:
                print(f"[risk] {symbol}: could not read open orders to arm the exit orders ({e}) -- will try again next cycle.")
                continue

            stop_ids = self._match_orders(existing, "stop", stop_price, amounts)
            if stop_ids is None:
                try:
                    stop_ids = exchange_client.place_stop_orders(symbol, side, amounts, stop_price, ps)
                except Exception as e:
                    trade["exit_orders"] = {
                        "mode": "legacy", "stop_ids": [],
                        "note": f"per-part stops could not be placed: {e}",
                    }
                    self._save_state()
                    print(
                        f"[risk] {symbol}: could not place the per-part stop orders ({e}) -- keeping the "
                        f"original stop; TP1/TP2 fall back to market closes on the scheduled run."
                    )
                    continue
            eo = {"mode": "resting", "stop_ids": stop_ids}
            trade["exit_orders"] = eo
            self._save_state()

            # The single full-size stop that came with the entry is now
            # superseded by the per-part stops.
            stale = [
                o["id"] for o in existing
                if o["kind"] == "stop" and o["id"] not in stop_ids
                and o.get("price") and o.get("amount")
                and abs(o["price"] - stop_price) <= 5e-4 * stop_price
                and abs(o["amount"] - trade["qty"]) <= 1e-3 * trade["qty"]
            ]
            exchange_client.cancel_orders(symbol, stale)

            placed = []
            for name, qty in plan:
                if name == "runner":
                    continue
                price = trade["partial_tp_price"] if name == "tp1" else trade["tp2_price"]
                found = self._match_orders(existing, "tp", price, [qty])
                if found:
                    order_id = found[0]
                else:
                    try:
                        order_id = exchange_client.place_take_profit_order(symbol, side, qty, price, ps)
                    except Exception as e:
                        order_id = None
                        print(
                            f"[risk] {symbol}: could not place the resting {name.upper()} order at {price:.6g} "
                            f"({e}) -- this take-profit falls back to a market close on the scheduled run."
                        )
                eo[f"{name}_id"] = order_id
                eo[f"{name}_qty"] = qty
                self._save_state()
                if order_id:
                    placed.append(f"{name.upper()} {qty:g} @ {price:.6g}")
            print(
                f"[risk] {symbol}: exit orders on BingX -- {len(stop_ids)} stop(s) at {stop_price:.6g} "
                f"for {' + '.join(f'{q:g}' for q in amounts)}; resting take-profits: {', '.join(placed) or 'none'}."
            )

    def _move_stops(self, trade: dict, exchange_client, price: float, amounts: list) -> bool:
        """Moves the trade's stop(s) to `price`, one stop per entry in
        `amounts`: places the new ones first, then cancels the old ones,
        so the position is never without a stop. Returns False (and leaves
        the old stops exactly as they were) if the new ones can't be
        placed."""
        symbol = trade["symbol"]
        side = self._close_side(trade)
        ps = trade["position_side"]
        eo = trade.get("exit_orders")
        if eo is None:
            eo = trade["exit_orders"] = {"mode": "legacy", "stop_ids": []}
        old_ids = list(eo.get("stop_ids") or [])
        try:
            new_ids = exchange_client.place_stop_orders(symbol, side, amounts, price, ps)
        except Exception as e:
            print(f"[risk] {symbol}: could not place the stop at {price:.6g} ({e}) -- the previous stop stays in place.")
            return False
        if old_ids:
            exchange_client.cancel_orders(symbol, old_ids)
        elif not self.pending_orders(symbol):
            # Stops this trade has no ids for (attached by BingX to the
            # entry, or placed by an older version of the bot).
            try:
                stale = [
                    o["id"] for o in exchange_client.open_exit_orders(symbol, ps, side)
                    if o["kind"] == "stop" and o["id"] not in new_ids
                ]
                exchange_client.cancel_orders(symbol, stale)
            except Exception as e:
                print(f"[risk] {symbol}: could not look up superseded stop orders ({e}).")
        eo["stop_ids"] = new_ids
        trade["stop_loss"] = price
        self._save_state()
        return True

    def _tp_fill(self, trade: dict, stage: int, high, low, last_price, exchange_client, position_qty):
        """Has TP1 (stage=1) / TP2 (stage=2) of this LIVE trade been taken?
        Returns the fill {"qty", "price", "fee", "real", "when", "resting"}
        or None if not (yet).

        Resting order on the exchange: its own status decides, read from
        BingX. If the order can't be read this cycle, the trade waits --
        unless the exchange position has shrunk by (at least) that
        take-profit's size, in which case the fill is taken as given with
        an estimated price. A resting order that is gone without filling
        (cancelled by hand, rejected when it triggered) switches this
        take-profit to the fallback below.

        Fallback (no resting order): the behaviour before resting
        take-profits -- when the latest candle has reached the level,
        close that part at market now."""
        symbol = trade["symbol"]
        eo = trade.get("exit_orders") or {}
        key = f"tp{stage}_id"
        tp_price = trade["partial_tp_price"] if stage == 1 else trade["tp2_price"]

        order_id = eo.get(key)
        if order_id:
            try:
                info = exchange_client.order_fill(order_id, symbol)
            except Exception as e:
                planned = eo.get(f"tp{stage}_qty") or 0.0
                if position_qty is not None and planned > 0 and position_qty <= (trade["qty"] - planned) * 1.02:
                    print(f"[risk] {symbol}: resting TP{stage} order unreadable ({e}), but the position shrank to {position_qty:g} -- treating it as filled.")
                    return {"qty": min(planned, trade["qty"]), "price": tp_price, "fee": None,
                            "real": False, "when": None, "resting": True}
                print(f"[risk] {symbol}: could not read the resting TP{stage} order ({e}) -- will check again next cycle.")
                return None
            if info["status"] == "open":
                return None
            if info["status"] == "filled":
                return {"qty": min(info["filled"], trade["qty"]), "price": info["price"] or tp_price,
                        "fee": info["fee"], "real": info["price"] is not None,
                        "when": info["timestamp"], "resting": True}
            eo[key] = None
            print(f"[risk] {symbol}: the resting TP{stage} order is no longer on BingX (cancelled or rejected) -- falling back to a market close on the scheduled run for this take-profit.")

        if high is None or low is None:
            return None
        is_long = trade["position_side"].upper() == "LONG"
        if not ((high >= tp_price) if is_long else (low <= tp_price)):
            return None
        f1, f2 = self._tp_fractions()
        initial_qty = trade.get("initial_qty", trade["qty"])
        tp_qty = round(min(initial_qty * (f1 if stage == 1 else f2), trade["qty"]), 8)
        tp_qty = self._exchange_qty(exchange_client, symbol, tp_qty)
        remaining_qty = round(trade["qty"] - tp_qty, 8)
        if stage == 1 and (tp_qty <= 0 or remaining_qty <= 0):
            return None
        if stage == 2 and remaining_qty <= 0:
            # TP1 + TP2 fractions add up to the whole position:
            # TP2 closes everything that's left.
            tp_qty = trade["qty"]
        if tp_qty <= 0:
            # Too small to close as its own order: nothing to sell, the
            # stage is simply done and the remainder moves on.
            return {"qty": 0.0, "price": tp_price, "fee": None, "real": False, "when": None, "resting": False}
        try:
            order = exchange_client.close_partial_position(
                symbol=symbol, side=self._close_side(trade), amount=tp_qty,
                position_side=trade["position_side"],
            )
        except Exception as e:
            print(f"[risk] {symbol}: TP{stage} market close failed, will retry next cycle: {e}")
            return None
        fill = {"qty": tp_qty, "price": last_price if last_price else tp_price, "fee": None,
                "real": False, "when": None, "resting": False}
        order_id = (order or {}).get("id")
        if order_id:
            try:
                info = exchange_client.order_fill(str(order_id), symbol)
                if info["status"] == "filled" and info["price"] is not None:
                    fill.update(price=info["price"], fee=info["fee"], real=True, when=info["timestamp"])
            except Exception as e:
                print(f"[risk] {symbol}: TP{stage} closed at market but its fill could not be read ({e}) -- journalled with an estimated price.")
        return fill

    def _apply_tp_fill(self, trade: dict, stage: int, fill: dict, exchange_client):
        """Books a TP1/TP2 fill and brings the stops in line with what is
        left: after TP1 the remaining parts' stops move to breakeven;
        after TP2 only the runner's stop remains and trailing is armed."""
        symbol = trade["symbol"]
        f1, f2 = self._tp_fractions()
        eo = trade.get("exit_orders")
        how = "resting order on BingX" if fill["resting"] else "market close on the scheduled run"
        if stage == 1:
            text, stop_after = self._tp1_trigger_text(trade), trade["entry_price"]
        else:
            text, stop_after = self._tp2_trigger_text(trade), trade["stop_loss"]
        if fill["qty"] > 0:
            self._book_live_exit(
                trade, "PARTIAL_EXIT", fill["qty"], fill["price"], fill["fee"],
                fill["real"], fill["when"], f"{text} ({how})", stop_after,
            )
        trade["qty"] = max(round(trade["qty"] - fill["qty"], 8), 0.0)
        if eo is not None:
            eo[f"tp{stage}_id"] = None
        if stage == 1:
            trade["partial_taken"] = True
            if f2 <= 0:
                trade["tp2_taken"] = True
                trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple
            target = trade["entry_price"]
        else:
            trade["tp2_taken"] = True
            trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple
            target = trade["entry_price"] if trade.get("breakeven_pending") else trade["stop_loss"]
        self._save_state()

        if trade["qty"] <= 0:
            if eo is not None:
                exchange_client.cancel_orders(symbol, eo.get("stop_ids") or [])
                eo["stop_ids"] = []
            return
        amounts = [q for _, q in self._tranche_plan(trade, exchange_client)]
        if self._move_stops(trade, exchange_client, target, amounts):
            trade.pop("breakeven_pending", None)
        elif trade["stop_loss"] != target:
            # Typically: price is already back through the entry, so BingX
            # refuses a stop there. The previous stops stay (still one per
            # remaining part), and the move is retried every cycle.
            trade["breakeven_pending"] = True
            print(f"[risk] {symbol}: stop NOT yet at breakeven after TP{stage}; previous stop at {trade['stop_loss']:.6g} still protects the position, retrying next cycle.")

    def reconcile_live_exits(self, symbol: str, high, low, exchange_client,
                             last_price=None, position_qty=None) -> list:
        """LIVE: advance each open trade on symbol through TP1 and TP2.

        With resting take-profit orders (the default, see
        arm_live_exit_orders) BingX executes TP1/TP2 itself the moment
        price gets there; this method notices the fill on the next run,
        journals it with the real fill price and fee, and moves the stops:

        TP1 filled -> the remaining parts' stops move to breakeven.
        TP2 filled -> only the runner's stop is left, and the trailing
                      stop is armed (trail_live_stops()).

        Without a resting order for a take-profit (it couldn't be placed,
        or RESTING_TP_ORDERS=false) that take-profit works as before: when
        the latest candle's high/low reaches the level, that part is
        closed at market on this run.
        A run that finds both levels done advances both stages.

        high/low/last_price may be None (no candles this cycle): resting
        orders are still checked, only the fallback is skipped.
        position_qty is the size of the exchange position for this symbol,
        used only as a cross-check when an order can't be read.
        """
        open_trades = self.state.get("open_trades", [])
        touched = [t for t in open_trades if t["symbol"] == symbol and not self._trailing_armed(t)]
        if not touched:
            return []

        f1, f2 = self._tp_fractions()
        acted = []
        for trade in touched:
            progressed = False

            if trade.get("breakeven_pending") and trade.get("partial_taken"):
                amounts = [q for _, q in self._tranche_plan(trade, exchange_client)]
                if self._move_stops(trade, exchange_client, trade["entry_price"], amounts):
                    trade.pop("breakeven_pending", None)

            # --- TP1 ---
            if not trade.get("partial_taken"):
                fill = self._tp_fill(trade, 1, high, low, last_price, exchange_client, position_qty)
                if fill is None:
                    continue
                self._apply_tp_fill(trade, 1, fill, exchange_client)
                progressed = True
                if f2 <= 0 or trade["qty"] <= 0:
                    acted.append(trade)
                    continue

            # --- TP2 ---
            fill = self._tp_fill(trade, 2, high, low, last_price, exchange_client, position_qty)
            if fill is not None:
                self._apply_tp_fill(trade, 2, fill, exchange_client)
                progressed = True

            if progressed:
                acted.append(trade)

        self._save_state()
        return acted

    def trail_live_stops(self, symbol: str, high, low, exchange_client):
        """LIVE: ratchet the stop on runners (trades past TP2) to
        TRAIL_R_MULTIPLE x risk behind the best price seen so far."""
        if high is None or low is None:
            return
        open_trades = self.state.get("open_trades", [])
        trailing = [
            t for t in open_trades
            if t["symbol"] == symbol and self._trailing_armed(t) and t.get("qty", 0) > 0
        ]
        for trade in trailing:
            is_long = trade["position_side"].upper() == "LONG"
            if is_long:
                candidate = high - trade["trail_distance"]
                new_stop = max(trade["stop_loss"], candidate)
                if trade.get("breakeven_pending"):
                    new_stop = max(new_stop, trade["entry_price"])
            else:
                candidate = low + trade["trail_distance"]
                new_stop = min(trade["stop_loss"], candidate)
                if trade.get("breakeven_pending"):
                    new_stop = min(new_stop, trade["entry_price"])
            if new_stop == trade["stop_loss"]:
                continue
            if self._move_stops(trade, exchange_client, new_stop, [trade["qty"]]):
                trade.pop("breakeven_pending", None)
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
        dry_run_fee_pct_per_side for why. LIVE P/L uses the fees BingX
        reports on the actual fills instead (see _book_live_exit)."""
        return (qty * entry_price + qty * exit_price) * self.cfg.dry_run_fee_pct_per_side

    def _close_dry_run_trade(self, trade: dict, exit_price: float, qty: float, reason: str):
        is_long = trade["position_side"].upper() == "LONG"
        direction = 1 if is_long else -1
        gross_pnl = (exit_price - trade["entry_price"]) * direction * qty
        fee = self._estimate_round_trip_fee(qty, trade["entry_price"], exit_price)
        pnl = gross_pnl - fee
        self._record_realized_pnl(pnl)
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
            "fee": round(fee, 6),
        })

    def _take_dry_run_partial(self, trade: dict, symbol: str, stage: int = 1) -> bool:
        """Simulate TP1 (stage=1) or TP2 (stage=2) on a DRY_RUN trade at the
        exact TP price, net of the estimated fee. Returns True if this
        closed the whole remaining position (only possible at TP2 when
        TP1 + TP2 fractions add up to 100%)."""
        f1, f2 = self._tp_fractions()
        initial_qty = trade.get("initial_qty", trade["qty"])
        if stage == 1:
            exit_price = trade["partial_tp_price"]
            part_qty = round(min(initial_qty * f1, trade["qty"]), 8)
        else:
            exit_price = trade["tp2_price"]
            part_qty = round(min(initial_qty * f2, trade["qty"]), 8)
        remaining_qty = round(trade["qty"] - part_qty, 8)

        if stage == 2 and remaining_qty <= 0:
            self._close_dry_run_trade(
                trade, exit_price=exit_price, qty=trade["qty"],
                reason=f"TP2 hit ({trade.get('tp2_source') or 'swing level'}) -- position fully closed",
            )
            return True

        is_long = trade["position_side"].upper() == "LONG"
        direction = 1 if is_long else -1
        gross_pnl = (exit_price - trade["entry_price"]) * direction * part_qty
        fee = self._estimate_round_trip_fee(part_qty, trade["entry_price"], exit_price)
        pnl = gross_pnl - fee
        self._record_realized_pnl(pnl)

        if stage == 1:
            self._journal_partial(trade, "DRY_RUN", part_qty, exit_price, trade["entry_price"],
                                  self._tp1_trigger_text(trade), pnl, fee)
            trade["qty"] = remaining_qty
            trade["stop_loss"] = trade["entry_price"]
            trade["partial_taken"] = True
            if f2 <= 0:
                trade["tp2_taken"] = True
                trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple
        else:
            self._journal_partial(trade, "DRY_RUN", part_qty, exit_price, trade["stop_loss"],
                                  self._tp2_trigger_text(trade), pnl, fee)
            trade["qty"] = remaining_qty
            trade["tp2_taken"] = True
            trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple
        return False

    def reconcile_dry_run_exits(self, symbol: str, high: float, low: float) -> list:
        """DRY_RUN: walk each paper trade on symbol through its exit plan
        using the latest candle's high/low:

        stage 0 (nothing taken)  -> stop-loss (checked first, conservative)
                                    or TP1
        stage 1 (TP1 taken)      -> TP2, or the breakeven stop
        stage 2 (TP2 taken)      -> trailing stop, ratcheted each cycle

        A candle that runs through TP1 and TP2 takes both. The breakeven
        stop and the trailing stop aren't checked on the same candle that
        moved/armed them (same as before), since the candle's high/low
        can't tell which came first.
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
            stage_changed_this_candle = False

            # --- stage 0: initial stop-loss or TP1 ---
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
                tp1 = trade["partial_tp_price"]
                hit_tp1 = (high >= tp1) if is_long else (low <= tp1)
                if not hit_tp1:
                    remaining.append(trade)
                    continue
                self._take_dry_run_partial(trade, symbol, stage=1)
                changed = True
                stage_changed_this_candle = True

            # --- stage 1: TP2 or breakeven stop ---
            if not self._trailing_armed(trade):
                tp2 = trade["tp2_price"]
                hit_tp2 = (high >= tp2) if is_long else (low <= tp2)
                if hit_tp2:
                    fully_closed = self._take_dry_run_partial(trade, symbol, stage=2)
                    changed = True
                    if fully_closed:
                        closed.append(trade)
                    else:
                        remaining.append(trade)
                    continue
                if stage_changed_this_candle:
                    remaining.append(trade)
                    continue
                sl = trade["stop_loss"]
                hit_sl = (low <= sl) if is_long else (high >= sl)
                if hit_sl:
                    self._close_dry_run_trade(
                        trade, exit_price=sl, qty=trade["qty"],
                        reason="breakeven stop hit (after TP1, before TP2)",
                    )
                    closed.append(trade)
                    changed = True
                    continue
                remaining.append(trade)
                continue

            if stage_changed_this_candle:
                # TP2 disabled (TP2_FRACTION=0): trailing armed at TP1 this
                # candle -- start trailing from the next cycle.
                remaining.append(trade)
                continue

            # --- stage 2: trailing runner ---
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
        existing_notional: float = 0.0,
    ) -> tuple[float, int]:
        """Fixed-margin sizing: every trade commits cfg.margin_per_trade_usd
        of margin (not a fraction of equity), and LEVERAGE IS SOLVED FOR
        per trade from the stop-loss distance so the trade's dollar risk
        hits cfg.risk_pct_per_trade of equity (hard-capped at 3% by
        max_risk_pct_per_trade_hard_cap regardless of what
        RISK_PCT_PER_TRADE is set to) -- a tighter stop (smaller
        stop_distance_pct) gets MORE leverage to reach that risk target, a
        wider stop gets LESS. Leverage is floored to an integer (so actual
        risk never exceeds the target, only ever falls a bit short of it)
        and clamped to at most safe_leverage() -- the lesser of LEVERAGE and
        the 10x hard cap -- so an extremely tight stop can't demand more
        leverage than the account is allowed to use; the trade simply
        risks less than the target in that case. If even 1x leverage on
        the fixed margin would already risk more than the target (a very
        wide stop relative to the margin), the trade is skipped entirely
        (returns qty=0) rather than forcing 1x leverage and exceeding the
        risk cap.

        The margin-utilization cap from before is kept as an independent
        backstop on TOTAL exposure across every open position combined
        (LIVE + DRY_RUN): even though each trade's own margin is fixed at
        cfg.margin_per_trade_usd, this still caps the combined notional of
        every open position so several concurrent trades (when
        MAX_CONCURRENT_POSITIONS > 1) can't collectively demand more
        margin than max_margin_utilization_pct of the account's
        theoretical capacity at the 10x hard-cap leverage.

        Returns (qty, leverage) -- caller must call exchange_client.
        set_leverage(symbol, leverage) with the leverage this method
        picked before submitting the order, since it's chosen per trade,
        not read from a fixed config value. Caller is still responsible
        for rounding qty to the exchange's amount precision/step size
        before submitting."""
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0 or entry <= 0 or equity <= 0:
            return 0.0, 1
        stop_distance_pct = stop_distance / entry
        if stop_distance_pct <= 0:
            return 0.0, 1

        margin = self.cfg.margin_per_trade_usd
        target_risk_dollars = equity * self.cfg.risk_pct_per_trade
        leverage_ceiling = self.safe_leverage()
        required_leverage = target_risk_dollars / (margin * stop_distance_pct)
        leverage = min(leverage_ceiling, math.floor(required_leverage))
        if leverage < 1:
            # Even 1x leverage on the fixed margin would already risk more
            # than the target -- skip the trade rather than silently
            # accepting more risk than the cap allows (forcing leverage up
            # to 1 here would violate the 3% hard cap for a wide stop).
            return 0.0, 1

        qty = (margin * leverage) / entry

        max_total_notional = (
            equity * self.cfg.max_leverage_hard_cap * self.cfg.max_margin_utilization_pct
        )
        available_notional = max(max_total_notional - existing_notional, 0.0)
        qty = min(qty, available_notional / entry)
        return max(qty, 0.0), leverage

    def safe_leverage(self) -> int:
        """Leverage CEILING: the auto-sizer in position_size() never
        exceeds this -- min(LEVERAGE, the 10x hard cap). Kept as a knob
        so you can dial back how much leverage the auto-sizer is allowed
        to reach for, even though the exact leverage used per trade is
        now computed from each trade's own stop-loss distance rather
        than applied directly."""
        return min(self.cfg.leverage, self.cfg.max_leverage_hard_cap)

    @staticmethod
    def new_trade_id() -> str:
        return uuid.uuid4().hex[:12]
