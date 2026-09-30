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
reconcile_dry_run_exits) and, for LIVE trades, actually places the
partial-close and replacement-stop orders on the exchange (see
reconcile_live_exits and exchange_client.py) -- like the rest of this
bot's BingX order wiring, the live path has not been verified against a
real BingX account; validate it with a small real trade before trusting
it (see the README).
"""
import csv
import json
import math
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
        the UTC day. Halts NEW entries; does not touch existing
        positions/stops (those are managed by the exchange directly)."""
        self._roll_day_if_needed(equity)
        start = self.state.get("day_start_equity") or equity
        if start <= 0:
            return False
        loss_pct = (start - equity) / start
        if self.cfg.dry_run:
            # Paper trades never move the real account balance, so the
            # equity comparison above can't see them. Use today's
            # simulated realized P/L as well, so DRY_RUN trips the breaker
            # exactly when LIVE would.
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
                  when there is no such swing. Closes TP2_FRACTION of the
                  original position (30%). The stop stays at breakeven
                  between TP1 and TP2.
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
                self._record_realized_pnl(pnl)
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

    def _journal_partial(self, trade: dict, mode: str, qty: float, price: float,
                         stop_after: float, trigger: str, pnl=""):
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

    def reconcile_live_exits(self, symbol: str, high: float, low: float, exchange_client) -> list:
        """LIVE: advance each open trade on symbol through TP1 and TP2
        using the latest candle's high/low.

        TP1 hit  -> close PARTIAL_TP_FRACTION of the original size at market,
                    move the exchange stop for the rest to breakeven.
        TP2 hit  -> close TP2_FRACTION of the original size at market,
                    re-place the breakeven stop for the smaller remainder,
                    and arm the trailing stop (trail_live_stops()).
        A candle that runs through both levels advances both stages.
        """
        open_trades = self.state.get("open_trades", [])
        touched = [t for t in open_trades if t["symbol"] == symbol and not self._trailing_armed(t)]
        if not touched:
            return []

        f1, f2 = self._tp_fractions()
        acted = []
        for trade in touched:
            is_long = trade["position_side"].upper() == "LONG"
            close_side = "sell" if trade["side"].upper() == "BUY" else "buy"
            initial_qty = trade.get("initial_qty", trade["qty"])
            progressed = False

            # --- TP1 ---
            if not trade.get("partial_taken"):
                tp1 = trade["partial_tp_price"]
                if not ((high >= tp1) if is_long else (low <= tp1)):
                    continue
                tp1_qty = round(min(initial_qty * f1, trade["qty"]), 8)
                remaining_qty = round(trade["qty"] - tp1_qty, 8)
                if tp1_qty <= 0 or remaining_qty <= 0:
                    continue
                try:
                    exchange_client.close_partial_position(
                        symbol=symbol, side=close_side, amount=tp1_qty,
                        position_side=trade["position_side"],
                    )
                    exchange_client.replace_stop_order(
                        symbol=symbol, side=close_side, amount=remaining_qty,
                        new_stop_price=trade["entry_price"], position_side=trade["position_side"],
                    )
                except Exception as e:
                    print(f"[risk] {symbol}: TP1 order failed, leaving trade untouched: {e}")
                    continue
                self._journal_partial(trade, "LIVE", tp1_qty, tp1, trade["entry_price"],
                                      self._tp1_trigger_text(trade))
                trade["qty"] = remaining_qty
                trade["stop_loss"] = trade["entry_price"]
                trade["partial_taken"] = True
                progressed = True
                if f2 <= 0:
                    trade["tp2_taken"] = True
                    trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple
                    acted.append(trade)
                    continue

            # --- TP2 ---
            tp2 = trade["tp2_price"]
            if (high >= tp2) if is_long else (low <= tp2):
                tp2_qty = round(min(initial_qty * f2, trade["qty"]), 8)
                remaining_qty = round(trade["qty"] - tp2_qty, 8)
                try:
                    if remaining_qty <= 0:
                        # TP1 + TP2 fractions add up to the whole position:
                        # TP2 closes everything that's left.
                        exchange_client.close_partial_position(
                            symbol=symbol, side=close_side, amount=trade["qty"],
                            position_side=trade["position_side"],
                        )
                    else:
                        exchange_client.close_partial_position(
                            symbol=symbol, side=close_side, amount=tp2_qty,
                            position_side=trade["position_side"],
                        )
                        exchange_client.replace_stop_order(
                            symbol=symbol, side=close_side, amount=remaining_qty,
                            new_stop_price=trade["stop_loss"], position_side=trade["position_side"],
                        )
                except Exception as e:
                    print(f"[risk] {symbol}: TP2 order failed, will retry next cycle: {e}")
                    if progressed:
                        acted.append(trade)
                    continue
                self._journal_partial(trade, "LIVE", min(tp2_qty, trade["qty"]), tp2,
                                      trade["stop_loss"], self._tp2_trigger_text(trade))
                trade["qty"] = max(remaining_qty, 0.0)
                trade["tp2_taken"] = True
                trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple
                progressed = True

            if progressed:
                acted.append(trade)

        if acted:
            self._save_state()
        return acted

    def trail_live_stops(self, symbol: str, high: float, low: float, exchange_client):
        """LIVE: ratchet the stop on runners (trades past TP2) to
        TRAIL_R_MULTIPLE x risk behind the best price seen so far."""
        open_trades = self.state.get("open_trades", [])
        trailing = [
            t for t in open_trades
            if t["symbol"] == symbol and self._trailing_armed(t) and t.get("qty", 0) > 0
        ]
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
                                  self._tp1_trigger_text(trade), pnl)
            trade["qty"] = remaining_qty
            trade["stop_loss"] = trade["entry_price"]
            trade["partial_taken"] = True
            if f2 <= 0:
                trade["tp2_taken"] = True
                trade["trail_distance"] = trade["risk_per_unit"] * self.cfg.trail_r_multiple
        else:
            self._journal_partial(trade, "DRY_RUN", part_qty, exit_price, trade["stop_loss"],
                                  self._tp2_trigger_text(trade), pnl)
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
