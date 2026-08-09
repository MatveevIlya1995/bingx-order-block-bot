"""
Risk management: position sizing, leverage cap, and a daily-loss circuit
breaker, plus a simple CSV trade journal.

The circuit breaker compares CURRENT account equity (fetched fresh from
BingX every cycle) against the equity recorded at the start of the UTC
day. This is deliberately simple and exchange-agnostic: it doesn't try to
parse individual closed-trade fills (whose exact field names vary and
couldn't be verified against live BingX data from this environment) --
it just asks "is my balance down too much vs. this morning?" every time
it runs, which also catches large *unrealized* drawdowns on an open
position, not only realized losses. State persists to a small JSON file
so it survives across separate scheduled-task invocations, which each
run as a fresh process.
"""
import csv
import json
import os
import datetime


class RiskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.cfg.state_path):
            with open(self.cfg.state_path) as f:
                return json.load(f)
        return {
            "date": None,
            "day_start_equity": None,
        }

    def _save_state(self):
        os.makedirs(os.path.dirname(self.cfg.state_path) or ".", exist_ok=True)
        with open(self.cfg.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _roll_day_if_needed(self, equity: float):
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if self.state.get("date") != today:
            self.state["date"] = today
            self.state["day_start_equity"] = equity
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

    def log_trade(self, row: dict):
        """Appends one row to the CSV trade journal (data/trades.csv by
        default), creating the header if the file is new. This is a
        lightweight local record of what the bot *attempted* -- BingX's
        own order/position history is always the authoritative source
        for actual fills and realized P&L."""
        path = self.cfg.trades_csv_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        file_exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

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
