"""
Risk management: position sizing, leverage cap, and a daily-loss circuit
breaker. State (today's realized P&L, trade history) persists to a small
JSON file so it survives across separate scheduled-task invocations of the
bot, which each run as a fresh process.
"""
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
            "realized_pnl_today": 0.0,
            "trades": [],
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
            self.state["realized_pnl_today"] = 0.0
            self._save_state()

    def circuit_breaker_tripped(self, equity: float) -> bool:
        """True if today's realized losses have hit the configured cap.
        Halts NEW entries; does not touch existing positions/stops."""
        self._roll_day_if_needed(equity)
        start = self.state.get("day_start_equity") or equity
        if start <= 0:
            return False
        loss_pct = -self.state["realized_pnl_today"] / start
        return loss_pct >= self.cfg.max_daily_loss_pct

    def record_trade(self, pnl: float):
        self.state["realized_pnl_today"] = self.state.get("realized_pnl_today", 0.0) + pnl
        self.state.setdefault("trades", []).append(
            {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), "pnl": pnl}
        )
        self._save_state()

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
