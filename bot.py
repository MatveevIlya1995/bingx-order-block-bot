"""
Entry point. Designed to be invoked ONCE PER CYCLE by a scheduler (cron,
GitHub Actions, systemd timer, etc.) running somewhere with real internet
access.

Each run:
  1. reads account equity and open positions from BingX
  2. checks the daily-loss circuit breaker
  3. fetches recent candles and looks for a fresh order-block retest signal
  4. sizes the position by fixed fractional risk
  5. if DRY_RUN=true (the default), logs what it WOULD do and stops there
  6. if DRY_RUN=false, sets leverage and submits entry + stop-loss + take-profit
"""
import sys
import traceback

from config import load_config
from exchange_client import ExchangeClient
from strategy import ohlcv_to_df, detect_order_blocks, generate_signal
from risk_manager import RiskManager


def run_once():
    cfg = load_config()

    if not cfg.api_key or not cfg.api_secret:
        print("[bot] BINGX_API_KEY / BINGX_API_SECRET are not set. Nothing to do.")
        return

    client = ExchangeClient(cfg)
    risk = RiskManager(cfg)

    equity = client.fetch_balance_usdt()
    if equity <= 0:
        print("[bot] Could not read a positive USDT balance -- aborting this cycle.")
        return

    if risk.circuit_breaker_tripped(equity):
        print("[bot] Daily loss circuit breaker is tripped. No new entries today.")
        return

    open_positions = client.fetch_open_positions()
    if len(open_positions) >= cfg.max_concurrent_positions:
        print(f"[bot] Already at max concurrent positions ({len(open_positions)}). Skipping.")
        return

    raw = client.fetch_ohlcv()
    df = ohlcv_to_df(raw)
    obs = detect_order_blocks(df, n=cfg.fractal_n)
    signal = generate_signal(df, obs, reward_risk_ratio=cfg.reward_risk_ratio)

    if not signal:
        print("[bot] No order-block retest signal this cycle.")
        return

    qty = risk.position_size(equity, signal["entry"], signal["stop_loss"])
    if qty <= 0:
        print("[bot] Computed position size was zero -- skipping.")
        return

    print(
        f"[bot] Signal: {signal['side'].upper()} {cfg.symbol} qty~{qty:.6f} "
        f"entry~{signal['entry']:.2f} SL={signal['stop_loss']:.2f} TP={signal['take_profit']:.2f}"
    )

    import datetime as _dt

    journal_row = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mode": "DRY_RUN" if cfg.dry_run else "LIVE",
        "symbol": cfg.symbol,
        "side": signal["side"].upper(),
        "qty": round(qty, 6),
        "entry": round(signal["entry"], 4),
        "stop_loss": round(signal["stop_loss"], 4),
        "take_profit": round(signal["take_profit"], 4),
        "equity_at_signal": round(equity, 2),
    }

    if cfg.dry_run:
        risk.log_trade(journal_row)
        print("[bot] DRY_RUN is on -- no order sent (logged to trade journal). Set DRY_RUN=false only once you've validated this.")
        return

    client.set_leverage(risk.safe_leverage())
    client.create_market_order_with_stops(
        side=signal["side"],
        amount=qty,
        stop_loss_price=signal["stop_loss"],
        take_profit_price=signal["take_profit"],
        position_side=signal["position_side"],
    )
    risk.log_trade(journal_row)
    print("[bot] Order submitted with attached stop-loss and take-profit, logged to trade journal.")


if __name__ == "__main__":
    try:
        run_once()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
