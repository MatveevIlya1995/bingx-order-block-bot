"""
Entry point. Designed to be invoked ONCE PER CYCLE by a scheduler (cron,
GitHub Actions, systemd timer, etc.) running somewhere with real internet
access.

Each run:
1. reads account equity and open positions from BingX
2. reconciles any previously-open trade that has since closed, logging
   its exit price/time/duration/realized P&L to the trade journal
3. checks the daily-loss circuit breaker
4. fetches recent candles and looks for a fresh order-block retest signal
5. sizes the position by fixed fractional risk
6. if DRY_RUN=true (the default), logs what it WOULD do and stops there
7. if DRY_RUN=false, sets leverage and submits entry + stop-loss +
   take-profit, and starts tracking the position so a later cycle can
   log its exit
"""
import datetime
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

    open_positions = client.fetch_open_positions()

    # Reconcile closed trades first and unconditionally, so exits get
    # logged promptly (every ~15 min) regardless of what happens below.
    closed = risk.reconcile_closed_trades(open_positions, client)
    for trade in closed:
        print(f"[bot] Detected closed trade {trade['trade_id']} ({trade['symbol']} {trade['position_side']}).")

    if risk.circuit_breaker_tripped(equity):
        print("[bot] Daily loss circuit breaker is tripped. No new entries today.")
        return

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

    ob = signal["order_block"]
    formed_ts = df.iloc[ob["formed_idx"]]["ts"]
    bos_ts = df.iloc[ob["bos_idx"]]["ts"]
    trigger = (
        f"{ob['type']} order block retest: zone {ob['bottom']:.4f}-{ob['top']:.4f} "
        f"(formed {formed_ts:%Y-%m-%d %H:%M} UTC, break of structure confirmed "
        f"{bos_ts:%Y-%m-%d %H:%M} UTC)"
    )

    trade_id = risk.new_trade_id()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    journal_row = {
        "trade_id": trade_id,
        "timestamp_utc": now_iso,
        "mode": "DRY_RUN" if cfg.dry_run else "LIVE",
        "symbol": cfg.symbol,
        "side": signal["side"].upper(),
        "position_side": signal["position_side"],
        "qty": round(qty, 6),
        "price": round(signal["entry"], 4),
        "stop_loss": round(signal["stop_loss"], 4),
        "take_profit": round(signal["take_profit"], 4),
        "equity": round(equity, 2),
        "trigger": trigger,
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
    risk.track_open_trade({
        "trade_id": trade_id,
        "entry_time_utc": now_iso,
        "mode": "LIVE",
        "symbol": cfg.symbol,
        "side": signal["side"].upper(),
        "position_side": signal["position_side"],
        "qty": round(qty, 6),
        "entry_price": round(signal["entry"], 4),
        "stop_loss": round(signal["stop_loss"], 4),
        "take_profit": round(signal["take_profit"], 4),
        "trigger": trigger,
    })
    print("[bot] Order submitted with attached stop-loss and take-profit, logged to trade journal.")


if __name__ == "__main__":
    try:
        run_once()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
