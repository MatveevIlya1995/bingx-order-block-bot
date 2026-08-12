"""
Entry point. Designed to be invoked ONCE PER CYCLE by a scheduler (cron,
GitHub Actions, systemd timer, etc.) running somewhere with real internet
access.

Each run:
  1. reads account equity and open positions (across all watched symbols)
  2. checks the daily-loss circuit breaker (account-wide)
  3. scans cfg.symbols in order; for each one not already holding a
     position, looks for a fresh order-block retest signal
  4. sizes each position by fixed fractional risk
  5. if DRY_RUN=true (the default), logs what it WOULD do and moves on
  6. if DRY_RUN=false, sets leverage and submits entry + stop-loss + take-profit
  7. stops scanning once max_concurrent_positions is reached for the cycle
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

    if risk.circuit_breaker_tripped(equity):
        print("[bot] Daily loss circuit breaker is tripped. No new entries today.")
        return

    open_positions = client.fetch_open_positions()
    open_symbols = {p.get("symbol") for p in open_positions}
    open_count = len(open_positions)

    print(
        f"[bot] Equity=${equity:.2f} | {open_count}/{cfg.max_concurrent_positions} "
        f"positions open ({sorted(open_symbols) or 'none'}) | watching {len(cfg.symbols)} symbols"
    )

    if open_count >= cfg.max_concurrent_positions:
        print("[bot] Already at max concurrent positions. Skipping scan this cycle.")
        return

    for symbol in cfg.symbols:
        if open_count >= cfg.max_concurrent_positions:
            print(f"[bot] Reached max concurrent positions ({cfg.max_concurrent_positions}) mid-scan. Stopping.")
            break

        if symbol in open_symbols:
            print(f"[bot] {symbol}: already have an open position, skipping.")
            continue

        try:
            raw = client.fetch_ohlcv(symbol)
        except Exception as e:
            print(f"[bot] {symbol}: could not fetch candles ({e}). Skipping.")
            continue

        df = ohlcv_to_df(raw)
        obs = detect_order_blocks(df, n=cfg.fractal_n)
        signal = generate_signal(df, obs, reward_risk_ratio=cfg.reward_risk_ratio)

        if not signal:
            print(f"[bot] {symbol}: no order-block retest signal this cycle.")
            continue

        qty = risk.position_size(equity, signal["entry"], signal["stop_loss"])
        if qty <= 0:
            print(f"[bot] {symbol}: computed position size was zero -- skipping.")
            continue

        print(
            f"[bot] {symbol}: signal {signal['side'].upper()} qty~{qty:.6f} "
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
            "symbol": symbol,
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
            print(f"[bot] {symbol}: DRY_RUN is on -- no order sent (logged to trade journal).")
            continue

        client.set_leverage(symbol, risk.safe_leverage())
        client.create_market_order_with_stops(
            symbol=symbol,
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
            "symbol": symbol,
            "side": signal["side"].upper(),
            "position_side": signal["position_side"],
            "qty": round(qty, 6),
            "entry_price": round(signal["entry"], 4),
            "stop_loss": round(signal["stop_loss"], 4),
            "take_profit": round(signal["take_profit"], 4),
            "trigger": trigger,
        })
        open_count += 1
        open_symbols.add(symbol)
        print(f"[bot] {symbol}: order submitted with attached stop-loss/take-profit, logged.")


if __name__ == "__main__":
    try:
        run_once()
    except Exception:
        print("[bot] Unhandled exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
