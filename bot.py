"""
Entry point. Designed to be invoked ONCE PER CYCLE by a scheduler (cron,
GitHub Actions, systemd timer, etc.) running somewhere with real internet
access.

Each run:
1. reads account equity and open positions (across all watched symbols)
2. reconciles any previously tracked LIVE trades that have since closed
   on the exchange, logging an EXIT row and updating today's realized P/L
3. manages every currently-open trade (LIVE or DRY_RUN paper) against its
   symbol's freshly-fetched candle, in this order:
     a. DRY_RUN paper trades: simulates stop-loss / partial-take-profit /
        trailing-stop hits against the candle's high/low and logs the
        result -- this is the only way a DRY_RUN trade's outcome ever
        gets recorded (see risk_manager.reconcile_dry_run_exits)
     b. LIVE trades: checks for a partial-take-profit trigger and, if hit,
        actually places the partial-close + breakeven-stop orders on the
        exchange; then ratchets the trailing stop forward for any trade
        already past its partial (see risk_manager.reconcile_live_exits /
        trail_live_stops). A full stop-out is still detected the normal
        way, via reconcile_closed_trades() noticing the exchange position
        is gone.
4. checks the daily-loss circuit breaker (account-wide)
5. scans cfg.symbols in order; for each one not already holding a
   position (LIVE or DRY_RUN -- see risk.occupied_symbols()), runs a
   top-down multi-timeframe analysis BEFORE ever looking for an entry
   (see analyze_higher_timeframes() below):
     a. weekly trend + weekly order blocks
     b. daily trend + daily order blocks
     c. 4h trend + 4h order blocks
     d. 1h trend + 1h order blocks
   Only the PRIMARY levels (cfg.primary_htf_timeframes, weekly+daily by
   default) can veto a trade on disagreement -- the higher timeframe sets
   the direction. The remaining levels are still analyzed and logged
   every cycle, but are for entry *timing* context, not a veto: they
   never block a trade by themselves. This mirrors how top-down
   multi-timeframe trading is usually practiced -- a lower-timeframe
   pullback against the primary trend is normal, and is often exactly the
   retracement this strategy is trying to trade, not a reason to skip it.
6. looks for a fresh order-block retest signal on the 15m entry
   timeframe (cfg.timeframe); the signal's direction must match the
   primary bias from step 5 (an all-neutral primary read imposes no
   directional constraint at all)
7. sizes each position by fixed fractional risk, hard-capped at
   cfg.max_risk_pct_per_trade_hard_cap (3%) of equity, AND separately
   capped so the position's notional (plus whatever's already open)
   never requires more margin than cfg.max_margin_utilization_pct of the
   account's theoretical capacity at the configured leverage -- see
   risk_manager.position_size() and config.py for why both caps exist.
8. skips the signal if a trade (DRY_RUN or LIVE) was already taken in
   this order-block zone before -- the bot only ever takes one trade per
   zone, regardless of how many times it gets retested or re-detected
   under a different formed/BOS labelling across cycles
9. if DRY_RUN=true (the default), logs what it WOULD do and starts
   tracking it so step 3 can resolve it on a future cycle
10. if DRY_RUN=false, sets leverage and submits entry + stop-loss +
    take-profit (take-profit is always cfg.required_reward_risk_ratio
    times the stop distance -- see config.py); a partial-take-profit at
    cfg.partial_tp_r_multiple, breakeven stop and trailing runner then
    take over from step 3b once the trade is underway.
11. stops scanning once max_concurrent_positions is reached for the cycle
    (now counting DRY_RUN paper trades the same as LIVE positions, so
    paper-trading behavior is a faithful preview of live behavior)
"""
import datetime
import sys
import traceback

from config import load_config
from exchange_client import ExchangeClient
from strategy import (
    ohlcv_to_df,
    detect_order_blocks,
    generate_signal,
    detect_trend,
    combine_htf_bias,
)
from risk_manager import RiskManager


def analyze_higher_timeframes(client, cfg, symbol):
    """Top-down weekly -> daily -> 4h -> 1h analysis for one symbol.

    Fetches each timeframe in cfg.htf_timeframes (highest first) and
    computes its trend (EMA-based, see strategy.detect_trend) and its
    order blocks (so a level's structure is on record even though only
    the 15m entry timeframe's order blocks currently drive the trade
    itself -- see the README), then folds ONLY the PRIMARY levels'
    (cfg.primary_htf_timeframes) trend readings into one overall bias
    with combine_htf_bias(). Non-primary ("refinement") levels are
    analyzed and included in `detail` for visibility, but never
    contribute to `bias` -- they can't veto a trade.

    Returns (bias, detail):
      bias   -- "bullish" / "bearish" / None (no primary-timeframe
                opinion, don't filter) / "conflict" (primary levels
                disagree -- caller should skip the symbol this cycle)
      detail -- a short human-readable summary of every level's trend and
                unmitigated order-block count, for logging/the journal
                (primary levels are marked so it's clear which ones can
                veto and which are for timing only)
    """
    primary_trends = []
    detail_parts = []
    for tf in cfg.htf_timeframes:
        raw = client.fetch_ohlcv(symbol, timeframe=tf, limit=cfg.trend_candle_limit)
        df = ohlcv_to_df(raw)
        trend = detect_trend(df, ema_period=cfg.trend_ema_period)
        obs = detect_order_blocks(df, n=cfg.fractal_n)
        unmitigated = sum(1 for ob in obs if not ob["mitigated"])
        is_primary = tf in cfg.primary_htf_timeframes
        if is_primary:
            primary_trends.append(trend)
        tag = "*" if is_primary else ""
        detail_parts.append(f"{tf}{tag}={trend}({unmitigated} OB)")
    bias = combine_htf_bias(primary_trends)
    return bias, " ".join(detail_parts) + " (*=primary/directional, rest=timing only)"


def _manage_open_trades(client, risk, cfg, symbol):
    """Fetches this symbol's latest entry-timeframe candle once and uses
    it to resolve/advance every currently-tracked open trade on it,
    LIVE or DRY_RUN. Returns the freshly-fetched OHLCV dataframe so the
    caller can also use it for entry-signal scanning without fetching
    twice, or None if the fetch failed."""
    try:
        raw = client.fetch_ohlcv(symbol)
    except Exception as e:
        print(f"[bot] {symbol}: could not fetch candles for trade management ({e}).")
        return None
    df = ohlcv_to_df(raw)
    high = float(df.iloc[-1]["high"])
    low = float(df.iloc[-1]["low"])

    closed_dry_run = risk.reconcile_dry_run_exits(symbol, high=high, low=low)
    if closed_dry_run:
        print(f"[bot] {symbol}: closed {len(closed_dry_run)} DRY_RUN paper trade(s), logged EXIT row(s).")

    partial_live = risk.reconcile_live_exits(symbol, high=high, low=low, exchange_client=client)
    if partial_live:
        print(f"[bot] {symbol}: took partial take-profit on {len(partial_live)} LIVE trade(s), stop moved to breakeven.")
    risk.trail_live_stops(symbol, high=high, low=low, exchange_client=client)

    return df


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

    equity_warning = risk.check_equity_jump(equity)
    if equity_warning:
        print(f"[bot] WARNING: {equity_warning}")

    open_positions = client.fetch_open_positions()
    exchange_open_symbols = {p.get("symbol") for p in open_positions}

    closed_trades = risk.reconcile_closed_trades(open_positions, client)
    if closed_trades:
        print(f"[bot] Reconciled {len(closed_trades)} closed trade(s), logged EXIT row(s).")

    # Manage every currently-open trade (LIVE or DRY_RUN) against this
    # cycle's fresh candle -- partial take-profit, trailing, or (for
    # DRY_RUN) a full simulated stop/trail exit. Done for every occupied
    # symbol regardless of the circuit breaker, since managing/closing
    # existing risk should never be skipped.
    df_cache = {}
    for symbol in sorted(risk.occupied_symbols()):
        df_cache[symbol] = _manage_open_trades(client, risk, cfg, symbol)

    if risk.circuit_breaker_tripped(equity):
        print("[bot] Daily loss circuit breaker is tripped. No new entries today.")
        return

    # Recompute AFTER trade management above, since a full close/partial
    # this cycle can change what counts as occupied. Counting by symbol
    # (not by tracked-trade record) matches the existing simplification
    # elsewhere in this bot that assumes at most one open position per
    # symbol at a time.
    occupied_symbols = risk.occupied_symbols() | exchange_open_symbols
    open_count = len(occupied_symbols)

    print(
        f"[bot] Equity=${equity:.2f} | {open_count}/{cfg.max_concurrent_positions} "
        f"positions open ({sorted(occupied_symbols) or 'none'}) | watching {len(cfg.symbols)} symbols"
    )

    if open_count >= cfg.max_concurrent_positions:
        print("[bot] Already at max concurrent positions. Skipping scan this cycle.")
        return

    for symbol in cfg.symbols:
        if open_count >= cfg.max_concurrent_positions:
            print(f"[bot] Reached max concurrent positions ({cfg.max_concurrent_positions}) mid-scan. Stopping.")
            break

        if symbol in occupied_symbols:
            print(f"[bot] {symbol}: already have an open position (LIVE or DRY_RUN), skipping.")
            continue

        # --- Steps 1-2 & 5 (part 1): top-down weekly -> daily -> 4h -> 1h
        # trend cascade. Only the PRIMARY levels can veto this symbol for
        # the cycle before we even look at the entry timeframe.
        bias = None
        htf_detail = ""
        if cfg.trend_filter_enabled:
            try:
                bias, htf_detail = analyze_higher_timeframes(client, cfg, symbol)
            except Exception as e:
                print(f"[bot] {symbol}: could not complete higher-timeframe analysis ({e}). Skipping.")
                continue
            if bias == "conflict":
                print(f"[bot] {symbol}: primary higher-timeframe trends disagree ({htf_detail}) -- no trade this cycle.")
                continue

        df = df_cache.get(symbol)
        if df is None:
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

        # --- Step 5 (part 2) & Step 6: never trade against the
        # PRIMARY higher-timeframe bias established above.
        if bias == "bullish" and signal["position_side"] == "SHORT":
            print(f"[bot] {symbol}: skipping SHORT signal -- primary higher-timeframe trend is bullish ({htf_detail}).")
            continue
        if bias == "bearish" and signal["position_side"] == "LONG":
            print(f"[bot] {symbol}: skipping LONG signal -- primary higher-timeframe trend is bearish ({htf_detail}).")
            continue

        leverage = risk.safe_leverage()
        existing_notional = risk.total_open_notional()
        qty = risk.position_size(
            equity, signal["entry"], signal["stop_loss"],
            leverage=leverage, existing_notional=existing_notional,
        )
        if qty <= 0:
            print(f"[bot] {symbol}: computed position size was zero (risk-based size or margin cap) -- skipping.")
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
            f"{bos_ts:%Y-%m-%d %H:%M} UTC) | HTF: {htf_detail or 'trend filter disabled'}"
        )

        if risk.zone_already_traded(symbol, ob):
            print(f"[bot] {symbol}: a trade was already taken in this order-block zone -- skipping duplicate.")
            continue

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
            risk.remember_zone_trade(symbol, ob)
            risk.track_dry_run_trade({
                "trade_id": trade_id,
                "entry_time_utc": now_iso,
                "mode": "DRY_RUN",
                "symbol": symbol,
                "side": signal["side"].upper(),
                "position_side": signal["position_side"],
                "qty": round(qty, 6),
                "entry_price": round(signal["entry"], 4),
                "stop_loss": round(signal["stop_loss"], 4),
                "take_profit": round(signal["take_profit"], 4),
                "trigger": trigger,
            })
            print(f"[bot] {symbol}: DRY_RUN is on -- no order sent (logged to trade journal).")
            occupied_symbols.add(symbol)
            open_count += 1
            continue

        client.set_leverage(symbol, leverage)
        client.create_market_order_with_stops(
            symbol=symbol,
            side=signal["side"],
            amount=qty,
            stop_loss_price=signal["stop_loss"],
            take_profit_price=signal["take_profit"],
            position_side=signal["position_side"],
        )
        risk.log_trade(journal_row)
        risk.remember_zone_trade(symbol, ob)
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
        occupied_symbols.add(symbol)
        print(f"[bot] {symbol}: order submitted with attached stop-loss/take-profit, logged.")


if __name__ == "__main__":
    try:
        run_once()
    except Exception:
        print("[bot] Unhandled exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
