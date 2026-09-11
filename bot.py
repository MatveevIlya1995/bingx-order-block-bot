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
5. scans cfg.symbols, running a top-down multi-timeframe analysis on
   every symbol that isn't already holding a position (LIVE or DRY_RUN --
   see risk.occupied_symbols()) BEFORE ever looking for an entry (see
   analyze_higher_timeframes() below):
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

   A weekly-vs-daily disagreement is a special case: rather than always
   blocking the trade, analyze_higher_timeframes() first checks how much
   room price still has left to run toward the weekly chart's own last
   confirmed swing high/low (see strategy.weekly_room_pct()). If there's
   still at least cfg.weekly_daily_override_min_room_pct percent of room
   left in the weekly trend's direction, the weekly trend wins outright
   and the daily disagreement is overridden -- logged, not vetoed on.
   Below that threshold, the disagreement still blocks the trade as
   before.

   Every symbol that clears the cascade and has a valid entry signal (see
   step 6) is a "candidate" for this cycle. Candidates are then taken in
   two passes rather than in cfg.symbols order: first every candidate
   whose bias did NOT need the weekly/daily override, then -- only with
   whatever max_concurrent_positions slots and margin remain -- the
   candidates that only qualified because of the override. A conflicted
   daily trend overridden by a still-room-to-run weekly trend is a
   real, but lower-conviction, setup relative to one where every primary
   level already agreed; when MAX_CONCURRENT_POSITIONS lets several
   positions share one margin budget, this ordering means the budget
   goes to the higher-conviction setups first instead of whichever
   symbol happens to come first in cfg.symbols.
6. looks for a fresh order-block retest signal on the 15m entry
   timeframe (cfg.timeframe); the signal's direction must match the
   primary bias from step 5 (an all-neutral primary read imposes no
   directional constraint at all)
7. sizes each position with a FIXED margin (cfg.margin_per_trade_usd,
   $50 by default) and a per-trade LEVERAGE solved from the stop-loss
   distance so the trade's dollar risk hits cfg.risk_pct_per_trade of
   equity, hard-capped at cfg.max_risk_pct_per_trade_hard_cap (3%)
   regardless of what RISK_PCT_PER_TRADE is set to, and never exceeding
   the leverage ceiling (min(cfg.leverage, the 10x hard cap)) -- a very
   tight stop-loss then risks LESS than the 3% target instead of using
   more leverage than allowed. Separately, the combined notional of
   every open position (plus this one) is still capped at
   cfg.max_margin_utilization_pct of the account's theoretical capacity
   at the 10x hard-cap leverage -- see risk_manager.position_size() and
   config.py for why both caps exist. If the resulting size is still
   non-zero but its notional falls below cfg.min_notional_usdt (this
   happens when the margin-utilization cap above bites hard, e.g.
   several concurrent positions have already claimed most of the shared
   margin budget), the signal is skipped entirely rather than opening a
   token-sized position that can't meaningfully win or lose -- see the
   note on min_notional_usdt in config.py for why that matters.
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
    weekly_room_pct,
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

    Weekly/daily disagreement override: if combine_htf_bias() comes back
    "conflict" and the primary set is exactly weekly+daily, this checks
    strategy.weekly_room_pct() -- how far price still has to run, in the
    weekly trend's direction, before it reaches the weekly chart's own
    last confirmed swing high/low. If there's still at least
    cfg.weekly_daily_override_min_room_pct percent of room left, the
    weekly trend is used as the bias instead of blocking the trade: a
    daily trend that disagrees with a weekly trend that still has plenty
    of room to run is more likely a pullback inside the bigger move than
    an actual reversal. Below that room threshold the disagreement still
    blocks the trade, since there's more reason to think the weekly move
    itself is running out of room. A differently configured
    HTF_PRIMARY_TIMEFRAMES (not exactly weekly+daily) skips this override
    and keeps the original hard-conflict behavior.

    Returns (bias, detail):
        bias -- "bullish" / "bearish" / None (no primary-timeframe
                opinion, don't filter) / "conflict" (primary levels
                disagree and the weekly/daily override didn't apply --
                caller should skip the symbol this cycle)
        detail -- a short human-readable summary of every level's trend and
                  unmitigated order-block count, for logging/the journal
                  (primary levels are marked so it's clear which ones can
                  veto and which are for timing only), plus a note when the
                  weekly/daily override was evaluated
    """
    primary_trends = []
    detail_parts = []
    trend_by_tf = {}
    df_by_tf = {}
    for tf in cfg.htf_timeframes:
        raw = client.fetch_ohlcv(symbol, timeframe=tf, limit=cfg.trend_candle_limit)
        df = ohlcv_to_df(raw)
        trend = detect_trend(df, ema_period=cfg.trend_ema_period)
        obs = detect_order_blocks(df, n=cfg.fractal_n)
        unmitigated = sum(1 for ob in obs if not ob["mitigated"])
        is_primary = tf in cfg.primary_htf_timeframes
        if is_primary:
            primary_trends.append(trend)
        trend_by_tf[tf] = trend
        df_by_tf[tf] = df
        tag = "*" if is_primary else ""
        detail_parts.append(f"{tf}{tag}={trend}({unmitigated} OB)")

    bias = combine_htf_bias(primary_trends)
    note = ""

    if (
        bias == "conflict"
        and set(cfg.primary_htf_timeframes) == {"1w", "1d"}
        and "1w" in trend_by_tf
        and "1d" in trend_by_tf
    ):
        weekly_trend = trend_by_tf["1w"]
        daily_trend = trend_by_tf["1d"]
        if weekly_trend in ("bullish", "bearish") and daily_trend in ("bullish", "bearish"):
            current_price = float(df_by_tf["1d"].iloc[-1]["close"])
            room = weekly_room_pct(df_by_tf["1w"], weekly_trend, current_price, n=cfg.fractal_n)
            threshold = cfg.weekly_daily_override_min_room_pct
            if room is not None and room >= threshold:
                bias = weekly_trend
                note = (
                    f" | OVERRIDE: weekly={weekly_trend} vs daily={daily_trend} disagreed, but "
                    f"{room:.2f}% room left to the weekly "
                    f"{'high' if weekly_trend == 'bullish' else 'low'} "
                    f"(>= {threshold:.2f}% threshold) -- trading with the weekly trend."
                )
            else:
                room_desc = f"{room:.2f}%" if room is not None else "no confirmed weekly swing yet"
                note = (
                    f" | weekly={weekly_trend} vs daily={daily_trend} disagreement NOT overridden "
                    f"(room={room_desc}, need >= {threshold:.2f}%) -- no trade."
                )

    return bias, " ".join(detail_parts) + " (*=primary/directional, rest=timing only)" + note


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


def _find_candidate(client, cfg, symbol, df_cache):
    """Runs the higher-timeframe cascade and entry-signal scan for one
    symbol (steps 5-6 of the module docstring). Returns a dict describing
    a tradeable candidate, or None if this symbol has nothing to trade
    this cycle. Never sizes or opens anything -- that's left to the
    caller so candidates from every symbol can be collected and reordered
    before any of them actually claims margin (see run_once())."""
    bias = None
    htf_detail = ""
    if cfg.trend_filter_enabled:
        try:
            bias, htf_detail = analyze_higher_timeframes(client, cfg, symbol)
        except Exception as e:
            print(f"[bot] {symbol}: could not complete higher-timeframe analysis ({e}). Skipping.")
            return None
        if bias == "conflict":
            print(f"[bot] {symbol}: primary higher-timeframe trends disagree ({htf_detail}) -- no trade this cycle.")
            return None

    df = df_cache.get(symbol)
    if df is None:
        try:
            raw = client.fetch_ohlcv(symbol)
        except Exception as e:
            print(f"[bot] {symbol}: could not fetch candles ({e}). Skipping.")
            return None
        df = ohlcv_to_df(raw)

    obs = detect_order_blocks(df, n=cfg.fractal_n)
    signal = generate_signal(df, obs, reward_risk_ratio=cfg.reward_risk_ratio)

    if not signal:
        print(f"[bot] {symbol}: no order-block retest signal this cycle.")
        return None

    # --- Step 5 (part 2) & Step 6: never trade against the
    # PRIMARY higher-timeframe bias established above.
    if bias == "bullish" and signal["position_side"] == "SHORT":
        print(f"[bot] {symbol}: skipping SHORT signal -- primary higher-timeframe trend is bullish ({htf_detail}).")
        return None
    if bias == "bearish" and signal["position_side"] == "LONG":
        print(f"[bot] {symbol}: skipping LONG signal -- primary higher-timeframe trend is bearish ({htf_detail}).")
        return None

    return {
        "symbol": symbol,
        "htf_detail": htf_detail,
        "df": df,
        "signal": signal,
        "is_override": "OVERRIDE:" in htf_detail,
    }


def _try_open_candidate(client, risk, cfg, candidate, equity):
    """Sizes, journals and (if not DRY_RUN) submits one candidate's trade.
    Returns True if a position was actually opened (so the caller can
    update its own open_count/occupied_symbols bookkeeping), False if the
    candidate was skipped (zero/too-small size, or the zone was already
    traded)."""
    symbol = candidate["symbol"]
    htf_detail = candidate["htf_detail"]
    df = candidate["df"]
    signal = candidate["signal"]

    existing_notional = risk.total_open_notional()
    qty, leverage = risk.position_size(
        equity, signal["entry"], signal["stop_loss"],
        existing_notional=existing_notional,
    )
    if qty <= 0:
        print(f"[bot] {symbol}: computed position size was zero (risk-based size or margin cap) -- skipping.")
        return False

    notional = qty * signal["entry"]
    if notional < cfg.min_notional_usdt:
        print(
            f"[bot] {symbol}: computed size (qty~{qty:.6f}, ~${notional:.2f} notional) is below "
            f"MIN_NOTIONAL_USDT=${cfg.min_notional_usdt:.2f} -- skipping rather than opening a "
            f"token-sized trade that can't move the needle and would burn this zone for nothing."
        )
        return False

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
        return False

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
        return True

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
    print(f"[bot] {symbol}: order submitted with attached stop-loss/take-profit, logged.")
    return True


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

    # --- Steps 5-6: find every symbol with a tradeable candidate this
    # cycle, WITHOUT sizing or opening anything yet. Bucketed into
    # "clean" (bias resolved without needing the weekly/daily override)
    # and "override" (only tradeable because of it) so the execution
    # pass below can give the higher-conviction setups first claim on
    # the shared max_concurrent_positions slots and margin budget.
    clean_candidates = []
    override_candidates = []
    for symbol in cfg.symbols:
        if symbol in occupied_symbols:
            print(f"[bot] {symbol}: already have an open position (LIVE or DRY_RUN), skipping.")
            continue
        candidate = _find_candidate(client, cfg, symbol, df_cache)
        if candidate is None:
            continue
        if candidate["is_override"]:
            override_candidates.append(candidate)
        else:
            clean_candidates.append(candidate)

    # --- Steps 7-11: size and open candidates, clean ones first, then
    # override-only ones with whatever slots/margin remain.
    for candidate in clean_candidates + override_candidates:
        if open_count >= cfg.max_concurrent_positions:
            print(f"[bot] Reached max concurrent positions ({cfg.max_concurrent_positions}) mid-scan. Stopping.")
            break
        opened = _try_open_candidate(client, risk, cfg, candidate, equity)
        if opened:
            open_count += 1
            occupied_symbols.add(candidate["symbol"])


if __name__ == "__main__":
    try:
        run_once()
    except Exception:
        print("[bot] Unhandled exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
