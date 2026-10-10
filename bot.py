"""
Entry point. Designed to be invoked ONCE PER CYCLE by a scheduler (cron,
GitHub Actions, systemd timer, etc.) running somewhere with real internet
access.

Entries are RESTING LIMIT ORDERS placed inside unmitigated order blocks,
not market orders fired when a candle happens to touch a zone: the limit
sits at the zone edge price reaches first (top of a bullish OB, bottom of
a bearish OB) with the stop-loss attached, so the trade is executed inside
the order block whenever price gets there -- even between two runs.

Each run:
1. reads account equity and open positions (across all watched symbols)
2. checks every resting limit order the bot is tracking: a filled order
   becomes a normal open trade (journalled ENTRY, zone remembered as
   traded) and every other resting order on that symbol is cancelled --
   one position per symbol; a cancelled/expired order is dropped
3. reconciles any previously tracked LIVE trades that have since closed
   on the exchange, logging an EXIT row (and PARTIAL_EXIT rows for
   take-profits that filled in between), updating today's realized P/L
   and cancelling whatever exit orders the trade left on the exchange
4. manages every currently-open trade: LIVE trades get their exits put on
   BingX once (one stop per part of the position, TP1 and TP2 as resting
   reduce-only take-profit orders -- see
   risk_manager.arm_live_exit_orders), fills BingX made since the last
   run are journalled at their real price and fee, the stop moves to
   breakeven after TP1 and trails after TP2 (reconcile_live_exits /
   trail_live_stops). A take-profit that could not be placed is closed at
   market by the run that sees price reach it, as before. DRY_RUN paper
   trades are simulated against the freshly-fetched candle
   (reconcile_dry_run_exits).
5. checks the daily-loss circuit breaker (account-wide); if tripped, every
   resting limit order is cancelled and no new ones are placed
6. for every symbol without an open position, runs the top-down
   multi-timeframe analysis (see analyze_higher_timeframes(): daily + 4h
   are primary/directional with the room override between them, 1h + 15m
   are timing context only) and then picks the setups for this symbol
   (strategy.limit_setups()): the LAST TWO valid, unmitigated order
   blocks on the 5m entry timeframe in the primary bias direction.
   "Valid" = the order block sits at a strong swing -- a swing low/high
   that went on to cause a BOS or CHoCH (the five rules for those are at
   the top of strategy.py); "unmitigated" = price hasn't
   visited the zone and left it, and no completed candle closed through
   it (see strategy._apply_mitigation()). Every setup is then tagged with
   its smart-money context (BOS or CHoCH, liquidity sweep, displacement,
   fair value gap, premium/discount, hour -- strategy.add_smc_context()),
   which goes into the log and the journal. The optional entry filters
   (ENTRY_SYMBOLS, MIN_STOP_DISTANCE_PCT, HTF_TIMING_AGREEMENT and the
   smart-money filters OB_STRUCTURE, REQUIRE_LIQUIDITY_SWEEP,
   MIN_DISPLACEMENT_ATR, REQUIRE_FVG, PREMIUM_DISCOUNT_FILTER,
   ENTRY_HOURS_UTC -- all off by default) then remove setups; they never
   add any.
7. syncs that symbol's resting orders with those setups: orders whose
   order block is no longer one of them (mitigated without a fill,
   invalidated, bias changed, replaced by newer order blocks, zone already
   traded) are cancelled; setups without an order get one.
   The stop-loss goes beyond the last swing -- the swing low/high the
   order block belongs to (STOP_PLACEMENT, strategy.stop_prices()).
8. sizes each new order exactly like before (fixed margin, per-trade
   leverage solved from the stop distance to hit RISK_PCT_PER_TRADE,
   10x ceiling, shared margin-utilization cap -- see
   risk_manager.position_size()); orders below MIN_NOTIONAL_USDT are
   skipped.
9. slots: a symbol with resting orders reserves ONE of the
   MAX_CONCURRENT_POSITIONS slots (only one of its orders can become a
   position), and its largest order's notional is reserved against the
   margin cap. Symbols already holding resting orders keep their slot;
   free slots go to clean (no-override) setups first, then to setups
   that only qualified through the primary-level room override.
10. DRY_RUN=true: orders are simulated -- a resting paper order "fills"
    once a candle since it was placed traded through its limit price, and
    then follows the same paper-trade management as before.
"""
import datetime
import sys
import traceback

import pandas as pd

from config import load_config, override_levels, smc_filter_settings
from exchange_client import ExchangeClient
from strategy import (
    ohlcv_to_df,
    detect_order_blocks,
    limit_setups,
    detect_trend,
    combine_htf_bias,
    weekly_room_pct,
    next_swing_beyond,
    apply_entry_filters,
    entry_symbol_allowed,
    add_smc_context,
    describe_smc,
    structure_state,
)
from risk_manager import RiskManager

# How many of the most recent valid, unmitigated order blocks get a
# resting limit order per symbol.
MAX_LIMIT_ORDERS_PER_SYMBOL = 2

# STOP_PLACEMENT, as written into the journal's trigger column.
STOP_NAMES = {"order_block": "order block", "swing": "last swing", "previous_swing": "previous swing"}


def analyze_higher_timeframes(client, cfg, symbol):
    """Top-down daily -> 4h -> 1h -> 15m analysis for one symbol.

    Fetches each timeframe in cfg.htf_timeframes (highest first) and
    computes its trend (EMA-based, see strategy.detect_trend) and its
    order blocks, then folds ONLY the PRIMARY levels'
    (cfg.primary_htf_timeframes) trend readings into one overall bias
    with combine_htf_bias(). Non-primary ("refinement") levels are
    analyzed and included in `detail` for visibility, but never
    contribute to `bias` -- they can't veto a trade.

    Room override: if combine_htf_bias() comes back "conflict" and there
    are exactly two primary levels (config.override_levels: daily and 4h
    by default), this checks strategy.weekly_room_pct() -- how far price
    still has to run, in the HIGHER level's trend direction, before it
    reaches that chart's own last confirmed swing high/low. If there's still at least
    cfg.weekly_daily_override_min_room_pct percent of room left, the
    higher level's trend is used as the bias instead of blocking the trade.
    Below that room threshold the disagreement still blocks the trade.

    Returns (bias, detail, trend_by_tf):
        bias        -- "bullish" / "bearish" / None (no primary-timeframe
                       opinion, don't filter) / "conflict" (skip the symbol)
        detail      -- a short human-readable summary of every level's trend
                       and unmitigated order-block count, plus a note when
                       the room override was evaluated
        trend_by_tf -- {timeframe: trend} for every level, used by the
                       optional HTF_TIMING_AGREEMENT entry filter
    """
    primary_trends = []
    detail_parts = []
    trend_by_tf = {}
    df_by_tf = {}
    for tf in cfg.htf_timeframes:
        raw = client.fetch_ohlcv(symbol, timeframe=tf, limit=cfg.trend_candle_limit)
        df = ohlcv_to_df(raw)
        trend = detect_trend(df, ema_period=cfg.trend_ema_period)
        obs = detect_order_blocks(df, n=cfg.fractal_n, confirmed_swings=cfg.structure_confirmed_swings,
                                  strong_swings_only=cfg.require_strong_swing)
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

    pair = override_levels(cfg)
    if bias == "conflict" and pair and pair[0] in trend_by_tf and pair[1] in trend_by_tf:
        top, lower = pair
        top_trend = trend_by_tf[top]
        lower_trend = trend_by_tf[lower]
        if top_trend in ("bullish", "bearish") and lower_trend in ("bullish", "bearish"):
            current_price = float(df_by_tf[lower].iloc[-1]["close"])
            room = weekly_room_pct(df_by_tf[top], top_trend, current_price, n=cfg.fractal_n)
            threshold = cfg.weekly_daily_override_min_room_pct
            if room is not None and room >= threshold:
                bias = top_trend
                note = (
                    f" | OVERRIDE: {top}={top_trend} vs {lower}={lower_trend} disagreed, but "
                    f"{room:.2f}% room left to the {top} "
                    f"{'high' if top_trend == 'bullish' else 'low'} "
                    f"(>= {threshold:.2f}% threshold) -- trading with the {top} trend."
                )
            else:
                room_desc = f"{room:.2f}%" if room is not None else f"no confirmed {top} swing yet"
                note = (
                    f" | {top}={top_trend} vs {lower}={lower_trend} disagreement NOT overridden "
                    f"(room={room_desc}, need >= {threshold:.2f}%) -- no trade."
                )

    return bias, " ".join(detail_parts) + " (*=primary/directional, rest=timing only)" + note, trend_by_tf


def _fetch_df(client, symbol, df_cache):
    df = df_cache.get(symbol)
    if df is not None:
        return df
    try:
        raw = client.fetch_ohlcv(symbol)
    except Exception as e:
        print(f"[bot] {symbol}: could not fetch candles ({e}).")
        return None
    df = ohlcv_to_df(raw)
    df_cache[symbol] = df
    return df


def _manage_open_trades(client, risk, cfg, symbol, df_cache, position_qty=None):
    """Resolves/advances every currently-tracked open trade on this symbol,
    LIVE or DRY_RUN.

    LIVE: makes sure the trade's exit orders are on the exchange (per-part
    stops + resting TP1/TP2, see RiskManager.arm_live_exit_orders), books
    take-profits BingX filled since the last run, and trails the runner's
    stop. The latest entry-timeframe candle is only needed for the DRY_RUN
    simulation, for the trailing stop, and for take-profits that have no
    resting order (the market-close fallback) -- so a failed candle fetch
    no longer stops the resting orders from being placed and checked.

    position_qty: size of the exchange position on this symbol (None if
    unknown), a cross-check for when a resting order can't be read."""
    df = _fetch_df(client, symbol, df_cache)
    if df is None:
        print(f"[bot] {symbol}: no candles for trade management this cycle.")
        high = low = last = None
    else:
        high = float(df.iloc[-1]["high"])
        low = float(df.iloc[-1]["low"])
        last = float(df.iloc[-1]["close"])

    if high is not None:
        closed_dry_run = risk.reconcile_dry_run_exits(symbol, high=high, low=low)
        if closed_dry_run:
            print(f"[bot] {symbol}: closed {len(closed_dry_run)} DRY_RUN paper trade(s), logged EXIT row(s).")

    risk.arm_live_exit_orders(symbol, exchange_client=client)
    partial_live = risk.reconcile_live_exits(
        symbol, high=high, low=low, exchange_client=client,
        last_price=last, position_qty=position_qty,
    )
    if partial_live:
        print(f"[bot] {symbol}: advanced TP1/TP2 on {len(partial_live)} LIVE trade(s) (stop at breakeven after TP1, trailing after TP2).")
    risk.trail_live_stops(symbol, high=high, low=low, exchange_client=client)


# ---------------------------------------------------------------------------
# resting limit orders
# ---------------------------------------------------------------------------

def _cancel_pending(client, risk, order, reason):
    """Cancels one resting limit order (on the exchange for LIVE) and drops
    it from state. Returns False if a LIVE cancel failed -- the order is
    then kept in state so the next run checks it again (it may have
    filled in the meantime)."""
    if order["mode"] == "LIVE":
        try:
            client.cancel_order(order["order_id"], order["symbol"])
        except Exception as e:
            print(f"[bot] {order['symbol']}: could not cancel limit order {order['order_id']} ({e}) -- will re-check next cycle.")
            return False
    risk.remove_pending_order(order["order_id"])
    risk.log_order_event("LIMIT_CANCELLED", order, reason)
    print(f"[bot] {order['symbol']}: cancelled {order['position_side']} limit @ {order['entry_price']:.6g} -- {reason}")
    return True


def _order_fee(o):
    """Fee BingX reported on a fetched order, as a positive USDT amount, or
    None when it isn't there (a commission of exactly 0 on a filled order
    is read as "not reported")."""
    try:
        cost = ((o or {}).get("fee") or {}).get("cost")
        if cost is None or float(cost) == 0:
            return None
        return abs(float(cost))
    except (TypeError, ValueError):
        return None


def _promote_fill(client, risk, cfg, order, qty, fill_price, fill_time_iso, entry_fee=None):
    """A resting limit order filled: start tracking it as a normal open
    trade (same TP1/TP2/trailing plan as before), journal the ENTRY and
    remember the zone as traded.

    entry_fee is the fee BingX reported for the entry fill (None if it
    didn't). It's journalled on the ENTRY row and kept on the trade, so
    every later exit can be booked net of its share of it."""
    symbol = order["symbol"]
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    trade = {
        "trade_id": order["trade_id"],
        "entry_time_utc": fill_time_iso or now_iso,
        "mode": order["mode"],
        "symbol": symbol,
        "side": order["side"].upper(),
        "position_side": order["position_side"],
        "qty": round(qty, 6),
        "entry_price": round(fill_price, 6),
        "stop_loss": order["stop_loss"],
        "take_profit": order["take_profit"],
        "trigger": order["trigger"] + f" | filled limit @ {fill_price:.6g}",
        "tp2_price": order.get("tp2_price"),
        "tp2_source": order.get("tp2_source"),
        "entry_fee": entry_fee,
    }
    risk.log_trade({
        "trade_id": order["trade_id"],
        "timestamp_utc": trade["entry_time_utc"],
        "mode": order["mode"],
        "symbol": symbol,
        "side": trade["side"],
        "position_side": trade["position_side"],
        "qty": trade["qty"],
        "price": trade["entry_price"],
        "stop_loss": round(order["stop_loss"], 6),
        "take_profit": round(order["take_profit"], 6),
        "equity": order.get("equity", ""),
        "trigger": trade["trigger"],
        "fee": round(entry_fee, 6) if entry_fee is not None else "",
    })
    risk.remember_zone_trade(symbol, order["ob"])
    risk.remove_pending_order(order["order_id"])
    if order["mode"] == "LIVE":
        risk.track_open_trade(trade)
    else:
        risk.track_dry_run_trade(trade)
    print(f"[bot] {symbol}: {order['position_side']} limit FILLED @ {fill_price:.6g} (qty {qty:.6f}) -- now managing as an open trade.")


def _dry_run_fill(order, df):
    """DRY_RUN: has any candle since the order was placed traded through
    its limit price? Returns the simulated fill price (the limit, or the
    candle's open if it gapped past the limit), or None."""
    if df is None or len(df) < 2:
        return None
    placed = pd.Timestamp(order["placed_time_utc"]).tz_convert("UTC").tz_localize(None)
    candle_len = df["ts"].iloc[-1] - df["ts"].iloc[-2]
    limit = order["entry_price"]
    is_long = order["position_side"] == "LONG"
    for _, row in df[df["ts"] + candle_len > placed].iterrows():
        hit = row["low"] <= limit if is_long else row["high"] >= limit
        if not hit:
            continue
        if row["ts"] >= placed:
            return min(limit, float(row["open"])) if is_long else max(limit, float(row["open"]))
        return limit
    return None


def _process_pending_fills(client, risk, cfg, df_cache):
    """Step 2: turn filled limit orders into open trades, cancel the
    other resting orders on a symbol once one of them fills, drop
    orders the exchange cancelled/expired."""
    for symbol in sorted(risk.pending_symbols()):
        filled_any = False
        for order in risk.pending_orders(symbol):
            if order["mode"] == "LIVE":
                try:
                    o = client.fetch_order(order["order_id"], symbol)
                except Exception as e:
                    print(f"[bot] {symbol}: could not check limit order {order['order_id']} ({e}).")
                    continue
                status = (o.get("status") or "").lower()
                filled = float(o.get("filled") or 0.0)
                if filled > 0:
                    if status == "open":
                        # Partially filled: keep what filled, cancel the rest,
                        # and make sure the stop covers exactly the filled size.
                        close_side = "sell" if order["side"].lower() == "buy" else "buy"
                        try:
                            client.cancel_order(order["order_id"], symbol)
                        except Exception as e:
                            print(f"[bot] {symbol}: could not cancel rest of partially-filled order ({e}).")
                        try:
                            client.replace_stop_order(
                                symbol=symbol, side=close_side, amount=filled,
                                new_stop_price=order["stop_loss"], position_side=order["position_side"],
                            )
                        except Exception as e:
                            print(f"[bot] {symbol}: WARNING could not re-place stop for the partial fill ({e}) -- CHECK THE STOP ON BINGX.")
                    fill_price = float(o.get("average") or o.get("price") or order["entry_price"])
                    ts = o.get("lastTradeTimestamp") or o.get("timestamp")
                    fill_iso = (
                        datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).isoformat()
                        if ts else None
                    )
                    _promote_fill(client, risk, cfg, order, filled, fill_price, fill_iso, _order_fee(o))
                    filled_any = True
                elif status in ("canceled", "cancelled", "expired", "rejected"):
                    risk.remove_pending_order(order["order_id"])
                    risk.log_order_event("LIMIT_CANCELLED", order, f"order {status} on the exchange")
                    print(f"[bot] {symbol}: limit order {order['order_id']} was {status} on the exchange -- dropped.")
            else:
                df = _fetch_df(client, symbol, df_cache)
                fill_price = _dry_run_fill(order, df)
                if fill_price is not None:
                    _promote_fill(client, risk, cfg, order, order["qty"], fill_price, None)
                    filled_any = True
        if filled_any:
            for other in risk.pending_orders(symbol):
                _cancel_pending(client, risk, other, "another order on this symbol filled (one position per symbol)")


def _find_setups(client, cfg, symbol, df_cache):
    """Higher-timeframe cascade + limit-order setups for one symbol.
    Returns (setups, htf_detail, is_override, df). setups is empty when
    there's nothing to have an order resting for this cycle."""
    if not entry_symbol_allowed(symbol, cfg.entry_symbols):
        print(f"[bot] {symbol}: not in ENTRY_SYMBOLS -- no new orders (open positions are still managed).")
        return [], "", False, None

    bias = None
    htf_detail = ""
    timing_trends = None
    if cfg.trend_filter_enabled:
        try:
            bias, htf_detail, trend_by_tf = analyze_higher_timeframes(client, cfg, symbol)
            timing_trends = [
                trend_by_tf[tf] for tf in cfg.htf_timeframes
                if tf not in cfg.primary_htf_timeframes and tf in trend_by_tf
            ]
        except Exception as e:
            print(f"[bot] {symbol}: could not complete higher-timeframe analysis ({e}).")
            return None, "", False, None
        if bias == "conflict":
            print(f"[bot] {symbol}: primary higher-timeframe trends disagree ({htf_detail}) -- no orders this cycle.")
            return [], htf_detail, False, None

    df = _fetch_df(client, symbol, df_cache)
    if df is None:
        return None, htf_detail, False, None

    obs = detect_order_blocks(df, n=cfg.fractal_n, confirmed_swings=cfg.structure_confirmed_swings,
                              strong_swings_only=cfg.require_strong_swing)
    setups = limit_setups(
        df, obs,
        max_setups=MAX_LIMIT_ORDERS_PER_SYMBOL,
        direction=bias,
        reward_risk_ratio=cfg.reward_risk_ratio,
        stop_placement=cfg.stop_placement,
    )
    if not setups:
        print(f"[bot] {symbol}: no valid unmitigated order block in the {bias or 'any'} direction ({_structure_note(df, cfg)}).")
    setups = add_smc_context(df, setups, n=cfg.fractal_n, dealing_range_candles=cfg.dealing_range_candles)
    setups, dropped = apply_entry_filters(
        setups,
        min_stop_distance_pct=cfg.min_stop_distance_pct,
        timing_trends=timing_trends,
        timing_mode=cfg.htf_timing_agreement,
        **smc_filter_settings(cfg),
    )
    for setup, reason in dropped:
        print(f"[bot] {symbol}: {setup['position_side']} setup @ {setup['entry']:.6g} filtered out -- {reason}.")
    return setups, htf_detail, "OVERRIDE:" in htf_detail, df


def _structure_note(df, cfg):
    """Where the entry timeframe's structure stands, for the log:
    "15m structure bullish, last break a CHoCH above 64210"."""
    try:
        last = structure_state(df, n=cfg.fractal_n, confirmed_swings=cfg.structure_confirmed_swings)
    except Exception:
        return f"{cfg.timeframe} structure n/a"
    if last is None:
        return f"{cfg.timeframe} structure: no break yet"
    side = "above" if last["direction"] == "bullish" else "below"
    return f"{cfg.timeframe} structure {last['direction']}, last break a {last['kind']} {side} {last['level']:.6g}"


def _find_tp2_level(df, setup, cfg):
    """TP2 = the next swing high (LONG) / swing low (SHORT) beyond TP1 on
    the entry timeframe. Returns (price, description), or (None, None) when
    there's no such swing -- RiskManager then falls back to
    TP2_FALLBACK_R_MULTIPLE x risk."""
    entry = setup["entry"]
    risk_per_unit = abs(entry - setup["stop_loss"])
    if risk_per_unit <= 0:
        return None, None
    is_long = setup["position_side"].upper() == "LONG"
    direction = 1 if is_long else -1
    tp1 = entry + direction * risk_per_unit * cfg.partial_tp_r_multiple
    level = next_swing_beyond(df, "bullish" if is_long else "bearish", tp1, n=cfg.fractal_n)
    if level is None:
        return None, None
    r_multiple = abs(level - entry) / risk_per_unit
    return level, f"next swing {'high' if is_long else 'low'} ({r_multiple:.2f}R)"


def _sync_symbol_orders(client, risk, cfg, symbol, setups, htf_detail, df, equity, may_add):
    """Step 7-8: make this symbol's resting orders match `setups`.
    Cancels orders whose order block is no longer a setup; places orders
    for setups that don't have one (only if `may_add`). Returns True if
    the symbol has at least one resting order afterwards."""
    wanted = {risk._zone_key(symbol, s["order_block"]): s for s in setups}

    for order in risk.pending_orders(symbol):
        if order["zone_key"] not in wanted:
            _cancel_pending(client, risk, order, "order block mitigated/invalidated, bias changed, replaced by newer order blocks, or filtered out by an entry filter")

    have = {o["zone_key"] for o in risk.pending_orders(symbol)}
    to_place = [s for k, s in wanted.items() if k not in have]
    if not to_place or not may_add:
        return bool(risk.pending_orders(symbol))

    sized = []
    for setup in to_place:
        ob = setup["order_block"]
        if risk.zone_already_traded(symbol, ob):
            print(f"[bot] {symbol}: zone {ob['bottom']:.6g}-{ob['top']:.6g} was already traded -- no new order.")
            continue
        existing_notional = risk.total_open_notional() + risk.pending_notional()
        qty, leverage = risk.position_size(
            equity, setup["entry"], setup["stop_loss"], existing_notional=existing_notional,
        )
        if qty <= 0:
            print(f"[bot] {symbol}: limit @ {setup['entry']:.6g} sized to zero (risk-based size or margin cap) -- skipped.")
            continue
        notional = qty * setup["entry"]
        if notional < cfg.min_notional_usdt:
            print(
                f"[bot] {symbol}: limit @ {setup['entry']:.6g} would be ~${notional:.2f} notional, below "
                f"MIN_NOTIONAL_USDT=${cfg.min_notional_usdt:.2f} -- skipped."
            )
            continue
        sized.append((setup, qty, leverage))
    if not sized:
        return bool(risk.pending_orders(symbol))

    # Leverage is per symbol+side on BingX, so every order on the same side
    # shares one setting. Use the LOWEST leverage any of them needs: their
    # dollar risk (qty x stop distance) doesn't depend on leverage, and the
    # lower setting keeps the liquidation price further beyond each stop.
    if not cfg.dry_run:
        by_side = {}
        for setup, _qty, lev in sized:
            ps = setup["position_side"]
            by_side[ps] = min(by_side.get(ps, lev), lev)
        for o in risk.pending_orders(symbol):
            ps = o["position_side"]
            if o.get("leverage") and ps in by_side:
                by_side[ps] = min(by_side[ps], o["leverage"])
        for ps, lev in by_side.items():
            client.set_leverage(symbol, lev, ps)

    for setup, qty, leverage in sized:
        ob = setup["order_block"]
        formed_ts = df.iloc[ob["formed_idx"]]["ts"]
        bos_ts = df.iloc[ob["bos_idx"]]["ts"]
        tp2_price, tp2_source = _find_tp2_level(df, setup, cfg)
        trigger = (
            f"{ob['type']} order block limit: zone {ob['bottom']:.4f}-{ob['top']:.4f} "
            f"(formed {formed_ts:%Y-%m-%d %H:%M} UTC, {ob.get('structure', 'BOS')} confirmed "
            f"{bos_ts:%Y-%m-%d %H:%M} UTC) | HTF: {htf_detail or 'trend filter disabled'}"
            f" | exits: TP1 {cfg.partial_tp_r_multiple:g}R (mirror of SL), TP2 "
            + (f"{tp2_price:.6g} ({tp2_source})" if tp2_price is not None else "fallback (no swing beyond TP1)")
            + ", runner trails after TP2"
            + f" | stop beyond the {STOP_NAMES.get(setup.get('stop_placement'), 'order block')}"
            + f" | SMC: {describe_smc(setup.get('smc'))}"
        )
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trade_id = risk.new_trade_id()
        if cfg.dry_run:
            order_id = f"paper-{trade_id}"
        else:
            try:
                placed = client.create_limit_order_with_stop(
                    symbol=symbol,
                    side=setup["side"],
                    amount=qty,
                    price=setup["entry"],
                    stop_loss_price=setup["stop_loss"],
                    position_side=setup["position_side"],
                )
            except Exception as e:
                print(f"[bot] {symbol}: placing {setup['position_side']} limit @ {setup['entry']:.6g} FAILED ({e}).")
                continue
            order_id = str((placed or {}).get("id"))
        order = {
            "order_id": order_id,
            "trade_id": trade_id,
            "mode": "DRY_RUN" if cfg.dry_run else "LIVE",
            "symbol": symbol,
            "side": setup["side"],
            "position_side": setup["position_side"],
            "qty": round(qty, 6),
            "leverage": leverage,
            "entry_price": setup["entry"],
            "stop_loss": setup["stop_loss"],
            "take_profit": setup["take_profit"],
            "tp2_price": tp2_price,
            "tp2_source": tp2_source,
            "equity": round(equity, 2),
            "zone_key": risk._zone_key(symbol, ob),
            "ob": {"type": ob["type"], "top": ob["top"], "bottom": ob["bottom"]},
            "placed_time_utc": now_iso,
            "trigger": trigger,
        }
        risk.track_pending_order(order)
        risk.log_order_event("LIMIT_PLACED", order)
        print(
            f"[bot] {symbol}: {'DRY_RUN ' if cfg.dry_run else ''}{setup['position_side']} limit placed @ "
            f"{setup['entry']:.6g} SL={setup['stop_loss']:.6g} qty~{qty:.6f} ({ob.get('structure', 'BOS')} OB "
            f"{ob['bottom']:.6g}-{ob['top']:.6g}; {describe_smc(setup.get('smc'))})"
        )
    return bool(risk.pending_orders(symbol))


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

    df_cache = {}

    # Step 2 runs BEFORE reconciling closed positions, so an order that
    # filled and got stopped out between two runs is first recorded as a
    # trade and then closed in this same cycle.
    _process_pending_fills(client, risk, cfg, df_cache)

    open_positions = client.fetch_open_positions()
    exchange_open_symbols = {p.get("symbol") for p in open_positions}

    closed_trades = risk.reconcile_closed_trades(open_positions, client)
    if closed_trades:
        print(f"[bot] Reconciled {len(closed_trades)} closed trade(s), logged EXIT row(s).")

    position_qty_by_symbol = {}
    for p in open_positions:
        try:
            position_qty_by_symbol[p.get("symbol")] = float(p.get("contracts") or 0.0)
        except (TypeError, ValueError):
            pass

    for symbol in sorted(risk.occupied_symbols()):
        try:
            _manage_open_trades(
                client, risk, cfg, symbol, df_cache,
                position_qty=position_qty_by_symbol.get(symbol),
            )
        except Exception:
            # One symbol's trade management failing must not stop the other
            # open trades from being managed this cycle.
            print(f"[bot] {symbol}: trade management failed this cycle -- will retry next cycle:", file=sys.stderr)
            traceback.print_exc()

    if risk.circuit_breaker_tripped(equity):
        print("[bot] Daily loss circuit breaker is tripped. Cancelling resting orders, no new entries today.")
        for order in risk.pending_orders():
            _cancel_pending(client, risk, order, "daily loss circuit breaker tripped")
        return

    position_symbols = risk.occupied_symbols() | exchange_open_symbols

    # A symbol that already has a position never keeps resting orders.
    for symbol in sorted(risk.pending_symbols() & position_symbols):
        for order in risk.pending_orders(symbol):
            _cancel_pending(client, risk, order, "symbol already has an open position")

    open_count = len(position_symbols | risk.pending_symbols())
    print(
        f"[bot] Equity=${equity:.2f} | {open_count}/{cfg.max_concurrent_positions} slots used "
        f"(positions: {sorted(position_symbols) or 'none'}; resting orders: "
        f"{sorted(risk.pending_symbols()) or 'none'}) | watching {len(cfg.symbols)} symbols"
    )

    # --- Step 6: analysis + setups for every symbol without a position.
    holding, clean, override = [], [], []
    for symbol in cfg.symbols:
        if symbol in position_symbols:
            print(f"[bot] {symbol}: already have an open position (LIVE or DRY_RUN), skipping.")
            continue
        setups, htf_detail, is_override, df = _find_setups(client, cfg, symbol, df_cache)
        if setups is None:
            # Data problem this cycle -- leave any resting orders alone
            # rather than cancelling on missing information.
            continue
        entry = (symbol, setups, htf_detail, df)
        if risk.pending_orders(symbol):
            holding.append(entry)
        elif setups:
            (override if is_override else clean).append(entry)

    # --- Steps 7-9: symbols that already hold a slot first (keep/cancel/top
    # up their orders), then free slots to clean setups, then override ones.
    for symbol, setups, htf_detail, df in holding:
        if not _sync_symbol_orders(client, risk, cfg, symbol, setups, htf_detail, df, equity, may_add=True):
            open_count -= 1
    for symbol, setups, htf_detail, df in clean + override:
        if open_count >= cfg.max_concurrent_positions:
            print(f"[bot] {symbol}: valid order block(s) but all {cfg.max_concurrent_positions} slots are used -- no order.")
            continue
        if _sync_symbol_orders(client, risk, cfg, symbol, setups, htf_detail, df, equity, may_add=True):
            open_count += 1


if __name__ == "__main__":
    try:
        run_once()
    except Exception:
        print("[bot] Unhandled exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
