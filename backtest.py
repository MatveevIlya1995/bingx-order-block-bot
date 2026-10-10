"""
Backtest: replays the bot's strategy on BingX history, with fees.

    python3 backtest.py --days 180

What it does
------------
1. Downloads public candle history from BingX (no API key needed) for every
   symbol in SYMBOLS: the entry timeframe (5m), every higher timeframe the
   trend cascade uses (1d/4h/1h/15m) and the "execution" timeframe used to
   walk price between two bot runs (5m, i.e. the entry candles themselves
   unless TIMEFRAME is set higher).
2. SIGNALS (the slow part, done once): for every entry-candle step it rebuilds
   exactly what the live bot would have seen at that moment -- the last
   CANDLE_LIMIT entry candles and the last TREND_CANDLE_LIMIT candles of each
   higher timeframe, each ending in a just-opened, still-forming candle -- and
   calls the bot's own code on it: strategy.detect_order_blocks(),
   strategy.limit_setups(), strategy.add_smc_context(),
   strategy.combine_htf_bias(), strategy.weekly_room_pct(),
   bot._find_tp2_level(). Nothing about the strategy is re-implemented
   here.
3. SIMULATION (fast, repeated per variant): replays the bot's run loop over
   those signals as one account -- resting limit entries, MAX_CONCURRENT_
   POSITIONS slots, one position per symbol, one trade per zone, the bot's
   own RiskManager.position_size(), the margin cap, MIN_NOTIONAL_USDT, the
   daily-loss circuit breaker, and the TP1 / TP2 / breakeven / trailing exit
   plan -- and applies the entry filters through the same
   strategy.apply_entry_filters() the live bot calls.
4. Compares variants: minimum stop distance, symbol set, timing-level
   agreement (each alone and combined), resting take-profit orders vs. the
   old market close on the scheduled run, the smart-money filters (order
   blocks from a BOS only / a CHoCH only, liquidity sweep, displacement,
   fair value gap, premium/discount, entry hours) each alone and in a few
   combinations, the three stop placements (beyond the order block, the
   last swing, the swing before it), and -- with an extra signal pass each -- swings confirmed
   before they count as structure vs. the behaviour before
   (STRUCTURE_CONFIRMED_SWINGS) and only order blocks at strong swings vs.
   all of them (REQUIRE_STRONG_SWING).

Settings come from the same environment variables as the bot (config.py), so
running it with the bot's variables backtests the bot's live configuration.

What it assumes -- read before trusting a number
------------------------------------------------
- The bot runs once per entry candle, right at the candle boundary, and sees
  the opening price of the new candle. (Live runs come from a scheduler and
  land some minutes later.)
- Between two runs, price is walked through the execution-timeframe candles.
  If a candle touches both a stop and a take-profit, the STOP is assumed to
  have come first -- except when the candle opens beyond the take-profit.
- A resting limit entry fills at its limit price (maker fee) when a candle
  trades through it. An entry placed while price is already inside the zone
  fills at once at the current price (taker fee + slippage).
- Stops and take-profits are market orders once triggered: taker fee, plus
  --slippage-pct against the trade (a stop that is gapped through fills at
  the candle's open).
- Take-profits rest on the exchange from the first run AFTER the entry fill
  (that is when the live bot places them); until then only the stop exists.
- When one of a symbol's two entry orders fills, the other is cancelled at
  once (live: at the next run).
- Quantities are not rounded to BingX's amount step. Funding payments are
  ignored. Candle data is last-traded price; BingX triggers stops on mark
  price.
- BingX only keeps a limited history of small-timeframe candles (roughly
  100 days of 15m and 45 days of 5m when this was written). If the period
  asked for is longer, the test starts where the history starts and says
  so; if 5m candles don't cover it, price is walked on the entry-timeframe
  candles instead (--exec-timeframe auto).
- No backtest shows what will happen next. Picking the best-looking variant
  from one stretch of history over-fits: check that a filter helps in BOTH
  halves of the period (the report shows them) before believing it.
"""
import argparse
import bisect
import csv
import datetime
import json
import math
import os
import pickle
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import load_config, override_levels
from risk_manager import RiskManager
from strategy import (
    add_smc_context,
    apply_entry_filters,
    combine_htf_bias,
    detect_order_blocks,
    entry_symbol_allowed,
    limit_setups,
    nearest_confirmed_swing_level,
)

MAX_LIMIT_ORDERS_PER_SYMBOL = 2  # same as bot.py
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
         "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000}
ZONE_MEMORY_MS = 7 * 86_400_000  # risk_manager.DRY_RUN_SIGNAL_MEMORY_DAYS


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. history
# ---------------------------------------------------------------------------

def fetch_history(exchange, symbol, timeframe, start_ms, end_ms, cache_dir=None):
    """All candles of `timeframe` with start_ms <= open time < end_ms, as a
    float array [[ts, open, high, low, close, volume], ...] sorted by time."""
    step = TF_MS[timeframe]
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        name = f"{symbol.replace('/', '_').replace(':', '_')}_{timeframe}_{start_ms}_{end_ms}.npy"
        cache_path = os.path.join(cache_dir, name)
        if os.path.exists(cache_path):
            return np.load(cache_path)
    rows = {}
    window = 1000 * step
    cursor = start_ms - start_ms % step
    while cursor < end_ms:
        until = min(cursor + window, end_ms) - 1
        batch = None
        for attempt in range(5):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000,
                                             params={"until": until})
                break
            except Exception as e:  # network hiccup / rate limit: back off and retry
                wait = 2 ** attempt
                log(f"  {symbol} {timeframe}: fetch failed ({type(e).__name__}: {str(e)[:120]}), retry in {wait}s")
                time.sleep(wait)
        if batch is None:
            raise RuntimeError(f"could not download {symbol} {timeframe} candles from BingX")
        for c in batch:
            if c and c[0] is not None and start_ms <= c[0] < end_ms:
                rows[int(c[0])] = [float(c[0])] + [float(x) for x in c[1:6]]
        cursor += window
    arr = np.array([rows[k] for k in sorted(rows)], dtype=float).reshape(-1, 6)
    if cache_path:
        np.save(cache_path, arr)
    return arr


def synthetic_history(symbols, start_ms, end_ms, seed, exec_tf, htf_timeframes, htf_lookback):
    """Random-walk candles, for tests and for trying the script offline."""
    rng = np.random.default_rng(seed)
    data = {}
    base_tf = "5m" if exec_tf == "5m" else exec_tf
    step = TF_MS[base_tf]
    longest = max(TF_MS[tf] for tf in htf_timeframes) * htf_lookback
    t0 = start_ms - longest
    t0 -= t0 % TF_MS["1w"]
    n = int((end_ms - t0) // step)
    for i, symbol in enumerate(symbols):
        drift = rng.normal(0, 2e-6)
        vol = 0.0012 * (1 + 0.5 * rng.random())
        regime = np.repeat(rng.normal(0, 4e-5, n // 2000 + 1), 2000)[:n]
        rets = rng.normal(drift, vol, n) + regime
        close = 100.0 * (1 + i) * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(rng.normal(0, vol * 0.6, (2, n)))
        high = np.maximum(open_, close) * (1 + wick[0])
        low = np.minimum(open_, close) * (1 - wick[1])
        ts = t0 + np.arange(n) * step
        base = np.column_stack([ts, open_, high, low, close, np.ones(n)])

        def resample(tf):
            k = TF_MS[tf] // step
            m = (n // k) * k
            b = base[:m].reshape(-1, k, 6)
            return np.column_stack([b[:, 0, 0], b[:, 0, 1], b[:, :, 2].max(1), b[:, :, 3].min(1),
                                    b[:, -1, 4], b[:, :, 5].sum(1)])

        data[symbol] = {tf: (base if tf == base_tf else resample(tf))
                        for tf in set([base_tf, "15m", exec_tf] + list(htf_timeframes))}
    return data


def load_data(cfg, symbols, start_ms, end_ms, exec_tf, cache_dir, synthetic_seed):
    lookback = cfg.trend_candle_limit + 5
    if synthetic_seed is not None:
        return synthetic_history(symbols, start_ms, end_ms, synthetic_seed, exec_tf, cfg.htf_timeframes, lookback)
    import ccxt
    exchange = ccxt.bingx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    exchange.load_markets()
    entry_tf = cfg.timeframe
    # Entry candles: CANDLE_LIMIT of warm-up, plus enough to rebuild the
    # still-forming candle of the longest higher timeframe at the first step.
    entry_from = start_ms - max(cfg.candle_limit * TF_MS[entry_tf],
                                max(TF_MS[tf] for tf in cfg.htf_timeframes) + TF_MS[entry_tf]) - TF_MS["1d"]
    data = {}
    for symbol in symbols:
        if symbol not in exchange.markets:
            log(f"{symbol}: not listed on BingX swap -- skipped")
            continue
        t = time.time()
        d = {entry_tf: fetch_history(exchange, symbol, entry_tf, entry_from, end_ms, cache_dir)}
        if exec_tf != entry_tf:
            d[exec_tf] = fetch_history(exchange, symbol, exec_tf, start_ms - TF_MS[entry_tf], end_ms, cache_dir)
        for tf in cfg.htf_timeframes:
            d[tf] = fetch_history(exchange, symbol, tf, start_ms - lookback * TF_MS[tf], end_ms, cache_dir)
        data[symbol] = d
        log(f"{symbol}: " + ", ".join(
            f"{tf} x{len(a)}" + (f" from {day(a[0, 0])}" if len(a) else "") for tf, a in d.items()
        ) + f"  ({time.time() - t:.0f}s)")
    return data


def day(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")


def usable_period(cfg, data, symbols, start_ms, exec_tf):
    """BingX only keeps a limited history of small-timeframe candles, so the
    requested period may not be fully available. Returns
    (symbols, start_ms, exec_tf, notes): the symbols with enough history, the
    first moment every one of them has the candles a bot run needs (warm-up
    included), the execution timeframe to walk price with, and notes for the
    report.

    exec_tf "auto" picks 5m when 5m candles cover the whole period and the
    entry timeframe otherwise -- one timeframe for the whole period, so
    both halves are simulated the same way."""
    entry_tf = cfg.timeframe
    warm = max(cfg.candle_limit * TF_MS[entry_tf], max(TF_MS[tf] for tf in cfg.htf_timeframes)) + TF_MS[entry_tf]
    notes = []
    ready = {s: int(data[s][entry_tf][0, 0]) + warm for s in symbols if len(data[s][entry_tf])}
    if not ready:
        return [], start_ms, entry_tf, notes
    typical = sorted(ready.values())[len(ready) // 2]
    short = [s for s in ready if ready[s] > typical + TF_MS["1d"]]
    for s in short:
        notes.append(f"{s.split('/')[0]} left out: its {entry_tf} history only starts {day(data[s][entry_tf][0, 0])}.")
    symbols = [s for s in symbols if s in ready and s not in short]
    begin = max([start_ms] + [ready[s] for s in symbols])
    if begin > start_ms:
        notes.append(
            f"Requested start {day(start_ms)}, but BingX only served {entry_tf} candles back to "
            f"{day(min(data[s][entry_tf][0, 0] for s in symbols))}; after warm-up the test starts {day(begin)}."
        )
    want = "5m" if exec_tf == "auto" else exec_tf
    if want != entry_tf:
        have = all(want in data[s] and len(data[s][want]) for s in symbols)
        first = max(int(data[s][want][0, 0]) for s in symbols) if have else None
        if not have or first > begin:
            if exec_tf == "auto":
                notes.append(
                    f"{want} candles only go back to {day(first) if first else 'n/a'}, so price between runs is walked on "
                    f"{entry_tf} candles for the whole period (coarser: one candle per run, and a candle that touches "
                    f"both stop and take-profit counts as a stop)."
                )
                want = entry_tf
            else:
                notes.append(f"{want} candles only go back to {day(first)}; the test starts there.")
                begin = max(begin, first + TF_MS[entry_tf])
    begin -= begin % TF_MS[entry_tf]
    return symbols, begin, want, notes


# ---------------------------------------------------------------------------
# 2. signals -- what the bot would have seen at every run
# ---------------------------------------------------------------------------

class HtfView:
    """One higher timeframe of one symbol, answering "what did the last
    TREND_CANDLE_LIMIT candles look like at time t?" -- i.e. exactly what
    client.fetch_ohlcv(symbol, timeframe=tf, limit=TREND_CANDLE_LIMIT)
    returned to the live bot: completed candles plus the still-forming one."""

    def __init__(self, tf, candles, limit, ema_period):
        self.tf = tf
        self.step = TF_MS[tf]
        self.c = candles
        self.ts = candles[:, 0] if len(candles) else np.array([])
        self.limit = limit
        self.ema_period = ema_period
        self.alpha = 2.0 / (ema_period + 1.0)
        self._ema_cache = {}
        self._swing_cache = {}

    def completed_count(self, t_run):
        """How many candles had fully closed by t_run."""
        return int(np.searchsorted(self.ts + self.step, t_run, side="right"))

    def window(self, n_done, partial):
        """(highs, lows, closes) of the candles the bot would have fetched:
        the last limit-1 completed ones + the forming candle `partial`
        = (open, high, low, close)."""
        lo = max(0, n_done - (self.limit - 1))
        highs = np.append(self.c[lo:n_done, 2], partial[1])
        lows = np.append(self.c[lo:n_done, 3], partial[2])
        closes = np.append(self.c[lo:n_done, 4], partial[3])
        return highs, lows, closes

    def _ema_before_last(self, n_done):
        """EMA (pandas ewm(span, adjust=False)) over the completed part of
        the window -- the value detect_trend() calls prev_ema."""
        if n_done not in self._ema_cache:
            lo = max(0, n_done - (self.limit - 1))
            closes = self.c[lo:n_done, 4]
            ema = closes[0]
            a = self.alpha
            for x in closes[1:]:
                ema = a * x + (1 - a) * ema
            self._ema_cache[n_done] = float(ema)
        return self._ema_cache[n_done]

    def trend(self, n_done, price):
        """strategy.detect_trend() on the bot's window, without building a
        DataFrame: the window's last close is the current price."""
        length = min(self.limit, n_done + 1)
        if n_done < 1 or length < self.ema_period + 2:
            return "neutral"
        prev_ema = self._ema_before_last(n_done)
        last_ema = self.alpha * price + (1 - self.alpha) * prev_ema
        if price > last_ema and last_ema > prev_ema:
            return "bullish"
        if price < last_ema and last_ema < prev_ema:
            return "bearish"
        return "neutral"

    def swing_level(self, n_done, partial, direction, fractal_n):
        key = (n_done, partial[1], partial[2], direction)
        if key not in self._swing_cache:
            highs, lows, _ = self.window(n_done, partial)
            df = pd.DataFrame({"high": highs, "low": lows})
            self._swing_cache[key] = nearest_confirmed_swing_level(df, direction, n=fractal_n)
        return self._swing_cache[key]


def analyze_htf(cfg, views, n_done, partials, price):
    """bot.analyze_higher_timeframes() for one moment in history.
    Returns (bias, is_override, trend_by_tf). tests/test_backtest.py checks
    it against the bot's own function candle for candle."""
    trend_by_tf = {tf: views[tf].trend(n_done[tf], price) for tf in cfg.htf_timeframes}
    bias = combine_htf_bias([trend_by_tf[tf] for tf in cfg.htf_timeframes if tf in cfg.primary_htf_timeframes])
    is_override = False
    pair = override_levels(cfg)
    if bias == "conflict" and pair and pair[0] in trend_by_tf and pair[1] in trend_by_tf:
        top = pair[0]
        higher, lower = trend_by_tf[top], trend_by_tf[pair[1]]
        if higher in ("bullish", "bearish") and lower in ("bullish", "bearish") and price > 0:
            level = views[top].swing_level(n_done[top], partials[top], higher, cfg.fractal_n)
            room = None
            if level is not None:
                room = ((level - price) if higher == "bullish" else (price - level)) / price * 100.0
            if room is not None and room >= cfg.weekly_daily_override_min_room_pct:
                bias, is_override = higher, True
    return bias, is_override, trend_by_tf


def compute_signals(args):
    """Every run's setups for ONE symbol. Returns
    {"symbol", "steps": [t_run, ...], "px": [...], "signals": {i: (is_override, timing_trends, [setup, ...])}}
    where setup = (position_side, entry, stop, ob_type, ob_top, ob_bottom, tp2_price, smc, stops):
    smc is the setup's smart-money context (strategy.add_smc_context) and
    stops is {STOP_PLACEMENT: (stop, tp2_price)} -- the stop and the TP2
    that goes with it under every stop placement, so the simulation can
    compare them without another signal pass."""
    cfg, symbol, d, start_ms, end_ms = args
    from bot import _find_tp2_level  # the bot's own TP2 rule

    entry_tf = cfg.timeframe
    c = d[entry_tf]
    ts = c[:, 0].astype(np.int64)
    views = {tf: HtfView(tf, d[tf], cfg.trend_candle_limit, cfg.trend_ema_period) for tf in cfg.htf_timeframes}
    timing_tfs = [tf for tf in cfg.htf_timeframes if tf not in cfg.primary_htf_timeframes]

    # running still-forming candle per higher timeframe: [period_start, open, high, low]
    forming = {tf: None for tf in cfg.htf_timeframes}
    first = int(np.searchsorted(ts, start_ms))
    first = max(first, cfg.candle_limit)

    steps, prices, signals = [], [], {}
    cols = ["ts", "open", "high", "low", "close", "volume"]
    for i in range(first, len(c)):
        t_run = int(ts[i])                 # the bot runs as candle i opens
        px = float(c[i, 1])
        n_done, partials = {}, {}
        for tf in cfg.htf_timeframes:
            v = views[tf]
            nd = v.completed_count(t_run)
            n_done[tf] = nd
            period_start = int(v.ts[nd - 1] + v.step) if nd > 0 else t_run - t_run % v.step
            f = forming[tf]
            if f is None or f[0] != period_start:
                f = forming[tf] = [period_start, px, px, px]
                # entry candles of this period that came before this step
                j = i - 1
                while j >= 0 and ts[j] >= period_start:
                    j -= 1
                if j + 1 < i:
                    seg = c[j + 1:i]
                    f[1] = float(seg[0, 1])
                    f[2] = max(px, float(seg[:, 2].max()))
                    f[3] = min(px, float(seg[:, 3].min()))
            else:
                prev = c[i - 1]
                f[2] = max(f[2], float(prev[2]), px)
                f[3] = min(f[3], float(prev[3]), px)
            partials[tf] = (f[1], max(f[2], px), min(f[3], px), px)
        if t_run >= end_ms:
            break

        steps.append(t_run)
        prices.append(px)
        bias, is_override, trend_by_tf = None, False, {}
        if cfg.trend_filter_enabled:
            bias, is_override, trend_by_tf = analyze_htf(cfg, views, n_done, partials, px)
            if bias == "conflict":
                continue
        # The candles client.fetch_ohlcv(symbol) returned: CANDLE_LIMIT-1
        # completed ones and the candle that has just opened.
        win = np.vstack([c[i - (cfg.candle_limit - 1):i], [t_run, px, px, px, px, 0.0]])
        df = pd.DataFrame(win, columns=cols)
        obs = detect_order_blocks(df, n=cfg.fractal_n, confirmed_swings=cfg.structure_confirmed_swings,
                                  strong_swings_only=cfg.require_strong_swing)
        setups = limit_setups(df, obs, max_setups=MAX_LIMIT_ORDERS_PER_SYMBOL, direction=bias,
                              reward_risk_ratio=cfg.reward_risk_ratio, stop_placement=cfg.stop_placement)
        if not setups:
            continue
        add_smc_context(df, setups, n=cfg.fractal_n, dealing_range_candles=cfg.dealing_range_candles)
        out = []
        for s in setups:
            tp2, _ = _find_tp2_level(df, s, cfg)
            ob = s["order_block"]
            stops, tp2_at = {}, {float(s["stop_loss"]): tp2}
            for placement, stop in s["stops"].items():
                stop = float(stop)
                if stop not in tp2_at:
                    tp2_at[stop], _ = _find_tp2_level(df, {**s, "stop_loss": stop}, cfg)
                stops[placement] = (stop, tp2_at[stop])
            out.append((s["position_side"], float(s["entry"]), float(s["stop_loss"]),
                        ob["type"], float(ob["top"]), float(ob["bottom"]), tp2, dict(s["smc"]), stops))
        timing = tuple(trend_by_tf[tf] for tf in timing_tfs) if cfg.trend_filter_enabled else None
        signals[len(steps) - 1] = (is_override, timing, out)
    return {"symbol": symbol, "steps": steps, "px": prices, "signals": signals}


# ---------------------------------------------------------------------------
# 3. simulation -- the bot's run loop on one account
# ---------------------------------------------------------------------------

class Costs:
    def __init__(self, maker_pct, taker_pct, slippage_pct):
        self.maker = maker_pct / 100.0
        self.taker = taker_pct / 100.0
        self.slip = slippage_pct / 100.0


class Sim:
    """One variant: one account replayed over all symbols.

    exit_mode "resting": TP1/TP2 rest on the exchange (the live behaviour).
    exit_mode "run":     TP1/TP2 are market closes on the run that sees the
                         price beyond the level (the behaviour before)."""

    def __init__(self, cfg, costs, start_equity, exit_mode="resting",
                 min_stop_pct=0.0, timing_mode="off", entry_symbols=None, stop_placement=None,
                 tp1_r=None, tp2_r=None, **smc_filters):
        self.cfg = cfg
        self.costs = costs
        self.exit_mode = exit_mode
        self.min_stop_pct = min_stop_pct
        self.timing_mode = timing_mode
        self.entry_symbols = entry_symbols or []
        # None: the stop the signals were computed with (cfg.stop_placement);
        # otherwise the stop (and its TP2) under that STOP_PLACEMENT
        self.stop_placement = stop_placement
        # Exit targets in multiples of the initial risk. tp1_r None: the
        # bot's PARTIAL_TP_R_MULTIPLE. tp2_r None: the bot's TP2 (the next
        # swing beyond TP1, or its fallback); a number: TP2 fixed there.
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        # smart-money filters, as strategy.apply_entry_filters() takes them:
        # ob_structure, require_sweep, min_displacement_atr, require_fvg,
        # premium_discount, entry_hours
        self.smc_filters = smc_filters
        self.cash = start_equity            # realized equity
        self.start_equity = start_equity
        self.positions = {}                 # symbol -> trade dict
        self.pending = {}                   # symbol -> [order dict]
        self.traded_zones = {}              # zone key -> time
        self.trades = []                    # closed trades
        self.equity_curve = []              # (t, mark-to-market equity)
        self.day = None
        self.day_start_equity = None
        self.realized_today = 0.0
        self.breaker_runs = 0
        self.rm = RiskManager.__new__(RiskManager)   # the bot's own sizing code
        self.rm.cfg = cfg
        f1 = min(max(cfg.partial_tp_fraction, 0.0), 1.0)
        self.f1, self.f2 = f1, min(max(cfg.tp2_fraction, 0.0), 1.0 - f1)

    # ---- bookkeeping ----
    def _book(self, trade, qty, price, fee, kind, t):
        pnl = (price - trade["entry"]) * trade["dir"] * qty - fee
        self.cash += pnl
        self.realized_today += pnl
        trade["pnl"] += pnl
        trade["fees"] += fee
        trade["qty"] = max(trade["qty"] - qty, 0.0)
        trade["exits"].append((t, kind, qty, price))
        if trade["qty"] <= 1e-12:
            trade["exit_time"] = t
            trade["exit_kind"] = kind
            self.trades.append(trade)
            del self.positions[trade["symbol"]]

    def _market_exit(self, trade, qty, price, kind, t):
        """A triggered stop / take-profit or a market close: taker fee and
        slippage against the trade."""
        fill = price * (1 - trade["dir"] * self.costs.slip)
        self._book(trade, qty, fill, qty * fill * self.costs.taker, kind, t)

    def _open(self, order, price, t, taker):
        d = 1 if order["ps"] == "LONG" else -1
        if taker:
            price = price * (1 + d * self.costs.slip)
        qty = order["qty"]
        fee = qty * price * (self.costs.taker if taker else self.costs.maker)
        self.cash -= fee
        self.realized_today -= fee
        risk_per_unit = abs(price - order["stop"])
        tp1_r = self.tp1_r if self.tp1_r else self.cfg.partial_tp_r_multiple
        tp1 = price + d * risk_per_unit * tp1_r
        tp2 = price + d * risk_per_unit * self.tp2_r if self.tp2_r else order["tp2"]
        fallback_r = max(self.cfg.tp2_fallback_r_multiple, tp1_r + 0.5)
        if tp2 is None or not (tp2 > tp1 if d == 1 else tp2 < tp1):
            tp2 = price + d * risk_per_unit * fallback_r        # RiskManager._arm_partial_and_trail
        symbol = order["symbol"]
        self.positions[symbol] = {
            "symbol": symbol, "ps": order["ps"], "dir": d, "entry": price, "entry_time": t,
            "qty": qty, "initial_qty": qty, "stop": order["stop"], "initial_stop": order["stop"],
            "risk_per_unit": risk_per_unit, "risk_usd": qty * risk_per_unit,
            "tp1": tp1, "tp2": tp2, "tp1_done": False, "tp2_done": False,
            "tp1_resting": False, "tp2_resting": False, "armed": False, "be_pending": False,
            "pnl": -fee, "fees": fee, "exits": [], "stop_pct": risk_per_unit / price * 100.0,
            # what MIN_STOP_DISTANCE_PCT looks at: the setup's limit price to its stop
            "setup_stop_pct": abs(order["entry"] - order["stop"]) / order["entry"] * 100.0,
            "timing": order["timing"], "override": order["override"], "taker_entry": taker,
            "leverage": order["leverage"], "smc": order.get("smc"),
        }
        self.traded_zones[order["zone"]] = t
        self.pending.pop(symbol, None)       # one position per symbol

    # ---- the market between two runs ----
    def walk(self, symbol, candles, t_end):
        """Price action for one symbol between two runs: `candles` are the
        execution-timeframe candles [ts, o, h, l, c, v] of that interval."""
        for candle in candles:
            ts, o, h, l = candle[0], candle[1], candle[2], candle[3]
            trade = self.positions.get(symbol)
            if trade is None:
                orders = self.pending.get(symbol)
                if not orders:
                    return
                filled = None
                for order in orders:
                    hit = l <= order["entry"] if order["ps"] == "LONG" else h >= order["entry"]
                    if hit:
                        # the order nearest to where price came from fills first
                        if filled is None or (
                            order["entry"] > filled["entry"] if order["ps"] == "LONG" else order["entry"] < filled["entry"]
                        ):
                            filled = order
                if filled is None:
                    continue
                self._open(filled, filled["entry"], ts, taker=False)
                trade = self.positions[symbol]
                # same candle: price went on through the stop?
                if (l <= trade["stop"]) if trade["dir"] == 1 else (h >= trade["stop"]):
                    self._market_exit(trade, trade["qty"], trade["stop"], "stop", ts)
                continue
            self._walk_position(trade, ts, o, h, l)

    def _walk_position(self, trade, ts, o, h, l):
        d = trade["dir"]
        long = d == 1
        # take-profits the candle OPENED beyond fill first, at the open
        for n in (1, 2):
            if trade.get(f"tp{n}_resting") and not trade[f"tp{n}_done"] and trade["qty"] > 0:
                tp = trade[f"tp{n}"]
                if (o >= tp) if long else (o <= tp):
                    self._tp_fill(trade, n, o, ts)
        if trade["qty"] <= 0:
            return
        stop = trade["stop"]
        if (l <= stop) if long else (h >= stop):
            gapped = (o < stop) if long else (o > stop)
            self._market_exit(trade, trade["qty"], o if gapped else stop, "stop", ts)
            return
        for n in (1, 2):
            if trade.get(f"tp{n}_resting") and not trade[f"tp{n}_done"] and trade["qty"] > 0:
                tp = trade[f"tp{n}"]
                if (h >= tp) if long else (l <= tp):
                    self._tp_fill(trade, n, tp, ts)

    def _tp_qty(self, trade, n):
        if n == 1:
            return min(trade["initial_qty"] * self.f1, trade["qty"])
        return min(trade["initial_qty"] * self.f2, trade["qty"])

    def _tp_fill(self, trade, n, price, ts):
        qty = self._tp_qty(trade, n)
        trade[f"tp{n}_done"] = True
        trade[f"tp{n}_resting"] = False
        if n == 1:
            trade["be_pending"] = True       # the stop moves at the next run
        if qty > 0:
            self._market_exit(trade, qty, price, f"tp{n}", ts)

    # ---- one bot run ----
    def equity(self, px):
        eq = self.cash
        for symbol, t in self.positions.items():
            eq += (px[symbol] - t["entry"]) * t["dir"] * t["qty"]
        return eq

    def _manage(self, trade, price, t):
        """bot._manage_open_trades for one LIVE trade at run time."""
        long = trade["dir"] == 1
        beyond = (lambda level: price >= level) if long else (lambda level: price <= level)
        resting = self.exit_mode == "resting"
        if resting and not trade["armed"]:
            trade["armed"] = True
            for n in (1, 2):
                if n == 2 and self.f2 <= 0:
                    continue
                # a take-profit can only rest if price hasn't passed it yet
                trade[f"tp{n}_resting"] = not trade[f"tp{n}_done"] and not beyond(trade[f"tp{n}"])
        # take-profits without a resting order: market close when the run sees the level
        for n in (1, 2):
            if n == 2 and (self.f2 <= 0 or not trade["tp1_done"]):
                continue
            if trade[f"tp{n}_done"] or trade.get(f"tp{n}_resting") or trade["qty"] <= 0:
                continue
            if beyond(trade[f"tp{n}"]):
                qty = self._tp_qty(trade, n)
                trade[f"tp{n}_done"] = True
                if n == 1:
                    trade["be_pending"] = True
                if qty > 0:
                    self._market_exit(trade, qty, price, f"tp{n}", t)
        if trade["qty"] <= 0:
            return
        # stop to breakeven after TP1 -- only possible while price is beyond the entry
        if trade["tp1_done"] and trade["be_pending"]:
            if (price > trade["entry"]) if long else (price < trade["entry"]):
                trade["stop"] = trade["entry"]
                trade["be_pending"] = False
        # trailing runner after TP2 (or after TP1 when TP2 is disabled)
        if trade["tp1_done"] and (trade["tp2_done"] or self.f2 <= 0):
            distance = trade["risk_per_unit"] * self.cfg.trail_r_multiple
            if long:
                new_stop = max(trade["stop"], price - distance)
            else:
                new_stop = min(trade["stop"], price + distance)
            if new_stop != trade["stop"]:
                trade["stop"] = new_stop
                trade["be_pending"] = False

    def run(self, t, px, signals_at, symbols):
        """One scheduled bot run at time t. px: {symbol: price now};
        signals_at: {symbol: (is_override, timing, [setup, ...])}."""
        cfg = self.cfg
        for symbol in list(self.positions):
            if symbol in px:
                self._manage(self.positions[symbol], px[symbol], t)

        equity = self.equity(px)
        self.equity_curve.append((t, equity))
        day = t // 86_400_000
        if day != self.day:
            self.day, self.day_start_equity, self.realized_today = day, equity, 0.0
        start = self.day_start_equity or equity
        loss = max((start - equity) / start, -self.realized_today / start) if start > 0 else 0.0
        if loss >= cfg.max_daily_loss_pct:
            self.pending.clear()
            self.breaker_runs += 1
            return

        for symbol in list(self.pending):
            if symbol in self.positions:
                del self.pending[symbol]
        open_count = len(set(self.positions) | set(self.pending))

        holding, clean, override = [], [], []
        for symbol in symbols:
            if symbol in self.positions or symbol not in px:
                continue
            sig = signals_at.get(symbol)
            setups, is_override, timing = [], False, None
            if sig is not None and entry_symbol_allowed(symbol, self.entry_symbols):
                is_override, timing, raw = sig
                setups = []
                for s in raw:
                    stop, tp2 = s[2], s[6]
                    if self.stop_placement and len(s) > 8 and self.stop_placement in s[8]:
                        stop, tp2 = s[8][self.stop_placement]
                    setups.append({"position_side": s[0], "entry": s[1], "stop_loss": stop, "_raw": s,
                                   "_tp2": tp2, "smc": s[7] if len(s) > 7 else None})
                setups, _ = apply_entry_filters(setups, min_stop_distance_pct=self.min_stop_pct,
                                                timing_trends=list(timing) if timing is not None else None,
                                                timing_mode=self.timing_mode, **self.smc_filters)
            item = (symbol, setups, is_override, timing)
            if symbol in self.pending:
                holding.append(item)
            elif setups:
                (override if is_override else clean).append(item)

        for item in holding:
            if not self._sync(item, equity, px, t):
                open_count -= 1
        for item in clean + override:
            if open_count >= cfg.max_concurrent_positions:
                continue
            if self._sync(item, equity, px, t):
                open_count += 1

    def _zone_key(self, symbol, raw):
        return f"{symbol}|{raw[3]}|{round(raw[5], 6)}-{round(raw[4], 6)}"

    def _sync(self, item, equity, px, t):
        """bot._sync_symbol_orders: keep this symbol's resting orders equal
        to its setups. Returns True if it has a resting order afterwards."""
        symbol, setups, is_override, timing = item
        cfg = self.cfg
        wanted = {self._zone_key(symbol, s["_raw"]): s for s in setups}
        orders = [o for o in self.pending.get(symbol, []) if o["zone"] in wanted]
        have = {o["zone"] for o in orders}
        if orders:
            self.pending[symbol] = orders
        else:
            self.pending.pop(symbol, None)
        # As in the bot: every new order of this symbol is sized against the
        # notional that was open/reserved BEFORE any of them was placed
        # (RiskManager.total_open_notional() + pending_notional()).
        existing = sum(p["qty"] * p["entry"] for p in self.positions.values())
        existing += sum(max(o["qty"] * o["entry"] for o in lst) for lst in self.pending.values())
        for zone, s in wanted.items():
            if zone in have:
                continue
            seen = self.traded_zones.get(zone)
            if seen is not None and t - seen < ZONE_MEMORY_MS:
                continue
            raw = s["_raw"]
            stop = s["stop_loss"]
            qty, leverage = self.rm.position_size(equity, raw[1], stop, existing_notional=existing)
            if qty <= 0 or qty * raw[1] < cfg.min_notional_usdt:
                continue
            order = {"symbol": symbol, "ps": raw[0], "entry": raw[1], "stop": stop, "tp2": s.get("_tp2", raw[6]),
                     "qty": qty, "zone": zone, "placed": t, "leverage": leverage,
                     "timing": timing, "override": is_override,
                     "smc": raw[7] if len(raw) > 7 else None}
            price = px[symbol]
            inside = price <= raw[1] if raw[0] == "LONG" else price >= raw[1]
            if inside:
                # price already at/through the limit: BingX fills it at once
                self._open(order, price, t, taker=True)
                return True              # the symbol holds a slot (as a position)
            self.pending.setdefault(symbol, []).append(order)
        return symbol in self.pending


def simulate(cfg, costs, start_equity, signals, exec_data, symbols, **variant):
    """Runs one variant over the whole period. `signals` is
    {symbol: compute_signals() result}; all symbols share one account."""
    sim = Sim(cfg, costs, start_equity, **variant)
    step_index = {s: {t: i for i, t in enumerate(signals[s]["steps"])} for s in symbols}
    all_steps = sorted(set().union(*[signals[s]["steps"] for s in symbols]))
    exec_ts = {s: exec_data[s][:, 0].tolist() for s in symbols}
    exec_rows = {s: exec_data[s].tolist() for s in symbols}
    prev_t = None
    for t in all_steps:
        if prev_t is not None:
            for symbol in sorted(set(sim.positions) | set(sim.pending)):
                ts = exec_ts[symbol]
                a, b = bisect.bisect_left(ts, prev_t), bisect.bisect_left(ts, t)
                if b > a:
                    sim.walk(symbol, exec_rows[symbol][a:b], t)
        px, sig_at = {}, {}
        for s in symbols:
            i = step_index[s].get(t)
            if i is None:
                continue
            px[s] = signals[s]["px"][i]
            sig = signals[s]["signals"].get(i)
            if sig is not None:
                sig_at[s] = sig
        sim.run(t, px, sig_at, symbols)
        prev_t = t
    return sim


# ---------------------------------------------------------------------------
# 4. statistics and report
# ---------------------------------------------------------------------------

def stats(sim, t_mid=None):
    trades = sim.trades
    pnl = [t["pnl"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    r = [t["pnl"] / t["risk_usd"] for t in trades if t["risk_usd"] > 0]
    peak, dd = -1e18, 0.0
    for _, e in sim.equity_curve:
        peak = max(peak, e)
        dd = max(dd, (peak - e) / peak if peak > 0 else 0.0)
    out = {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "net_pnl": sum(pnl),
        "return_pct": sum(pnl) / sim.start_equity * 100,
        "avg_r": float(np.mean(r)) if r else 0.0,
        "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf") if wins else 0.0,
        "max_dd_pct": dd * 100,
        "fees": sum(t["fees"] for t in trades),
        "breaker_runs": sim.breaker_runs,
    }
    if t_mid is not None:
        out["pnl_h1"] = sum(t["pnl"] for t in trades if t["entry_time"] < t_mid)
        out["pnl_h2"] = sum(t["pnl"] for t in trades if t["entry_time"] >= t_mid)
        out["trades_h1"] = sum(1 for t in trades if t["entry_time"] < t_mid)
        out["trades_h2"] = out["trades"] - out["trades_h1"]
    return out


def breakdown(trades, key, order=None):
    groups = {}
    for t in trades:
        groups.setdefault(key(t), []).append(t)
    rows = []
    for name in (order or sorted(groups)):
        g = groups.get(name, [])
        if not g:
            continue
        pnl = sum(t["pnl"] for t in g)
        r = [t["pnl"] / t["risk_usd"] for t in g if t["risk_usd"] > 0]
        rows.append((name, len(g), sum(1 for t in g if t["pnl"] > 0) / len(g) * 100, pnl,
                     float(np.mean(r)) if r else 0.0, sum(t["fees"] for t in g)))
    return rows


def timing_state(trade):
    """How the timing levels' trends (1h/15m by default) stood relative to
    the trade when its order was placed."""
    timing = trade["timing"]
    if timing is None:
        return "n/a"
    want = "bullish" if trade["ps"] == "LONG" else "bearish"
    against = "bearish" if want == "bullish" else "bullish"
    if against in timing:
        return "against (a timing level opposite)"
    if all(x == want for x in timing):
        return "aligned (every timing level with the trade)"
    return "neutral (not against, not both aligned)"


STOP_BUCKETS = [(0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.40),
                (0.40, 0.50), (0.50, 0.75), (0.75, 1.0), (1.0, 1e9)]


def stop_bucket(trade):
    for lo, hi in STOP_BUCKETS:
        if lo <= trade["setup_stop_pct"] < hi:
            return f"{lo:.2f}-{hi:.2f}%" if hi < 1e9 else f">= {lo:.2f}%"
    return "?"


# ---- smart-money context of a trade (the tags its order carried when placed) ----

def structure_tag(trade):
    return (trade.get("smc") or {}).get("structure") or "n/a"


LIQUIDITY_NAMES = {"sweep": "sweep (an earlier swing pierced by wicks only)",
                   "run": "run (candles closed beyond the earlier swing)",
                   "none": "none (no earlier swing reached)"}


def liquidity_tag(trade):
    return LIQUIDITY_NAMES.get((trade.get("smc") or {}).get("liquidity"), "n/a")


def flag_tag(field):
    def key(trade):
        smc = trade.get("smc")
        return "n/a" if not smc else ("yes" if smc.get(field) else "no")
    return key


DISPLACEMENT_BUCKETS = [(0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]


def displacement_label(lo, hi):
    return f"{lo:g}-{hi:g} ATR" if hi < 1e9 else f">= {lo:g} ATR"


def displacement_bucket(trade):
    d = (trade.get("smc") or {}).get("displacement_atr")
    if d is None:
        return "n/a"
    for lo, hi in DISPLACEMENT_BUCKETS:
        if lo <= d < hi:
            return displacement_label(lo, hi)
    return "n/a"


RANGE_RIGHT = "right half (LONG in discount / SHORT in premium)"
RANGE_WRONG = "wrong half (LONG in premium / SHORT in discount)"


def range_half(trade):
    pos = (trade.get("smc") or {}).get("range_pos")
    if pos is None:
        return "n/a"
    return RANGE_RIGHT if (pos <= 0.5 if trade["ps"] == "LONG" else pos >= 0.5) else RANGE_WRONG


SESSIONS = [(0, 7, "00-07 UTC (Asia)"), (7, 12, "07-12 UTC (London morning)"),
            (12, 16, "12-16 UTC (New York morning)"), (16, 21, "16-21 UTC (New York afternoon)"),
            (21, 24, "21-24 UTC (late)")]


def session_of(trade):
    """By the hour the entry FILLED."""
    hour = int(trade["entry_time"] // 3_600_000) % 24
    for lo, hi, name in SESSIONS:
        if lo <= hour < hi:
            return name
    return "n/a"


# The combined candidate, fixed BEFORE it was run: the three changes that lost
# least in every earlier backtest (a stop beyond the previous swing, no stops
# tighter than 0.5%, timing levels not against the trade) plus bigger targets
# (TP1 at 2x the risk and TP2 at 3x, instead of 1x and the next swing). It
# passes only if it is profitable in BOTH halves of the period.
CANDIDATE_PARTS = [
    ("stop beyond the previous swing", {"stop_placement": "previous_swing"}),
    ("minimum stop 0.5%", {"min_stop_pct": 0.5}),
    ("timing levels not against the trade", {"timing_mode": "not_against"}),
    ("targets at 2R and 3R", {"tp1_r": 2.0, "tp2_r": 3.0}),
]


def exit_kind(trade):
    kinds = [e[1] for e in trade["exits"]]
    if "tp1" not in kinds:
        return "stopped out before TP1"
    return "TP1 + TP2, runner trailed" if "tp2" in kinds else "TP1 only, rest at breakeven/stop"


EXIT_KINDS = ["TP1 + TP2, runner trailed", "TP1 only, rest at breakeven/stop", "stopped out before TP1"]


# Each smart-money filter alone: (row name, Sim keyword arguments).
SMC_VARIANTS = [
    ("order blocks from a BOS only (OB_STRUCTURE=bos)", {"ob_structure": "bos"}),
    ("order blocks from a CHoCH only (OB_STRUCTURE=choch)", {"ob_structure": "choch"}),
    ("liquidity sweep required (REQUIRE_LIQUIDITY_SWEEP=true)", {"require_sweep": True}),
    ("displacement >= 1 ATR (MIN_DISPLACEMENT_ATR=1)", {"min_displacement_atr": 1.0}),
    ("displacement >= 1.5 ATR (MIN_DISPLACEMENT_ATR=1.5)", {"min_displacement_atr": 1.5}),
    ("displacement >= 2 ATR (MIN_DISPLACEMENT_ATR=2)", {"min_displacement_atr": 2.0}),
    ("displacement >= 3 ATR (MIN_DISPLACEMENT_ATR=3)", {"min_displacement_atr": 3.0}),
    ("fair value gap required (REQUIRE_FVG=true)", {"require_fvg": True}),
    ("premium/discount (PREMIUM_DISCOUNT_FILTER=true)", {"premium_discount": True}),
    ("entry hours 7-11 UTC (ENTRY_HOURS_UTC=7-11)", {"entry_hours": [(7, 11)]}),
    ("entry hours 12-16 UTC (ENTRY_HOURS_UTC=12-16)", {"entry_hours": [(12, 16)]}),
    ("entry hours 7-16 UTC (ENTRY_HOURS_UTC=7-16)", {"entry_hours": [(7, 16)]}),
]

# A few stacks of them, as the usual smart-money entry models combine them.
SMC_COMBOS = [
    ("sweep + fair value gap", {"require_sweep": True, "require_fvg": True}),
    ("sweep + displacement >= 1.5 ATR", {"require_sweep": True, "min_displacement_atr": 1.5}),
    ("fair value gap + displacement >= 1.5 ATR", {"require_fvg": True, "min_displacement_atr": 1.5}),
    ("CHoCH + sweep", {"ob_structure": "choch", "require_sweep": True}),
    ("BOS + fair value gap", {"ob_structure": "bos", "require_fvg": True}),
    ("sweep + fair value gap + premium/discount",
     {"require_sweep": True, "require_fvg": True, "premium_discount": True}),
    ("sweep + fair value gap + displacement >= 1.5 ATR + premium/discount",
     {"require_sweep": True, "require_fvg": True, "min_displacement_atr": 1.5, "premium_discount": True}),
]


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(out)


def fmt_variant_row(name, s):
    pf = "inf" if math.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
    return [name, s["trades"], f"{s['win_rate']:.1f}%", f"{s['net_pnl']:+.2f}", f"{s['return_pct']:+.1f}%",
            f"{s['avg_r']:+.3f}", pf, f"{s['max_dd_pct']:.1f}%", f"{s['fees']:.2f}",
            f"{s.get('pnl_h1', 0):+.2f} / {s.get('pnl_h2', 0):+.2f}"]


VARIANT_HEADER = ["variant", "trades", "win rate", "net P/L $", "return", "avg R/trade", "profit factor",
                  "max drawdown", "fees $", "net P/L 1st half / 2nd half"]

SYMBOL_SETS = {
    "all": [],
    "top 8 (BTC ETH SOL BNB XRP DOGE ADA LINK)": ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK"],
    "majors (BTC ETH SOL BNB)": ["BTC", "ETH", "SOL", "BNB"],
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest the BingX order-block bot on BingX history.")
    ap.add_argument("--days", type=int, default=180, help="length of the test period (default 180)")
    ap.add_argument("--end", default=None, help="last day of the period, YYYY-MM-DD UTC (default: today)")
    ap.add_argument("--symbols", default=None, help="comma-separated symbols (default: SYMBOLS / the bot's basket)")
    ap.add_argument("--exec-timeframe", default="auto",
                    help="candles used to walk price between two runs: auto (5m if BingX has it for the whole "
                         "period, else the entry timeframe), or e.g. 5m / 15m")
    ap.add_argument("--start-equity", type=float, default=230.0, help="account equity at the start, USDT")
    ap.add_argument("--maker-fee-pct", type=float, default=0.02, help="fee for resting limit fills, %% (BingX base tier 0.02)")
    ap.add_argument("--taker-fee-pct", type=float, default=0.05, help="fee for market / triggered fills, %% (BingX base tier 0.05)")
    ap.add_argument("--slippage-pct", type=float, default=0.02, help="slippage on market / triggered fills, %% of price")
    ap.add_argument("--out", default="backtest_results", help="output folder")
    ap.add_argument("--cache-dir", default=None, help="keep downloaded candles here between runs")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel workers for the signal pass")
    ap.add_argument("--skip-structure-compare", action="store_true",
                    help="skip the two extra signal passes that compare STRUCTURE_CONFIRMED_SWINGS and "
                         "REQUIRE_STRONG_SWING true vs false (saves roughly two thirds of the signal time)")
    ap.add_argument("--synthetic", type=int, default=None, metavar="SEED",
                    help="use random-walk candles instead of BingX (offline smoke test; results are meaningless)")
    args = ap.parse_args(argv)

    cfg = load_config()
    # The filters under test are applied by the simulation, variant by variant.
    cfg = replace(cfg, dry_run=True, min_stop_distance_pct=0.0, entry_symbols=[], htf_timing_agreement="off",
                  ob_structure="any", require_liquidity_sweep=False, min_displacement_atr=0.0,
                  require_fvg=False, premium_discount_filter=False, entry_hours_utc=[])
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else list(cfg.symbols)
    end = (datetime.datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
           if args.end else datetime.datetime.now(datetime.timezone.utc))
    end_ms = int(end.timestamp() * 1000)
    end_ms -= end_ms % TF_MS[cfg.timeframe]
    start_ms = end_ms - args.days * 86_400_000
    costs = Costs(args.maker_fee_pct, args.taker_fee_pct, args.slippage_pct)
    os.makedirs(args.out, exist_ok=True)

    log(f"Period: {datetime.datetime.fromtimestamp(start_ms / 1000, datetime.timezone.utc):%Y-%m-%d %H:%M} -> "
        f"{datetime.datetime.fromtimestamp(end_ms / 1000, datetime.timezone.utc):%Y-%m-%d %H:%M} UTC, "
        f"{len(symbols)} symbols")
    fetch_exec = "5m" if args.exec_timeframe == "auto" else args.exec_timeframe
    data = load_data(cfg, symbols, start_ms, end_ms, fetch_exec, args.cache_dir, args.synthetic)
    symbols = [s for s in symbols if s in data and len(data[s][cfg.timeframe]) > cfg.candle_limit + 10]
    symbols, start_ms, exec_tf, notes = usable_period(cfg, data, symbols, start_ms, args.exec_timeframe)
    for note in notes:
        log("NOTE: " + note)
    if not symbols or end_ms - start_ms < 2 * TF_MS["1d"]:
        log("No usable history -- nothing to backtest.")
        return 1
    days = (end_ms - start_ms) / TF_MS["1d"]

    log(f"Computing signals with {args.jobs} worker(s)...")
    t0 = time.time()
    jobs = [(cfg, s, data[s], start_ms, end_ms) for s in symbols]
    if args.jobs > 1 and len(jobs) > 1:
        with Pool(min(args.jobs, len(jobs))) as pool:
            results = pool.map(compute_signals, jobs)
    else:
        results = [compute_signals(j) for j in jobs]
    signals = {r["symbol"]: r for r in results}
    n_setups = sum(len(v[2]) for r in results for v in r["signals"].values())
    log(f"Signals done in {time.time() - t0:.0f}s: {sum(len(r['steps']) for r in results)} runs, "
        f"{n_setups} setup-observations.")
    with open(os.path.join(args.out, "signals.pkl"), "wb") as f:
        pickle.dump(signals, f)

    exec_data = {s: data[s][exec_tf] for s in symbols}
    t_mid = (start_ms + end_ms) // 2

    def run(**variant):
        return simulate(cfg, costs, args.start_equity, signals, exec_data, symbols, **variant)

    timing_label = "/".join(tf for tf in cfg.htf_timeframes if tf not in cfg.primary_htf_timeframes) or "timing levels"
    log("Simulating variants...")
    base = run()
    base_stats = stats(base, t_mid)

    # Settings that change which order blocks exist at all need their own
    # signal pass: the same account on the signals the OTHER value produces.
    def other_setting(label, **change):
        log(f"Computing signals for the other {label} setting...")
        t0 = time.time()
        other_cfg = replace(cfg, **change)
        other_jobs = [(other_cfg, s, data[s], start_ms, end_ms) for s in symbols]
        if args.jobs > 1 and len(other_jobs) > 1:
            with Pool(min(args.jobs, len(other_jobs))) as pool:
                other_results = pool.map(compute_signals, other_jobs)
        else:
            other_results = [compute_signals(j) for j in other_jobs]
        other_signals = {r["symbol"]: r for r in other_results}
        count = sum(len(v[2]) for r in other_results for v in r["signals"].values())
        log(f"Done in {time.time() - t0:.0f}s: {count} setup-observations (this run's setting: {n_setups}).")
        sim = simulate(other_cfg, costs, args.start_equity, other_signals, exec_data, symbols)
        return stats(sim, t_mid), count

    def swing_name(on):
        return ("true -- a swing counts once it is confirmed" if on
                else "false -- a swing counts at its own candle (before)")

    def strong_name(on):
        return ("true -- only order blocks at a strong swing" if on
                else "false -- every order block that caused a break (before)")

    rows_swing = [fmt_variant_row(swing_name(cfg.structure_confirmed_swings) + " [this run's setting]", base_stats)]
    rows_strong = [fmt_variant_row(strong_name(cfg.require_strong_swing) + " [this run's setting]", base_stats)]
    swing_note = strong_note = ""
    if args.skip_structure_compare:
        swing_note = strong_note = "The other setting was not simulated (--skip-structure-compare)."
    else:
        st, count = other_setting("STRUCTURE_CONFIRMED_SWINGS",
                                  structure_confirmed_swings=not cfg.structure_confirmed_swings)
        rows_swing.append(fmt_variant_row(swing_name(not cfg.structure_confirmed_swings), st))
        swing_note = (f"Setup-observations (a setup seen at one run): {n_setups} with this run's setting, "
                      f"{count} with the other.")
        st, count = other_setting("REQUIRE_STRONG_SWING", require_strong_swing=not cfg.require_strong_swing)
        rows_strong.append(fmt_variant_row(strong_name(not cfg.require_strong_swing), st))
        strong_note = (f"Setup-observations: {n_setups} with this run's setting, {count} with the other.")

    # Stop placement: the same setups with the stop beyond the order block,
    # the last swing, or the swing before it (all three are in the signals).
    stop_names = {"order_block": "order_block -- beyond the order block candle (before)",
                  "swing": "swing -- beyond the last swing",
                  "previous_swing": "previous_swing -- beyond the swing before it"}
    rows_stop_place, rows_stop_width, stop_json = [], [], []
    for mode, name in stop_names.items():
        sim = base if mode == cfg.stop_placement else run(stop_placement=mode)
        st = base_stats if mode == cfg.stop_placement else stats(sim, t_mid)
        if mode == cfg.stop_placement:
            name += " [this run's setting]"
        rows_stop_place.append(fmt_variant_row(name, st))
        widths = sorted(t["setup_stop_pct"] for t in sim.trades)
        risked = sum(t["risk_usd"] for t in sim.trades)
        rows_stop_width.append([name, f"{widths[len(widths) // 2]:.2f}%" if widths else "n/a",
                                f"${risked / len(sim.trades):.2f}" if sim.trades else "n/a",
                                f"{st['fees'] / risked * 100:.0f}%" if risked > 0 else "n/a"])
        stop_json.append({"stop_placement": mode, **st})

    # The combined candidate and what each of its parts does alone.
    rows_candidate = [fmt_variant_row("baseline (as the bot is set now)", base_stats)]
    for name, kw in CANDIDATE_PARTS:
        rows_candidate.append(fmt_variant_row("only: " + name, stats(run(**kw), t_mid)))
    no_targets = {k: v for name, kw in CANDIDATE_PARTS[:3] for k, v in kw.items()}
    rows_candidate.append(fmt_variant_row("the first three together (targets unchanged)", stats(run(**no_targets), t_mid)))
    candidate_kw = {k: v for name, kw in CANDIDATE_PARTS for k, v in kw.items()}
    candidate = run(**candidate_kw)
    candidate_stats = stats(candidate, t_mid)
    rows_candidate.append(fmt_variant_row("CANDIDATE: all four", candidate_stats))
    passed = candidate_stats["pnl_h1"] > 0 and candidate_stats["pnl_h2"] > 0
    verdict = (f"**{'PASSED' if passed else 'FAILED'}**: the candidate made {candidate_stats['pnl_h1']:+.2f} in the first half "
               f"and {candidate_stats['pnl_h2']:+.2f} in the second ({candidate_stats['trades_h1']} and "
               f"{candidate_stats['trades_h2']} trades). It passes only if both are above zero.")
    candidate_exits = breakdown(candidate.trades, exit_kind, order=EXIT_KINDS)

    rows_smc = [fmt_variant_row("all smart-money filters off", base_stats)]
    smc_json = []
    for name, kw in SMC_VARIANTS:
        st = stats(run(**kw), t_mid)
        rows_smc.append(fmt_variant_row(name, st))
        smc_json.append({"variant": name, **st})
    rows_smc_combo = [fmt_variant_row("all smart-money filters off", base_stats)]
    for name, kw in SMC_COMBOS:
        st = stats(run(**kw), t_mid)
        rows_smc_combo.append(fmt_variant_row(name, st))
        smc_json.append({"variant": name, **st})
    rows_exit = [fmt_variant_row("resting TP1/TP2 orders (live now)", base_stats),
                 fmt_variant_row("market close on the scheduled run (before)", stats(run(exit_mode="run"), t_mid))]

    rows_stop = [fmt_variant_row("off (0)", base_stats)]
    for m in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75):
        rows_stop.append(fmt_variant_row(f"{m:.2f}%", stats(run(min_stop_pct=m), t_mid)))

    rows_timing = [fmt_variant_row("off", base_stats)]
    for mode in ("not_against", "aligned"):
        rows_timing.append(fmt_variant_row(mode, stats(run(timing_mode=mode), t_mid)))

    rows_sets = []
    for name, members in SYMBOL_SETS.items():
        s = base_stats if not members else stats(run(entry_symbols=members), t_mid)
        rows_sets.append(fmt_variant_row(name, s))

    combos = []
    for m in (0.0, 0.20, 0.30):
        for mode in ("off", "not_against", "aligned"):
            for name, members in SYMBOL_SETS.items():
                s = stats(run(min_stop_pct=m, timing_mode=mode, entry_symbols=members), t_mid)
                combos.append((s["net_pnl"], fmt_variant_row(
                    f"min stop {m:.2f}% + {timing_label} {mode} + {name.split(' (')[0]}", s), m, mode, name, s))
    combos.sort(key=lambda x: -x[0])

    by_symbol = breakdown(base.trades, lambda t: t["symbol"].split("/")[0])
    by_symbol.sort(key=lambda r: -r[3])
    by_stop = breakdown(base.trades, stop_bucket,
                        order=[f"{lo:.2f}-{hi:.2f}%" if hi < 1e9 else f">= {lo:.2f}%" for lo, hi in STOP_BUCKETS])
    by_timing = breakdown(base.trades, timing_state)
    by_structure = breakdown(base.trades, structure_tag, order=["BOS", "CHoCH", "n/a"])
    by_sweep = breakdown(base.trades, liquidity_tag, order=[LIQUIDITY_NAMES[k] for k in ("sweep", "run", "none")] + ["n/a"])
    by_fvg = breakdown(base.trades, flag_tag("fvg"), order=["yes", "no", "n/a"])
    by_displacement = breakdown(base.trades, displacement_bucket,
                                order=[displacement_label(lo, hi) for lo, hi in DISPLACEMENT_BUCKETS] + ["n/a"])
    by_range = breakdown(base.trades, range_half, order=[RANGE_RIGHT, RANGE_WRONG, "n/a"])
    by_session = breakdown(base.trades, session_of, order=[name for _, _, name in SESSIONS])
    by_side = breakdown(base.trades, lambda t: t["ps"])
    by_exit = breakdown(base.trades, exit_kind, order=EXIT_KINDS)

    def brk(rows, first):
        return md_table([first, "trades", "win rate", "net P/L $", "avg R/trade", "fees $"],
                        [[r[0], r[1], f"{r[2]:.1f}%", f"{r[3]:+.2f}", f"{r[4]:+.3f}", f"{r[5]:.2f}"] for r in rows])

    period = (f"{datetime.datetime.fromtimestamp(start_ms / 1000, datetime.timezone.utc):%Y-%m-%d} to "
              f"{datetime.datetime.fromtimestamp(end_ms / 1000, datetime.timezone.utc):%Y-%m-%d}")
    source = "SYNTHETIC random-walk candles (smoke test -- numbers are meaningless)" if args.synthetic is not None \
        else "BingX USDT-M perpetual candles"
    note_text = ("\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
    report = f"""# Backtest: {period} ({days:.0f} days)

Data: {source}. {len(symbols)} symbols: {", ".join(s.split("/")[0] for s in symbols)}.
{note_text}
Timeframes: entries on {cfg.timeframe}; direction from {"/".join(cfg.primary_htf_timeframes)}; timing levels {timing_label}. Stop: beyond the {cfg.stop_placement.replace("_", " ")}. Order blocks: {"only at strong swings" if cfg.require_strong_swing else "every one that caused a break"}; swings {"confirmed first" if cfg.structure_confirmed_swings else "counted at their own candle"}.
Settings (from the bot's configuration): risk {cfg.risk_pct_per_trade * 100:g}% per trade, margin ${cfg.margin_per_trade_usd:g}, leverage ceiling {min(cfg.leverage, cfg.max_leverage_hard_cap)}x, max {cfg.max_concurrent_positions} positions, daily loss limit {cfg.max_daily_loss_pct * 100:g}%, primary-level override room {cfg.weekly_daily_override_min_room_pct:g}%, TP1 {cfg.partial_tp_fraction:.0%} at {cfg.partial_tp_r_multiple:g}R, TP2 {cfg.tp2_fraction:.0%}, trail {cfg.trail_r_multiple:g}R.
Costs: maker {args.maker_fee_pct}%, taker {args.taker_fee_pct}%, slippage {args.slippage_pct}% on market/triggered fills. Starting equity ${args.start_equity:g}. Execution walked on {exec_tf} candles.

"R" is the amount risked on a trade (size x stop distance); "avg R/trade" is net P/L after fees in units of that. The two halves split the period in the middle ({day(t_mid)}) by entry time.

## Baseline -- the live configuration, all filters off

{md_table(VARIANT_HEADER, [fmt_variant_row("baseline", base_stats)])}

Daily-loss breaker active on {base_stats["breaker_runs"]} of {len(base.equity_curve)} runs.

## Combined candidate

{md_table(VARIANT_HEADER, rows_candidate)}

{verdict}

The candidate was fixed before this run: stop beyond the previous swing, no stop tighter than 0.5%, timing levels ({timing_label}) not against the trade, TP1 at 2x the risk (half the position, stop to breakeven) and TP2 at 3x (30%), runner trailing as before.

How the candidate's trades ended:

{brk(candidate_exits, "how the trade ended")}

## Exit handling

{md_table(VARIANT_HEADER, rows_exit)}

## Minimum stop distance (MIN_STOP_DISTANCE_PCT), other filters off

{md_table(VARIANT_HEADER, rows_stop)}

Baseline trades by the setup's stop distance (limit price to stop -- what the minimum is compared with):

{brk(by_stop, "stop distance")}

## {timing_label} agreement (HTF_TIMING_AGREEMENT), other filters off

{md_table(VARIANT_HEADER, rows_timing)}

Baseline trades by how {timing_label} stood at entry:

{brk(by_timing, f"{timing_label} vs trade")}

## Symbol set (ENTRY_SYMBOLS), other filters off

{md_table(VARIANT_HEADER, rows_sets)}

Baseline trades by symbol:

{brk(by_symbol, "symbol")}

## Combinations (best net P/L first)

{md_table(VARIANT_HEADER, [c[1] for c in combos])}

## Stop placement (STOP_PLACEMENT), other filters off

{md_table(VARIANT_HEADER, rows_stop_place)}

{md_table(["stop placement", "median stop distance", "average $ risked per trade", "fees as a share of what was risked"], rows_stop_width)}

The stop is always 0.1% of the price beyond its level. "swing" is the swing low (LONG) / swing high (SHORT) the order block belongs to; it equals "order_block" whenever the order block candle made that low/high itself. A wider stop also moves TP1 further away (TP1 mirrors the stop).

## Market structure: swing confirmation (STRUCTURE_CONFIRMED_SWINGS)

{md_table(VARIANT_HEADER, rows_swing)}

{swing_note}
With "false", an order block whose breaking candle turns out to be a swing itself disappears again once that swing is confirmed, and its order is cancelled; with "true" it stays. So "true" keeps more order blocks -- more orders and more trades, not the same trades filtered.

## Market structure: strong swings (REQUIRE_STRONG_SWING)

{md_table(VARIANT_HEADER, rows_strong)}

{strong_note}
A strong swing is a swing low/high from which price went on to cause a BOS or CHoCH. With "true", an order block only counts if the move that broke structure started from a real swing point; one that came out of a pause of a candle or two is ignored.

## Smart-money filters, each alone (other filters off)

{md_table(VARIANT_HEADER, rows_smc)}

A few of them stacked:

{md_table(VARIANT_HEADER, rows_smc_combo)}

Baseline trades by what their order block looked like when the order was placed -- the same facts the filters above test. BOS = the order block caused a break WITH the previous break (continuation), CHoCH = the first break AGAINST it (reversal):

{brk(by_structure, "order block from a")}

{brk(by_sweep, "liquidity taken before it")}

{brk(by_fvg, "fair value gap left by the move")}

{brk(by_displacement, "displacement")}

{brk(by_range, "entry in the dealing range")}

{brk(by_session, "hour the entry filled")}

## More baseline detail

{brk(by_side, "side")}

{brk(by_exit, "how the trade ended")}

## Reading this

- These are simulated fills on historical candles, under the assumptions listed at the top of backtest.py. Real results will differ.
- A filter that only looks good in one half of the period, or that leaves very few trades, is probably fitting noise.
- Per-symbol results come from few trades each; a symbol set chosen from this table is fitted to this period.
- Many filters are compared here. With this many tries a few will look good by luck alone; the more rows a table has, the less its best row means. Prefer a filter that makes sense to you beforehand, helps in both halves and still leaves a decent number of trades.
"""
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(report)
    with open(os.path.join(args.out, "trades_baseline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "side", "entry_time_utc", "exit_time_utc", "entry", "initial_stop", "setup_stop_pct", "stop_pct",
                    "qty", "leverage", "risk_usd", "pnl", "r_multiple", "fees", "exit_kind", "timing_4h_1h",
                    "override", "taker_entry", "exits", "ob_structure", "liquidity", "fair_value_gap",
                    "displacement_atr", "range_pos"])
        for t in sorted(base.trades, key=lambda x: x["entry_time"]):
            iso = lambda ms: datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).isoformat()
            smc = t.get("smc") or {}
            w.writerow([t["symbol"], t["ps"], iso(t["entry_time"]), iso(t["exit_time"]), t["entry"], t["initial_stop"],
                        round(t["setup_stop_pct"], 4), round(t["stop_pct"], 4), t["initial_qty"], t["leverage"], round(t["risk_usd"], 4),
                        round(t["pnl"], 4), round(t["pnl"] / t["risk_usd"], 3) if t["risk_usd"] else "",
                        round(t["fees"], 4), t["exit_kind"], "/".join(t["timing"]) if t["timing"] else "",
                        t["override"], t["taker_entry"],
                        "; ".join(f"{k} {q:.6g}@{p:.6g}" for _, k, q, p in t["exits"]),
                        smc.get("structure", ""), smc.get("liquidity", ""), smc.get("fvg", ""),
                        round(smc["displacement_atr"], 3) if smc.get("displacement_atr") is not None else "",
                        round(smc["range_pos"], 3) if smc.get("range_pos") is not None else ""])
    with open(os.path.join(args.out, "variants.json"), "w") as f:
        json.dump({"baseline": base_stats,
                   "combinations": [{"min_stop_pct": c[2], "timing": c[3], "symbols": c[4], **c[5]} for c in combos],
                   "smart_money": smc_json, "stop_placement": stop_json,
                   "candidate": {"settings": candidate_kw, "passed": passed, **candidate_stats}},
                  f, indent=2, default=str)
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
