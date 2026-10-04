# BingX Order-Block Futures Bot

A rule-based bot for BingX USDT-M perpetual futures that trades an "order
block" (smart-money-concept) strategy with a top-down multi-timeframe
filter: the weekly and daily trend (the "primary" levels) set the
allowed direction, then it looks for an order-block retest on the 15m
chart in that direction, using the 4h and 1h trend as entry-timing
context rather than a veto. Every entry gets an automatic stop-loss
(hard-capped at 3% of equity) sized so it also never asks for more
margin than the account can actually support. At 1R (the stop-loss mirrored) half the position takes profit and the
stop for the rest moves to breakeven; 30% more closes at the next swing
high/low (TP2), and the last 20% trails with no fixed cap. A daily-loss circuit
breaker halts new entries if losses get too large for the day.

## How a trade gets taken

1. **Weekly** -- trend is read from an EMA of weekly closes, and weekly
   order blocks are located for context. This is a **primary** level: it
   sets the allowed trade direction.
2. **Daily** -- same analysis. Also **primary**: if the daily trend
   clearly contradicts the weekly trend, the symbol is skipped entirely
   for this cycle -- the bot never trades against its primary timeframes'
   trend. If weekly and daily agree (or either/both are neutral), that
   becomes the one direction (or "either direction") a trade is allowed
   to take below.
3. **4h** -- same analysis again, but this level is for **timing** only:
   it's still fetched, analyzed, and logged every cycle, but it can never
   veto a trade on its own. A 4h pullback against a strong weekly/daily
   uptrend is normal -- often exactly the retracement this strategy is
   trying to trade -- not a reason to skip it.
4. **1h** -- same analysis again, also timing-only, same reasoning as 4h.
5. **15m (entry timeframe)** -- the bot looks for a fresh order-block
   retest signal, and the signal's direction must match the PRIMARY
   (weekly/daily) bias established in steps 1-2. An all-neutral primary
   read imposes no directional constraint at all.
6. **Stop-loss & take-profit** -- the stop-loss sits just beyond the 15m
   order block that produced the signal (below it for a LONG, above it
   for a SHORT). Every trade commits a FIXED margin
   (`MARGIN_PER_TRADE_USD`, $50 by default); the LEVERAGE is then
   solved per trade from the stop-loss distance so the trade's dollar
   risk hits `RISK_PCT_PER_TRADE` of equity (hard-capped at 3%
   regardless of what `RISK_PCT_PER_TRADE` is set to) -- a tighter stop
   gets more leverage to reach that risk target, a wider stop gets
   less, and leverage never exceeds the `LEVERAGE` ceiling (itself
   hard-capped at 10x). Separately, the combined notional of every open
   position -- this one plus whatever's already open -- never requires
   more margin than `MAX_MARGIN_UTILIZATION_PCT` of the account's
   theoretical capacity at the 10x hard-cap leverage, as a backstop
   against several concurrent positions collectively over-margining the
   account.
7. **TP1 / TP2 / trailing runner** -- three exits per trade:
   - **TP1** is the stop-loss mirrored to the other side of the entry
     (`PARTIAL_TP_R_MULTIPLE` x the initial risk, 1R by default: a stop
     $5 below entry puts TP1 $5 above it). `PARTIAL_TP_FRACTION` of the
     position (50%) closes there and the stop for the rest moves to
     breakeven.
   - **TP2** is the next swing high (LONG) / swing low (SHORT) beyond TP1
     on the entry timeframe (or `TP2_FALLBACK_R_MULTIPLE`, 2R, if there
     is none). `TP2_FRACTION` of the original position (30%) closes
     there. The stop stays at breakeven between TP1 and TP2.
   - **Runner** -- the remaining 20% trails `TRAIL_R_MULTIPLE` behind the
     best price reached, with no fixed take-profit cap, starting after TP2.
   In LIVE mode TP1 and TP2 rest on BingX as reduce-only take-profit
   orders (see "Exit orders on the exchange" below), so they fill the
   moment price gets there instead of waiting for the next scheduled run;
   the stop-loss is always on the exchange. Set `PARTIAL_TP_FRACTION=0` to
   disable all of this and fall back to a fixed-2R-take-profit-only exit.

See `strategy.combine_htf_bias()` and `bot.analyze_higher_timeframes()`
for the primary/timing cascade logic, `risk_manager.position_size()` for
the margin cap, `risk_manager.reconcile_dry_run_exits()` /
`reconcile_live_exits()` / `trail_live_stops()` for the partial-TP and
trailing-stop mechanics, and `config.py` for every hard cap and default.

## Exit orders on the exchange (LIVE)

The first run that sees a filled entry puts the whole exit plan on BingX:

- **TP1 and TP2** as resting reduce-only `TAKE_PROFIT_MARKET` orders. BingX
  closes that part at market when price reaches the level, whether or not
  the bot is running. (Trigger orders rather than limit orders on purpose:
  a resting limit close reserves that part of the position, and the
  stop-loss could then be refused for lack of available position.)
- **One stop order per part** of the position -- TP1 part, TP2 part, runner
  -- all at the stop price, replacing the single full-size stop BingX
  attached to the entry. A take-profit can fill between two runs; until the
  next run moves the stops, they must still add up to what is left. With one
  stop per part they do, whichever parts have already closed.
- On the next run the bot reads the fills from BingX, journals them at the
  **real fill price and fee**, moves the remaining stops to breakeven after
  TP1 and arms the trailing stop after TP2. If the breakeven stop can't be
  placed yet (price is already back through the entry), the previous stops
  stay and the move is retried every run.
- When a trade closes, every stop / take-profit order it left behind is
  cancelled.

**Fallback.** If the per-part stops can't be placed, nothing changes for
that trade: the original stop stays and TP1/TP2 are closed at market by the
scheduled run that sees price there (the behaviour before). If only one
take-profit can't be placed -- typically because price is already past it --
only that one falls back. `RESTING_TP_ORDERS=false` turns the feature off for
new trades.

Between the entry fill and the first run after it, only the stop is on the
exchange -- the take-profits appear with that run.

## Entry filters (all off by default)

| Setting | What it does |
|---|---|
| `MIN_STOP_DISTANCE_PCT` | No order for a setup whose stop is closer to the entry than this % of the entry price. `0` = off. |
| `ENTRY_SYMBOLS` | Only these symbols may take new entries (`BTC,ETH` or full symbols). Empty = all. Symbols left out are still watched, so their open positions keep being managed -- use this rather than shrinking `SYMBOLS`. |
| `HTF_TIMING_AGREEMENT` | `off`: 4h/1h never block a trade. `not_against`: no order if 4h or 1h trends against the trade. `aligned`: no order unless 4h and 1h both trend with it. |

The filters only remove setups; a filtered-out setup is not replaced by an
older order block. `backtest.py` applies them through the same function the
bot uses.

## Backtest

`backtest.py` replays the strategy on BingX history with fees and compares
the entry filters:

```
python3 backtest.py --days 180
```

or run the **Backtest** workflow from the Actions tab (the report lands on
the run's summary page). It needs no API key (public candles only) and uses
the bot's settings from the same environment variables. For every 15-minute
step it rebuilds the candles the bot would have fetched at that moment and
calls the bot's own strategy code, then simulates one account: resting limit
entries, position slots, sizing, the margin cap, the daily-loss breaker and
the TP1/TP2/breakeven/trailing exits, with maker/taker fees and slippage.

It is a simulation -- the assumptions (run timing, what happens when one
candle touches both stop and take-profit, no amount rounding, no funding)
are listed at the top of `backtest.py`. Choosing the best-looking filter
values from one stretch of history over-fits; the report splits the period
in two halves so you can check a filter helps in both.

## Tests

```
pip install pytest
python3 -m pytest tests
```

`tests/` runs the real bot code against an in-memory stand-in for BingX
(`tests/fake_bingx.py`): the full exit lifecycle, every fallback, take-profits
filling between runs followed by a stop-out, journal/fee accounting against
the account's actual P/L, and a check that the backtest finds exactly the
setups the bot's own code finds.

## Files

| File | Purpose |
|---|---|
| `bot.py` | Entry point -- runs one full check-and-maybe-trade cycle, including the weekly/daily (primary) + 4h/1h (timing) cascade and per-cycle trade management |
| `strategy.py` | Order block detection, limit-order setups, EMA trend detection, the multi-timeframe bias combiner, and the optional entry filters |
| `risk_manager.py` | Position sizing (risk + margin caps), leverage cap, daily-loss circuit breaker, LIVE exit orders (per-part stops, resting TP1/TP2, breakeven, trailing) with real-fill journalling, DRY_RUN P/L simulation (with an estimated fee deduction) |
| `exchange_client.py` | Thin wrapper around ccxt's BingX integration: entries with attached stop, stop / take-profit orders, fill and fee lookup |
| `backtest.py` | Replays the strategy on BingX history with fees and compares the entry filters |
| `tests/` | Scenario tests against an in-memory fake exchange |
| `config.py` | Loads settings from environment variables, and enforces the hard caps (3% risk, margin utilization, fixed 2:1 reward:risk, 10x leverage) |
| `.env.example` | Template for your API keys and settings |
| `.gitignore` | Keeps `.env`, Python bytecode caches, and OS metadata files out of version control |
| `run.sh` | Loads `.env` and runs one cycle -- point cron at this |
| `.github/workflows/bot.yml` | GitHub Actions alternative to cron -- runs on a schedule and commits state/journal updates back to the repo |
| `.github/workflows/backtest.yml` | Runs `backtest.py` on demand from the Actions tab |
| `data/state.json` | Auto-created; tracks today's P&L and open trades (including partial-TP/trailing state and the ids of their exit orders) for the circuit breaker and exit reconciliation |
| `data/trades.csv` | Auto-created trade journal (`ENTRY`/`PARTIAL_EXIT`/`EXIT` rows, plus limit-order events). LIVE exit rows carry the real fill price, the fee (`fee` column) and P/L net of fees |

The strategy, risk-management, multi-timeframe cascade, margin-cap
sizing, and partial-take-profit/trailing-stop logic were all tested in
this session against hand-built and synthetic scenarios (bullish retest
-> correct LONG signal, a close-through -> correctly produces no trade,
a primary-level disagreement blocking a trade while a timing-level-only
disagreement does not, an oversized-notional trade (reproduced from this
bot's own live trade history) getting capped by the margin check instead
of the pure risk-based size, and a full partial-take-profit -> breakeven
-> trail -> stop-out lifecycle, including the fee deduction reducing the
simulated P/L). The exit orders resting on the exchange (per-part stops,
resting TP1/TP2) are tested against a fake exchange in `tests/`, including
both ways BingX might treat a triggered close order bigger than the
position (reject it, or close what is left) -- which of the two BingX
really does could not be checked from where this was written, which is why
the stops are split so that neither case leaves a position unprotected.
Watch the first live trade that takes TP1 on BingX itself (see Step 3
below).

## Where this needs to run

It cannot run inside this chat. This sandbox's network is locked down to
an allowlist that blocks both PyPI and BingX's API entirely, so nothing
here can pip install or reach bingx.com. You'll need to run it somewhere
with normal internet access -- your own computer, a small VPS (e.g. a
$5/mo box), or similar. The scheduling instructions below assume a plain
cron job on Linux/macOS; swap in Task Scheduler or a systemd timer if you
prefer (a GitHub Actions workflow is also included -- see `.github/workflows/bot.yml`).

## Setup

1. Get the code onto a machine with internet access, then:

```
cd bingx-order-block-bot
pip install -r requirements.txt
cp .env.example .env
```

2. Create a BingX API key (BingX -> Account -> API Management):

- Enable Perpetual Futures trading permission
- Leave withdrawal permission OFF -- always
- Set an IP whitelist to the machine that will run the bot
- Paste the key/secret into `.env`. Never commit `.env` or paste your secret key into a chat.

3. Validate before trading real money, in this order:

- Run `python3 bot.py` a few times with `DRY_RUN=true` (the default) and confirm it reads your balance, fetches weekly/daily/4h/1h/15m candles, and logs sensible trend/signal output without error.
- If you can, point `SYMBOLS`/test against BingX's demo trading (VST) environment first, or at minimum keep `RISK_PCT_PER_TRADE` very small for the first week live.
- Open a Python REPL and place one manual small test order through `ExchangeClient` to confirm `positionSide` and stop-order params match your account's position mode (BingX supports both "one-way" and "hedge" mode, which changes some order parameters), and confirm `fetch_trade_exit()`'s field-name assumptions (`reduceOnly`, `positionSide`, `realizedPnl`) against a real closed trade. Also specifically exercise a partial take-profit once live (or watch closely the first time one fires): confirm `close_partial_position()` reduces the position by the expected amount and `replace_stop_order()` cancels the correct existing order (not the take-profit) and re-places the stop at breakeven. This is the one set of steps that could not be verified from this environment -- do not skip it.
- Only then set `DRY_RUN=false`.

4. Schedule it. The strategy runs on the `TIMEFRAME` candle (default 15m), so a cron entry that fires a bit more often than that is enough -- the bot itself is idempotent per cycle (it checks existing positions before opening a new one):

```
*/5 * * * * /path/to/bingx-order-block-bot/run.sh >> /path/to/bingx-order-block-bot/data/cron.log 2>&1
```

Or use the included `.github/workflows/bot.yml` (runs every 15 minutes via GitHub Actions; needs `BINGX_API_KEY`/`BINGX_API_SECRET` as repo Secrets and the config knobs below as repo Variables).

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `SYMBOLS` | (15-symbol default basket) | Comma-separated ccxt unified symbols for the USDT-M perpetuals to scan, highest-priority first |
| `TIMEFRAME` | `15m` | Entry candle timeframe -- where the order-block retest signal itself fires |
| `CANDLE_LIMIT` | `300` | How many entry-timeframe candles to fetch per cycle |
| `FRACTAL_N` | `2` | Fractal window (bars each side) used for swing-high/low detection, on every timeframe |
| `REWARD_RISK_RATIO` | `2` | Ignored: always forced to `2` (the fixed take-profit is twice the stop-loss distance -- see partial-TP/trailing above for what usually happens before that fixed target is reached) |
| `TREND_FILTER_ENABLED` | `true` | Master toggle for the weekly/daily/4h/1h cascade |
| `HTF_TIMEFRAMES` | `1w,1d,4h,1h` | Every level analyzed and logged each cycle, highest timeframe first |
| `HTF_PRIMARY_TIMEFRAMES` | `1w,1d` | Which of the levels above can actually veto a trade on disagreement; the rest are timing-only context |
| `TREND_EMA_PERIOD` | `50` | EMA period used for the trend read at every cascade level |
| `TREND_CANDLE_LIMIT` | `150` | How many candles to fetch per cascade level |
| `MARGIN_PER_TRADE_USD` | `50` | Fixed margin (USDT) committed to every trade; leverage is solved per trade from the stop-loss distance to hit `RISK_PCT_PER_TRADE` (see `risk_manager.position_size()`) |
| `RISK_PCT_PER_TRADE` | `3` | Target % of account equity risked per trade, used to solve each trade's leverage; hard-capped at 3% regardless of this value (defaults to that same ceiling) |
| `LEVERAGE` | `3` | Leverage CEILING for the auto-sizer above -- the leverage it solves for per trade never exceeds this (or the 10x hard cap, whichever is lower) |
| `MAX_MARGIN_UTILIZATION_PCT` | `50` | Open positions combined may never require more than this % of equity*leverage in margin, regardless of what the risk-based size alone would compute |
| `MAX_DAILY_LOSS_PCT` | `3` | Daily loss % that halts new entries: checked against both the drop in account equity since the start of the UTC day and the day's realized P/L (net of fees), whichever is worse |
| `MAX_CONCURRENT_POSITIONS` | `1` | Won't open a new position while this many are already open -- now counts DRY_RUN paper trades the same as LIVE positions, so paper-trading behavior is a faithful preview of live behavior |
| `PARTIAL_TP_R_MULTIPLE` | `1.0` | R-multiple at which the partial take-profit fires |
| `PARTIAL_TP_FRACTION` | `0.5` | Fraction of the position closed at the partial take-profit; set to `0` to disable partial-TP/trailing entirely |
| `TRAIL_R_MULTIPLE` | `1.0` | How many R behind the best price reached so far the trailing stop sits, once armed |
| `TP2_FRACTION` | `0.3` | Fraction of the ORIGINAL position closed at TP2 -- the next swing high (LONG) / swing low (SHORT) beyond TP1 on the entry timeframe. The trailing stop only arms for what is left after TP2 (20% by default). Set to 0 to skip TP2 and trail right after TP1 |
| `TP2_FALLBACK_R_MULTIPLE` | `2.0` | Where TP2 goes, in multiples of the initial risk, when there is no swing beyond TP1 in the fetched candles |
| `RESTING_TP_ORDERS` | `true` | LIVE: rest TP1/TP2 on BingX with one stop per part of the position (see "Exit orders on the exchange"). `false` = market closes on the scheduled run, as before |
| `MIN_STOP_DISTANCE_PCT` | `0` | Entry filter: minimum stop distance in % of the entry price; `0` = off |
| `ENTRY_SYMBOLS` | (empty) | Entry filter: symbols allowed to take new entries; empty = all |
| `HTF_TIMING_AGREEMENT` | `off` | Entry filter: `off` / `not_against` / `aligned` -- whether 4h/1h must agree with the trade |
| `DRY_RUN_FEE_PCT_PER_SIDE` | `0.05` (%) | Taker-fee estimate per side: subtracted from DRY_RUN's simulated P/L, and used in LIVE only when BingX doesn't report a fill's fee (the journal row says so) |
| `DRY_RUN` | `true` | Logs signals only (and simulates their outcome); set `false` to place real orders |

## Important limitations and risks -- please read

- Leverage amplifies losses, including to zero (liquidation). A 3x leveraged position can be liquidated by a ~33% adverse move; the code's hard leverage cap (10x) and margin-utilization cap are backstops, not a recommendation to use them.
- Order blocks, as implemented here, are one reasonable interpretation of a discretionary trading concept, not a guaranteed edge. `backtest.py` replays the strategy on BingX history, but a backtest is a simulation under stated assumptions, not proof of future results. DRY_RUN's simulated P/L nets out an estimated trading fee, but still doesn't model slippage -- real fills on a fast-moving stop or trailing exit can be worse than the simulated price.
- API/library details can drift. BingX's position-mode handling (`positionSide`, one-way vs. hedge accounts) and ccxt's exact param names for `STOP_MARKET`/`TAKE_PROFIT_MARKET` orders, the `info` field names `exchange_client.fetch_trade_exit()` reads for exit fills (`reduceOnly`, `positionSide`, `realizedPnl`), and the stop / take-profit order calls are the most likely things to need a small adjustment for your account -- this is what Step 3 above is for.
- The weekly/daily/4h/1h cascade means the bot needs more historical candles and more API calls per symbol per cycle (five timeframes instead of one), and trade management (checking every open position's candle for a partial-TP/trailing trigger) adds one more fetch per open symbol per cycle. If you watch many symbols on a tight schedule, keep an eye on BingX's rate limits.
- The margin-utilization cap is a heuristic based on each open trade's entry price, not a live mark-to-market margin read from the exchange -- it substantially reduces the risk of a rejected order or an unexpectedly over-leveraged account, but isn't a guarantee BingX's own margin calculation will match exactly.
- Nothing here is financial advice, and I'm not a financial advisor -- this is software that automates a rule set you should understand and agree with before it trades your money.
- Keep `.env` out of version control and off any machine you don't control; anyone with the API key/secret can trade on your account (not withdraw, since that permission is off).
