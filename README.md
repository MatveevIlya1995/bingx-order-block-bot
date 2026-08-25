# BingX Order-Block Futures Bot

A rule-based bot for BingX USDT-M perpetual futures that trades an "order
block" (smart-money-concept) strategy with a top-down multi-timeframe
filter: it only takes a trade when the weekly, daily, 4h and 1h trends
agree (or have no opinion), then looks for an order-block retest on the
15m chart in that direction. Every entry gets an automatic stop-loss
(hard-capped at 3% of equity) and a take-profit that's always exactly
twice the stop-loss distance. A daily-loss circuit breaker halts new
entries if losses get too large for the day.

## How a trade gets taken

1. **Weekly** -- trend is read from an EMA of weekly closes, and weekly
   order blocks are located for context.
2. **Daily** -- same analysis, and the daily trend must not contradict
   the weekly trend.
3. **4h** -- same analysis again, must not contradict weekly/daily.
4. **1h** -- same analysis again, must not contradict the levels above.
   If any two of these four levels disagree (one bullish, one bearish),
   the symbol is skipped entirely for this cycle -- the bot never trades
   against a higher timeframe's trend. If every level agrees, or every
   level is neutral/unclear, that becomes the one direction (or "either
   direction") a trade is allowed to take below.
5. **15m (entry timeframe)** -- the bot looks for a fresh order-block
   retest signal, exactly as before, but the signal's direction must
   match the bias established in steps 1-4.
6. **Stop-loss & take-profit** -- the stop-loss sits just beyond the 15m
   order block that produced the signal (below it for a LONG, above it
   for a SHORT), and position size is chosen so that stop-out risk is
   never more than 3% of account equity, regardless of what
   `RISK_PCT_PER_TRADE` is set to. Take-profit is always exactly 2x that
   risk, regardless of what `REWARD_RISK_RATIO` is set to.

See `strategy.combine_htf_bias()` and `bot.analyze_higher_timeframes()`
for the exact cascade logic, and `config.py` for the hard caps.

## Where this needs to run

It cannot run inside this chat. This sandbox's network is locked down to
an allowlist that blocks both PyPI and BingX's API entirely, so nothing
here can pip install or reach bingx.com. You'll need to run it somewhere
with normal internet access -- your own computer, a small VPS (e.g. a
$5/mo box), or similar. The scheduling instructions below assume a plain
cron job on Linux/macOS; swap in Task Scheduler or a systemd timer if you
prefer (a GitHub Actions workflow is also included -- see `.github/workflows/bot.yml`).

## Files

| File | Purpose |
|---|---|
| `bot.py` | Entry point -- runs one full check-and-maybe-trade cycle, including the weekly/daily/4h/1h cascade |
| `strategy.py` | Order block detection, retest signal generation, EMA trend detection, and the multi-timeframe bias combiner |
| `risk_manager.py` | Position sizing, leverage cap, daily-loss circuit breaker |
| `exchange_client.py` | Thin wrapper around ccxt's BingX integration, including exit-fill lookup for the trade journal |
| `config.py` | Loads settings from environment variables, and enforces the hard caps (3% risk, fixed 2:1 reward:risk, 10x leverage) |
| `.env.example` | Template for your API keys and settings |
| `run.sh` | Loads `.env` and runs one cycle -- point cron at this |
| `.github/workflows/bot.yml` | GitHub Actions alternative to cron -- runs on a schedule and commits state/journal updates back to the repo |
| `data/state.json` | Auto-created; tracks today's P&L and open trades for the circuit breaker and exit reconciliation |
| `data/trades.csv` | Auto-created trade journal (ENTRY/EXIT rows); open it on github.com and it renders as a table |

The strategy, risk-management and multi-timeframe cascade logic were
tested in this session against hand-built OHLCV scenarios (bullish
retest -> correct LONG signal with SL beyond the order block and TP at
exactly 2x risk, a close-through -> correctly produces no trade, the
weekly/daily/4h/1h cascade correctly blocking a trade when levels
disagree and correctly passing an all-agreeing or all-neutral case,
leverage/risk-pct/reward:risk hard-cap clamping, and the exit-fill lookup
used by the trade journal). They have not been tested against live BingX
data or backtested over history, because this sandbox can't reach BingX.
Do that yourself before risking real money (see Step 3 below).

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
- Open a Python REPL and place one manual small test order through `ExchangeClient` to confirm `positionSide` and stop-order params match your account's position mode (BingX supports both "one-way" and "hedge" mode, which changes some order parameters), and confirm `fetch_trade_exit()`'s field-name assumptions (`reduceOnly`, `positionSide`, `realizedPnl`) against a real closed trade. This is the one step that could not be verified from this environment -- do not skip it.
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
| `REWARD_RISK_RATIO` | `2` | Ignored: always forced to `2` (take-profit is always twice the stop-loss distance) |
| `TREND_FILTER_ENABLED` | `true` | Master toggle for the weekly/daily/4h/1h cascade |
| `HTF_TIMEFRAMES` | `1w,1d,4h,1h` | The cascade itself, highest timeframe first |
| `TREND_EMA_PERIOD` | `50` | EMA period used for the trend read at every cascade level |
| `TREND_CANDLE_LIMIT` | `150` | How many candles to fetch per cascade level |
| `RISK_PCT_PER_TRADE` | `0.5` | % of account equity risked per trade (via stop distance); hard-capped at 3% regardless of this value |
| `LEVERAGE` | `3` | Requested leverage; hard-capped at 10x in code regardless of this value |
| `MAX_DAILY_LOSS_PCT` | `3` | Daily realized-loss % that halts new entries for the rest of the UTC day |
| `MAX_CONCURRENT_POSITIONS` | `1` | Won't open a new position while this many are already open |
| `DRY_RUN` | `true` | Logs signals only; set `false` to place real orders |

## Important limitations and risks -- please read

- Leverage amplifies losses, including to zero (liquidation). A 3x leveraged position can be liquidated by a ~33% adverse move; the code's hard leverage cap (10x) is a backstop, not a recommendation to use it.
- Order blocks, as implemented here, are one reasonable interpretation of a discretionary trading concept, not a guaranteed edge, and the weekly/daily/4h/1h agreement rule is a deliberate design choice (fewer, more conservative trades) that has not been backtested. There is no backtest included proving this makes money -- you should backtest it against BingX historical data yourself before trusting it with real funds.
- API/library details can drift. BingX's position-mode handling (`positionSide`, one-way vs. hedge accounts) and ccxt's exact param names for `STOP_MARKET`/`TAKE_PROFIT_MARKET` orders, and the `info` field names `exchange_client.fetch_trade_exit()` reads for exit fills (`reduceOnly`, `positionSide`, `realizedPnl`) are the most likely things to need a small adjustment for your account -- this is what Step 3 above is for.
- The weekly/daily/4h/1h cascade means the bot needs more historical candles and more API calls per symbol per cycle (five timeframes instead of one). If you watch many symbols on a tight schedule, keep an eye on BingX's rate limits.
- Nothing here is financial advice, and I'm not a financial advisor -- this is software that automates a rule set you should understand and agree with before it trades your money.
- Keep `.env` out of version control and off any machine you don't control; anyone with the API key/secret can trade on your account (not withdraw, since that permission is off).
