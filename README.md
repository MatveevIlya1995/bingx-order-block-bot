# BingX Order-Block Futures Bot

A rule-based bot for BingX USDT-M perpetual futures that trades an "order
block" (smart-money-concept) strategy: it finds swing highs/lows, watches
for a break of structure, marks the last opposite candle before that break
as an order block, and enters when price retests that zone and reacts.
Every entry gets an automatic stop-loss and take-profit sized by a fixed
percentage of account risk. A daily-loss circuit breaker halts new entries
if losses get too large for the day.

## Where this needs to run

**It cannot run inside this chat.** This sandbox's network is locked down
to an allowlist that blocks both PyPI and BingX's API entirely, so nothing
here can `pip install` or reach `bingx.com`. You'll need to run it
somewhere with normal internet access — your own computer, a small VPS
(e.g. a $5/mo box), or similar. The scheduling instructions below assume a
plain cron job on Linux/macOS; swap in Task Scheduler or a systemd timer
if you prefer.

## Files

| File | Purpose |
|---|---|
| `bot.py` | Entry point — runs one full check-and-maybe-trade cycle |
| `strategy.py` | Order block detection + retest signal generation |
| `risk_manager.py` | Position sizing, leverage cap, daily-loss circuit breaker |
| `exchange_client.py` | Thin wrapper around `ccxt`'s BingX integration |
| `config.py` | Loads settings from environment variables |
| `.env.example` | Template for your API keys and settings |
| `run.sh` | Loads `.env` and runs one cycle — point cron at this |
| `data/state.json` | Auto-created; tracks today's P&L for the circuit breaker |

The strategy and risk-management logic were tested in this session against
hand-built OHLCV scenarios (bullish retest → correct long signal, a
close-through invalidation → correctly produces *no* trade, leverage
clamping, position sizing, and the daily circuit breaker) — see the
transcript for the test cases. They have **not** been tested against live
BingX data or backtested over history, because this sandbox can't reach
BingX. Do that yourself before risking real money (see Step 3 below).

## Setup

**1. Get the code onto a machine with internet access**, then:

```bash
cd bingx-order-block-bot
pip install -r requirements.txt
cp .env.example .env
```

**2. Create a BingX API key** (BingX → Account → API Management):
- Enable **Perpetual Futures trading** permission
- Leave **withdrawal** permission OFF — always
- Set an **IP whitelist** to the machine that will run the bot
- Paste the key/secret into `.env`. Never commit `.env` or paste your
  secret key into a chat.

**3. Validate before trading real money, in this order:**
   - Run `python3 bot.py` a few times with `DRY_RUN=true` (the default) and
     confirm it reads your balance, fetches candles, and logs sensible
     signals without error.
   - If you can, point `SYMBOL`/test against BingX's demo trading (VST)
     environment first, or at minimum size RISK_PCT_PER_TRADE very small
     for the first week live.
   - Open a Python REPL and place **one manual small test order** through
     `ExchangeClient` to confirm `positionSide` and stop-order params match
     your account's position mode (BingX supports both "one-way" and
     "hedge" mode, which changes some order parameters). This is the one
     step I could not verify from this environment — do not skip it.
   - Only then set `DRY_RUN=false`.

**4. Schedule it.** The strategy runs on the `TIMEFRAME` candle (default
15m), so a cron entry that fires a bit more often than that is enough —
the bot itself is idempotent per cycle (it checks existing positions
before opening a new one):

```cron
*/5 * * * * /path/to/bingx-order-block-bot/run.sh >> /path/to/bingx-order-block-bot/data/cron.log 2>&1
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `SYMBOL` | `BTC/USDT:USDT` | ccxt unified symbol for the USDT-M perpetual |
| `TIMEFRAME` | `15m` | Candle timeframe the strategy runs on |
| `RISK_PCT_PER_TRADE` | `0.5` | % of account equity risked per trade (via stop distance) |
| `LEVERAGE` | `3` | Requested leverage; hard-capped at 10x in code regardless of this value |
| `MAX_DAILY_LOSS_PCT` | `3` | Daily realized-loss % that halts new entries for the rest of the UTC day |
| `MAX_CONCURRENT_POSITIONS` | `1` | Won't open a new position while this many are already open |
| `REWARD_RISK_RATIO` | `2` | Take-profit distance as a multiple of the stop distance |
| `DRY_RUN` | `true` | Logs signals only; set `false` to place real orders |

## Important limitations and risks — please read

- **Leverage amplifies losses, including to zero (liquidation).** A 3x
  leveraged position can be liquidated by a ~33% adverse move; the code's
  hard leverage cap (10x) is a backstop, not a recommendation to use it.
- **Order blocks, as implemented here, are one reasonable interpretation
  of a discretionary trading concept**, not a guaranteed edge. There is no
  backtest included proving this makes money — you should backtest it
  against BingX historical data yourself before trusting it with real
  funds, and consider it a starting point to tune (fractal sensitivity,
  timeframe, reward:risk, symbol) rather than a finished strategy.
- **API/library details can drift.** BingX's position-mode handling
  (`positionSide`, one-way vs. hedge accounts) and ccxt's exact param
  names for `STOP_MARKET`/`TAKE_PROFIT_MARKET` orders are the most likely
  things to need a small adjustment for your account — this is what Step 3
  above is for.
- **Nothing here is financial advice**, and I'm not a financial advisor —
  this is software that automates a rule set you should understand and
  agree with before it trades your money.
- Keep `.env` out of version control and off any machine you don't
  control; anyone with the API key/secret can trade on your account (not
  withdraw, since that permission is off).
