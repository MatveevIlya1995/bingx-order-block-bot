"""
Configuration for the BingX order-block futures bot.

All values are read from environment variables so no secrets ever live in
the code itself. Copy .env.example to .env, fill it in, and make sure your
run environment loads it (the included run.sh does this, or you can
`export $(cat .env | xargs)` before running).
"""
import os
from dataclasses import dataclass, field


def _env(env_key: str, default: str) -> str:
    """Like os.getenv, but also falls back to `default` when the variable
    is set to an empty string. GitHub Actions' `${{ vars.X }}` syntax
    always defines the env var, substituting "" when the repo Variable
    was never created -- plain os.getenv(key, default) would miss that
    and pass an empty string straight through to int()/float()."""
    val = os.getenv(env_key)
    if val is None or val.strip() == "":
        return default
    return val


def _bool(env_key: str, default: bool) -> bool:
    val = os.getenv(env_key)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_list(env_key: str, default_csv: str) -> list[str]:
    """Comma-separated list env var -> list of stripped strings."""
    val = _env(env_key, default_csv)
    return [s.strip() for s in val.split(",") if s.strip()]


# A curated set of ~15 of the most liquid BingX USDT-M perpetuals -- not
# "every symbol BingX lists" on purpose, since thin/illiquid pairs carry
# much higher slippage and manipulation risk. Override with the SYMBOLS
# repo variable (comma-separated) to use a different list.
DEFAULT_SYMBOLS = (
    "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT,XRP/USDT:USDT,"
    "DOGE/USDT:USDT,ADA/USDT:USDT,AVAX/USDT:USDT,LINK/USDT:USDT,LTC/USDT:USDT,"
    "DOT/USDT:USDT,TRX/USDT:USDT,NEAR/USDT:USDT,ATOM/USDT:USDT,APT/USDT:USDT"
)

# Full top-down cascade, highest timeframe first -- every level here is
# fetched and logged each cycle (see bot.analyze_higher_timeframes()), so
# you always see weekly through 1h trend context. Override with the
# HTF_TIMEFRAMES repo variable (comma-separated, highest timeframe first).
DEFAULT_HTF_TIMEFRAMES = "1w,1d,4h,1h"

# Of the levels above, only these are DIRECTIONAL/AUTHORITATIVE: they set
# the trade's allowed bias, and a disagreement between them blocks the
# trade outright (see strategy.combine_htf_bias() and
# bot.analyze_higher_timeframes()) -- unless the weekly/daily room
# override below kicks in. The remaining HTF_TIMEFRAMES levels are
# "refinement" levels -- still fetched, analyzed and logged every cycle,
# but purely informational: they never veto a trade. This mirrors how
# most top-down multi-timeframe traders actually work -- the higher
# timeframe sets direction, lower ones are for timing entries, not for
# overruling the primary trend on every small pullback. Override with the
# HTF_PRIMARY_TIMEFRAMES repo variable if you want a different split (e.g.
# just the weekly, or weekly+daily+4h).
DEFAULT_HTF_PRIMARY_TIMEFRAMES = "1w,1d"

# Valid values for HTF_TIMING_AGREEMENT (see Config.htf_timing_agreement).
HTF_TIMING_AGREEMENT_MODES = ("off", "not_against", "aligned")


@dataclass
class Config:
    # --- credentials ---
    api_key: str = field(default_factory=lambda: _env("BINGX_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: _env("BINGX_API_SECRET", ""))

    # --- market ---
    # List of ccxt unified symbols for USDT-margined perpetuals, e.g.
    # "BTC/USDT:USDT". The bot scans all of them each cycle, in order,
    # until it either runs out of symbols or hits max_concurrent_positions.
    symbols: list = field(default_factory=lambda: _env_list("SYMBOLS", DEFAULT_SYMBOLS))
    timeframe: str = field(default_factory=lambda: _env("TIMEFRAME", "15m"))
    candle_limit: int = field(default_factory=lambda: int(_env("CANDLE_LIMIT", "300")))

    # --- strategy ---
    fractal_n: int = field(default_factory=lambda: int(_env("FRACTAL_N", "2")))
    reward_risk_ratio: float = field(default_factory=lambda: float(_env("REWARD_RISK_RATIO", "2")))
    # Hard rule, not just a default: take-profit is always exactly this
    # multiple of the stop-loss distance. load_config() below forces
    # reward_risk_ratio to this value even if REWARD_RISK_RATIO is set to
    # something else in the environment. NOTE: since partial_tp_r_multiple
    # and trailing (below) now manage the runner after the partial exit,
    # this fixed 2R target is really only ever hit as a full exit if
    # trailing never gets the chance to activate (e.g. PARTIAL_TP_FRACTION
    # is set to 0, disabling partials).
    required_reward_risk_ratio: float = 2.0

    # --- top-down higher-timeframe trend filter ---
    # Weekly -> daily -> 4h -> 1h, highest timeframe first (see
    # DEFAULT_HTF_TIMEFRAMES/DEFAULT_HTF_PRIMARY_TIMEFRAMES above and
    # bot.analyze_higher_timeframes()). A signal is only vetoed when a
    # PRIMARY level's trend clearly contradicts another primary level's,
    # or contradicts the signal's own direction -- an unclear/neutral
    # trend, or a disagreeing REFINEMENT-only level, does not block a
    # trade by itself.
    trend_filter_enabled: bool = field(default_factory=lambda: _bool("TREND_FILTER_ENABLED", True))
    htf_timeframes: list = field(default_factory=lambda: _env_list("HTF_TIMEFRAMES", DEFAULT_HTF_TIMEFRAMES))
    primary_htf_timeframes: list = field(
        default_factory=lambda: _env_list("HTF_PRIMARY_TIMEFRAMES", DEFAULT_HTF_PRIMARY_TIMEFRAMES)
    )
    trend_ema_period: int = field(default_factory=lambda: int(_env("TREND_EMA_PERIOD", "50")))
    trend_candle_limit: int = field(default_factory=lambda: int(_env("TREND_CANDLE_LIMIT", "150")))

    # If the weekly and daily primary trends disagree, the default
    # behavior (see strategy.combine_htf_bias()) is to treat that as a
    # hard "conflict" and skip the symbol for the cycle entirely. This
    # setting relaxes that specifically for weekly vs. daily:
    # bot.analyze_higher_timeframes() locates the most recently
    # CONFIRMED weekly swing high/low (the top/bottom of the weekly
    # range as far as N-bar fractal confirmation can currently tell --
    # see strategy.weekly_room_pct()), and if price still has at least
    # this much room (%) left to run toward that level in the WEEKLY
    # trend's direction, the weekly trend wins outright and the daily
    # disagreement is overridden (still logged, just not vetoed on).
    # Below this room threshold -- price is already close to the
    # weekly range's edge -- a weekly/daily disagreement still blocks
    # the trade as before, since there's more reason to think the
    # weekly move itself is running out of room. Only ever applies
    # when HTF_PRIMARY_TIMEFRAMES is exactly weekly+daily (the default);
    # a differently configured primary set falls back to the original
    # hard-conflict behavior. Set to a very large number (e.g. 999) to
    # disable the override entirely.
    weekly_daily_override_min_room_pct: float = field(
        default_factory=lambda: float(_env("WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT", "3"))
    )

    # Position-sizing safety floor. risk_manager.position_size() can
    # legitimately return a small-but-nonzero quantity once the
    # margin-utilization cap bites (see max_margin_utilization_pct
    # below) -- e.g. when MAX_CONCURRENT_POSITIONS lets several
    # positions compete for the same shared margin budget and an
    # earlier symbol in the scan already claimed most of it. Before
    # this floor existed, bot.py's only guard was `qty <= 0`, so a
    # trade sized down to a few millionths of a coin (~$0 notional)
    # still went through: it can't meaningfully win or lose, yet it
    # still burns that order-block zone's one-trade-per-zone allowance
    # and one of the max_concurrent_positions slots for nothing. If the
    # computed notional (qty * entry price) is below this many USDT,
    # bot.py skips the trade entirely instead of opening it. Set to 0
    # to disable this floor.
    min_notional_usdt: float = field(
        default_factory=lambda: float(_env("MIN_NOTIONAL_USDT", "10"))
    )

    # --- risk management ---
    # Fixed margin committed per trade, in USDT -- position_size() no
    # longer sizes off a fraction of equity directly; instead it solves
    # for the LEVERAGE that turns this fixed margin into a position whose
    # stop-out loses risk_pct_per_trade of equity (see risk_manager.
    # position_size()). A tighter stop-loss (closer to entry) gets more
    # leverage to reach that risk target; a wider stop gets less.
    margin_per_trade_usd: float = field(
        default_factory=lambda: float(_env("MARGIN_PER_TRADE_USD", "50"))
    )
    # Target risk, as a % of equity, that position_size() solves the
    # per-trade leverage for. Defaults to the hard cap itself (3%) --
    # i.e. every trade targets the maximum allowed risk given its own
    # stop-loss distance, backing off only when the required leverage
    # would exceed the leverage ceiling (see risk_manager.safe_leverage()).
    # Lower this if you want trades to risk less than the 3% ceiling.
    risk_pct_per_trade: float = field(
        default_factory=lambda: float(_env("RISK_PCT_PER_TRADE", "3")) / 100
    )
    # Hard ceiling enforced in code regardless of what RISK_PCT_PER_TRADE
    # env var says: the stop-loss on any single trade may never risk more
    # than 3% of total account equity.
    max_risk_pct_per_trade_hard_cap: float = 0.03
    # Leverage CEILING for the auto-sizer in risk_manager.position_size():
    # the leverage it solves for, per trade, from the stop-loss distance
    # never exceeds this (or the 10x hard cap below, whichever is lower).
    # It is no longer applied directly as a fixed leverage on every trade.
    leverage: int = field(default_factory=lambda: int(_env("LEVERAGE", "3")))
    # Hard ceiling enforced in code regardless of what LEVERAGE env var says.
    max_leverage_hard_cap: int = 10
    # Second, independent cap on top of risk_pct_per_trade: the NOTIONAL
    # value of a single position, and of all open positions combined, may
    # never require more than this fraction of equity*leverage in margin.
    # Without this, a tight stop-loss (a small stop_distance) makes
    # risk_amount / stop_distance blow up into a notional far larger than
    # the account can actually margin -- risk_pct_per_trade alone only
    # bounds what you lose if stopped out, not how big the position is.
    # 0.5 means: never let open positions collectively require more than
    # 50% of the account's theoretical max notional (equity * leverage) --
    # deliberately leaving headroom so one position's stop-out doesn't
    # eat margin needed by others, and so a fill doesn't get rejected
    # outright for insufficient margin.
    max_margin_utilization_pct: float = field(
        default_factory=lambda: float(_env("MAX_MARGIN_UTILIZATION_PCT", "50")) / 100
    )
    max_daily_loss_pct: float = field(
        default_factory=lambda: float(_env("MAX_DAILY_LOSS_PCT", "3")) / 100
    )
    max_concurrent_positions: int = field(
        default_factory=lambda: int(_env("MAX_CONCURRENT_POSITIONS", "1"))
    )

    # --- partial take-profit + trailing stop ---
    # At PARTIAL_TP_R_MULTIPLE times the original risk (1R = the distance
    # from entry to the initial stop-loss), PARTIAL_TP_FRACTION of the
    # position is closed and the stop for the remainder moves to
    # breakeven (the original entry price) -- the classic "take some off,
    # can't lose on the rest" adjustment. From then on the remainder is a
    # trailing-stop runner with no fixed cap, so a strong trend can pay
    # out well beyond the fixed 2R take-profit instead of being capped
    # there every time. Set PARTIAL_TP_FRACTION to 0 to disable this and
    # fall back to the old fixed-2R-take-profit-only behavior.
    partial_tp_r_multiple: float = field(
        default_factory=lambda: float(_env("PARTIAL_TP_R_MULTIPLE", "1.0"))
    )
    partial_tp_fraction: float = field(
        default_factory=lambda: float(_env("PARTIAL_TP_FRACTION", "0.5"))
    )
    # Once trailing, the stop is ratcheted to (favorable extreme so far)
    # minus/plus this many R-multiples -- i.e. the runner is allowed this
    # much room to breathe before the trail locks in the next step of
    # profit. Smaller = locks in profit faster but gets stopped out of
    # trends sooner; larger = rides trends further but gives back more.
    trail_r_multiple: float = field(
        default_factory=lambda: float(_env("TRAIL_R_MULTIPLE", "1.0"))
    )

    # --- second take-profit (TP2) ---
    # After TP1, TP2_FRACTION of the ORIGINAL position closes at the next
    # swing high (LONG) / swing low (SHORT) beyond TP1 on the entry
    # timeframe (FRACTAL_N fractals on the fetched candles). If there is no
    # such swing, TP2 sits at TP2_FALLBACK_R_MULTIPLE x the initial risk
    # instead. The stop stays at breakeven between TP1 and TP2; the
    # trailing stop (TRAIL_R_MULTIPLE) only arms for what's left after TP2.
    # Defaults: TP1 50%, TP2 30%, runner 20%. Set TP2_FRACTION to 0 to skip
    # TP2 and start trailing right after TP1 (the previous behavior).
    tp2_fraction: float = field(
        default_factory=lambda: float(_env("TP2_FRACTION", "0.3"))
    )
    tp2_fallback_r_multiple: float = field(
        default_factory=lambda: float(_env("TP2_FALLBACK_R_MULTIPLE", "2.0"))
    )

    # --- resting take-profit orders (LIVE) ---
    # True (default): once the bot sees a filled entry it places TP1 and TP2
    # on BingX as resting reduce-only TAKE_PROFIT_MARKET orders, and splits
    # the stop-loss into one stop per part of the position (TP1 part, TP2
    # part, runner) so the stops still add up to exactly what is left after
    # a take-profit fills between two runs. If any of those orders cannot be
    # placed, that trade (or just that take-profit) falls back to the old
    # behaviour: a market close on the scheduled run that sees the candle
    # touch the level. Set RESTING_TP_ORDERS=false to use the old behaviour
    # for every new trade.
    resting_tp_orders: bool = field(default_factory=lambda: _bool("RESTING_TP_ORDERS", True))

    # --- entry filters (all OFF by default) ---
    # Minimum stop distance, in % of the entry price. A setup whose stop sits
    # closer to the entry than this gets no order. 0 = off. Example: 0.25
    # skips every setup with a stop less than 0.25% away from its entry.
    min_stop_distance_pct: float = field(
        default_factory=lambda: float(_env("MIN_STOP_DISTANCE_PCT", "0"))
    )
    # Symbols allowed to take NEW entries, as a comma-separated list of full
    # symbols ("BTC/USDT:USDT") or just the coin ("BTC"). Empty = off (every
    # symbol in SYMBOLS may trade). Symbols left out keep being watched, so
    # positions already open on them are still managed -- which is why this
    # is separate from SYMBOLS: never shrink SYMBOLS while a position on a
    # removed symbol is open.
    entry_symbols: list = field(default_factory=lambda: _env_list("ENTRY_SYMBOLS", ""))
    # Whether the timing levels (the HTF_TIMEFRAMES that are not primary --
    # 4h and 1h by default) must agree with the trade direction:
    #   off         -- they never block a trade (the behaviour so far)
    #   not_against -- no order if 4h or 1h trends AGAINST the trade
    #                  (a neutral 4h/1h is fine)
    #   aligned     -- no order unless 4h AND 1h both trend WITH the trade
    htf_timing_agreement: str = field(
        default_factory=lambda: _env("HTF_TIMING_AGREEMENT", "off").strip().lower()
    )

    # --- cost modeling ---
    # DRY_RUN's simulated exits previously assumed a fill at the exact
    # stop/take-profit price with no trading fees at all, which makes
    # paper-trading stats look better than live trading ever will. This
    # is a round-trip (entry + exit) taker-fee estimate applied to
    # simulated DRY_RUN P/L only, so paper stats are a more honest
    # preview. LIVE P/L uses the fee BingX reports on each fill (see
    # exchange_client.order_fill); this rate is only the LIVE fallback when
    # an order's fee can't be read, and such rows are marked as estimated.
    # 0.05% is BingX's standard default USDT-M perpetual taker fee at the
    # time this was written -- check your own account's actual fee tier
    # (maker/taker, any VIP discount) and adjust if it differs.
    dry_run_fee_pct_per_side: float = field(
        default_factory=lambda: float(_env("DRY_RUN_FEE_PCT_PER_SIDE", "0.05")) / 100
    )

    # --- safety ---
    # Defaults to True on purpose. You must consciously set DRY_RUN=false
    # to let the bot place real orders.
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))

    # --- paths ---
    state_path: str = field(default_factory=lambda: _env("STATE_PATH", "data/state.json"))
    log_path: str = field(default_factory=lambda: _env("LOG_PATH", "data/trades.log"))
    # CSV trade journal -- open it on github.com and it renders as a table.
    trades_csv_path: str = field(default_factory=lambda: _env("TRADES_CSV_PATH", "data/trades.csv"))


def load_config() -> Config:
    cfg = Config()
    if cfg.leverage > cfg.max_leverage_hard_cap:
        print(
            f"[config] LEVERAGE={cfg.leverage} exceeds hard cap "
            f"({cfg.max_leverage_hard_cap}); clamping."
        )
        cfg.leverage = cfg.max_leverage_hard_cap
    if cfg.risk_pct_per_trade > cfg.max_risk_pct_per_trade_hard_cap:
        print(
            f"[config] RISK_PCT_PER_TRADE={cfg.risk_pct_per_trade * 100:.2f}% exceeds hard cap "
            f"({cfg.max_risk_pct_per_trade_hard_cap * 100:.0f}%); clamping."
        )
        cfg.risk_pct_per_trade = cfg.max_risk_pct_per_trade_hard_cap
    if cfg.reward_risk_ratio != cfg.required_reward_risk_ratio:
        print(
            f"[config] REWARD_RISK_RATIO={cfg.reward_risk_ratio} ignored -- this strategy always "
            f"uses a fixed {cfg.required_reward_risk_ratio}:1 reward:risk ratio."
        )
        cfg.reward_risk_ratio = cfg.required_reward_risk_ratio
    if not (0.0 < cfg.max_margin_utilization_pct <= 1.0):
        print(
            f"[config] MAX_MARGIN_UTILIZATION_PCT={cfg.max_margin_utilization_pct * 100:.0f}% "
            f"is out of range (0-100]; clamping to 50%."
        )
        cfg.max_margin_utilization_pct = 0.5
    missing_primary = [tf for tf in cfg.primary_htf_timeframes if tf not in cfg.htf_timeframes]
    if missing_primary:
        print(
            f"[config] HTF_PRIMARY_TIMEFRAMES contains {missing_primary}, not present in "
            f"HTF_TIMEFRAMES={cfg.htf_timeframes}; adding them so they still get analyzed."
        )
        cfg.htf_timeframes = cfg.htf_timeframes + missing_primary
    if cfg.weekly_daily_override_min_room_pct < 0:
        print(
            f"[config] WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT="
            f"{cfg.weekly_daily_override_min_room_pct} is negative; clamping to 0."
        )
        cfg.weekly_daily_override_min_room_pct = 0.0
    if cfg.min_notional_usdt < 0:
        print(
            f"[config] MIN_NOTIONAL_USDT={cfg.min_notional_usdt} is negative; clamping to 0."
        )
        cfg.min_notional_usdt = 0.0
    if cfg.margin_per_trade_usd <= 0:
        print(
            f"[config] MARGIN_PER_TRADE_USD={cfg.margin_per_trade_usd} must be positive; "
            f"clamping to 50."
        )
        cfg.margin_per_trade_usd = 50.0
    if cfg.min_stop_distance_pct < 0:
        print(
            f"[config] MIN_STOP_DISTANCE_PCT={cfg.min_stop_distance_pct} is negative; "
            f"clamping to 0 (off)."
        )
        cfg.min_stop_distance_pct = 0.0
    if cfg.htf_timing_agreement not in HTF_TIMING_AGREEMENT_MODES:
        print(
            f"[config] HTF_TIMING_AGREEMENT={cfg.htf_timing_agreement!r} is not one of "
            f"{HTF_TIMING_AGREEMENT_MODES}; using 'off'."
        )
        cfg.htf_timing_agreement = "off"
    return cfg
