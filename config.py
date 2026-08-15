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

    # --- higher-timeframe trend filter ---
    # Entries are only vetoed when they clearly go against the 4h trend
    # (e.g. a LONG signal while the 4h trend is bearish) -- an unclear/
    # neutral 4h trend does not block a trade.
    trend_filter_enabled: bool = field(default_factory=lambda: _bool("TREND_FILTER_ENABLED", True))
    trend_timeframe: str = field(default_factory=lambda: _env("TREND_TIMEFRAME", "4h"))
    trend_ema_period: int = field(default_factory=lambda: int(_env("TREND_EMA_PERIOD", "50")))
    trend_candle_limit: int = field(default_factory=lambda: int(_env("TREND_CANDLE_LIMIT", "150")))

    # --- risk management ---
    risk_pct_per_trade: float = field(
        default_factory=lambda: float(_env("RISK_PCT_PER_TRADE", "0.5")) / 100
    )
    leverage: int = field(default_factory=lambda: int(_env("LEVERAGE", "3")))
    # Hard ceiling enforced in code regardless of what LEVERAGE env var says.
    max_leverage_hard_cap: int = 10
    max_daily_loss_pct: float = field(
        default_factory=lambda: float(_env("MAX_DAILY_LOSS_PCT", "3")) / 100
    )
    max_concurrent_positions: int = field(
        default_factory=lambda: int(_env("MAX_CONCURRENT_POSITIONS", "1"))
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
    return cfg
