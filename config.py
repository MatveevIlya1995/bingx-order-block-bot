"""
Configuration for the BingX order-block futures bot.

All values are read from environment variables so no secrets ever live in
the code itself. Copy .env.example to .env, fill it in, and make sure your
run environment loads it (the included run.sh does this with python-dotenv
if it's installed, or you can `export $(cat .env | xargs)` before running).
"""
import os
from dataclasses import dataclass, field


def _bool(env_key: str, default: bool) -> bool:
    val = os.getenv(env_key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- credentials ---
    api_key: str = field(default_factory=lambda: os.getenv("BINGX_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINGX_API_SECRET", ""))

    # --- market ---
    # ccxt unified symbol for a USDT-margined perpetual, e.g. "BTC/USDT:USDT"
    symbol: str = field(default_factory=lambda: os.getenv("SYMBOL", "BTC/USDT:USDT"))
    timeframe: str = field(default_factory=lambda: os.getenv("TIMEFRAME", "15m"))
    candle_limit: int = field(default_factory=lambda: int(os.getenv("CANDLE_LIMIT", "300")))

    # --- strategy ---
    fractal_n: int = field(default_factory=lambda: int(os.getenv("FRACTAL_N", "2")))
    reward_risk_ratio: float = field(default_factory=lambda: float(os.getenv("REWARD_RISK_RATIO", "2")))

    # --- risk management ---
    risk_pct_per_trade: float = field(
        default_factory=lambda: float(os.getenv("RISK_PCT_PER_TRADE", "0.5")) / 100
    )
    leverage: int = field(default_factory=lambda: int(os.getenv("LEVERAGE", "3")))
    # Hard ceiling enforced in code regardless of what LEVERAGE env var says.
    max_leverage_hard_cap: int = 10
    max_daily_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "3")) / 100
    )
    max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "1"))
    )

    # --- safety ---
    # Defaults to True on purpose. You must consciously set DRY_RUN=false
    # to let the bot place real orders.
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))

    # --- paths ---
    state_path: str = field(default_factory=lambda: os.getenv("STATE_PATH", "data/state.json"))
    log_path: str = field(default_factory=lambda: os.getenv("LOG_PATH", "data/trades.log"))


def load_config() -> Config:
    cfg = Config()
    if cfg.leverage > cfg.max_leverage_hard_cap:
        print(
            f"[config] LEVERAGE={cfg.leverage} exceeds hard cap "
            f"({cfg.max_leverage_hard_cap}); clamping."
        )
        cfg.leverage = cfg.max_leverage_hard_cap
    return cfg
