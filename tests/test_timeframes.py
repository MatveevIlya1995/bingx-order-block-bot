"""The timeframe cascade: daily + 4h set direction, 1h + 15m are timing
levels, entries are on 5m -- and never on anything below 5m."""
import pytest

import bot
import config
from strategy import ohlcv_to_df

TF_ENV = ("TIMEFRAME", "HTF_TIMEFRAMES", "HTF_PRIMARY_TIMEFRAMES", "WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT",
          "REQUIRE_STRONG_SWING", "STRUCTURE_CONFIRMED_SWINGS", "TREND_FILTER_ENABLED")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in TF_ENV:
        monkeypatch.delenv(key, raising=False)


def test_default_cascade():
    cfg = config.load_config()
    assert cfg.timeframe == "5m"
    assert cfg.htf_timeframes == ["1d", "4h", "1h", "15m"]
    assert cfg.primary_htf_timeframes == ["1d", "4h"]
    assert config.override_levels(cfg) == ("1d", "4h")
    assert cfg.require_strong_swing is True


def test_empty_github_variables_mean_the_defaults(monkeypatch):
    for key in TF_ENV:
        monkeypatch.setenv(key, "")
    cfg = config.load_config()
    assert (cfg.timeframe, cfg.htf_timeframes, cfg.primary_htf_timeframes) == ("5m", ["1d", "4h", "1h", "15m"], ["1d", "4h"])
    assert cfg.require_strong_swing is True


@pytest.mark.parametrize("asked,used", [("1m", "5m"), ("3m", "5m"), ("5m", "5m"), ("15m", "15m"), ("1h", "1h"),
                                        ("30s", "5m"), ("fast", "5m")])
def test_entry_timeframe_never_goes_below_5m(monkeypatch, capsys, asked, used):
    monkeypatch.setenv("TIMEFRAME", asked)
    assert config.load_config().timeframe == used
    assert ("below the 5m floor" in capsys.readouterr().out) == (asked != used)


def test_the_previous_cascade_is_still_a_valid_setting(monkeypatch):
    monkeypatch.setenv("TIMEFRAME", "15m")
    monkeypatch.setenv("HTF_TIMEFRAMES", "1w,1d,4h,1h")
    monkeypatch.setenv("HTF_PRIMARY_TIMEFRAMES", "1w,1d")
    cfg = config.load_config()
    assert config.override_levels(cfg) == ("1w", "1d")
    monkeypatch.setenv("HTF_PRIMARY_TIMEFRAMES", "1w")
    assert config.override_levels(config.load_config()) is None            # one primary level: nothing to override
    monkeypatch.setenv("HTF_PRIMARY_TIMEFRAMES", "1w,1d,4h")
    assert config.override_levels(config.load_config()) is None


class Trending:
    """Candles per timeframe: a steady climb or fall from 100, optionally
    with one old swing high (a spike on candle 5 of one timeframe)."""

    def __init__(self, trends, swing_high=None):
        self.trends, self.swing_high = trends, swing_high

    def fetch_ohlcv(self, symbol, timeframe=None, limit=None):
        n = limit or 150
        up = self.trends[timeframe] == "bullish"
        rows = []
        for i in range(n):
            c = 100.0 + (i if up else -i) * 0.1
            o = c - 0.1 if up else c + 0.1
            rows.append([i * 60_000, o, max(o, c) + 0.01, min(o, c) - 0.01, c, 1.0])
        if self.swing_high is not None and timeframe == self.swing_high[0]:
            rows[5][2] = self.swing_high[1]
        return rows


def test_direction_comes_from_daily_and_4h(monkeypatch):
    cfg = config.load_config()
    agree = {"1d": "bullish", "4h": "bullish", "1h": "bearish", "15m": "bearish"}
    bias, detail, trends = bot.analyze_higher_timeframes(Trending(agree), cfg, "BTC/USDT:USDT")
    assert bias == "bullish" and trends == agree           # 1h/15m against it do not veto
    assert "1d*=bullish" in detail and "4h*=bullish" in detail and "1h=bearish" in detail and "15m=bearish" in detail

    clash = {"1d": "bullish", "4h": "bearish", "1h": "bullish", "15m": "bullish"}
    bias, detail, _ = bot.analyze_higher_timeframes(Trending(clash), cfg, "BTC/USDT:USDT")
    assert bias == "conflict" and "1d=bullish vs 4h=bearish disagreement NOT overridden" in detail

    # with room left to the daily chart's last swing high, the daily trend wins
    monkeypatch.setenv("WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT", "3")
    cfg = config.load_config()
    price = ohlcv_to_df(Trending(clash).fetch_ohlcv("x", "4h", cfg.trend_candle_limit))["close"].iloc[-1]
    spiked = Trending(clash, swing_high=("1d", 120.0))         # an old daily swing high at 120
    room = (120.0 - price) / price * 100
    bias, detail, _ = bot.analyze_higher_timeframes(spiked, cfg, "BTC/USDT:USDT")
    assert bias == "bullish" and "OVERRIDE: 1d=bullish vs 4h=bearish" in detail
    assert f"{room:.2f}% room left to the 1d high" in detail and "trading with the 1d trend" in detail
    # ...unless that room is below the threshold
    monkeypatch.setenv("WEEKLY_DAILY_OVERRIDE_MIN_ROOM_PCT", str(int(room) + 5))
    assert bot.analyze_higher_timeframes(spiked, config.load_config(), "BTC/USDT:USDT")[0] == "conflict"
