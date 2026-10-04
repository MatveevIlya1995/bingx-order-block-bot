"""The optional entry filters: off by default, and only ever removing setups."""
import pytest

import bot
import config
from strategy import (
    apply_entry_filters,
    entry_symbol_allowed,
    stop_distance_pct,
    timing_agreement_ok,
)


def setup(entry, stop, side="LONG"):
    return {"side": "buy" if side == "LONG" else "sell", "position_side": side, "entry": entry,
            "stop_loss": stop, "take_profit": 0.0, "order_block": {"type": "bullish", "top": entry, "bottom": stop}}


def test_defaults_are_off(monkeypatch):
    for key in ("MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT", "RESTING_TP_ORDERS"):
        monkeypatch.delenv(key, raising=False)
    cfg = config.load_config()
    assert cfg.min_stop_distance_pct == 0.0
    assert cfg.entry_symbols == []
    assert cfg.htf_timing_agreement == "off"
    assert cfg.resting_tp_orders is True
    setups = [setup(100.0, 99.99), setup(100.0, 99.0, "SHORT")]
    kept, dropped = apply_entry_filters(
        setups, min_stop_distance_pct=cfg.min_stop_distance_pct,
        timing_trends=["bearish", "bearish"], timing_mode=cfg.htf_timing_agreement)
    assert kept == setups and dropped == []


def test_empty_github_variables_mean_off(monkeypatch):
    # `${{ vars.X }}` passes "" for a variable that was never created
    for key in ("MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT", "RESTING_TP_ORDERS"):
        monkeypatch.setenv(key, "")
    cfg = config.load_config()
    assert (cfg.min_stop_distance_pct, cfg.entry_symbols, cfg.htf_timing_agreement, cfg.resting_tp_orders) == (0.0, [], "off", True)


def test_bad_values_fall_back_to_off(monkeypatch):
    monkeypatch.setenv("MIN_STOP_DISTANCE_PCT", "-1")
    monkeypatch.setenv("HTF_TIMING_AGREEMENT", "Strict")
    cfg = config.load_config()
    assert cfg.min_stop_distance_pct == 0.0 and cfg.htf_timing_agreement == "off"
    monkeypatch.setenv("HTF_TIMING_AGREEMENT", " Not_Against ")
    assert config.load_config().htf_timing_agreement == "not_against"


def test_min_stop_distance():
    tight, wide = setup(100.0, 99.85), setup(100.0, 99.70)
    assert stop_distance_pct(tight) == pytest.approx(0.15)
    kept, dropped = apply_entry_filters([tight, wide], min_stop_distance_pct=0.25)
    assert kept == [wide] and dropped[0][0] is tight and "MIN_STOP_DISTANCE_PCT" in dropped[0][1]
    kept, _ = apply_entry_filters([tight, wide], min_stop_distance_pct=0.15)
    assert kept == [tight, wide]            # exactly at the minimum is allowed


@pytest.mark.parametrize("mode,trends,long_ok,short_ok", [
    ("off", ["bearish", "bearish"], True, True),
    ("not_against", ["bullish", "neutral"], True, False),
    ("not_against", ["neutral", "neutral"], True, True),
    ("not_against", ["bullish", "bearish"], False, False),
    ("aligned", ["bullish", "bullish"], True, False),
    ("aligned", ["bullish", "neutral"], False, False),
    ("aligned", [], False, False),
])
def test_timing_agreement(mode, trends, long_ok, short_ok):
    assert timing_agreement_ok("LONG", trends, mode) is long_ok
    assert timing_agreement_ok("SHORT", trends, mode) is short_ok


def test_timing_agreement_without_trends_does_not_block():
    assert timing_agreement_ok("LONG", None, "aligned") is True


def test_entry_symbols():
    assert entry_symbol_allowed("BTC/USDT:USDT", []) is True
    assert entry_symbol_allowed("BTC/USDT:USDT", ["btc", "ETH/USDT:USDT"]) is True
    assert entry_symbol_allowed("ETH/USDT:USDT", ["btc", "ETH/USDT:USDT"]) is True
    assert entry_symbol_allowed("SOL/USDT:USDT", ["btc", "ETH/USDT:USDT"]) is False


def test_find_setups_applies_the_filters(monkeypatch, capsys):
    """bot._find_setups end to end with stubbed analysis: 1w/1d bullish,
    4h bullish, 1h bearish; two LONG setups, one with a tight stop."""
    for key in ("MIN_STOP_DISTANCE_PCT", "ENTRY_SYMBOLS", "HTF_TIMING_AGREEMENT"):
        monkeypatch.delenv(key, raising=False)
    tight, wide = setup(100.0, 99.9), setup(98.0, 97.5)
    monkeypatch.setattr(bot, "analyze_higher_timeframes", lambda client, cfg, symbol: (
        "bullish", "detail", {"1w": "bullish", "1d": "bullish", "4h": "bullish", "1h": "bearish"}))
    monkeypatch.setattr(bot, "_fetch_df", lambda client, symbol, cache: object())
    monkeypatch.setattr(bot, "detect_order_blocks", lambda df, n: [])
    monkeypatch.setattr(bot, "limit_setups", lambda *a, **k: [tight, wide])

    def find(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        cfg = config.load_config()
        out = bot._find_setups(None, cfg, "BTC/USDT:USDT", {})[0]
        for k in env:
            monkeypatch.delenv(k)
        return out

    assert find() == [tight, wide]                                   # everything off
    assert find(MIN_STOP_DISTANCE_PCT="0.25") == [wide]
    assert find(HTF_TIMING_AGREEMENT="not_against") == []            # 1h is against LONG
    assert find(HTF_TIMING_AGREEMENT="aligned") == []
    assert find(ENTRY_SYMBOLS="ETH,SOL") == []
    assert find(ENTRY_SYMBOLS="eth,btc") == [tight, wide]
    assert "filtered out" in capsys.readouterr().out
