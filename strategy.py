"""
Order block (smart-money-concept) strategy.

Definitions used here (one common, defensible interpretation among several
that exist in trading communities -- tune to taste):

1. Swing points are found with a simple N-bar fractal: a candle is a swing
   high if its high is the max of the (2N+1)-bar window centered on it,
   and symmetrically for swing lows.
2. A "break of structure" (BOS) happens when a candle *closes* beyond the
   most recent confirmed swing high (bullish BOS) or swing low (bearish
   BOS).
3. The order block is the last opposite-colored candle before the
   impulsive move that produced the BOS:
   - bullish OB = last bearish (red) candle before a bullish BOS
   - bearish OB = last bullish (green) candle before a bearish BOS
   Its zone is that candle's [low, high].
4. A signal fires the first time price trades back into an unmitigated
   OB zone and closes back out of it in the direction of the original
   impulse (a "retest and reaction"). If price closes all the way through
   the zone instead, the OB is invalidated ("mitigated") without a trade.

Stop-loss is placed just beyond the OB zone (below it for a LONG, above it
for a SHORT); take-profit is a fixed risk multiple -- config.py hard-
enforces a 2:1 reward:risk ratio regardless of what REWARD_RISK_RATIO is
set to (see required_reward_risk_ratio there), and the position sizer in
risk_manager.py hard-caps the stop-loss risk itself at 3% of equity.

Multi-timeframe use: every function below operates on a single timeframe's
DataFrame. bot.py calls detect_trend() and detect_order_blocks() once per
level of its weekly -> daily -> 4h -> 1h cascade (see
bot.analyze_higher_timeframes()), and again on the 15m entry timeframe for
generate_signal(). combine_htf_bias() folds the cascade's trend readings
into one overall directional bias that bot.py then requires the entry
signal not to contradict.
"""
import pandas as pd
import numpy as np


def ohlcv_to_df(raw) -> pd.DataFrame:
    """raw: list of [timestamp_ms, open, high, low, close, volume] as returned by ccxt."""
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def detect_trend(df: pd.DataFrame, ema_period: int = 50) -> str:
    """Trend filter based on an EMA of closes.

    Meant to be called on weekly/daily/4h/1h (or similar) candles,
    separate from the 15m candles the order-block logic runs on. Returns
    "bullish" if the last close is above a rising EMA(ema_period),
    "bearish" if it's below a falling EMA, and "neutral" if there isn't
    enough data yet or the EMA is flat/ambiguous. Callers should treat
    "neutral" as "no opinion" -- this filter is meant to veto clear
    counter-trend entries, not to require a strong trend before trading.
    """
    if len(df) < ema_period + 2:
        return "neutral"
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    last_close = df["close"].iloc[-1]
    last_ema = ema.iloc[-1]
    prev_ema = ema.iloc[-2]
    if last_close > last_ema and last_ema > prev_ema:
        return "bullish"
    if last_close < last_ema and last_ema < prev_ema:
        return "bearish"
    return "neutral"


def combine_htf_bias(trends: list) -> str | None:
    """Fold an ordered (highest-timeframe-first) list of "bullish" /
    "bearish" / "neutral" readings into one overall directional bias for
    the weekly -> daily -> 4h -> 1h cascade:

    - "neutral" readings are skipped: an unclear level doesn't block a
      trade, it simply has no opinion.
    - if every non-neutral reading agrees, that direction is returned.
    - if any two levels disagree (one bullish, one bearish), returns
      "conflict" -- the strategy's rule is to never trade against a
      higher timeframe's trend, so a clear disagreement between levels
      kills the setup outright rather than picking a side.
    - if every level is neutral, returns None: no directional bias,
      both LONG and SHORT entry signals are allowed through unfiltered.
    """
    bias = None
    for t in trends:
        if t == "neutral":
            continue
        if bias is None:
            bias = t
        elif t != bias:
            return "conflict"
    return bias


def find_fractals(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    highs = df["high"].values
    lows = df["low"].values
    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)

    for i in range(n, len(df) - n):
        window_h = highs[i - n : i + n + 1]
        window_l = lows[i - n : i + n + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == n:
            swing_high[i] = True
        if lows[i] == window_l.min() and np.argmin(window_l) == n:
            swing_low[i] = True

    out = df.copy()
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out


def detect_order_blocks(df: pd.DataFrame, n: int = 2) -> list[dict]:
    """Returns a list of order block dicts in chronological order:
    {type, top, bottom, formed_idx, bos_idx, mitigated}
    """
    df = find_fractals(df, n=n)
    obs: list[dict] = []

    last_swing_high = None
    last_swing_high_idx = None
    last_swing_low = None
    last_swing_low_idx = None

    for i in range(len(df)):
        row = df.iloc[i]

        if row["swing_high"]:
            last_swing_high = row["high"]
            last_swing_high_idx = i
        if row["swing_low"]:
            last_swing_low = row["low"]
            last_swing_low_idx = i

        # Bullish BOS
        if last_swing_high is not None and row["close"] > last_swing_high:
            search_from = last_swing_high_idx if last_swing_high_idx is not None else 0
            ob_idx = None
            for j in range(i, search_from, -1):
                if df.iloc[j]["close"] < df.iloc[j]["open"]:
                    ob_idx = j
                    break
            if ob_idx is not None:
                obs.append(
                    {
                        "type": "bullish",
                        "top": float(df.iloc[ob_idx]["high"]),
                        "bottom": float(df.iloc[ob_idx]["low"]),
                        "formed_idx": ob_idx,
                        "bos_idx": i,
                        "mitigated": False,
                    }
                )
            last_swing_high = None  # structure broken; wait for the next confirmed swing

        # Bearish BOS
        if last_swing_low is not None and row["close"] < last_swing_low:
            search_from = last_swing_low_idx if last_swing_low_idx is not None else 0
            ob_idx = None
            for j in range(i, search_from, -1):
                if df.iloc[j]["close"] > df.iloc[j]["open"]:
                    ob_idx = j
                    break
            if ob_idx is not None:
                obs.append(
                    {
                        "type": "bearish",
                        "top": float(df.iloc[ob_idx]["high"]),
                        "bottom": float(df.iloc[ob_idx]["low"]),
                        "formed_idx": ob_idx,
                        "bos_idx": i,
                        "mitigated": False,
                    }
                )
            last_swing_low = None

    return obs


def generate_signal(df: pd.DataFrame, order_blocks: list[dict], reward_risk_ratio: float = 2.0):
    """Evaluate only the most recent (last) candle for a fresh retest signal.

    Mutates order_blocks in place to mark mitigation so repeated calls across
    scheduled runs don't refire the same zone.
    """
    if len(df) < 2:
        return None
    i = len(df) - 1
    row = df.iloc[i]

    for ob in order_blocks:
        if ob["mitigated"]:
            continue
        if ob["bos_idx"] >= i:
            continue  # only consider order blocks confirmed before this candle

        if ob["type"] == "bullish":
            touched = row["low"] <= ob["top"]
            closed_through = row["close"] < ob["bottom"]
            if closed_through:
                ob["mitigated"] = True
                continue
            if touched and row["close"] > ob["bottom"]:
                ob["mitigated"] = True
                entry = float(row["close"])
                stop = ob["bottom"] * 0.999
                risk = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * reward_risk_ratio
                return {
                    "side": "buy",
                    "position_side": "LONG",
                    "entry": entry,
                    "stop_loss": stop,
                    "take_profit": target,
                    "order_block": ob,
                }

        elif ob["type"] == "bearish":
            touched = row["high"] >= ob["bottom"]
            closed_through = row["close"] > ob["top"]
            if closed_through:
                ob["mitigated"] = True
                continue
            if touched and row["close"] < ob["top"]:
                ob["mitigated"] = True
                entry = float(row["close"])
                stop = ob["top"] * 1.001
                risk = stop - entry
                if risk <= 0:
                    continue
                target = entry - risk * reward_risk_ratio
                return {
                    "side": "sell",
                    "position_side": "SHORT",
                    "entry": entry,
                    "stop_loss": stop,
                    "take_profit": target,
                    "order_block": ob,
                }

    return None
