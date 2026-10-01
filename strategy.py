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
signal not to contradict -- UNLESS the weekly/daily disagreement override
applies (see weekly_room_pct() below and bot.analyze_higher_timeframes()):
when the weekly and daily trends disagree but price still has enough room
left to run toward the weekly chart's own last confirmed swing high/low,
the weekly trend is allowed to win instead of blocking the trade outright.
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
      kills the setup outright rather than picking a side. (bot.py's
      weekly/daily override can still turn a "conflict" back into a
      directional bias afterwards -- see
      bot.analyze_higher_timeframes() and weekly_room_pct() below.)
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


def nearest_confirmed_swing_level(df: pd.DataFrame, direction: str, n: int = 2):
    """The most recently CONFIRMED swing high (direction="bullish") or
    swing low (direction="bearish") in df -- i.e. the top/bottom of the
    current range as far as N-bar fractal confirmation can currently
    tell. A swing point needs n bars to close on both sides before it's
    confirmed, so this always lags the most extreme price seen so far by
    up to n bars.

    Returns None if no such point has been confirmed yet (not enough
    data, or price has only made one-directional swings within the
    window fetched).
    """
    out = find_fractals(df, n=n)
    if direction == "bullish":
        matches = out[out["swing_high"]]
        price_col = "high"
    elif direction == "bearish":
        matches = out[out["swing_low"]]
        price_col = "low"
    else:
        return None
    if matches.empty:
        return None
    return float(matches.iloc[-1][price_col])


def weekly_room_pct(weekly_df: pd.DataFrame, direction: str, current_price: float, n: int = 2):
    """% distance from current_price to the current weekly range's
    boundary in `direction` -- the most recently confirmed weekly swing
    high (for "bullish") or swing low (for "bearish"). In plain terms:
    "how much room is left to run" before price reaches a level the
    weekly chart has already marked as a turning point.

    Used by bot.analyze_higher_timeframes() to decide whether a
    weekly/daily trend disagreement should still veto a trade: if
    there's still plenty of room left in the weekly trend's direction,
    a contradicting daily trend reading is more likely a pullback inside
    the bigger weekly move than a genuine reversal, so the weekly trend
    is allowed to win instead of the trade being skipped outright.

    Returns None if no confirmed weekly swing point exists yet in that
    direction -- callers should treat this conservatively, as "room
    unknown" (i.e. don't override on it). A negative result means price
    has already traded through the last confirmed swing level in that
    direction.
    """
    if current_price is None or current_price <= 0:
        return None
    level = nearest_confirmed_swing_level(weekly_df, direction, n=n)
    if level is None:
        return None
    if direction == "bullish":
        return (level - current_price) / current_price * 100.0
    if direction == "bearish":
        return (current_price - level) / current_price * 100.0
    return None


def _find_order_block(opens, highs, lows, closes, direction, search_from, bos_idx):
    """Locates the order block that CAUSED a structure break at bos_idx,
    or returns None if there isn't one that qualifies.

    The candidate is the last opposite-colored candle at or before the
    extreme of the leg that broke structure (the lowest low for a bullish
    break, the highest high for a bearish one), searched between the swing
    that was broken (search_from) and the breaking candle. It is only
    accepted as the cause of the break if:
      - no candle between it and the break closed back beyond its far
        side (the impulse never gave the zone back), and
      - the breaking candle closed beyond the zone's near side (price
        actually displaced out of the zone, not just past a swing that
        happened to sit inside it).
    """
    lo = search_from + 1
    if bos_idx - lo < 1:
        return None
    if direction == "bullish":
        extreme_idx = lo + int(np.argmin(lows[lo:bos_idx]))
        ob_idx = None
        for j in range(extreme_idx, search_from, -1):
            if closes[j] < opens[j]:
                ob_idx = j
                break
        if ob_idx is None:
            return None
        top, bottom = float(highs[ob_idx]), float(lows[ob_idx])
        if (closes[ob_idx + 1 : bos_idx + 1] < bottom).any():
            return None
        if closes[bos_idx] <= top:
            return None
    else:
        extreme_idx = lo + int(np.argmax(highs[lo:bos_idx]))
        ob_idx = None
        for j in range(extreme_idx, search_from, -1):
            if closes[j] > opens[j]:
                ob_idx = j
                break
        if ob_idx is None:
            return None
        top, bottom = float(highs[ob_idx]), float(lows[ob_idx])
        if (closes[ob_idx + 1 : bos_idx + 1] > top).any():
            return None
        if closes[bos_idx] >= bottom:
            return None
    return {"formed_idx": ob_idx, "top": top, "bottom": bottom}


def _apply_mitigation(highs, lows, closes, obs):
    """Scans every candle after each order block's break of structure and
    records how price has interacted with the zone since:

      first_touch_idx -- first candle whose wick traded into the zone
      invalidated     -- a COMPLETED candle closed all the way through it
      mitigated       -- invalidated, OR price has touched the zone and is
                         no longer inside it (it is outside right now, or
                         a completed candle since the first touch closed
                         outside the zone). How recently that happened
                         doesn't matter: once price has visited the zone
                         and left without a trade, the zone is used up.

    The one case a touched zone stays unmitigated is when price is still
    inside it right now on its first visit -- a limit order at the zone's
    near edge would then fill immediately, inside the zone. The
    in-progress last candle is never used to invalidate a zone (its close
    isn't final), but its current price does count for "outside right now".
    """
    last = len(closes) - 1
    price = closes[last]
    for ob in obs:
        top, bottom = ob["top"], ob["bottom"]
        bullish = ob["type"] == "bullish"
        first_touch = None
        invalidated = False
        left_zone = False
        for j in range(ob["bos_idx"] + 1, last + 1):
            if bullish:
                touched = lows[j] <= top
                through = closes[j] < bottom
            else:
                touched = highs[j] >= bottom
                through = closes[j] > top
            if through and j < last:
                invalidated = True
                break
            if first_touch is None and touched:
                first_touch = j
            if first_touch is not None and j < last and not (bottom <= closes[j] <= top):
                left_zone = True  # a completed candle closed back outside the zone
        inside_now = bottom <= price <= top
        ob["first_touch_idx"] = first_touch
        ob["invalidated"] = invalidated
        ob["mitigated"] = invalidated or (
            first_touch is not None and (left_zone or not inside_now)
        )


def detect_order_blocks(df: pd.DataFrame, n: int = 2) -> list[dict]:
    """Returns a list of order block dicts in chronological order:
    {type, top, bottom, formed_idx, bos_idx, structure, broken_level,
     first_touch_idx, invalidated, mitigated}

    An order block is only recorded when it CAUSED a structure break (see
    _find_order_block): either a BOS (break in the same direction as the
    previous break -- trend continuation) or a CHoCH (change of character:
    the first break against the previous break's direction). The very
    first break in the lookback window has no prior direction to compare
    against and is labelled BOS.

    `mitigated` is computed from the full candle history every call (see
    _apply_mitigation), so a zone that was already visited stays
    mitigated across scheduled runs instead of looking fresh again each
    time the order blocks are re-detected.
    """
    df = find_fractals(df, n=n)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    swing_highs = df["swing_high"].values
    swing_lows = df["swing_low"].values
    obs: list[dict] = []

    last_swing_high = None
    last_swing_high_idx = None
    last_swing_low = None
    last_swing_low_idx = None
    structure = None  # direction of the most recent break: "bullish" / "bearish"

    for i in range(len(df)):
        if swing_highs[i]:
            last_swing_high = highs[i]
            last_swing_high_idx = i
        if swing_lows[i]:
            last_swing_low = lows[i]
            last_swing_low_idx = i

        # Bullish break (BOS if structure was already bullish, CHoCH if it was bearish)
        if last_swing_high is not None and closes[i] > last_swing_high:
            kind = "CHoCH" if structure == "bearish" else "BOS"
            found = _find_order_block(
                opens, highs, lows, closes, "bullish", last_swing_high_idx, i
            )
            if found is not None:
                obs.append(
                    {
                        "type": "bullish",
                        "top": found["top"],
                        "bottom": found["bottom"],
                        "formed_idx": found["formed_idx"],
                        "bos_idx": i,
                        "structure": kind,
                        "broken_level": float(last_swing_high),
                        "mitigated": False,
                    }
                )
            structure = "bullish"
            last_swing_high = None  # structure broken; wait for the next confirmed swing

        # Bearish break (BOS if structure was already bearish, CHoCH if it was bullish)
        if last_swing_low is not None and closes[i] < last_swing_low:
            kind = "CHoCH" if structure == "bullish" else "BOS"
            found = _find_order_block(
                opens, highs, lows, closes, "bearish", last_swing_low_idx, i
            )
            if found is not None:
                obs.append(
                    {
                        "type": "bearish",
                        "top": found["top"],
                        "bottom": found["bottom"],
                        "formed_idx": found["formed_idx"],
                        "bos_idx": i,
                        "structure": kind,
                        "broken_level": float(last_swing_low),
                        "mitigated": False,
                    }
                )
            structure = "bearish"
            last_swing_low = None

    _apply_mitigation(highs, lows, closes, obs)
    return obs


def _setup_from_ob(ob: dict, reward_risk_ratio: float):
    """Limit-order setup for one order block: entry at the zone edge price
    reaches first (top of a bullish OB, bottom of a bearish OB), stop just
    beyond the far edge, fixed reward:risk take-profit. None if the
    stop distance comes out non-positive."""
    if ob["type"] == "bullish":
        entry = float(ob["top"])
        stop = ob["bottom"] * 0.999
        risk = entry - stop
        side, position_side = "buy", "LONG"
        target = entry + risk * reward_risk_ratio
    else:
        entry = float(ob["bottom"])
        stop = ob["top"] * 1.001
        risk = stop - entry
        side, position_side = "sell", "SHORT"
        target = entry - risk * reward_risk_ratio
    if risk <= 0:
        return None
    return {
        "side": side,
        "position_side": position_side,
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "order_block": ob,
    }


def _zones_overlap(a: dict, b: dict) -> bool:
    return a["bottom"] <= b["top"] and b["bottom"] <= a["top"]


def limit_setups(
    df: pd.DataFrame,
    order_blocks: list[dict],
    max_setups: int = 2,
    direction: str | None = None,
    reward_risk_ratio: float = 2.0,
) -> list[dict]:
    """The most recent `max_setups` valid, unmitigated order blocks, as
    resting limit-order setups (newest first).

    - only order blocks that caused a BOS/CHoCH exist at all (see
      detect_order_blocks)
    - mitigated / invalidated order blocks are skipped (see
      _apply_mitigation)
    - an order block whose break happened on the still-forming last
      candle is skipped until that candle closes
    - `direction` ("bullish"/"bearish") keeps only order blocks of that
      type -- bot.py passes the higher-timeframe bias here; None allows both
    - a zone overlapping one already picked is skipped, so two orders
      never sit on the same price area
    """
    last = len(df) - 1
    picks: list[dict] = []
    ordered = sorted(order_blocks, key=lambda o: (o["bos_idx"], o["formed_idx"]), reverse=True)
    for ob in ordered:
        if ob["mitigated"] or ob["bos_idx"] >= last:
            continue
        if direction in ("bullish", "bearish") and ob["type"] != direction:
            continue
        if any(_zones_overlap(ob, p["order_block"]) for p in picks):
            continue
        setup = _setup_from_ob(ob, reward_risk_ratio)
        if setup is None:
            continue
        picks.append(setup)
        if len(picks) >= max_setups:
            break
    return picks


def generate_signal(df: pd.DataFrame, order_blocks: list[dict], reward_risk_ratio: float = 2.0):
    """Market-entry variant (no longer used by bot.py, which now rests
    limit orders via limit_setups()): returns a setup for the first valid
    order block price is sitting inside right now on its first visit, or
    None."""
    if len(df) < 2:
        return None
    i = len(df) - 1
    price = float(df.iloc[i]["close"])
    for ob in order_blocks:
        if ob["mitigated"] or ob["bos_idx"] >= i or ob.get("first_touch_idx") is None:
            continue
        if not (ob["bottom"] <= price <= ob["top"]):
            continue
        setup = _setup_from_ob(ob, reward_risk_ratio)
        if setup is None:
            continue
        risk = abs(price - setup["stop_loss"])
        direction = 1 if setup["position_side"] == "LONG" else -1
        setup["entry"] = price
        setup["take_profit"] = price + direction * risk * reward_risk_ratio
        return setup
    return None


def next_swing_beyond(df: pd.DataFrame, direction: str, beyond_price: float, n: int = 2):
    """The closest confirmed swing high ABOVE beyond_price (direction=
    "bullish", i.e. a LONG's next resistance) or swing low BELOW it
    (direction="bearish", a SHORT's next support) in df, or None if the
    fetched candles contain no such swing. Used to place TP2 at the next
    swing level past TP1.
    """
    out = find_fractals(df, n=n)
    if direction == "bullish":
        levels = out.loc[out["swing_high"], "high"]
        levels = levels[levels > beyond_price]
        return float(levels.min()) if not levels.empty else None
    if direction == "bearish":
        levels = out.loc[out["swing_low"], "low"]
        levels = levels[levels < beyond_price]
        return float(levels.max()) if not levels.empty else None
    return None
