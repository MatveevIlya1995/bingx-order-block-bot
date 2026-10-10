"""
Order block (smart-money-concept) strategy.

Definitions used here (one common, defensible interpretation among several
that exist in trading communities -- tune to taste):

MARKET STRUCTURE: BOS AND CHoCH (see detect_structure_breaks)
-------------------------------------------------------------
A BOS and a CHoCH are the same event -- a candle CLOSES beyond a swing
point. Which of the two it is depends only on which way structure was
pointing before:

  BOS   (break of structure)   the break is in the SAME direction as the
                               previous break: the trend continues.
  CHoCH (change of character)  the break is AGAINST the previous break:
                               the first sign the trend may be turning.

After a CHoCH the direction flips, so the next break the same way is a
BOS. A reversal therefore reads CHoCH first, then BOS.

The five rules that make that exact:

1. Swing point. A candle is a swing high if its high is the highest of the
   (2N+1)-candle window centred on it (FRACTAL_N candles on each side),
   and symmetrically for swing lows. It becomes the structure level only
   once the N candles after it exist -- never at the candle where it
   occurred (STRUCTURE_CONFIRMED_SWINGS, on by default).
2. Break. A candle CLOSES beyond the most recent confirmed swing high
   (bullish break) or swing low (bearish break). A wick through a level
   that closes back inside is NOT a break -- it is a liquidity sweep.
3. Direction. Starts undefined; after that it is the direction of the most
   recent break.
4. Label. A break in the current direction is a BOS, a break against it is
   a CHoCH. The first break in the fetched candles has nothing to compare
   with and is labelled BOS.
5. One break per swing. A level that was broken is used up; the next break
   needs a new confirmed swing.

With FRACTAL_N=2 these are small ("internal") swings of the entry
timeframe. A larger FRACTAL_N reads bigger ("external") structure and
gives fewer, more significant breaks.

ORDER BLOCKS
------------
0. Only order blocks at a STRONG SWING count. A strong swing is a swing
   low/high from which price went on to cause a BOS or CHoCH; a break
   made out of a pause that never formed a swing gives no order block
   (REQUIRE_STRONG_SWING, on by default -- see detect_order_blocks).
1. The order block is the last opposite-colored candle before the
   impulsive move that produced the break:
   - bullish OB = last bearish (red) candle before a bullish BOS/CHoCH
   - bearish OB = last bullish (green) candle before a bearish BOS/CHoCH
   Its zone is that candle's [low, high]. An order block that did not
   cause a BOS or CHoCH is not recorded at all.
2. Entries are resting limit orders at the zone edge price reaches first
   (see limit_setups). A zone price has visited and left, or that a
   completed candle closed through, is mitigated and gets no order.

OTHER SMART-MONEY CONCEPTS (see add_smc_context / apply_entry_filters)
----------------------------------------------------------------------
Each setup is tagged with: whether its order block came from a BOS or a
CHoCH, whether the move took liquidity first (a sweep of a prior swing),
how strong the displacement out of the zone was, whether it left a fair
value gap, where the entry sits in the dealing range (premium/discount),
and the UTC hour. Every one of those can be made a condition for placing
the order; all are OFF by default so they can be backtested first.

Stop-loss is placed just beyond the OB zone (below it for a LONG, above it
for a SHORT); take-profit is a fixed risk multiple -- config.py hard-
enforces a 2:1 reward:risk ratio regardless of what REWARD_RISK_RATIO is
set to (see required_reward_risk_ratio there), and the position sizer in
risk_manager.py hard-caps the stop-loss risk itself at 3% of equity.

Multi-timeframe use: every function below operates on a single timeframe's
DataFrame. bot.py calls detect_trend() and detect_order_blocks() once per
level of its daily -> 4h -> 1h -> 15m cascade (see
bot.analyze_higher_timeframes()), and again on the 5m entry timeframe for
the limit-order setups. combine_htf_bias() folds the primary levels' trend
readings (daily + 4h) into one overall directional bias that the entry
must not contradict -- UNLESS the room override applies (see
weekly_room_pct() below and bot.analyze_higher_timeframes()): when the
two primary levels disagree but price still has enough room left to run
toward the higher chart's own last confirmed swing high/low, the higher
level's trend is allowed to win instead of blocking the trade outright.
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
    return {"formed_idx": ob_idx, "top": top, "bottom": bottom, "origin_idx": extreme_idx}


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


def detect_structure_breaks(df: pd.DataFrame, n: int = 2, confirmed_swings: bool = True) -> list[dict]:
    """Every break of structure in df, oldest first:
    {idx, direction, kind, level, swing_idx}

      idx        -- the candle that CLOSED beyond the level
      direction  -- "bullish" (closed above a swing high) / "bearish"
      kind       -- "BOS" (same direction as the previous break) or
                    "CHoCH" (against it); the first break in df is "BOS"
      level      -- the swing high/low that was broken
      swing_idx  -- the candle that made that swing

    The five rules are spelled out at the top of this file.

    confirmed_swings=True (the default): a swing becomes the level to break
    only once the n candles after it exist, so what is found at candle i
    depends on candles up to i and nothing later -- a break, once a
    completed candle made it, never disappears again.

    confirmed_swings=False is the behaviour before this option existed: a
    swing replaced the level at its own candle. Two things follow from
    that: a candle that closes beyond the old level AND turns out to be a
    swing itself (price pulls straight back after the break) is first seen
    as a break and then, once that swing is confirmed n candles later, no
    longer is -- its order block and the order resting in it vanish
    shortly after they appeared;
    and with the break gone the direction doesn't flip, so later BOS/CHoCH
    labels are off. Kept only so the two can be compared in the backtest.
    """
    if "swing_high" not in df.columns or "swing_low" not in df.columns:
        df = find_fractals(df, n=n)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    swing_highs = df["swing_high"].values
    swing_lows = df["swing_low"].values
    breaks: list[dict] = []

    last_swing_high = None
    last_swing_high_idx = None
    last_swing_low = None
    last_swing_low_idx = None
    structure = None  # direction of the most recent break: "bullish" / "bearish"

    for i in range(len(df)):
        # The swing that becomes known at candle i: the one n candles back
        # (its right-hand side is complete now), or -- old behaviour --
        # candle i itself.
        s = i - n if confirmed_swings else i
        if s >= 0:
            if swing_highs[s]:
                last_swing_high = highs[s]
                last_swing_high_idx = s
            if swing_lows[s]:
                last_swing_low = lows[s]
                last_swing_low_idx = s

        # Bullish break (BOS if structure was already bullish, CHoCH if it was bearish)
        if last_swing_high is not None and closes[i] > last_swing_high:
            breaks.append({
                "idx": i,
                "direction": "bullish",
                "kind": "CHoCH" if structure == "bearish" else "BOS",
                "level": float(last_swing_high),
                "swing_idx": last_swing_high_idx,
            })
            structure = "bullish"
            last_swing_high = None  # structure broken; wait for the next confirmed swing

        # Bearish break (BOS if structure was already bearish, CHoCH if it was bullish)
        if last_swing_low is not None and closes[i] < last_swing_low:
            breaks.append({
                "idx": i,
                "direction": "bearish",
                "kind": "CHoCH" if structure == "bullish" else "BOS",
                "level": float(last_swing_low),
                "swing_idx": last_swing_low_idx,
            })
            structure = "bearish"
            last_swing_low = None

    return breaks


def structure_state(df: pd.DataFrame, n: int = 2, confirmed_swings: bool = True):
    """Where structure stands right now, from COMPLETED candles only (the
    still-forming last candle's close isn't final): the most recent break
    as {direction, kind, level, idx}, or None if there hasn't been one.
    For logging -- "15m structure: bullish, last break a CHoCH"."""
    last = len(df) - 1
    done = [b for b in detect_structure_breaks(df, n=n, confirmed_swings=confirmed_swings) if b["idx"] < last]
    return done[-1] if done else None


def detect_order_blocks(
    df: pd.DataFrame, n: int = 2, confirmed_swings: bool = True, strong_swings_only: bool = True,
) -> list[dict]:
    """Returns a list of order block dicts in chronological order:
    {type, top, bottom, formed_idx, bos_idx, structure, broken_level,
     swing_idx, origin_idx, strong_swing, first_touch_idx, invalidated,
     mitigated}

    STRONG SWINGS. A strong swing is a swing low (or high) from which
    price went on to cause a BOS or CHoCH: the low of the move that closed
    above the last swing high, or the high of the move that closed below
    the last swing low. `origin_idx` is that turning point of the move and
    `strong_swing` says whether it really is a swing -- a confirmed
    fractal low/high by rule 1 (n candles on each side, the n after it
    complete). When it isn't, structure was broken out of a pause of a
    candle or two that never formed a swing, and with
    strong_swings_only=True (the default) that order block does not count:
    it is not returned at all, so it also never takes one of the "last two
    order blocks" places.

    An order block is only recorded when it CAUSED a structure break (see
    _find_order_block): either a BOS (break in the same direction as the
    previous break -- trend continuation) or a CHoCH (change of character:
    the first break against the previous break's direction). The very
    first break in the lookback window has no prior direction to compare
    against and is labelled BOS. The breaks themselves come from
    detect_structure_breaks(); `confirmed_swings` is passed straight to it.

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
    last = len(df) - 1
    obs: list[dict] = []

    for brk in detect_structure_breaks(df, n=n, confirmed_swings=confirmed_swings):
        found = _find_order_block(
            opens, highs, lows, closes, brk["direction"], brk["swing_idx"], brk["idx"]
        )
        if found is None:
            continue
        origin = found["origin_idx"]
        # a real swing point, with the n candles after it complete (the
        # last candle is still forming and could yet undo it)
        strong = origin + n < last and bool(
            swing_lows[origin] if brk["direction"] == "bullish" else swing_highs[origin]
        )
        if strong_swings_only and not strong:
            continue
        obs.append(
            {
                "type": brk["direction"],
                "top": found["top"],
                "bottom": found["bottom"],
                "formed_idx": found["formed_idx"],
                "bos_idx": brk["idx"],
                "structure": brk["kind"],
                "broken_level": brk["level"],
                "swing_idx": brk["swing_idx"],
                "origin_idx": origin,
                "strong_swing": strong,
                "mitigated": False,
            }
        )

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


# ---------------------------------------------------------------------------
# smart-money context of a setup
# ---------------------------------------------------------------------------

ATR_PERIOD = 14
OB_STRUCTURE_MODES = ("any", "bos", "choch")


def _atr(highs, lows, closes, end_idx: int, period: int = ATR_PERIOD):
    """Average true range of the `period` candles ending at end_idx
    (inclusive) -- the "normal" candle size just before a move. None when
    there aren't two candles to measure, or the range is zero."""
    start = max(1, end_idx - period + 1)
    if end_idx < start:
        return None
    h, l, prev_close = highs[start:end_idx + 1], lows[start:end_idx + 1], closes[start - 1:end_idx]
    true_range = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    value = float(true_range.mean())
    return value if value > 0 else None


def _liquidity_taken(highs, lows, closes, swing_highs, swing_lows, ob: dict) -> str:
    """Did the move that formed this order block take liquidity first?
    Returns "sweep", "run" or "none".

    Bullish order block: between the swing high that was broken and the
    break, price traded BELOW a swing low that formed earlier and that
    nothing had traded below until then -- the stops resting under that
    low were hit before price turned up.

      "sweep"  it did, and no candle of the move CLOSED below that low:
               the level was pierced by wicks only and rejected. This is
               the liquidity sweep of the textbooks (rule 2 at the top of
               this file: a wick through a level is a sweep, not a break).
      "run"    it did, but candles also closed below every such low: the
               level was accepted before price turned.
      "none"   the move never reached an earlier untouched swing low.

    If several earlier lows were taken, one that was only wicked is enough
    for "sweep". Bearish order blocks: the mirror image, above earlier
    swing highs.
    """
    start = ob.get("swing_idx")
    if start is None:
        return "none"
    lo, bos_idx = start + 1, ob["bos_idx"]
    if bos_idx <= lo:
        return "none"
    result = "none"
    if ob["type"] == "bullish":
        leg_extreme = float(lows[lo:bos_idx].min())
        leg_close = float(closes[lo:bos_idx].min())
        untouched = float("inf")          # lowest low between swing s and the move
        for s in range(start, -1, -1):
            if untouched <= leg_extreme:
                break                     # everything older was already traded through
            if swing_lows[s] and leg_extreme < lows[s] <= untouched:
                if leg_close >= lows[s]:
                    return "sweep"
                result = "run"
            untouched = min(untouched, float(lows[s]))
        return result
    leg_extreme = float(highs[lo:bos_idx].max())
    leg_close = float(closes[lo:bos_idx].max())
    untouched = float("-inf")
    for s in range(start, -1, -1):
        if untouched >= leg_extreme:
            break
        if swing_highs[s] and leg_extreme > highs[s] >= untouched:
            if leg_close <= highs[s]:
                return "sweep"
            result = "run"
        untouched = max(untouched, float(highs[s]))
    return result


def _displacement_atr(opens, highs, lows, closes, ob: dict):
    """How hard price left the zone: the biggest candle BODY between the
    order block and the candle that broke structure (both ends of the move
    included), in multiples of the average true range of the ATR_PERIOD
    candles up to the order block. 2.0 = a candle body twice the size of a
    normal candle's whole range. None if it can't be measured."""
    a, b = ob["formed_idx"] + 1, ob["bos_idx"] + 1
    atr = _atr(highs, lows, closes, ob["formed_idx"])
    if atr is None or b <= a:
        return None
    return float(np.abs(closes[a:b] - opens[a:b]).max() / atr)


def _has_fair_value_gap(highs, lows, ob: dict, last: int) -> bool:
    """Did the move out of the zone leave a fair value gap? Three candles
    in a row where the first and the third don't overlap -- for a bullish
    move the third candle's low is above the first candle's high -- with
    the middle candle anywhere from the one after the order block to the
    candle that broke structure. The third candle must be complete, so a
    gap whose middle candle is the breaking candle only counts once the
    candle after it has closed (`last` is the still-forming candle)."""
    bullish = ob["type"] == "bullish"
    for mid in range(ob["formed_idx"] + 1, ob["bos_idx"] + 1):
        if mid + 1 >= last:
            break
        if (lows[mid + 1] > highs[mid - 1]) if bullish else (highs[mid + 1] < lows[mid - 1]):
            return True
    return False


def _hour_utc(ts) -> int:
    """UTC hour of a candle timestamp (a datetime, or epoch milliseconds)."""
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return int(ts // 3_600_000) % 24
    return int(pd.Timestamp(ts).hour)


def add_smc_context(df: pd.DataFrame, setups: list[dict], n: int = 2, dealing_range_candles: int = 0) -> list[dict]:
    """Tags every setup with the smart-money facts about its order block,
    as setup["smc"]:

      structure         "BOS" (continuation) or "CHoCH" (reversal) -- the
                        break the order block caused
      liquidity         "sweep" / "run" / "none": whether the move took
                        liquidity first, and how (see _liquidity_taken)
      sweep             True for "sweep" -- an earlier swing was pierced
                        by wicks only and rejected
      displacement_atr  strength of the move out of the zone, in ATRs
                        (see _displacement_atr); None if unmeasurable
      fvg               True if the move left a fair value gap
                        (see _has_fair_value_gap)
      range_pos         where the ENTRY price sits in the dealing range --
                        the highest high and lowest low of the last
                        `dealing_range_candles` candles (0 = all fetched):
                        0.0 = the very bottom, 1.0 = the very top. Below
                        0.5 is "discount", above is "premium". None if the
                        range has no height.
      hour_utc          UTC hour of the newest candle, i.e. of this run

    Everything is read from df alone, and only from completed candles
    plus the current price, so bot.py and backtest.py get identical tags.
    Returns the same list (the dicts are tagged in place).
    """
    if not setups:
        return setups
    fr = find_fractals(df, n=n)
    opens, highs, lows, closes = (fr[c].values for c in ("open", "high", "low", "close"))
    swing_highs, swing_lows = fr["swing_high"].values, fr["swing_low"].values
    last = len(fr) - 1
    r = dealing_range_candles if 0 < dealing_range_candles < len(fr) else len(fr)
    range_high, range_low = float(highs[-r:].max()), float(lows[-r:].min())
    hour = _hour_utc(fr["ts"].iloc[-1]) if "ts" in fr.columns else None
    for setup in setups:
        ob = setup["order_block"]
        liquidity = _liquidity_taken(highs, lows, closes, swing_highs, swing_lows, ob)
        setup["smc"] = {
            "structure": ob.get("structure", "BOS"),
            "liquidity": liquidity,
            "sweep": liquidity == "sweep",
            "displacement_atr": _displacement_atr(opens, highs, lows, closes, ob),
            "fvg": _has_fair_value_gap(highs, lows, ob, last),
            "range_pos": ((setup["entry"] - range_low) / (range_high - range_low)
                          if range_high > range_low else None),
            "hour_utc": hour,
        }
    return setups


def describe_smc(smc) -> str:
    """One line for logs and the journal's trigger column."""
    if not smc:
        return "n/a"
    d = smc.get("displacement_atr")
    pos = smc.get("range_pos")
    liquidity = {"sweep": "swept (wick only)", "run": "run (closed beyond)"}.get(smc.get("liquidity"), "not taken")
    return (
        f"{smc.get('structure', '?')}, liquidity {liquidity}, "
        f"fair value gap {'yes' if smc.get('fvg') else 'no'}, displacement "
        + (f"{d:.1f} ATR" if d is not None else "n/a")
        + ", entry at "
        + (f"{pos * 100:.0f}% of the range ({'discount' if pos <= 0.5 else 'premium'})" if pos is not None else "n/a")
    )


def parse_hour_windows(text) -> list:
    """"7-10,13-16" -> [(7, 10), (13, 16)]: UTC hour windows, start
    included, end excluded. A window may wrap midnight ("22-2"). Empty
    text = no windows. Raises ValueError on anything else."""
    windows = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        start, sep, end = part.partition("-")
        if not sep:
            raise ValueError(f"{part!r} is not START-END")
        start, end = int(start), int(end)
        if not (0 <= start <= 23 and 0 <= end <= 24) or start == end:
            raise ValueError(f"{part!r}: hours must be 0-23 (end up to 24) and different")
        windows.append((start, end))
    return windows


def hour_allowed(hour: int, windows) -> bool:
    """True if `hour` falls in any of the windows (no windows = always)."""
    if not windows:
        return True
    for start, end in windows:
        if (start <= hour < end) if start < end else (hour >= start or hour < end):
            return True
    return False


def smc_filter_reason(
    setup: dict,
    ob_structure: str = "any",
    require_sweep: bool = False,
    min_displacement_atr: float = 0.0,
    require_fvg: bool = False,
    premium_discount: bool = False,
    entry_hours=None,
):
    """Why the smart-money filters reject this setup, or None if they
    don't. With every filter at its default (off) this is always None."""
    active = (ob_structure in ("bos", "choch") or require_sweep or min_displacement_atr > 0
              or require_fvg or premium_discount or bool(entry_hours))
    if not active:
        return None
    smc = setup.get("smc")
    if not smc:
        return "no smart-money context for this setup"
    if entry_hours and (smc.get("hour_utc") is None or not hour_allowed(smc["hour_utc"], entry_hours)):
        windows = ",".join(f"{a}-{b}" for a, b in entry_hours)
        return f"{smc.get('hour_utc')}h UTC is outside ENTRY_HOURS_UTC={windows}"
    if ob_structure in ("bos", "choch") and str(smc.get("structure", "")).lower() != ob_structure:
        return f"order block came from a {smc.get('structure')}, OB_STRUCTURE={ob_structure}"
    if require_sweep and not smc.get("sweep"):
        return (f"no liquidity sweep before the order block (liquidity: {smc.get('liquidity', 'none')}) "
                f"-- REQUIRE_LIQUIDITY_SWEEP")
    if min_displacement_atr > 0:
        d = smc.get("displacement_atr")
        if d is None or d < min_displacement_atr:
            shown = f"{d:.2f}" if d is not None else "n/a"
            return f"displacement {shown} ATR, below MIN_DISPLACEMENT_ATR={min_displacement_atr:g}"
    if require_fvg and not smc.get("fvg"):
        return "the move left no fair value gap (REQUIRE_FVG)"
    if premium_discount:
        pos = smc.get("range_pos")
        is_long = setup["position_side"].upper() == "LONG"
        if pos is None or (pos > 0.5 if is_long else pos < 0.5):
            shown = f"{pos * 100:.0f}%" if pos is not None else "n/a"
            return (f"entry at {shown} of the dealing range -- "
                    f"{'a LONG must be in the lower half (discount)' if is_long else 'a SHORT must be in the upper half (premium)'}"
                    f" (PREMIUM_DISCOUNT_FILTER)")
    return None


# ---------------------------------------------------------------------------
# entry filters (all off by default -- see config.py)
# ---------------------------------------------------------------------------

def stop_distance_pct(setup: dict) -> float:
    """Distance from a setup's entry to its stop-loss, in % of the entry."""
    entry = setup["entry"]
    if entry <= 0:
        return 0.0
    return abs(entry - setup["stop_loss"]) / entry * 100.0


def timing_agreement_ok(position_side: str, timing_trends, mode: str) -> bool:
    """Does the timing levels' trend (4h/1h by default) allow a trade in
    this direction?

      off         -- always
      not_against -- no timing level trends against the trade (neutral ok)
      aligned     -- every timing level trends with the trade

    timing_trends=None means the trends aren't available (trend filter
    disabled): nothing to disagree with, so the trade is allowed.
    """
    if mode == "off" or timing_trends is None:
        return True
    want = "bullish" if position_side.upper() == "LONG" else "bearish"
    against = "bearish" if want == "bullish" else "bullish"
    if mode == "not_against":
        return against not in timing_trends
    if mode == "aligned":
        return len(timing_trends) > 0 and all(t == want for t in timing_trends)
    return True


def apply_entry_filters(
    setups: list[dict],
    min_stop_distance_pct: float = 0.0,
    timing_trends=None,
    timing_mode: str = "off",
    **smc_filters,
) -> tuple[list[dict], list[tuple]]:
    """Applies the optional entry filters to the setups limit_setups()
    picked. Returns (kept, dropped) where dropped is a list of
    (setup, reason). With the defaults nothing is dropped.

    smc_filters are the smart-money filters, passed on to
    smc_filter_reason(): ob_structure, require_sweep,
    min_displacement_atr, require_fvg, premium_discount, entry_hours.
    They read setup["smc"] (see add_smc_context).

    The filters only REMOVE setups: a setup that is filtered out is not
    replaced by an older order block, so "the last two unmitigated order
    blocks" still means exactly that. bot.py and backtest.py both call
    this, so the backtest filters exactly like the live bot.
    """
    kept, dropped = [], []
    for setup in setups:
        distance = stop_distance_pct(setup)
        if min_stop_distance_pct > 0 and distance < min_stop_distance_pct:
            dropped.append((setup, f"stop {distance:.3f}% from entry, below MIN_STOP_DISTANCE_PCT={min_stop_distance_pct:g}%"))
            continue
        if not timing_agreement_ok(setup["position_side"], timing_trends, timing_mode):
            dropped.append((setup, f"timing levels {list(timing_trends)} fail HTF_TIMING_AGREEMENT={timing_mode}"))
            continue
        reason = smc_filter_reason(setup, **smc_filters) if smc_filters else None
        if reason:
            dropped.append((setup, reason))
            continue
        kept.append(setup)
    return kept, dropped


def entry_symbol_allowed(symbol: str, entry_symbols: list) -> bool:
    """True if `symbol` may take new entries. entry_symbols is the
    ENTRY_SYMBOLS list: empty = every symbol; entries may be full symbols
    ("BTC/USDT:USDT") or just the coin ("BTC"), case-insensitive."""
    if not entry_symbols:
        return True
    full = symbol.upper()
    base = full.split("/")[0]
    wanted = {str(s).strip().upper() for s in entry_symbols}
    return full in wanted or base in wanted
