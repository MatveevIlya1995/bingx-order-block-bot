"""
Strategy research: simple trend-following rules on years of BingX history.

    python3 research.py --since 2021-01-01

Why this exists
---------------
The order-block bot's own backtest (backtest.py) can only replay ~45 days of
5m history, and on that window every smart-money variant lost money, mostly
to fees and noise on tight 5m stops. This script tests a different family of
rules -- trend following on daily / 4h candles, the one family with a long
published record of working in futures markets incl. crypto -- over several
years, so a result can be checked on data the rule was not chosen on.

Every strategy is simulated as ONE account (like the bot): fixed-fraction
risk per trade (stop distance sets the size), at most MAX_POS positions,
leverage capped, taker fee + slippage on every fill. Funding is ignored
(see the report).

The period is split: the first 2/3 is IN-SAMPLE (parameters are compared
there), the last 1/3 is OUT-OF-SAMPLE (only looked at for the chosen rows).
A rule is only worth trading if it makes money in BOTH.
"""
import argparse
import datetime as dt
import itertools
import json
import math
import os
import time

import ccxt
import numpy as np
import pandas as pd

SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK",
           "LTC", "DOT", "TRX", "NEAR", "ATOM", "APT"]
FEE = 0.0005      # taker, per side
SLIP = 0.0003     # per side
RISK = 0.01       # equity risked per trade
MAX_POS = 5
MAX_LEV = 3.0     # total notional / equity cap
FUNDING_LONG_PER_DAY = 0.0003  # 0.01%/8h paid by longs; shorts get nothing (conservative)
TF_MS = {"4h": 14_400_000, "1d": 86_400_000}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def fetch(ex, sym, tf, since_ms, cache_dir):
    path = os.path.join(cache_dir, f"{sym}_{tf}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    rows = _fetch_forward(ex, sym, tf, since_ms) or _fetch_backward(ex, sym, tf, since_ms)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts").astype(float)
    df = df.iloc[:-1]  # drop the still-forming candle
    df.to_pickle(path)
    return df


def _call(ex, *a, **k):
    for attempt in range(5):
        try:
            return ex.fetch_ohlcv(*a, **k)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def _fetch_forward(ex, sym, tf, since_ms):
    rows, cur = [], since_ms
    while cur < time.time() * 1000:
        batch = _call(ex, f"{sym}/USDT:USDT", tf, since=cur, limit=1000)
        if not batch:
            break
        rows += batch
        nxt = batch[-1][0] + TF_MS[tf]
        if nxt <= cur or len(batch) < 2:
            break
        cur = nxt
        time.sleep(0.25)
    return rows


def _fetch_backward(ex, sym, tf, since_ms):
    """Pages back from now with endTime -- BingX returns nothing for a
    'since' further back than it keeps for small timeframes."""
    rows, end = [], int(time.time() * 1000)
    while end > since_ms:
        for attempt in range(5):
            try:
                batch = ex.fetch_ohlcv(f"{sym}/USDT:USDT", tf, limit=1000, params={"endTime": end})
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        batch = [b for b in batch if b[0] < end]
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0]
        time.sleep(0.25)
    return [r for r in rows if r[0] >= since_ms]


def atr(df, n):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# --------------------------------------------------------------------------
# signals: each returns per-bar columns the simulator reads
#   long_entry / short_entry: stop price to enter at next open (NaN = none)
#   exit_long / exit_short:   close at next open
#   trail_long / trail_short: stop level to ratchet to
# --------------------------------------------------------------------------
def donchian(df, n_entry, n_exit, atr_n=20, atr_mult=2.0, trend_ema=None, allow_short=True):
    d = pd.DataFrame(index=df.index)
    hi = df["high"].rolling(n_entry).max().shift()
    lo = df["low"].rolling(n_entry).min().shift()
    xhi = df["high"].rolling(n_exit).max().shift()
    xlo = df["low"].rolling(n_exit).min().shift()
    a = atr(df, atr_n)
    up = df["close"] >= hi
    dn = df["close"] <= lo
    if trend_ema:
        ema = df["close"].ewm(span=trend_ema, adjust=False).mean()
        up &= df["close"] > ema
        dn &= df["close"] < ema
    d["long_entry"] = np.where(up, df["close"] - atr_mult * a, np.nan)
    d["short_entry"] = np.where(dn & allow_short, df["close"] + atr_mult * a, np.nan)
    d["trail_long"] = np.maximum(xlo, df["close"] - atr_mult * a)
    d["trail_short"] = np.minimum(xhi, df["close"] + atr_mult * a)
    d["exit_long"] = False
    d["exit_short"] = False
    return d


def ema_cross(df, fast, slow, atr_n=20, atr_mult=3.0, allow_short=True):
    d = pd.DataFrame(index=df.index)
    f = df["close"].ewm(span=fast, adjust=False).mean()
    s = df["close"].ewm(span=slow, adjust=False).mean()
    a = atr(df, atr_n)
    cu = (f > s) & (f.shift() <= s.shift())
    cd = (f < s) & (f.shift() >= s.shift())
    d["long_entry"] = np.where(cu, df["close"] - atr_mult * a, np.nan)
    d["short_entry"] = np.where(cd & allow_short, df["close"] + atr_mult * a, np.nan)
    d["trail_long"] = df["close"] - atr_mult * a
    d["trail_short"] = df["close"] + atr_mult * a
    d["exit_long"] = f < s
    d["exit_short"] = f > s
    return d


# --------------------------------------------------------------------------
# portfolio simulator
# --------------------------------------------------------------------------
def simulate(data, sigs, start, end, equity0=1000.0):
    """data/sigs: {sym: DataFrame} on the same timeframe. Signals on bar t
    act at bar t+1's open; stops are checked on each bar's high/low (a bar
    that opens beyond the stop fills at the open)."""
    idx = sorted(set().union(*[d.index for d in data.values()]))
    idx = [t for t in idx if start <= t < end]
    bar_days = (idx[1] - idx[0]) / pd.Timedelta(days=1) if len(idx) > 1 else 1.0
    equity = equity0
    peak, max_dd = equity0, 0.0
    pos = {}       # sym -> dict(side, qty, entry, stop)
    pending = {}   # sym -> (side, stop) to open at next bar's open
    trades = []
    curve = []
    for t in idx:
        # 1) open pending entries at this bar's open
        for sym, (side, stop) in list(pending.items()):
            df = data[sym]
            if t not in df.index or sym in pos or len(pos) >= MAX_POS:
                continue
            o = df.at[t, "open"]
            px = o * (1 + SLIP) if side > 0 else o * (1 - SLIP)
            dist = abs(px - stop)
            if dist <= 0 or (side > 0 and stop >= px) or (side < 0 and stop <= px):
                continue
            qty = equity * RISK / dist
            open_notional = sum(p["qty"] * p["entry"] for p in pos.values())
            qty = min(qty, max(equity * MAX_LEV - open_notional, 0) / px)
            if qty * px < 5:
                continue
            equity -= qty * px * FEE
            pos[sym] = {"side": side, "qty": qty, "entry": px, "stop": stop, "t": t, "fees": qty * px * FEE,
                        "risk": qty * dist}
        pending.clear()

        # funding on open longs (shorts conservatively receive nothing)
        for sym, p in pos.items():
            if p["side"] > 0 and t in data[sym].index:
                f = p["qty"] * data[sym].at[t, "open"] * FUNDING_LONG_PER_DAY * bar_days
                equity -= f
                p["fees"] += f

        # 2) stops / exits on this bar
        for sym, p in list(pos.items()):
            df = data[sym]
            if t not in df.index:
                continue
            o, h, l, c = df.loc[t, ["open", "high", "low", "close"]]
            exit_px = None
            if p["side"] > 0 and l <= p["stop"]:
                exit_px = min(o, p["stop"]) * (1 - SLIP)
            elif p["side"] < 0 and h >= p["stop"]:
                exit_px = max(o, p["stop"]) * (1 + SLIP)
            if exit_px is None and p.get("exit_next"):
                exit_px = o * (1 - SLIP) if p["side"] > 0 else o * (1 + SLIP)
            if exit_px is not None:
                pnl = (exit_px - p["entry"]) * p["qty"] * p["side"]
                fee = exit_px * p["qty"] * FEE
                equity += pnl - fee
                trades.append({"sym": sym, "side": p["side"], "entry_t": p["t"], "exit_t": t,
                               "pnl": pnl - fee - p["fees"], "r": (pnl - fee - p["fees"]) / p["risk"],
                               "fees": fee + p["fees"]})
                del pos[sym]
                continue
            s = sigs[sym].loc[t]
            if p["side"] > 0:
                if not math.isnan(s["trail_long"]):
                    p["stop"] = max(p["stop"], s["trail_long"])
                p["exit_next"] = bool(s["exit_long"])
            else:
                if not math.isnan(s["trail_short"]):
                    p["stop"] = min(p["stop"], s["trail_short"])
                p["exit_next"] = bool(s["exit_short"])

        # 3) new signals (act next bar)
        for sym, s in sigs.items():
            if t not in s.index or sym in pos:
                continue
            row = s.loc[t]
            if not math.isnan(row["long_entry"]):
                pending[sym] = (1, row["long_entry"])
            elif not math.isnan(row["short_entry"]):
                pending[sym] = (-1, row["short_entry"])

        # mark to market
        mtm = equity + sum((data[s].at[t, "close"] - p["entry"]) * p["qty"] * p["side"]
                           for s, p in pos.items() if t in data[s].index)
        peak = max(peak, mtm)
        max_dd = max(max_dd, 1 - mtm / peak)
        curve.append((t, mtm))
    # close anything still open at the last close
    for sym, p in pos.items():
        c = data[sym]["close"][data[sym].index < end].iloc[-1]
        pnl = (c - p["entry"]) * p["qty"] * p["side"] - c * p["qty"] * FEE
        equity += pnl
        trades.append({"sym": sym, "side": p["side"], "entry_t": p["t"], "exit_t": end,
                       "pnl": pnl - p["fees"], "r": (pnl - p["fees"]) / p["risk"], "fees": p["fees"]})
    return equity, max_dd, trades, curve


def stats(equity0, equity, dd, trades, days):
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    ret = equity / equity0 - 1
    cagr = (equity / equity0) ** (365 / max(days, 1)) - 1 if equity > 0 else -1
    return {
        "trades": n,
        "win%": round(100 * len(wins) / n, 1) if n else 0,
        "PF": round(gp / gl, 2) if gl > 0 else None,
        "avgR": round(sum(t["r"] for t in trades) / n, 3) if n else 0,
        "return%": round(100 * ret, 1),
        "CAGR%": round(100 * cagr, 1),
        "maxDD%": round(100 * dd, 1),
        "fees%": round(100 * sum(t["fees"] for t in trades) / equity0, 1),
    }


STRATS = {}
for tf in ("1d", "4h"):
    for n_e, n_x in ((20, 10), (30, 15), (40, 20), (55, 20), (55, 30), (100, 50)):
        for ema in (None, 200):
            for shorts in (True, False):
                STRATS[f"donchian {n_e}/{n_x} {tf}{' ema200' if ema else ''}{'' if shorts else ' long-only'}"] = (
                    tf, lambda df, a=n_e, b=n_x, e=ema, s=shorts: donchian(df, a, b, trend_ema=e, allow_short=s))
    for f, s_ in ((20, 100), (50, 200)):
        for shorts in (True, False):
            STRATS[f"ema {f}/{s_} {tf}{'' if shorts else ' long-only'}"] = (
                tf, lambda df, a=f, b=s_, s=shorts: ema_cross(df, a, b, allow_short=s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--out", default="research_results")
    ap.add_argument("--cache", default="research_cache")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    ex = ccxt.bingx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since_ms = int(pd.Timestamp(args.since).value // 1_000_000)
    warm = 250
    data = {"1d": {}, "4h": {}}
    for tf in data:
        for sym in SYMBOLS:
            try:
                df = fetch(ex, sym, tf, since_ms - warm * TF_MS[tf], args.cache)
            except Exception as e:
                print(f"{sym} {tf}: {e}")
                continue
            if len(df) > warm + 100:
                data[tf][sym] = df
            print(f"{sym} {tf}: {len(df)} candles from {df.index[0] if len(df) else '-'}", flush=True)

    first = max(pd.Timestamp(args.since), min(d.index[0] for d in data["1d"].values()) + pd.Timedelta(days=warm))
    last = min(d.index[-1] for d in data["1d"].values()) + pd.Timedelta(days=1)
    split = first + (last - first) * 2 / 3
    lines = [f"# Trend-following research: {first.date()} to {last.date()}",
             "",
             f"In-sample {first.date()} to {split.date()}, out-of-sample {split.date()} to {last.date()}. "
             f"{len(data['1d'])} symbols. Risk {RISK*100:g}% per trade, max {MAX_POS} positions, "
             f"notional cap {MAX_LEV:g}x equity, taker fee {FEE*100:g}% + slippage {SLIP*100:g}% per side, "
             "funding 0.03%/day charged on longs (shorts get none). Entries at the next candle's open after a close-based signal.",
             "",
             "| strategy | IS trades | IS PF | IS avgR | IS CAGR% | IS maxDD% | OOS trades | OOS PF | OOS avgR | OOS return% | OOS maxDD% | full CAGR% | full maxDD% |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    results = {}
    for name, (tf, fn) in STRATS.items():
        d = data[tf]
        sigs = {s: fn(df) for s, df in d.items()}
        row = {}
        for label, a, b in (("IS", first, split), ("OOS", split, last), ("full", first, last)):
            eq, dd, tr, _ = simulate(d, sigs, a, b)
            row[label] = stats(1000.0, eq, dd, tr, (b - a).days)
            if label == "full":
                yearly = {}
                for t in tr:
                    yearly.setdefault(t["exit_t"].year, 0.0)
                    yearly[t["exit_t"].year] += t["pnl"]
                row["by_year_pnl_on_1000"] = {str(k): round(v, 1) for k, v in sorted(yearly.items())}
        results[name] = row
        i, o, f = row["IS"], row["OOS"], row["full"]
        lines.append(f"| {name} | {i['trades']} | {i['PF']} | {i['avgR']} | {i['CAGR%']} | {i['maxDD%']} | "
                     f"{o['trades']} | {o['PF']} | {o['avgR']} | {o['return%']} | {o['maxDD%']} | {f['CAGR%']} | {f['maxDD%']} |")
        print(lines[-1], flush=True)
    lines += ["", "## P/L by exit year (full period, $1000 start)", ""]
    for name, row in results.items():
        lines.append(f"- {name}: {row['by_year_pnl_on_1000']}")
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(lines))
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
