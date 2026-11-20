"""gta6-trophy-rarity-forecast-01 entry: a saturation-trend ensemble forecaster.

Model summary (see README.md for the longer version):

Exophase republishes its tracked-player counts in jumps, hours apart, so a case's read history is
mostly the same state repeated. The history is first compressed to distinct states (a read whose
players_tracked and full rate vector match the previous state adds nothing). On the distinct
states, each trophy gets two views of its earn rate at the target, both built on the logit scale
against x = ln(1 + hours since the game's release, 2026-11-19T00:00Z) -- log-time because earn
rates saturate: fast early growth that bends over, which a line on log-time tracks well.

  * Trend view: weighted least squares of logit(rate) on x over all distinct states (weights
    sqrt(players_tracked) times a 36-hour-half-life recency factor). The slope is shrunk toward
    the pooled slope of the trophy's tier, by n/(n+4) with n the trophy's distinct-state count,
    and clamped to [0, 2.5]. Trophies with a flat series or fewer than 3 states borrow the tier
    slope outright, anchored on their own last logit.
  * Earners view: the same machinery fitted to ln(earners) (earners = rate x players), divided
    by a separately fitted players_tracked at the target (log-fit, slope clamped to [0, 1.2] and
    shrunk 0.8x). This separates "more players arrived" from "a bigger share of them earned it".

The median forecast is the mean of the two views' logits, passed through sanity caps per tier
(a platinum cannot sit at 5% a week in; story-progression rates get a floor under their last
read). The spread combines the trend fit's residual variance with a horizon term and a floor,
and the five scored quantiles are read off a normal on the logit scale.

Robustness is a feature, not an afterthought: null rates carry forward, a trophy with no valid
rate gets a flat 0.05 row, an empty history gets a fixed low row set, and any unexpected failure
inside a case falls back to a persistence forecast for that case. The program never crashes.

Grader contract (from the kit): runs as `python3 -P -s /bot/main.py`, Python 3.13 standard
library only; reads every *.in in the input dir, writes <id>.out as
{"trophies": [{"id": .., "q": [q05, q25, q50, q75, q95]}]} with q in percent, 0..100,
non-decreasing, at most 65,536 bytes. Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, in case a sibling module is added.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Median-rate sanity caps per tier (percent). The 100%-completion gold is capped separately.
TIER_CAPS = {"platinum": 3.0, "gold": 65.0, "silver": 60.0, "bronze": 92.0}
COMPLETION_GOLD_CAP = 8.0
FALLBACK_RESID_VAR = 0.35 ** 2


# ---------------------------------------------------------------- small numeric helpers

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(pct):
    """Log-odds of a percent, the percent first clamped into (0, 100)."""
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg):
    """Percent (0..100) whose log-odds are lg; overflow-safe."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg)) if lg < 40 else 100.0
    e = math.exp(lg) if lg > -40 else 0.0
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matching the kit's JavaScript-style rounding).

    Whole numbers come back as ints so json.dumps writes 100, not 100.0.
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def quantiles(median_logit, sigma):
    qs = []
    for z in Z:
        q = clamp(inv_logit_pct(median_logit + z * sigma), 0.0, 100.0)
        qs.append(round4(q))
    for i in range(1, len(qs)):  # rounding must never break monotonicity
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


# ---------------------------------------------------------------- time handling

def parse_ts(text):
    txt = str(text).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def x_of(dt):
    """The model's time axis: ln(1 + hours since release), guarded near the release instant."""
    return math.log(max(1.0 + hours_since_release(dt), 0.05))


# ---------------------------------------------------------------- weighted least squares

def wls(xs, ys, ws):
    """Weighted least squares of y on x. Returns (slope, x_mean, y_mean, resid_var) or None.

    The line is reported through the weighted centroid (x_mean, y_mean) so a caller can swap in
    a shrunk slope and keep the line pinned to the data's centre of mass. resid_var is the
    weighted mean squared residual, dof-corrected when there are enough points.
    """
    sw = sum(ws)
    if sw <= 0:
        return None
    xm = sum(w * x for x, w in zip(xs, ws)) / sw
    ym = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    resid = sum(w * (y - (ym + slope * (x - xm))) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    n = len(xs)
    if n > 2:
        resid *= n / (n - 2)
    return slope, xm, ym, resid


# ---------------------------------------------------------------- history preparation

def distinct_states(case):
    """Compress the read history to distinct states, nulls carried forward first.

    A read whose players_tracked and (carried-forward) rate vector equal the previous state's
    is the same Exophase snapshot seen again; it collapses into that state, keeping the latest
    timestamp. Each returned state: {"t", "hours", "x", "players", "rates"}.
    """
    n = len(case["trophies"])
    carried = [None] * n
    states = []
    for read in case.get("history") or []:
        raw = read.get("rates") or []
        eff = []
        for i in range(n):
            r = raw[i] if i < len(raw) else None
            if r is not None:
                carried[i] = r
            eff.append(carried[i])
        players = read.get("players_tracked")
        when = parse_ts(read.get("slot") or read.get("at"))
        if states and states[-1]["players"] == players and states[-1]["rates"] == eff:
            states[-1]["t"] = when
            states[-1]["hours"] = hours_since_release(when)
            states[-1]["x"] = x_of(when)
        else:
            states.append({
                "t": when,
                "hours": hours_since_release(when),
                "x": x_of(when),
                "players": players,
                "rates": eff,
            })
    return states


def state_weight(state, case_at):
    """sqrt(players_tracked) x recency with a 36-hour half-life, relative to the case's `at`."""
    age = max(0.0, (case_at - state["t"]).total_seconds() / 3600.0)
    base = math.sqrt(state["players"]) if isinstance(state["players"], (int, float)) and state["players"] > 0 else 1.0
    return base * (0.5 ** (age / 36.0))


def series_for(states, idx, case_at, transform):
    """(x, y, w) points for one trophy across the states where its rate is valid."""
    pts = []
    for st in states:
        rate = st["rates"][idx]
        if rate is None:
            continue
        y = transform(rate, st)
        if y is None:
            continue
        pts.append((st["x"], y, state_weight(st, case_at)))
    return pts


def pooled_slope(series_by_trophy, trophies, tier, min_points=3):
    """One WLS slope over every point of every same-tier trophy with enough states."""
    xs, ys, ws = [], [], []
    for pts, trophy in zip(series_by_trophy, trophies):
        if trophy.get("tier") == tier and len(pts) >= min_points:
            for x, y, w in pts:
                xs.append(x)
                ys.append(y)
                ws.append(w)
    if len(xs) < min_points:
        return None
    fit = wls(xs, ys, ws)
    return fit[0] if fit else None


def view_forecast(pts, pool_slope, x_target, slope_lo, slope_hi):
    """One view's prediction at x_target from (x, y, w) points, plus residual variance or None.

    With >= 3 points and real variation, fit the trophy's own line and shrink its slope toward
    the tier pool by n/(n+4). Otherwise (too few points, or a flat series) anchor on the last
    point and ride the pooled slope; with no pool either, hold flat.
    """
    last_x, last_y = pts[-1][0], pts[-1][1]
    ys = [p[1] for p in pts]
    flat = max(ys) - min(ys) < 1e-9
    fit = wls([p[0] for p in pts], ys, [p[2] for p in pts]) if len(pts) >= 3 else None
    if fit is not None and not flat:
        slope_own, xm, ym, resid = fit
        if pool_slope is not None:
            shrink = len(pts) / (len(pts) + 4.0)
            slope = shrink * slope_own + (1.0 - shrink) * pool_slope
        else:
            slope = slope_own
        slope = clamp(slope, slope_lo, slope_hi)
        return ym + slope * (x_target - xm), resid
    slope = clamp(pool_slope, slope_lo, slope_hi) if pool_slope is not None else 0.0
    return last_y + slope * (x_target - last_x), None


def players_at_target(states, x_target):
    """players_tracked at the target: log-scale WLS over distinct states, slope in [0, 1.2]."""
    pts = [(st["x"], math.log(st["players"])) for st in states
           if isinstance(st["players"], (int, float)) and st["players"] > 0]
    if not pts:
        return None
    if len(pts) == 1:
        return math.exp(pts[0][1])
    fit = wls([p[0] for p in pts], [p[1] for p in pts], [1.0] * len(pts))
    if fit is None:
        return math.exp(pts[-1][1])
    slope = clamp(0.8 * fit[0], 0.0, 1.2)
    ln_p = fit[2] + slope * (x_target - fit[1])
    return math.exp(clamp(ln_p, 0.0, math.log(5e7)))


# ---------------------------------------------------------------- the forecast

def forecast_case(case):
    trophies = case["trophies"]
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)

    if not case.get("history"):
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.05, 0.1, 0.2, 0.5]} for t in trophies]}

    states = distinct_states(case)
    try:
        case_at = parse_ts(case["at"])
    except Exception:
        case_at = states[-1]["t"] if states else RELEASE
    try:
        x_target = x_of(parse_ts(case["target"]))
    except Exception:
        x_target = x_of(case_at) + math.log1p(max(0.0, (case.get("horizon_minutes") or 0) / 60.0))

    # Per-trophy point series for both views.
    trend_series = [series_for(states, i, case_at, lambda r, st: logit_pct(r)) for i in range(len(trophies))]

    def earners_y(rate, st):
        if not isinstance(st["players"], (int, float)) or st["players"] <= 0:
            return None
        return math.log(max(clamp(rate, 0.01, 99.99) / 100.0 * st["players"], 1e-6))

    earn_series = [series_for(states, i, case_at, earners_y) for i in range(len(trophies))]

    tiers = sorted({t.get("tier") for t in trophies if t.get("tier")})
    trend_pool = {tier: pooled_slope(trend_series, trophies, tier) for tier in tiers}
    earn_pool = {tier: pooled_slope(earn_series, trophies, tier) for tier in tiers}
    p_target = players_at_target(states, x_target)

    last_state = states[-1] if states else None
    rows = []
    for i, trophy in enumerate(trophies):
        pts = trend_series[i]
        if not pts:  # no valid rate anywhere in the history
            rows.append({"id": trophy["id"], "q": [0.05] * 5})
            continue

        trend_logit, resid_var = view_forecast(pts, trend_pool.get(trophy.get("tier")), x_target, 0.0, 2.5)

        median_logit = trend_logit
        epts = earn_series[i]
        if epts and p_target:
            ln_e, _ = view_forecast(epts, earn_pool.get(trophy.get("tier")), x_target, 0.0, 2.5)
            rate_b = clamp(100.0 * math.exp(clamp(ln_e, -20.0, 20.0)) / p_target, 0.01, 99.9)
            median_logit = 0.5 * (trend_logit + logit_pct(rate_b))

        # Sanity caps and the story-progression floor, applied in rate space.
        rate_med = inv_logit_pct(median_logit)
        cap = TIER_CAPS.get(trophy.get("tier"))
        if trophy.get("tier") == "gold" and (trophy.get("id") == 14 or "100%" in (trophy.get("name") or "")):
            cap = COMPLETION_GOLD_CAP
        if cap is not None:
            rate_med = min(rate_med, cap)
        last_rate = last_state["rates"][i] if last_state else None
        if (trophy.get("description") or "").startswith("Complete") and last_rate is not None \
                and last_state is not None and last_state["hours"] >= 12.0:
            rate_med = max(rate_med, last_rate * 0.55)
        median_logit = logit_pct(clamp(rate_med, 0.01, 99.9))

        var = resid_var if resid_var is not None else FALLBACK_RESID_VAR
        sigma = clamp(math.sqrt(var + (0.30 * math.sqrt(horizon_days)) ** 2 + 0.15 ** 2), 0.22, 1.6)
        rows.append({"id": trophy["id"], "q": quantiles(median_logit, sigma)})
    return {"trophies": rows}


def persistence_rows(case):
    """Fallback forecast: every trophy holds its last non-null rate, random-walk spread."""
    days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)
    sigma = 0.5 * math.sqrt(days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = None
        for read in reversed(case.get("history") or []):
            raw = read.get("rates") or []
            if i < len(raw) and raw[i] is not None:
                rate = raw[i]
                break
        if rate is None:
            rows.append({"id": trophy["id"], "q": [0.05] * 5})
        else:
            rows.append({"id": trophy["id"], "q": quantiles(logit_pct(rate), sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = persistence_rows(case)
            except Exception:
                forecast = {"trophies": [{"id": t["id"], "q": [0.05] * 5} for t in case.get("trophies", [])]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
