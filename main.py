#!/usr/bin/env python3
"""GTA VI Twitch CCU launch-day forecast (make.fun gta6-twitch-ccu-forecast-01).

Reads every *.in case file, writes <id>.out with 23 hourly rows (hours 2-24),
each {"hour": h, "q": [q05, q25, q50, q75, q95]}.

Model, fitted on the case's own comparables (past big launches' release-day
hourly curves, log scale):
  - anchor: hour-1 print p (the checkpoint's known row).
  - shape: for each hour h, offset o_h = median over comparables of
    ln(comparable viewers in hour h) - ln(comparable hour-1 viewers).
    Offsets are scaled by SHAPE_SHRINK (LOO-tuned on pooled practice
    comparables) to undo the median's inward pull.
  - spread: s_h = pstdev of the comparable offsets at hour h, scaled by
    SIGMA_SCALE (LOO-tuned).
  - quantile at tau: q = p * exp(SHAPE_SHRINK * o_h + z_tau * s_h),
    z_tau the standard normal quantiles, clamped to [1, 1e9].
Deterministic: no randomness, pure arithmetic over the input.
"""
import json
import math
import os
import statistics
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [ -1.6448536269514722, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.6448536269514722 ]

# Tuned by leave-one-launch-out pinball loss over the comparables pooled from
# the four practice cases (37 curves, 4070 items): scale 0.8 / shrink 1.15
# scored 0.804 vs 1.001 for the flat house baseline.
SIGMA_SCALE = 0.8
SHAPE_SHRINK = 1.15
FALLBACK_SIGMA = 0.5      # per-hour spread when too few comparables
FALLBACK_OFFSET = 0.0
MIN_CURVES = 5


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_offsets(comparables):
    """Collect per-hour log offsets relative to each comparable's hour 1."""
    per_hour = {h: [] for h in range(2, 25)}
    for c in comparables:
        try:
            day = parse_iso(c["day_start"])
        except Exception:
            continue
        hourly = {}
        for row in c.get("hourly") or []:
            try:
                t = parse_iso(row["from"])
                v = row.get("viewers")
            except Exception:
                continue
            if (t.year, t.month, t.day) == (day.year, day.month, day.day) \
                    and isinstance(v, (int, float)) and v > 0:
                hourly[int((t - day).total_seconds() // 3600)] = v
        if hourly.get(1):
            base = math.log(hourly[1])
            for h in range(2, 25):
                if h in hourly:
                    per_hour[h].append(math.log(hourly[h]) - base)
    return per_hour


def forecast_case(case):
    known = case.get("known") or []
    p = None
    for row in known:
        if row.get("hour") == 1:
            v = row.get("viewers")
            if isinstance(v, (int, float)) and v > 0:
                p = float(v)
    if p is None:
        # fall back to last valid history point before the checkpoint
        cp = parse_iso(case["checkpoint"])
        hist = [pt for pt in (case.get("history") or [])
                if isinstance(pt.get("viewers"), (int, float)) and pt["viewers"] > 0
                and parse_iso(pt["at"]) <= cp]
        p = float(hist[-1]["viewers"]) if hist else 1.0
    p = max(min(p, 1e9), 1.0)

    per_hour = release_day_offsets(case.get("comparables") or [])
    rows = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        vals = per_hour.get(h) or []
        if len(vals) >= MIN_CURVES:
            off = statistics.median(vals) * SHAPE_SHRINK
            sig = statistics.pstdev(vals) * SIGMA_SCALE
        else:
            off = FALLBACK_OFFSET
            sig = FALLBACK_SIGMA
        med = p * math.exp(off)
        q = [med * math.exp(z * sig) for z in Z]
        q = [min(max(x, 1.0), 1e9) for x in q]
        rows.append({"hour": int(h), "q": [round(x, 2) for x in q]})
    return {"hours": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(fn for fn in os.listdir(INPUT_DIR) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(INPUT_DIR, fn), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        cid = case.get("id") or fn[:-3]
        dst = os.path.join(OUTPUT_DIR, cid + ".out")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
