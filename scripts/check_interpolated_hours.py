"""
Evidence for dropping the interpolated hours, and what dropping them changed.

Checks that every flagged hour really is a straight-line interpolation, counts
which assay is interpolated, and compares the October 2026 event set (which kept
those hours) with the current one on the same 80/20 split.

Run:  python scripts/check_interpolated_hours.py > results/interpolated_hours.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_models import CSV, TARGETS, add_features, extract_events, interpolated_rows  # noqa: E402

FE, SI = TARGETS


def main():
    df = pd.read_csv(CSV, sep=",", decimal=",", parse_dates=["date"])
    interp = interpolated_rows(df)
    nu = df[interp].groupby("date")[TARGETS].nunique()
    rows = df[interp].groupby("date").size()
    both = (nu[FE] > 1) & (nu[SI] > 1)
    print(f"Interpolated hours: {len(nu)} ({int(interp.sum()):,} rows)")
    print(f"  both assays {int(both.sum())} | SiO2 only {int(((nu[SI] > 1) & ~both).sum())} | "
          f"Fe only {int(((nu[FE] > 1) & ~both).sum())}")
    print(f"  hours where an assay takes a new value on every row: {int((nu.max(axis=1) == rows).sum())} of {len(nu)}")

    worst = 0.0
    for _, part in df[interp].groupby("date"):
        for t in TARGETS:
            y = part[t].values
            if np.ptp(y) > 0:
                x = np.arange(len(y))
                worst = max(worst, np.abs(y - np.polyval(np.polyfit(x, y, 1), x)).max() / np.ptp(y))
    print(f"  largest departure from a straight line, as a fraction of the hour's range: {worst:.1e}")

    old = add_features(extract_events(exclude_interpolated=False))
    new = add_features(extract_events())
    flagged = set(nu.index)
    in_interp = old.date.isin(flagged).values
    split = int(0.8 * len(old))
    print(f"\nOctober 2026 event set: {len(old)} rows; current: {len(new)} rows")
    print(f"  October rows inside interpolated hours: {int(in_interp.sum())} "
          f"(train {int(in_interp[:split].sum())}, test {int(in_interp[split:].sum())})")
    print(f"  rows only in the current set: {len(set(new.date) - set(old.date))}")
    for t in TARGETS:
        gap = (old[t] - old[f"{t}_lag1"]).abs()
        print(f"  {t}: mean |assay - previous assay| {gap[in_interp].mean():.3f} on those rows, "
              f"{gap[~in_interp].mean():.3f} on real assays")


if __name__ == "__main__":
    sys.exit(main())
