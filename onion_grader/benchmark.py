"""Score the grader against photos you already know the answer for.

    python -m onion_grader.benchmark --healthy  "path/to/1. Healthy"  --unhealthy "path/to/2. Unhealthy" --limit 50

Folders are searched recursively. Each photo counts as "flagged" when at least
one onion in it comes out UNHEALTHY. A good grader flags ~0% of the healthy
photos and ~100% of the unhealthy ones. Folders named "Single" (one onion
per photo) also give a per-onion accuracy.

Caution: the learned classifier (onion_gate.pth) was TRAINED on the
onion_quality dataset, so scores on those folders are optimistic for it. For
an honest number use photos it has never seen -- your own new photos.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onion_grader.grader import OnionGrader  # noqa: E402


def photos(folder, limit, seed=0):
    out = [
        p
        for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--healthy", required=True)
    ap.add_argument("--unhealthy", required=True)
    ap.add_argument("--limit", type=int, default=50, help="photos per folder (0 = all)")
    ap.add_argument("--out", default="benchmark.csv")
    a = ap.parse_args()

    g = OnionGrader()
    rows = []
    for label, folder in [("healthy", a.healthy), ("unhealthy", a.unhealthy)]:
        files = photos(folder, a.limit)
        for i, f in enumerate(files, 1):
            r = g.grade(f)
            flagged = any(not o.healthy for o in r.onions)
            rows.append(
                {
                    "label": label,
                    "image": f,
                    "onions": r.n_onions,
                    "unhealthy_onions": r.n_unhealthy,
                    "flagged": flagged,
                    "single": "single" in f.lower(),
                }
            )
            print(
                "  [%s %d/%d] %-30s onions %2d  unhealthy %2d"
                % (
                    label,
                    i,
                    len(files),
                    os.path.basename(f)[:30],
                    r.n_onions,
                    r.n_unhealthy,
                )
            )

    def rate(sel):
        sel = [x for x in sel if x["onions"]]
        return (
            (100.0 * sum(x["flagged"] for x in sel) / len(sel), len(sel))
            if sel
            else (0, 0)
        )

    h, nh = rate([x for x in rows if x["label"] == "healthy"])
    u, nu = rate([x for x in rows if x["label"] == "unhealthy"])
    print()
    print("healthy photos flagged unhealthy:   %5.1f%%  of %d   (want ~0%%)" % (h, nh))
    print(
        "unhealthy photos flagged unhealthy: %5.1f%%  of %d   (want ~100%%)" % (u, nu)
    )
    single = [x for x in rows if x["single"] and x["onions"]]
    if single:
        acc = sum(x["flagged"] == (x["label"] == "unhealthy") for x in single) / len(
            single
        )
        print(
            "single-onion photos, accuracy:      %5.1f%%  of %d"
            % (100 * acc, len(single))
        )
    print("no onion found: %d photos" % sum(1 for x in rows if not x["onions"]))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("per-photo results -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
