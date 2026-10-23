"""Check the onion gate on YOUR photographs, before you trust it.

    python -m onion_grader.gatecheck --onions  "extra stuff/datasets/Onion_mix"
    python -m onion_grader.gatecheck --onions  DIR  --not-onions  "extra stuff/datasets/data/onion_quality/3. Not Onion"  --limit 200

Two questions, answered with counts rather than claims:

  1. On real onion photographs, does the gate throw away real onions?
     (every rejected box is saved as a picture, so you can look at each one)
  2. On things that are not onions, does anything still get graded?

Run it again whenever you change camera, variety, background, detector or the
onion_gate thresholds in calibration.json. Output goes to gatecheck_out/.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onion_grader.grader import OnionGrader   # noqa: E402

EXT = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def images(folder, limit=None):
    out = sorted({p for e in EXT for p in glob.glob(os.path.join(folder, e))})
    return out[:limit] if limit else out


def save_crop(src, box, path):
    im = Image.open(src).convert("RGB")
    x1, y1, x2, y2 = box
    im.crop((max(0, x1), max(0, y1), x2, y2)).save(path, quality=90)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onions", help="folder of real onion photographs")
    ap.add_argument("--not-onions", help="folder of images with NO onions in them")
    ap.add_argument("--limit", type=int, help="at most this many images per folder")
    ap.add_argument("--out", default="gatecheck_out")
    a = ap.parse_args()
    if not a.onions and not a.not_onions:
        ap.error("give --onions and/or --not-onions")
    os.makedirs(a.out, exist_ok=True)
    g = OnionGrader()
    rows = []
    ok = True

    if a.onions:
        files = images(a.onions, a.limit)
        kept = rej = flagged = refused = 0
        rdir = os.path.join(a.out, "rejected_from_onion_photos")
        os.makedirs(rdir, exist_ok=True)
        for i, f in enumerate(files, 1):
            r = g.grade(f)
            kept += r.n_onions
            rej += len(r.rejected)
            flagged += sum(o.maybe_not_onion for o in r.onions)
            refused += (not r.is_onion_photo)
            for j, x in enumerate(r.rejected):
                save_crop(f, x["box"], os.path.join(
                    rdir, "%s_%d_p%.2f.jpg" % (os.path.splitext(os.path.basename(f))[0],
                                               j, x["p_not_onion"])))
            rows.append({"set": "onions", "image": f, "graded": r.n_onions,
                         "rejected": len(r.rejected), "refused": not r.is_onion_photo,
                         "domain_ok": r.domain_ok})
            print("  [%d/%d] %-40s graded %2d  rejected %d%s"
                  % (i, len(files), os.path.basename(f)[:40], r.n_onions, len(r.rejected),
                     "  REFUSED" if not r.is_onion_photo else ""))
        print()
        print("ONION PHOTOGRAPHS: %d" % len(files))
        print("  graded %d onions, flagged %d as maybe-not-onion" % (kept, flagged))
        print("  rejected %d boxes  -> look at every one in %s" % (rej, rdir))
        print("  refused %d whole photographs  (should be 0)" % refused)
        ok &= refused == 0

    if a.not_onions:
        files = images(a.not_onions, a.limit)
        graded_imgs = 0
        adir = os.path.join(a.out, "graded_but_not_onion")
        os.makedirs(adir, exist_ok=True)
        for f in files:
            r = g.grade(f)
            if r.n_onions:
                graded_imgs += 1
                g.render(r, os.path.join(adir, os.path.basename(f)), f)
            rows.append({"set": "not_onions", "image": f, "graded": r.n_onions,
                         "rejected": len(r.rejected), "refused": not r.is_onion_photo,
                         "domain_ok": r.domain_ok})
        print()
        print("NOT-ONION IMAGES: %d" % len(files))
        print("  still produced a grade: %d (%.1f%%)  -> %s"
              % (graded_imgs, 100 * graded_imgs / max(len(files), 1), adir))
        print("  Every one of those is a hard negative: add it to the detector's")
        print("  training set as a background image (see tools/add_background_images.py).")

    with open(os.path.join(a.out, "gatecheck.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "image", "graded", "rejected",
                                           "refused", "domain_ok"])
        w.writeheader()
        w.writerows(rows)
    print("\nper-image results -> %s" % os.path.join(a.out, "gatecheck.csv"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
