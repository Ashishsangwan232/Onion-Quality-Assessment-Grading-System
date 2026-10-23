"""Check what hardware this install is actually using, and how fast it is.

    python -m onion_grader.selftest                  # synthetic image
    python -m onion_grader.selftest photo.jpg        # your own photograph

It reports where the models really landed, times each stage warm, and -- if a
GPU is present -- runs the SAME photograph on CPU as well, so the speedup is a
measurement rather than a claim. It also checks that the two devices agree on
the grades.

Written because the GPU path in this project was plumbed on a machine that had
no GPU. Everything here is something the author could not verify.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onion_grader.grader import OnionGrader, pick_device   # noqa: E402


def synthetic(path):
    """A tray of onion-ish blobs, so the test needs no data files."""
    from PIL import Image
    rng = np.random.default_rng(0)
    img = np.full((640, 960, 3), 150, np.uint8)
    img += rng.integers(-12, 12, img.shape, dtype=np.int16).astype(np.uint8)
    yy, xx = np.mgrid[0:640, 0:960]
    for cx, cy, r in [(170, 200, 110), (470, 190, 120), (760, 210, 105),
                      (250, 470, 115), (620, 480, 125)]:
        m = ((xx - cx) ** 2 / r ** 2 + (yy - cy) ** 2 / (r * 0.9) ** 2) < 1
        img[m] = np.array([172, 122, 86], np.uint8)
        d = m & (((xx - cx - r * 0.3) ** 2 + (yy - cy) ** 2) < (r * 0.35) ** 2)
        img[d] = np.array([70, 52, 40], np.uint8)          # a dark patch
    Image.fromarray(img).save(path, quality=95)
    return path


def bench(g, path, repeats=3):
    g.grade(path)                                          # warm
    runs = [g.grade(path) for _ in range(repeats)]
    t = {k: statistics.median([r.timings.get(k, 0) for r in runs])
         for k in ("detect", "segment", "measure", "total")}
    return t, runs[-1]


def main() -> int:
    import torch
    path = sys.argv[1] if len(sys.argv) > 1 else synthetic(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selftest.jpg"))

    print("=" * 66)
    print("HARDWARE")
    print("=" * 66)
    print("  torch                %s" % torch.__version__)
    print("  built with CUDA      %s" % torch.version.cuda)
    print("  cuda available       %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print("  gpu %d                %s, %.1f GB" % (i, p.name, p.total_memory / 1e9))
    mps = getattr(torch.backends, "mps", None)
    print("  apple mps            %s" % (bool(mps and mps.is_available())))
    print("  cpu cores            %s" % os.cpu_count())
    print("  pick_device() chose  %s" % pick_device())

    print()
    print("=" * 66)
    print("WHAT ACTUALLY RAN")
    print("=" * 66)
    synthetic_input = len(sys.argv) <= 1
    g = OnionGrader()
    if synthetic_input:
        # The synthetic blobs are not onions, and the onion gate correctly
        # says so. Switch it off here: this script measures SPEED, not grades.
        g.cfg.setdefault("onion_gate", {})["enabled"] = False
        print("  (synthetic image: onion gate disabled for timing -- pass a real photo to include it)")
    # Ultralytics moves YOLO when it performs its first prediction, so checking
    # immediately after loading reports the storage device rather than the
    # actual inference device.
    g.grade(path)
    pr = g._sam.predictor
    print("  requested            %s   FP16=%s" % (g.device, g.half))
    print("  YOLO weights on      %s" % next(g._yolo.model.parameters()).device)
    print("  SAM  weights on      %s (%s)"
          % (next(pr.model.parameters()).device, next(pr.model.parameters()).dtype))
    print("  torch threads        %d" % torch.get_num_threads())
    ok = str(next(g._yolo.model.parameters()).device).startswith(g.device.split(":")[0])
    print("  models on the chosen device?  %s" % ("YES" if ok else "NO - MISMATCH"))

    print()
    print("=" * 66)
    print("SPEED  (median of 3, models warm)")
    print("=" * 66)
    t, res = bench(g, path)
    print("  %-10s detect %5.2f  segment %5.2f  measure %5.2f   TOTAL %5.2f s   %d onions"
          % (g.device, t["detect"], t["segment"], t["measure"], t["total"], res.n_onions))

    if g.device != "cpu":
        gc = OnionGrader(device="cpu")
        gc.cfg = g.cfg
        gc._load()
        tc, rc = bench(gc, path)
        print("  %-10s detect %5.2f  segment %5.2f  measure %5.2f   TOTAL %5.2f s   %d onions"
              % ("cpu", tc["detect"], tc["segment"], tc["measure"], tc["total"], rc.n_onions))
        print()
        print("  SPEEDUP              %.1fx" % (tc["total"] / max(t["total"], 1e-9)))

        print()
        print("=" * 66)
        print("DO THE TWO DEVICES AGREE?")
        print("=" * 66)
        print("  Both paths use FP32; differences indicate a device-specific issue.")
        if res.n_onions != rc.n_onions:
            print("  COUNT DIFFERS: %d on %s vs %d on cpu  <- investigate"
                  % (res.n_onions, g.device, rc.n_onions))
        else:
            d = [abs(a.defect_area_fraction - b.defect_area_fraction)
                 for a, b in zip(res.onions, rc.onions)]
            band = sum(a.band_index != b.band_index
                       for a, b in zip(res.onions, rc.onions))
            print("  defect fraction: max difference %.4f, median %.4f" % (max(d), np.median(d)))
            print("  grades that differ: %d of %d" % (band, res.n_onions))
            if band == 0 and max(d) < 0.02:
                print("  -> the two devices agree. Numbers are comparable.")
            else:
                print("  -> they DISAGREE. Run with --device cpu for anything you")
                print("     intend to report until the CUDA result is investigated.")
    else:
        print()
        print("  No GPU found, so there is nothing to compare against. If you")
        print("  expected one: check `nvidia-smi`, then whether this torch was")
        print("  built with CUDA (above). A CPU-only torch will never use a GPU")
        print("  however many are installed.")

    print()
    print(res.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
