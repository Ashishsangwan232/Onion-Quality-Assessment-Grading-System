"""
Demo mode: serve a pre-computed result instead of running the models.

Used when ONION_DEMO=1 (e.g. Render free tier, where 512 MB of RAM is not
enough to load YOLO + MobileSAM + the onion gate). The saved result was made
by the real pipeline on demo/demo_onions.jpg -- see tools/make_demo.py.
Nothing here imports torch or ultralytics.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
from PIL import Image

from onion_grader.grader import Onion, Result

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "demo")
IMAGE = os.path.join(DEMO_DIR, "demo_onions.jpg")
RESULT = os.path.join(DEMO_DIR, "result.json")
CROPS = os.path.join(DEMO_DIR, "crops")


def enabled() -> bool:
    return os.environ.get("ONION_DEMO", "").lower() in ("1", "true", "yes")


def annotated_path(boxes: bool, outlines: bool) -> str:
    return os.path.join(DEMO_DIR, "annotated_%d%d.jpg" % (bool(boxes), bool(outlines)))


def load_result() -> Result:
    with open(RESULT) as fh:
        d = json.load(fh)
    d["onions"] = [Onion(**o) for o in d["onions"]]
    d["image"] = IMAGE
    return Result(**d)


class DemoGrader:
    """Stands in for OnionGrader: same render/crop calls, saved outputs."""

    device = "pre-computed"

    def grade(self, path=None, px_per_mm=None) -> Result:
        return load_result()

    def render(self, res, out_path, source=None, boxes=True, outlines=True, header=True):
        shutil.copyfile(annotated_path(boxes, outlines), out_path)

    def crop(self, res, onion, source, pad=0.12):
        return tuple(
            np.asarray(Image.open(os.path.join(CROPS, "%03d_%d.jpg" % (onion.index, k))).convert("RGB"))
            for k in range(3)
        )
