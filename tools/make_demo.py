"""
Build the pre-computed demo shown when the app runs with ONION_DEMO=1.

Runs the real pipeline once on onion_grader/demo/demo_onions.jpg and saves
everything the dashboard needs, so a server too small for the models
(Render free tier, 512 MB) can still show a genuine result.

    venv\\Scripts\\python tools/make_demo.py
"""

from __future__ import annotations

import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from onion_grader.grader import OnionGrader  # noqa: E402
from onion_grader.demo import DEMO_DIR, IMAGE, CROPS, RESULT, annotated_path  # noqa: E402


def save(arr, path, max_side=420):
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((max_side, max_side))
    im.save(path, "JPEG", quality=85)


def main():
    g = OnionGrader()
    res = g.grade(IMAGE)
    print(res.summary())

    with open(RESULT, "w") as fh:
        fh.write(res.to_json())

    for boxes in (False, True):
        for outlines in (False, True):
            g.render(res, annotated_path(boxes, outlines), IMAGE,
                     boxes=boxes, outlines=outlines)

    os.makedirs(CROPS, exist_ok=True)
    for f in os.listdir(CROPS):
        os.remove(os.path.join(CROPS, f))
    for o in res.onions:
        for k, arr in enumerate(g.crop(res, o, IMAGE)):
            save(arr, os.path.join(CROPS, "%03d_%d.jpg" % (o.index, k)))

    print("demo written to", DEMO_DIR)


if __name__ == "__main__":
    main()
