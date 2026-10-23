"""
Onion gate: "is this box actually an onion?"

Why this exists
---------------
The detector (onion_yolo.pt) was trained only on photographs of onions. It
never saw a face, a hand, a lamp or a web page, so it had no reason to learn
that those are *not* onions. Shown a screenshot, it happily draws boxes on
round, brownish things -- and everything after it (SAM, the defect rule, the
AGMARK grade) assumes every box is an onion and grades it.

The domain guard cannot catch this. It asks "does this photograph look like
the calibration photographs?", using onion-skin colour statistics. A face is
not "an unusual onion"; it is not an onion at all, and on a low-resolution
screenshot the guard can pass it silently.

So this module asks the missing question, per detection, before any grading.

The model
---------
It reuses the project's own EfficientNet-B0 v3 classifier
(extra stuff/unused-weights/new V3 file/best_onion_model_v3.pth, copied here
as onion_gate.pth). It was trained on 50k crops with a "Not Onion" class
(people, hands, chairs, bottles, fruit, bare background, leaves) and scored
99.2% on its held-out test set.

Only its "Not Onion" probability is used. Its Healthy/Unhealthy output is
ignored: grading stays with the calibrated pipeline, which is what the
reported kappa was measured on.

Measured on this project's data (see gatecheck.py to re-run it):
  * 50 tray photographs, 438 non-coin detections:
      p(not onion) >= 0.8 on 2 detections -- both bare floor, not onions
      no real onion reached 0.8; the highest real onion was 0.55
  * the web-page screenshot that produced the bug: all 4 detections
      (3 faces, 1 lamp) scored 0.978 - 0.995 and are rejected
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NOT_ONION = 2          # class order in model_meta_v3.json: Healthy, Unhealthy, Not Onion


class OnionGate:
    def __init__(self, weights: str, device: str = "cpu", img_size: int = 224,
                 pad: float = 0.05):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(1280, 3))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        m.load_state_dict(state)
        self.device = device
        self.model = m.to(device).eval()
        self.pad = pad
        # a straight squash, exactly as the model was trained -- not a centre crop
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def crops(self, rgb: np.ndarray, boxes) -> list:
        im = Image.fromarray(rgb)
        H, W = rgb.shape[:2]
        out = []
        for x1, y1, x2, y2 in boxes:
            px, py = self.pad * (x2 - x1), self.pad * (y2 - y1)
            out.append(im.crop((max(0, x1 - px), max(0, y1 - py),
                                min(W, x2 + px), min(H, y2 + py))))
        return out

    def probs(self, rgb: np.ndarray, boxes) -> np.ndarray:
        """(N, 3) probabilities: Healthy, Unhealthy, Not Onion."""
        import torch
        if len(boxes) == 0:
            return np.zeros((0, 3))
        x = torch.stack([self.tf(c) for c in self.crops(rgb, boxes)]).to(self.device)
        with torch.inference_mode():
            p = torch.softmax(self.model(x).float(), 1)
        return p.cpu().numpy()

    def p_not_onion(self, rgb: np.ndarray, boxes) -> np.ndarray:
        """Probability that each box is NOT an onion, 0-1."""
        return self.probs(rgb, boxes)[:, NOT_ONION] if len(boxes) else np.zeros(0)


def dedupe_boxes(boxes, inside=0.8):
    """Which boxes to drop as duplicates, fragments or merges.

    Returns {index: reason}. A box that contains two or more other,
    non-overlapping boxes (each >= `inside` covered by it) that together fill
    at least half of it is a merge spanning several onions: drop the big one. Otherwise a box lying >= `inside` inside another is a
    duplicate or a fragment: drop the smaller one."""
    b = np.asarray(boxes, float)
    n = len(b)
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9)
    cov = np.zeros((n, n))                 # cov[i, j]: share of i inside j
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ix = max(0.0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
            iy = max(0.0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
            cov[i, j] = ix * iy / area[i]
    drop = {}
    for j in range(n):
        inner = [i for i in range(n) if i != j and cov[i, j] >= inside]
        # Only a MERGE if the inner boxes are different onions: they must not
        # overlap each other much, and together they must fill most of the
        # big box. Two fragments of ONE onion (common in dense piles) fail
        # this test, so the big, correct box is kept and the fragments go.
        distinct = [i for i in inner
                    if all(max(cov[i, k], cov[k, i]) < 0.5 for k in inner if k != i)]
        if len(distinct) >= 2 and area[distinct].sum() >= 0.5 * area[j]:
            drop[j] = "one box spanning several onions"
    for i in range(n):
        if i in drop:
            continue
        for j in range(n):
            if j not in drop and cov[i, j] >= inside and area[i] <= area[j]:
                drop[i] = "duplicate / fragment of another box"
                break
    return drop


class FaceCheck:
    """Finds human faces, so a box drawn on a face is never graded.

    Why a second model: the onion classifier above was never shown close-up
    faces next to onions, and on a market photograph it called a vendor's
    face 12% "not onion" -- i.e. an onion. Faces are the most common non-onion
    thing in a real market photo, so they get a dedicated, purpose-built
    detector: OpenCV's YuNet (face_yunet.onnx, 230 KB, runs in ~20 ms).

    Measured: real faces score 0.88-0.92. On the 50 tray photographs YuNet
    also fires on some onions, but never above 0.86 -- hence the 0.9 cut. A
    face scoring 0.8-0.9 is accepted only if the onion classifier ALSO calls
    that face crop "not onion" (tray false faces never went above 0.26).
    """

    def __init__(self, weights: str, max_side: int = 1280):
        import cv2
        try:        # OpenCV 5 prints a harmless "Targets are not supported" line per call
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        self.cv2 = cv2
        self.weights = weights
        self.max_side = max_side
        self._det = None            # built once, resized per photo
        self._score = None

    def detect(self, rgb: np.ndarray, min_score: float = 0.8):
        """[(x1, y1, x2, y2, score)] in full-image pixels."""
        cv2 = self.cv2
        H, W = rgb.shape[:2]
        s = min(1.0, self.max_side / max(H, W))
        im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if s < 1:
            im = cv2.resize(im, (int(W * s), int(H * s)))
        if self._det is None or self._score != min_score:
            self._det = cv2.FaceDetectorYN.create(self.weights, "",
                                                  (im.shape[1], im.shape[0]), min_score)
            self._score = min_score
        self._det.setInputSize((im.shape[1], im.shape[0]))
        _, f = self._det.detect(im)
        if f is None:
            return []
        return [(x / s, y / s, (x + w) / s, (y + h) / s, float(sc))
                for x, y, w, h, sc in ((r[0], r[1], r[2], r[3], r[-1]) for r in f)]


def face_overlap(box, face):
    """(share of the face inside the box, share of the box that is face)."""
    ix = max(0.0, min(box[2], face[2]) - max(box[0], face[0]))
    iy = max(0.0, min(box[3], face[3]) - max(box[1], face[1]))
    inter = ix * iy
    fa = max((face[2] - face[0]) * (face[3] - face[1]), 1e-9)
    ba = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    return inter / fa, inter / ba
