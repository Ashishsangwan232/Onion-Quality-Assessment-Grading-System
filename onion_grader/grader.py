"""
Onion quality grading: photograph in, per-onion grades out.

    from onion_grader.grader import OnionGrader
    g = OnionGrader()
    result = g.grade("tray.jpg")
    g.render(result, "tray_graded.jpg")

or from the command line:

    python -m onion_grader.grader tray.jpg --out graded.jpg --json out.json

The whole pipeline, and where each piece came from:

    YOLO detection                    -> boxes        recall 0.998 on the test set
      + coin rule (small and grey)    -> scale        removes 54/72 spurious, 0/537 onions
      + ONION GATE (gate.py)          -> is it one?   drops faces, hands, lamps, floor
    MobileSAM, box-prompted           -> mask         per-onion boundary
    dull-pixel fraction in the mask   -> defect area  the only module that carries the grade
    isotonic calibration              -> area frac    calibrates the instrument, not the standard
    AGMARK 1/5 and 1/2                -> GRADE        statutory, never fitted
    distance to the nearest boundary  -> CONFIDENCE   0.450 -> 0.725 exact agreement
    Gaussian over healthy pixels      -> ANOMALY      flags a different KIND of damage
    mask geometry + coin              -> SIZE CODE    reported separately, as the standard has it
    photo-level anomaly + response    -> DOMAIN GUARD says so when the photo is outside calibration

Honest performance, cross-validated by physical onion (results/phase12):
  quadratic kappa 0.757 | exact band 0.617 | within one band 0.925
  for scale: two humans agree with each other at 0.701, one with herself at 0.852

Every grade is provisional and visual only. Firmness, dryness, internal
soundness and smell are part of the legal grade and are not in a photograph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(explicit, env, name, roots):
    """Locate a weights file: explicit arg, env var, then the usual folders."""
    for c in [explicit, os.environ.get(env)]:
        if c and os.path.exists(c):
            return c
    for r in roots:
        p = os.path.normpath(os.path.join(r, name))
        if os.path.exists(p):
            return p
    return None
DEFAULT_CAL = os.path.join(HERE, "calibration.json")
GRADE_NAMES = ["Extra Class eligible", "Class I", "Class II", "Below minimum"]
BAND_NAMES = ["none", "up to 1/5", "1/5 to 1/2", "over 1/2"]


@dataclass
class Onion:
    index: int
    box: list                      # x1, y1, x2, y2 in image pixels
    grade: str                     # AGMARK provisional grade
    band: str                      # defect-area band
    band_index: int
    defect_area_fraction: float    # calibrated, 0-1
    raw_defect_fraction: float     # before instrument calibration
    confidence: float              # 0-1, distance to the nearest grade boundary
    needs_review: bool
    anomaly_score: float           # fraction of pixels unlike healthy onion skin
    unusual_kind: bool             # "this is a different kind of damage"
    area_px: float
    eq_diameter_px: float
    circularity: float
    aspect_ratio: float
    anomaly_reliable: bool = True  # False when the whole photograph is out of domain
    size_ratio: float = 1.0        # diameter / median in THIS photo - needs no coin
    size_rank: int = 0             # 1 = largest
    eq_diameter_mm: float | None = None
    size_code: str | None = None
    detector_conf: float = 0.0     # YOLO box confidence
    p_not_onion: float = 0.0       # onion gate: probability this box is NOT an onion
    maybe_not_onion: bool = False  # gate was unsure -> kept, but needs a human
    p_unhealthy: float | None = None   # learned classifier: P(unhealthy | onion)
    verdict_source: str = "area rule"  # what decided HEALTHY / UNHEALTHY
    grade_note: str = ""           # set when the grade was moved to agree with the verdict
    partial: bool = False          # touches the photo edge: part of the onion is unseen
    unhealthy_at: float = 0.5      # classifier threshold (calibration.json -> verdict.unhealthy_at)
    borderline: tuple = (0.3, 0.7) # classifier unsure between these -> verdict BORDERLINE

    @property
    def healthy(self) -> bool:
        """The learned classifier decides when it ran; else the area rule."""
        if self.p_unhealthy is not None:
            return self.verdict_source.startswith("learned") and self.p_unhealthy < self.unhealthy_at
        return self.band_index == 0

    @property
    def is_borderline(self) -> bool:
        """The classifier is unsure. On the 538 tray onions v5 was validated on,
        0 fell in 0.3-0.7; on a new camera / floor, 15% did, and the same onion
        photographed four times wandered between 0.2 and 0.8. Forcing a
        HEALTHY/UNHEALTHY call there would be a coin flip, so it is not made."""
        return (self.p_unhealthy is not None and self.verdict_source.startswith("learned")
                and self.borderline[0] <= self.p_unhealthy < self.borderline[1])

    @property
    def verdict(self) -> str:
        if self.is_borderline:
            return "BORDERLINE"
        return "HEALTHY" if self.healthy else "UNHEALTHY"


@dataclass
class Result:
    image: str
    width: int
    height: int
    n_onions: int
    px_per_mm: float | None
    scale_source: str
    onions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    domain_ok: bool = True         # False: this photograph is unlike the calibration set
    domain_stats: dict = field(default_factory=dict)
    device: str = "cpu"            # what actually ran the two models
    timings: dict = field(default_factory=dict)   # seconds, per stage
    is_onion_photo: bool = True    # False: the gate says this is not a photo of onions
    rejected: list = field(default_factory=list)  # detections the gate threw out
    gate: str = "off"              # "on" when the onion gate ran

    @property
    def n_healthy(self) -> int:
        return sum(1 for o in self.onions if o.healthy and not o.is_borderline)

    @property
    def n_unhealthy(self) -> int:
        return sum(1 for o in self.onions if not o.healthy and not o.is_borderline)

    @property
    def n_borderline(self) -> int:
        return sum(1 for o in self.onions if o.is_borderline)

    @property
    def pct_borderline(self) -> float:
        return 100.0 * self.n_borderline / max(self.n_onions, 1)

    @property
    def pct_healthy(self) -> float:
        return 100.0 * self.n_healthy / max(self.n_onions, 1)

    @property
    def pct_unhealthy(self) -> float:
        return 100.0 * self.n_unhealthy / max(self.n_onions, 1)

    def to_json(self, indent=2):
        d = asdict(self)
        d["onions"] = [asdict(o) if not isinstance(o, dict) else o for o in self.onions]
        return json.dumps(d, indent=indent)

    def summary(self) -> str:
        if not self.is_onion_photo:
            return ("NOT A PHOTOGRAPH OF ONIONS: %d of %d detections rejected by the "
                    "onion gate - nothing was graded" % (len(self.rejected),
                                                        len(self.rejected) + self.n_onions))
        if not self.onions:
            return "no onions found"
        from collections import Counter
        g = Counter(o.grade for o in self.onions)
        lines = ["%d onions  |  HEALTHY %d (%.0f%%)  |  UNHEALTHY %d (%.0f%%)  |  BORDERLINE %d"
                 % (self.n_onions, self.n_healthy, self.pct_healthy,
                    self.n_unhealthy, self.pct_unhealthy, self.n_borderline)]
        for name in GRADE_NAMES:
            if g[name]:
                lines.append("  %-22s %d" % (name, g[name]))
        if self.rejected:
            lines.append("  %d detection(s) ignored - the onion gate says they are "
                         "not onions" % len(self.rejected))
        rev = sum(1 for o in self.onions if o.needs_review)
        unu = sum(1 for o in self.onions if o.unusual_kind)
        if rev:
            lines.append("  %d near a grade boundary - worth a human look" % rev)
        if unu:
            lines.append("  %d flagged as an unusual KIND of damage" % unu)
        if not self.domain_ok:
            s = self.domain_stats
            lines.append("  OUT OF THE CALIBRATED DOMAIN (%s): median anomaly %.2f, "
                         "highest raw defect %.1f%%"
                         % ("everything looks unfamiliar" if "A" in s.get("direction", "")
                            else "nothing registered as damage",
                            s.get("median_anomaly", 0),
                            100 * s.get("max_raw_defect_fraction", 0)))
        if self.px_per_mm:
            lines.append("  scale %.2f px/mm (%s)" % (self.px_per_mm, self.scale_source))
        if self.timings:
            lines.append("  %.2fs on %s  (detect %.2f, segment %.2f, measure %.2f)"
                         % (self.timings.get("total", 0), self.device,
                            self.timings.get("detect", 0), self.timings.get("segment", 0),
                            self.timings.get("measure", 0)))
        for w in self.warnings:
            lines.append("  ! " + w)
        return "\n".join(lines)


def pick_device(explicit: str | None = None) -> str:
    """cuda if there is one, then Apple mps, else cpu. Override with ONION_DEVICE."""
    import torch
    want = explicit or os.environ.get("ONION_DEVICE")
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OnionGrader:
    def __init__(self, calibration: str = DEFAULT_CAL,
                 detector: str | None = None, sam: str | None = None,
                 device: str | None = None, half: bool | None = None,
                 threads: int | None = None):
        self.cfg = json.load(open(calibration))
        self.device = pick_device(device)
        # MobileSAM produces empty masks in FP16 with the installed Ultralytics
        # runtime on CUDA. Keep inference FP32 until that combination is
        # validated; `half` remains accepted only for API compatibility.
        self.half = False
        self.threads = threads
        self.timings: dict = {}
        self.detector_path = _find(detector, "ONION_YOLO", "onion_yolo.pt",
                                   [HERE, os.path.join(HERE, ".."),
                                    os.path.join(HERE, "..", "new V3 file"),
                                    os.path.join(HERE, "..", "weights")])
        # mobile_sam.pt is fetched by ultralytics on first use if it is absent
        self.sam_path = _find(sam, "ONION_SAM", "mobile_sam.pt",
                              [HERE, os.path.join(HERE, ".."),
                               os.path.join(HERE, "..", "weights")]) or "mobile_sam.pt"
        # the onion gate: EfficientNet-B0 with a "Not Onion" class. Optional --
        # without it the grader runs as before, and says so in every result.
        # which classifier weights: calibration.json -> onion_gate.weights
        # (onion_gate.pth = the original v3; training/ writes onion_gate_v4.pth)
        gate_file = self.cfg.get("onion_gate", {}).get("weights", "onion_gate.pth")
        self.gate_path = _find(None, "ONION_GATE", gate_file,
                               [HERE, os.path.join(HERE, ".."),
                                os.path.join(HERE, "..", "weights")])
        self.face_path = _find(None, "ONION_FACE", "face_yunet.onnx",
                               [HERE, os.path.join(HERE, ".."),
                                os.path.join(HERE, "..", "weights")])
        self._yolo = None
        self._sam = None
        self._gate = None
        self._faces = None

    # ---------------------------------------------------------------- models
    def _load(self):
        if self._yolo is not None:
            return
        import torch
        from ultralytics import YOLO, SAM
        if not os.path.exists(self.detector_path):
            raise SystemExit("detector weights not found: %s\n"
                             "  set ONION_YOLO or pass detector=..." % self.detector_path)
        # SAM("mobile_sam.pt") downloads it the first time if missing
        self._yolo = YOLO(self.detector_path)
        self._sam = SAM(self.sam_path)
        if self.gate_path and self.cfg.get("onion_gate", {}).get("enabled", True):
            from onion_grader.gate import OnionGate
            self._gate = OnionGate(self.gate_path, device=self.device)
        if self.face_path and self.cfg.get("onion_gate", {}).get("faces", True):
            from onion_grader.gate import FaceCheck
            self._faces = FaceCheck(self.face_path)

        # Build SAM's predictor ONCE. This costs a full inference, and the old
        # code paid it on every photograph: the image was encoded twice, which
        # was 45% of SAM's time and ~28% of the whole pipeline.
        import numpy as _np
        from PIL import Image as _Image
        warm = _np.zeros((64, 64, 3), _np.uint8)
        self._sam(warm, bboxes=[[8.0, 8.0, 56.0, 56.0]], verbose=False,
                  device=self.device)
        self._warm = True

        # AFTER the models load: ultralytics resets the thread count during
        # setup, so setting it earlier does nothing.
        if self.device == "cpu":
            n = self.threads or (os.cpu_count() or 1)
            torch.set_num_threads(max(1, int(n)))

    # ---------------------------------------------------------------- steps
    def _detect(self, path, rgb):
        d = self.cfg["detector"]
        r = self._yolo(path, conf=d["conf"], imgsz=d["imgsz"], verbose=False,
                       device=self.device)[0]
        if r.boxes is None or not len(r.boxes):
            return np.zeros((0, 4)), None, np.zeros(0)
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        med = float(np.median(areas))
        f = rgb.astype(np.float32) / 255
        v = f.max(2)
        sat = (v - f.min(2)) / np.maximum(v, 1e-6)

        rule = d["coin_rule"]
        onions, coins, oconf = [], [], []
        for b, a, cf in zip(boxes, areas, confs):
            x1, y1, x2, y2 = [int(max(t, 0)) for t in b]
            if y2 <= y1 or x2 <= x1:
                continue
            s = float(sat[y1:y2, x1:x2].mean())
            if a < rule["max_area_ratio"] * med and s < rule["max_saturation"]:
                coins.append((b, a))
            else:
                onions.append(b)
                oconf.append(float(cf))
        coin = max(coins, key=lambda c: c[1])[0] if coins else None
        return (np.array(onions) if onions else np.zeros((0, 4))), coin, np.array(oconf)

    def _gate_boxes(self, rgb, boxes, confs, res):
        """Throw out boxes that are not onions, BEFORE anything is graded.

        Returns (boxes, confs, p_not) for the boxes that survive. Sets
        res.is_onion_photo = False when most detections are not onions, in
        which case nothing should be graded at all."""
        if self._gate is None:
            res.warnings.append(
                "Onion gate is OFF (%s): every detection is "
                "assumed to be an onion, so faces, hands or other round objects "
                "would be graded too." % ("disabled in calibration.json"
                                          if self.gate_path else "onion_gate.pth not found"))
            return boxes, confs, np.zeros(len(boxes)), None
        g = self.cfg.get("onion_gate", {})
        reject_at = g.get("reject_at", 0.8)
        res.gate = "on"
        P = self._gate.probs(rgb, boxes)
        p = P[:, 2].copy()
        # Look again with context: the box widened by context_pad on every side.
        # A tight box on a toenail or a fingertip is just a small pink rounded
        # shape, which the classifier cannot tell from an onion (toe: 0.33-0.77
        # not-onion); with the foot in view it is 0.83-0.94. On 575 real onions
        # (trays, Bulb folder, field, market, floor) the widened view never went
        # above 0.65. Rejection uses the higher of the two views; the review
        # flag keeps using the tight view.
        ctx = float(g.get("context_pad", 0.5))
        p_rej = p
        if ctx > 0 and len(boxes):
            H, W = rgb.shape[:2]
            bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            wide = np.stack([np.maximum(0, boxes[:, 0] - ctx * bw), np.maximum(0, boxes[:, 1] - ctx * bh),
                             np.minimum(W, boxes[:, 2] + ctx * bw), np.minimum(H, boxes[:, 3] + ctx * bh)], 1)
            p_rej = np.maximum(p, self._gate.probs(rgb, wide)[:, 2])
        keep = p_rej < reject_at
        p = np.where(keep, p, p_rej)          # rejected boxes report the score that rejected them
        why = ["not an onion"] * len(boxes)

        # faces: a dedicated detector, because the classifier misses them
        if self._faces is not None:
            from onion_grader.gate import face_overlap
            faces = self._faces.detect(rgb, g.get("face_min_score", 0.8))
            sure = g.get("face_sure_score", 0.9)
            unsure = [f for f in faces if f[4] < sure]
            if unsure:          # 0.8-0.9: only if the classifier agrees it's no onion
                pf = self._gate.p_not_onion(rgb, [f[:4] for f in unsure])
                unsure = [f for f, q in zip(unsure, pf) if q >= 0.5]
            faces = [f for f in faces if f[4] >= sure] + unsure
            res.domain_stats["faces_found"] = len(faces)
            for i, b in enumerate(boxes):
                for f in faces:
                    in_box, of_box = face_overlap(b, f)
                    if in_box >= g.get("face_in_box", 0.5) and of_box >= g.get("face_of_box", 0.2):
                        keep[i] = False
                        why[i] = "a human face"
                        p[i] = max(p[i], f[4])
                        break

        # duplicates, fragments, and one box drawn over two onions
        from onion_grader.gate import dedupe_boxes
        idx = np.nonzero(keep)[0]
        dup = {}
        if len(idx) > 1:
            dup = dedupe_boxes(boxes[idx], g.get("dedupe_inside", 0.8))
            for k, reason in dup.items():
                keep[idx[k]] = False
                why[idx[k]] = reason

        for b, cf, pn, w in zip(boxes[~keep], confs[~keep], p[~keep],
                                np.array(why, dtype=object)[~keep]):
            res.rejected.append({"box": [round(float(t), 1) for t in b],
                                 "detector_conf": round(float(cf), 3),
                                 "p_not_onion": round(float(pn), 3),
                                 "why": w})
        n = len(boxes) - len(dup)          # duplicates are not evidence of "not onions"
        bad = (~keep).sum() - len(dup)
        if n and (keep.sum() == 0 or bad / n > g.get("photo_reject_fraction", 0.5)):
            res.is_onion_photo = False
            res.warnings.append(
                "NOT A PHOTOGRAPH OF ONIONS: the onion gate rejected %d of %d "
                "detections (faces, hands, objects, background). Nothing was graded. "
                "Recalibrating would not help - the input is the problem, not the "
                "constants. Photograph the onions themselves on a plain surface."
                % (bad, n))
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0), None
        pu = P[keep, 1] / np.maximum(P[keep, 0] + P[keep, 1], 1e-9)
        return boxes[keep], confs[keep], p[keep], pu

    def _masks(self, path, boxes, shape):
        import torch
        from ultralytics.utils import ops
        H, W = shape
        pred = self._sam.predictor          # built once in _load(), not per image
        with torch.inference_mode():
            pred.setup_source(path)
            for batch in pred.dataset:
                # prompt_inference reads pred.src_shape, which is pred.batch[1][0].
                # Setting this is what the old per-image warm-up call was really
                # doing; without it the box prompts are scaled by the ratio of
                # whatever image the predictor last saw.
                pred.batch = batch
                im = pred.preprocess(batch[1])
                pred.features = pred.get_im_features(im)
                break
            low, _ = pred.prompt_inference(im, bboxes=boxes, multimask_output=False)
        out = []
        for i in range(low.shape[0]):
            with torch.inference_mode():
                m = ops.scale_masks(low[i:i + 1][None].float(), (H, W), padding=False)[0]
                m = (m > 0.0).cpu().numpy()[0]
            out.append(m)
        return out

    @staticmethod
    def _channels(rgb):
        """saturation, value and local texture for the WHOLE frame, once.

        These used to be recomputed inside _measure for every onion, which on a
        five-onion photograph meant five full-frame passes to read one mask."""
        from scipy import ndimage
        f = rgb.astype(np.float32) / 255
        v = f.max(2)
        s = (v - f.min(2)) / np.maximum(v, 1e-6)
        g = f.mean(2)
        tex = np.sqrt(np.maximum(ndimage.uniform_filter(g ** 2, 5)
                                 - ndimage.uniform_filter(g, 5) ** 2, 0))
        return v, s, tex

    def _measure(self, chan, mask, box):
        from scipy import ndimage
        c = self.cfg["defect"]
        v, s, tex = chan
        # keep only the blob containing the box centre
        lab, n = ndimage.label(mask)
        if n > 1:
            cy = int((box[1] + box[3]) / 2)
            cx = int((box[0] + box[2]) / 2)
            want = lab[min(max(cy, 0), lab.shape[0] - 1), min(max(cx, 0), lab.shape[1] - 1)]
            if want == 0:
                sizes = ndimage.sum(mask, lab, range(1, n + 1))
                want = int(np.argmax(sizes)) + 1
            mask = lab == want
        area = float(mask.sum())
        if area < 50:
            return None

        vm, sm, tm = v[mask], s[mask], tex[mask]
        raw = float(((sm < c["s_thr"]) & (vm < c["v_thr"])).mean())
        frac = float(np.interp(raw, c["isotonic_x"], c["isotonic_y"]))
        cuts = [c["detection_limit"]] + c["agmark_cuts"]
        band = int(np.digitize(frac, cuts))
        # confidence: distance to the nearest boundary, scaled to the widest gap
        gaps = [abs(frac - t) for t in cuts]
        conf = float(min(1.0, min(gaps) / 0.10))

        a = self.cfg["anomaly"]
        z = np.stack([sm, vm, tm], 1) - np.array(a["mu"])
        dist = np.einsum("ij,jk,ik->i", z, np.array(a["inv_cov"]), z)
        anom = float((dist > a["threshold"]).mean())

        er = ndimage.binary_erosion(mask, np.ones((3, 3), bool), border_value=0)
        per = max(float((mask & ~er).sum()), 1.0)
        ys, xs = np.nonzero(mask)
        y0, x0 = ys.mean(), xs.mean()
        cyy = ((ys - y0) ** 2).mean()
        cxx = ((xs - x0) ** 2).mean()
        cxy = ((ys - y0) * (xs - x0)).mean()
        tr = cyy + cxx
        disc = max(tr * tr / 4 - (cyy * cxx - cxy * cxy), 0.0) ** 0.5
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-9)

        return {"mask": mask, "band": band, "frac": frac, "raw": raw, "conf": conf, "anom": anom,
                "area": area, "eq_d": 2 * (area / np.pi) ** 0.5,
                "circ": 4 * np.pi * area / per ** 2,
                "ar": (4 * l1 ** 0.5) / max(4 * l2 ** 0.5, 1e-9)}

    # ---------------------------------------------------------------- api
    def grade(self, path: str, px_per_mm: float | None = None) -> Result:
        """px_per_mm: supply a known scale when no coin is in frame.

        Absolute size cannot be recovered from a single photograph without a
        reference of known size -- a small onion near the lens and a large one
        far away are the same pixels. Relative size is always reported and
        needs no reference at all."""
        import time
        t0 = time.perf_counter()
        self._load()
        rgb = np.asarray(Image.open(path).convert("RGB"))
        H, W = rgb.shape[:2]
        res = Result(image=os.path.basename(path), width=W, height=H,
                     n_onions=0, px_per_mm=None, scale_source="none",
                     device=self.device)

        t1 = time.perf_counter()
        boxes, coin, confs = self._detect(path, rgb)
        if len(boxes):
            boxes, confs, p_not, p_unh = self._gate_boxes(rgb, boxes, confs, res)
        res.timings["detect"] = round(time.perf_counter() - t1, 3)
        if not len(boxes):
            if res.is_onion_photo:
                res.warnings.append("no onions detected")
            res.timings["total"] = round(time.perf_counter() - t0, 3)
            return res
        review_at = self.cfg.get("onion_gate", {}).get("review_at", 0.5)

        if px_per_mm:
            res.px_per_mm = float(px_per_mm)
            res.scale_source = "supplied by the operator"
        elif coin is not None:
            dd = float(max(coin[2] - coin[0], coin[3] - coin[1]))
            res.px_per_mm = dd / self.cfg["scale"]["coin_mm"]
            res.scale_source = "coin detected in frame"
        else:
            res.warnings.append(
                "No reference object in frame, so absolute size is not recoverable - "
                "a small onion near the lens looks identical to a large one far away. "
                "Relative sizes below are exact; for millimetres, include a coin or "
                "pass px_per_mm.")

        # ultralytics rescales the bbox array IN PLACE to SAM's 1024px input, so
        # hand it a copy or every reported box comes back multiplied by 1024/W
        t2 = time.perf_counter()
        masks = self._masks(path, boxes.copy(), (H, W))
        res.timings["segment"] = round(time.perf_counter() - t2, 3)
        t3 = time.perf_counter()
        # Don't let the background decide health. The classifier learned that
        # soil/field backgrounds go with "unhealthy": clean glossy onions lying on
        # soil scored 0.85-0.96 unhealthy, and 0.09-0.12 with the soil greyed out.
        # But greying out ALONE is worse on clean trays (sound onions passed:
        # 22/45 vs 41/45 on the hand-graded set) -- the model never saw grey
        # backgrounds. So an onion is called unhealthy only if it looks unhealthy
        # BOTH as photographed AND with its background removed: min() of the two.
        # On the 112 usable hand-graded onions this scores exactly as the box
        # crop alone (93%: 63/67 rotten, 41/45 sound) and fixes the soil photos.
        vcfg = self.cfg.get("verdict", {})
        if p_unh is not None and vcfg.get("classify_on", "masked") == "masked":
            fill = np.array(vcfg.get("mask_fill", [200, 200, 200]), np.uint8)
            crops = []
            for b, m in zip(boxes, masks):
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                x1, y1 = max(x1, 0), max(y1, 0)
                sub = rgb[y1:y2, x1:x2].copy()
                sub[~m[y1:y2, x1:x2]] = fill
                crops.append(Image.fromarray(sub))
            import torch
            x = torch.stack([self._gate.tf(c) for c in crops]).to(self._gate.device)
            with torch.inference_mode():
                P = torch.softmax(self._gate.model(x).float(), 1).cpu().numpy()
            p_unh = np.minimum(p_unh, P[:, 1] / np.maximum(P[:, 0] + P[:, 1], 1e-9))
        chan = self._channels(rgb)          # once per frame, not once per onion
        alpha = self.cfg["scale"]["resolution_exponent"]
        for i, (b, m) in enumerate(zip(boxes, masks)):
            mm = self._measure(chan, m, b)
            unsure = bool(p_not[i] >= review_at)
            if mm is None:
                continue
            band, note, src, pu = mm["band"], "", "area rule", None
            vcfg = self.cfg.get("verdict", {})
            bl = tuple(vcfg.get("borderline", [0.3, 0.7]))
            if p_unh is not None and vcfg.get("source", "learned") == "learned":
                # The learned classifier decides HEALTHY / UNHEALTHY; the grade
                # is kept consistent with it. The dark-area % is still reported.
                pu = float(p_unh[i])
                src = "learned classifier"
                if pu >= vcfg.get("unhealthy_at", 0.5) and band == 0:
                    band, note = 1, ("defect seen by the classifier but not by the "
                                     "dark-area rule: the % underestimates it")
                elif pu < vcfg.get("healthy_sure_below", 0.2) and band > 0:
                    band, note = 0, ("classifier is confident this onion is sound: "
                                     "the dark area is shadow/skin, not damage")
                elif pu < vcfg.get("unhealthy_at", 0.5) and band > 0:
                    # verdict HEALTHY but the dark-area rule measured a lot of dark:
                    # usually shadow between onions, soil or a dark outer skin.
                    # The classifier is the better judge (98% vs 83%), so the grade
                    # is capped at Class I -- never "HEALTHY + Below minimum" -- and
                    # the onion is sent for a human check.
                    if band > 1:
                        band = 1
                    note = ("classifier and dark-area rule disagree - check by eye "
                            "(classifier %.0f%% unhealthy; the dark area is probably "
                            "shadow, soil or skin)" % (100 * pu))
            edge = self.cfg.get("verdict", {}).get("edge_px", 3)
            partial = bool(b[0] <= edge or b[1] <= edge or b[2] >= W - edge or b[3] >= H - edge)
            o = Onion(index=len(res.onions), box=[round(float(t), 1) for t in b],
                      grade=GRADE_NAMES[band], band=BAND_NAMES[band],
                      band_index=band,
                      defect_area_fraction=round(mm["frac"], 4),
                      raw_defect_fraction=round(mm["raw"], 4),
                      confidence=round(mm["conf"], 3),
                      needs_review=(mm["conf"] < 0.5 or unsure or partial
                                    or note.startswith("classifier and")
                                    or (pu is not None and bl[0] <= pu < bl[1])),
                      anomaly_score=round(mm["anom"], 4),
                      unusual_kind=mm["anom"] > 0.35,
                      area_px=round(mm["area"], 1),
                      eq_diameter_px=round(mm["eq_d"], 1),
                      circularity=round(mm["circ"], 4),
                      aspect_ratio=round(mm["ar"], 3),
                      detector_conf=round(float(confs[i]), 3),
                      p_not_onion=round(float(p_not[i]), 3),
                      maybe_not_onion=unsure,
                      p_unhealthy=None if pu is None else round(pu, 3),
                      verdict_source=src, grade_note=note, partial=partial,
                      unhealthy_at=float(vcfg.get("unhealthy_at", 0.5)),
                      borderline=bl)
            if res.px_per_mm:
                # correct for SAM's sub-quadratic area/resolution response (Phase 11)
                ref = 3.33
                corr = (res.px_per_mm / ref) ** (1 - alpha / 2)
                d_mm = mm["eq_d"] / res.px_per_mm * corr
                o.eq_diameter_mm = round(d_mm, 1)
                maj = d_mm * max(mm["ar"], 1.0) ** 0.5
                o.size_code = next(n for n, lo, hi in self.cfg["size_codes"] if lo <= maj < hi)
            o._mask = m  # kept for rendering, not serialised
            res.onions.append(o)
        res.timings["measure"] = round(time.perf_counter() - t3, 3)
        self._domain_guard(res)
        # Two independent signals agreeing: the gate was unsure about a box AND
        # the colour statistics say the photo is unlike any onion photograph.
        # In-domain the guard fires on 0 of 53 photographs, so this costs
        # nothing on real trays; out of domain it removes the doubtful boxes.
        if not res.domain_ok and "A" in res.domain_stats.get("direction", ""):
            doubtful = [o for o in res.onions if o.maybe_not_onion]
            for o in doubtful:
                res.rejected.append({"box": o.box, "detector_conf": o.detector_conf,
                                     "p_not_onion": o.p_not_onion,
                                     "why": "gate unsure + photo out of domain"})
            res.onions = [o for o in res.onions if not o.maybe_not_onion]
            for i, o in enumerate(res.onions):
                o.index = i
            if doubtful and not res.onions:
                res.is_onion_photo = False
                res.warnings.insert(0,
                    "NOT A PHOTOGRAPH OF ONIONS: the onion gate was unsure about "
                    "every detection and the photo is unlike any onion photograph "
                    "the grader was calibrated on. Nothing was graded.")
        # relative size needs NO reference object: ratios are scale-free
        if res.onions:
            med = float(np.median([o.eq_diameter_px for o in res.onions]))
            order = sorted(res.onions, key=lambda o: -o.eq_diameter_px)
            for rank, o in enumerate(order, 1):
                o.size_rank = rank
            for o in res.onions:
                o.size_ratio = round(o.eq_diameter_px / max(med, 1e-9), 3)
        res.n_onions = len(res.onions)
        res.timings["total"] = round(time.perf_counter() - t0, 3)
        return res

    # ------------------------------------------------------------ domain guard
    def _domain_guard(self, res: Result) -> None:
        """Is this photograph like the ones the constants were fitted on?

        The anomaly model was fitted on healthy onions from ONE session, and
        Phase 13 showed what that costs: a narrow healthy distribution makes
        everything *unfamiliar* look anomalous, healthy onions included. So a
        photograph where most onions score anomalous is not a photograph full of
        strange damage - it is a photograph from outside the calibrated domain,
        and the anomaly score is measuring the wrong thing.

        Across the 53 calibration photographs the per-photo median anomaly never
        exceeded 0.287 and the flagged fraction never exceeded 0.333. The cuts
        sit above both, so this fires on 0 of 53 in-domain photographs.
        """
        g = self.cfg.get("domain_guard")
        if not g or not res.onions:
            return
        a = np.array([o.anomaly_score for o in res.onions])
        raw = np.array([o.raw_defect_fraction for o in res.onions])
        med = float(np.median(a))
        flagged = float(np.mean([o.unusual_kind for o in res.onions]))
        max_raw = float(raw.max())

        # A: everything looks strange. B: nothing registers at all. They are
        # opposite directions and one statistic does not see both - under a
        # brightening shift every raw fraction goes to 0.0 while the anomaly
        # score goes DOWN, so A alone would pass the photograph silently.
        fail_a = med > g["median_anom_max"] or flagged > g["flagged_fraction_max"]
        fail_b = (len(res.onions) >= g["min_onions_for_response_check"]
                  and max_raw < g["max_raw_floor"])

        res.domain_stats = {
            "median_anomaly": round(med, 4),
            "flagged_fraction": round(flagged, 4),
            "max_raw_defect_fraction": round(max_raw, 4),
            "median_anomaly_max_in_domain": g["median_anom_max"],
            "flagged_fraction_max_in_domain": g["flagged_fraction_max"],
            "max_raw_floor_in_domain": g["max_raw_floor"],
            "direction": ("A" if fail_a else "") + ("B" if fail_b else "")}
        res.domain_ok = not (fail_a or fail_b)
        if res.domain_ok:
            return

        if fail_a:
            why = ("its onions look unlike anything in the calibration set "
                   "(median anomaly %.2f, against %.2f for every one of the 53 "
                   "calibration photographs)" % (med, g["median_anom_max"]))
            # Out of domain the score detects unfamiliarity, not a kind of
            # damage. Withdraw the claim rather than print it in a colour.
            for o in res.onions:
                o.unusual_kind = False
                o.anomaly_reliable = False
        else:
            why = ("the damage rule did not register on a single onion "
                   "(highest raw defect %.1f%%, and across the 53 calibration "
                   "photographs the highest never fell below 9.9%%). Either "
                   "every onion here really is spotless, or the brightness and "
                   "saturation thresholds have slid off this photograph"
                   % (100 * max_raw))
        res.warnings.append(
            "OUT OF THE CALIBRATED DOMAIN: %s. A different variety, camera, "
            "background or light will do this. Counts, outlines and relative "
            "sizes are unaffected. The defect percentages, and therefore the "
            "GRADES, may be systematically harsh or lenient here. First check "
            "the photo: onions on a plain surface, even light, no filters or "
            "screenshots. Only if this IS your normal set-up (new camera, "
            "variety or light), re-grade ~100 onions from it by hand and rerun "
            "audit/phase16_export_calibration.py." % why)

    # ---------------------------------------------------------------- output
    PALETTE = [(60, 200, 110), (240, 200, 70), (245, 140, 50), (235, 65, 65)]

    def crop(self, res: Result, onion, source: str, pad: float = 0.12):
        """Return (crop_rgb, mask_overlay_rgb, defect_overlay_rgb) for one onion."""
        from scipy import ndimage
        rgb = np.asarray(Image.open(source).convert("RGB"))
        H, W = rgb.shape[:2]
        x1, y1, x2, y2 = onion.box
        px, py = pad * (x2 - x1), pad * (y2 - y1)
        a, b = max(int(x1 - px), 0), max(int(y1 - py), 0)
        c, d = min(int(x2 + px), W), min(int(y2 + py), H)
        sub = rgb[b:d, a:c]
        m = getattr(onion, "_mask", None)
        if m is None:
            return sub, sub, sub
        ms = m[b:d, a:c]
        col = self.PALETTE[onion.band_index]

        outline = sub.copy()
        edge = ms & ~ndimage.binary_erosion(ms, np.ones((3, 3), bool), border_value=0)
        outline[ndimage.binary_dilation(edge, np.ones((3, 3), bool))] = col

        # where the defect actually is, using the same rule that made the grade
        cfg = self.cfg["defect"]
        f = sub.astype(np.float32) / 255
        v = f.max(2)
        sat = (v - f.min(2)) / np.maximum(v, 1e-6)
        dmap = (sat < cfg["s_thr"]) & (v < cfg["v_thr"]) & ms
        heat = sub.copy()
        heat[dmap] = (0.4 * heat[dmap] + 0.6 * np.array([255, 60, 50])).astype(np.uint8)
        heat[ndimage.binary_dilation(edge, np.ones((3, 3), bool))] = col
        return sub, outline, heat

    def render(self, res: Result, out_path: str, source: str | None = None,
               boxes: bool = True, outlines: bool = True, header: bool = True):
        """Annotated image: box + outline + per-onion label, and a header bar."""
        from PIL import ImageDraw, ImageFont
        from scipy import ndimage
        src = source or res.image
        im = Image.open(src).convert("RGB")
        arr = np.asarray(im).copy()
        H, W = arr.shape[:2]

        if outlines:
            for o in res.onions:
                m = getattr(o, "_mask", None)
                if m is None:
                    continue
                edge = m & ~ndimage.binary_erosion(m, np.ones((3, 3), bool), border_value=0)
                arr[ndimage.binary_dilation(edge, np.ones((3, 3), bool))] = self.PALETTE[o.band_index]

        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        scale = max(1, int(round(W / 900)))
        font = _font(True, 11 * scale)
        small = _font(False, 10 * scale)
        big = _font(True, 17 * scale)
        mid = _font(False, 11 * scale)

        # the header goes ABOVE the photograph, on its own strip: drawn over the
        # top of the image it used to hide whatever was there (e.g. a face)
        hb = ((74 if res.domain_ok else 100) * scale) if (header and res.onions) else 0
        if header and not res.is_onion_photo:
            hb = 40 * scale
        bar = 0

        # what the gate threw out: thin grey box, so the operator can see it
        for r in res.rejected:
            x1, y1, x2, y2 = [int(v) for v in r["box"]]
            d.rectangle([x1, y1, x2, y2], outline=(150, 150, 150), width=max(1, scale))
            tag = ("face - not an onion" if r.get("why") == "a human face"
                   else "not an onion (%.0f%%)" % (100 * r["p_not_onion"]))
            ty = max(y2 + 2 * scale, bar)
            d.rectangle([x1, ty, x1 + d.textlength(tag, font=small) + 8 * scale,
                         ty + 14 * scale], fill=(90, 90, 90))
            d.text((x1 + 4 * scale, ty + 1 * scale), tag, fill=(235, 235, 235), font=small)

        for o in res.onions:
            x1, y1, x2, y2 = [int(v) for v in o.box]
            col = self.PALETTE[o.band_index]
            if boxes:
                d.rectangle([x1, y1, x2, y2], outline=col, width=2 * scale)
            verdict = "CHECK" if (o.is_borderline or o.grade_note.startswith("classifier and")) else o.verdict
            tag = "#%d %s %s" % (o.index, verdict, o.grade.replace(" eligible", "")) \
                + (" (partial)" if o.partial else "")
            tw = d.textlength(tag, font=font)
            th = 15 * scale
            # keep the label out from under the header bar
            ty = y1 - th if y1 - th >= bar else (y1 + 2 * scale if y1 >= bar else y2)
            d.rectangle([x1, ty, x1 + tw + 8 * scale, ty + th], fill=col)
            d.text((x1 + 4 * scale, ty + 2 * scale), tag, fill=(12, 14, 18), font=font)

        if hb:
            canvas = Image.new("RGB", (W, H + hb), (14, 16, 20))
            canvas.paste(im, (0, hb))
            im = canvas
            d = ImageDraw.Draw(im)
            bar = hb
        if header and not res.is_onion_photo:
            d.rectangle([0, 0, W, bar], fill=(120, 30, 30))
            d.text((12 * scale, 11 * scale),
                   "NOT A PHOTOGRAPH OF ONIONS - nothing graded",
                   fill=(255, 235, 235), font=big)
        if header and res.onions:
            d.rectangle([0, 0, W, bar], fill=(14, 16, 20))
            pad = 12 * scale
            x = pad
            y = 9 * scale
            for text, col in [("%d onions" % res.n_onions, (240, 242, 245)),
                              ("HEALTHY %d (%.0f%%)" % (res.n_healthy, res.pct_healthy),
                               self.PALETTE[0]),
                              ("UNHEALTHY %d (%.0f%%)" % (res.n_unhealthy, res.pct_unhealthy),
                               self.PALETTE[3])] + ([("BORDERLINE %d" % res.n_borderline,
                                                      (150, 170, 255))] if res.n_borderline else []):
                hf = big
                if x + d.textlength(text, font=big) > W - pad:     # narrow photo: shrink to fit
                    hf = mid
                d.text((x, y), text, fill=col, font=hf)
                x += d.textlength(text, font=hf) + 18 * scale
            sub = ("size: %s" % (("%.2f px/mm - %s" % (res.px_per_mm, res.scale_source))
                                 if res.px_per_mm else
                                 "relative only, no reference object in frame"))
            d.text((pad, 38 * scale), sub, fill=(150, 156, 164), font=mid)
            if not res.domain_ok:
                d.rectangle([0, 62 * scale, W, bar], fill=(120, 78, 10))
                d.text((pad, 70 * scale),
                       "OUT OF THE CALIBRATED DOMAIN - counts and outlines hold, "
                       "grades may be systematically off", fill=(255, 226, 150), font=mid)
        im.save(out_path, quality=92)
        return out_path


_FONTS = {True: ["DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf", "Arial Bold.ttf",
                 "Helvetica.ttc", "LiberationSans-Bold.ttf"],
          False: ["DejaVuSans.ttf", "arial.ttf", "segoeui.ttf", "Arial.ttf",
                  "Helvetica.ttc", "LiberationSans-Regular.ttf"]}


def _font(bold: bool, size: int):
    """A scalable font on Linux, Windows and macOS. DejaVu alone is not on
    Windows, and the fallback bitmap font is ~10 px whatever size is asked."""
    from PIL import ImageFont
    for name in _FONTS[bold]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)     # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade the onions in a photograph.")
    ap.add_argument("image")
    ap.add_argument("--out", help="write an annotated image here")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--calibration", default=DEFAULT_CAL)
    ap.add_argument("--px-per-mm", type=float,
                    help="known scale, when no coin is in frame")
    ap.add_argument("--device", help="cuda, cpu, mps (default: cuda if present)")
    ap.add_argument("--half", action="store_true",
                    help="deprecated and ignored: FP16 MobileSAM produces empty masks")
    ap.add_argument("--threads", type=int, help="CPU threads (default: all cores)")
    a = ap.parse_args()

    g = OnionGrader(calibration=a.calibration, device=a.device,
                    half=True if a.half else None, threads=a.threads)
    res = g.grade(a.image, px_per_mm=a.px_per_mm)
    print(res.summary())
    print()
    print("  %-3s %-10s %-22s %8s %6s %10s %s"
          % ("#", "verdict", "grade", "defect", "conf", "size", "flags"))
    print("  " + "-" * 78)
    for o in res.onions:
        size = ("%.0f mm %s" % (o.eq_diameter_mm, o.size_code)) if o.eq_diameter_mm \
            else ("x%.2f" % o.size_ratio)
        flags = " ".join(f for f in [("review" if o.needs_review else ""),
                                     ("unusual" if o.unusual_kind else ""),
                                     ("maybe-not-onion" if o.maybe_not_onion else "")] if f)
        print("  %-3d %-10s %-22s %7.1f%% %6.2f %10s %s"
              % (o.index, o.verdict, o.grade, 100 * o.defect_area_fraction,
                 o.confidence, size, flags))
    for r in res.rejected:
        print("  --  ignored     %s (p=%.2f)  box %s" % (r.get("why", "not an onion"),
                                                       r["p_not_onion"], r["box"]))
    if a.out:
        print("\nannotated image -> %s" % g.render(res, a.out, a.image))
    if a.json:
        open(a.json, "w").write(res.to_json())
        print("result json     -> %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
