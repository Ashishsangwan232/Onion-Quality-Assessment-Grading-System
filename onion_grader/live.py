"""Live onion grading from a camera.

    python -m onion_grader.live                       # laptop webcam
    python -m onion_grader.live --source 1            # second camera
    python -m onion_grader.live --source http://192.168.1.7:8080/video   # phone
    python -m onion_grader.live --source clip.mp4 --out graded.mp4 --no-show

Keys:  q quit   g grade now   s save a snapshot   space pause

Needs `lap` for the tracker:  pip install lap

---------------------------------------------------------------------------
Why this is two loops, not one
---------------------------------------------------------------------------
Grading a photograph takes seconds, because MobileSAM has to encode the whole
image. Nothing that takes seconds can run per frame.

So the work is split by how fast it needs to be:

    EVERY FRAME     YOLO tracking  -> boxes + a stable id per onion   (fast)
    ON A CADENCE    SAM + grading  -> the grade, attached to that id  (slow)

A grade is computed on one frame and then *carried* by the tracker until the
next grading pass. The id is what makes that legitimate: the same physical onion
keeps the same number as it moves, so the grade follows the right object.

Grades therefore lag the video by up to `--every` seconds. That is stated on
screen rather than hidden, because a number that looks live and is not is worse
than a number labelled stale.

---------------------------------------------------------------------------
What live grading is worth, honestly
---------------------------------------------------------------------------
Every accuracy figure this project reports was measured on **still
photographs**: a fixed camera, even light, onions lying still. A hand-held
camera adds motion blur, changing angle, changing distance and changing white
balance, and none of that has been measured. Live grades are an interface onto
the same instrument, not a validated mode of it.

Expect the domain guard to fire often on a webcam. It is not a malfunction --
your camera is not the camera the constants were fitted on, and the guard is
saying so correctly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onion_grader.grader import OnionGrader, GRADE_NAMES   # noqa: E402

PALETTE_BGR = [(110, 200, 60), (70, 200, 240), (50, 140, 245), (65, 65, 235)]
GREY = (160, 160, 160)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


class LiveGrader:
    """Tracks every frame, grades every `every` seconds, keyed by track id."""

    def __init__(self, grader: OnionGrader, every: float = 3.0, imgsz: int = 480):
        self.g = grader
        self.every = every
        self.imgsz = imgsz
        self.grades: dict[int, dict] = {}     # track id -> last known grade
        self.last_graded = 0.0
        self.last_result = None
        self.grading_ms = None
        self.ignored: dict[int, dict] = {}    # track id -> why the gate ignored it
        self.tracking = True                  # False: tracker unavailable, detect only
        self.track_error = ""
        # Tracking gets its OWN copy of the detector, so a grading pass can run
        # in a background thread (it uses grader._yolo) while tracking goes on.
        from ultralytics import YOLO
        self.model = YOLO(grader.detector_path)

    # ---------------------------------------------------------------- fast
    def track(self, frame):
        """YOLO tracking on one frame. Returns [(track_id, box)].

        Falls back to plain detection (ids = position in the list, so grades
        are re-matched by overlap every pass) if the tracker cannot run --
        usually because the `lap` package is missing."""
        kw = dict(verbose=False, conf=self.g.cfg["detector"]["conf"],
                  imgsz=self.imgsz, device=self.g.device)
        if self.tracking:
            try:
                r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kw)[0]
                if r.boxes is None or r.boxes.id is None:
                    return []
                ids = r.boxes.id.int().cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy()
                return list(zip(ids.tolist(), boxes))
            except Exception as e:          # noqa: BLE001
                self.tracking = False
                self.track_error = "%s: %s" % (type(e).__name__, e)
        r = self.model(frame, **kw)[0]
        if r.boxes is None or not len(r.boxes):
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        return [(-(i + 1), b) for i, b in enumerate(boxes)]

    def lookup(self, tid, box):
        """Grade for a track; without a tracker, the graded onion it overlaps."""
        if tid >= 0 or self.last_result is None:
            return self.grades.get(tid), self.ignored.get(tid)
        best = max(((iou(o.box, box), o) for o in self.last_result.onions),
                   default=(0, None), key=lambda t: t[0])
        if best[0] >= 0.3:
            return self._entry(best[1]), None
        for rj in self.last_result.rejected:
            if iou(rj["box"], box) >= 0.3:
                return None, rj
        return None, None

    def _entry(self, o):
        return {"band": o.band_index, "verdict": o.verdict, "healthy": bool(o.healthy),
                "chk": bool(o.is_borderline),
                "pct": 100 * o.defect_area_fraction, "grade": o.grade,
                "conf": o.confidence, "review": o.needs_review,
                "p": o.p_unhealthy, "at": self.last_graded}

    # ---------------------------------------------------------------- slow
    def grade_now(self, frame, tracked, tmpdir):
        """Full grade on this frame; attach each result to the track it overlaps."""
        import cv2
        path = os.path.join(tmpdir, "_live_frame_%d.jpg" % os.getpid())
        cv2.imwrite(path, frame)
        t = time.perf_counter()
        res = self.g.grade(path)
        self.grading_ms = (time.perf_counter() - t) * 1000
        self.last_result = res
        self.last_graded = time.time()

        # match graded onions to live tracks by box overlap, each track once
        pairs = sorted(((iou(o.box, b), i, tid)
                        for i, o in enumerate(res.onions) for tid, b in tracked),
                       reverse=True)
        used_o, used_t = set(), set()
        for v, i, tid in pairs:
            if v < 0.3 or i in used_o or tid in used_t:
                continue
            used_o.add(i)
            used_t.add(tid)
            self.grades[tid] = self._entry(res.onions[i])
            self.ignored.pop(tid, None)
        # tracks the gate threw out (faces, hands, duplicates): remember why
        for rj in res.rejected:
            for tid, b in tracked:
                if tid not in used_t and iou(rj["box"], b) >= 0.3:
                    self.ignored[tid] = rj
                    self.grades.pop(tid, None)
        # forget tracks that have gone away
        live = {tid for tid, _ in tracked}
        for tid in list(self.grades):
            if tid not in live and time.time() - self.grades[tid]["at"] > 10:
                del self.grades[tid]
        return res

    # ---------------------------------------------------------------- draw
    def draw(self, frame, tracked, fps, paused=False):
        import cv2
        out = frame.copy()
        H, W = out.shape[:2]
        sc = max(0.5, W / 1280)
        n_h = n_u = n_b = n_ungraded = 0

        for tid, b in tracked:
            x1, y1, x2, y2 = [int(v) for v in b]
            g, ign = self.lookup(tid, b)
            if ign is not None:
                cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 1)
                cv2.putText(out, "not an onion" if ign.get("why", "").startswith("not")
                            or "face" in ign.get("why", "") else "ignored",
                            (x1 + 3, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45 * sc, (150, 150, 150), 1, cv2.LINE_AA)
                continue
            if g is None:
                n_ungraded += 1
                cv2.rectangle(out, (x1, y1), (x2, y2), GREY, 1)
                cv2.putText(out, "not graded yet", (x1 + 3, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, GREY, 1, cv2.LINE_AA)
                continue
            col = PALETTE_BGR[g["band"]]
            if g.get("chk"):
                n_b += 1
            else:
                n_h += g["healthy"]
                n_u += not g["healthy"]
            cv2.rectangle(out, (x1, y1), (x2, y2), col, max(2, int(2 * sc)))
            tag = "%s %s" % (g["verdict"], g["grade"].replace(" eligible", ""))
            if g["review"]:
                tag += " ?"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, 1)
            ty = y1 - 4 if y1 - th - 8 > 0 else y2 + th + 8
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), col, -1)
            cv2.putText(out, tag, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * sc, (18, 18, 18), 1, cv2.LINE_AA)

        bar = int(64 * sc)
        cv2.rectangle(out, (0, 0), (W, bar), (20, 18, 16), -1)
        head = "%d tracked   HEALTHY %d   UNHEALTHY %d" % (len(tracked), n_h, n_u)
        if n_b:
            head += "   BORDERLINE %d" % n_b
        if n_ungraded:
            head += "   (%d not yet graded)" % n_ungraded
        cv2.putText(out, head, (10, int(24 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * sc, (240, 242, 245), max(1, int(sc)), cv2.LINE_AA)

        age = time.time() - self.last_graded if self.last_graded else None
        sub = "%.1f fps tracking" % fps
        if self.grading_ms:
            sub += "   grade %.1fs" % (self.grading_ms / 1000)
        if age is not None:
            sub += "   grades %.1fs old" % age
        sub += "   %s" % self.g.device
        if paused:
            sub += "   [PAUSED]"
        cv2.putText(out, sub, (10, int(48 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * sc, (150, 156, 164), 1, cv2.LINE_AA)

        r = self.last_result
        if r is not None and not r.domain_ok:
            cv2.rectangle(out, (0, bar), (W, bar + int(26 * sc)), (10, 78, 120), -1)
            cv2.putText(out, "OUT OF THE CALIBRATED DOMAIN - counts hold, grades may be off",
                        (10, bar + int(18 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * sc, (150, 226, 255), 1, cv2.LINE_AA)
        return out


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description="Live onion grading from a camera.")
    ap.add_argument("--source", default="0",
                    help="0 for the built-in camera, a number for another, "
                         "an http://.../video URL for a phone, or a video file")
    ap.add_argument("--every", type=float, default=3.0,
                    help="seconds between full grading passes (default 3)")
    ap.add_argument("--imgsz", type=int, default=480,
                    help="detector size for tracking (default 480). Measured on "
                         "8 synthetic trays against the validated 960: 480 is 4x "
                         "faster and finds 100%% of the boxes 960 finds. That is "
                         "agreement with 960, NOT ground-truth recall -- use 960 "
                         "for anything you intend to report")
    ap.add_argument("--device", default=None, help="cuda, cpu, mps (default: auto)")
    ap.add_argument("--out", help="write the annotated video here")
    ap.add_argument("--no-show", action="store_true", help="headless")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds")
    ap.add_argument("--snapshots", default=".", help="where 's' saves stills")
    a = ap.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    g = OnionGrader(device=a.device)
    print("loading models on %s ..." % g.device)
    g._load()

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("cannot open source %r.\n"
              "  laptop camera: try --source 0, then 1, then 2\n"
              "  phone: install an IP-camera app and pass its http://.../video URL,\n"
              "         with the phone and this machine on the same wi-fi" % a.source)
        return 1

    live = LiveGrader(g, every=a.every, imgsz=a.imgsz)
    writer = None
    fps_hist = deque(maxlen=30)
    tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    t_start = time.time()
    paused = False
    frames = 0

    print("tracking every frame, grading every %.1fs   [q quit, g grade now, "
          "s snapshot, space pause]" % a.every)
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                t0 = time.perf_counter()
                tracked = live.track(frame)
                fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))
                if time.time() - live.last_graded > a.every:
                    live.grade_now(frame, tracked, tmpdir)
                shown = live.draw(frame, tracked, float(np.mean(fps_hist)), paused)

                if a.out:
                    if writer is None:
                        h, w = shown.shape[:2]
                        writer = cv2.VideoWriter(
                            a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                            max(cap.get(cv2.CAP_PROP_FPS) or 15, 1), (w, h))
                    writer.write(shown)

            if not a.no_show:
                cv2.imshow("onion grader - live", shown)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord(" "):
                    paused = not paused
                if k == ord("g"):
                    live.grade_now(frame, tracked, tmpdir)
                if k == ord("s"):
                    p = os.path.join(a.snapshots,
                                     "onion_live_%s.jpg" % time.strftime("%H%M%S"))
                    cv2.imwrite(p, shown)
                    print("saved %s" % p)
            if a.max_seconds and time.time() - t_start > a.max_seconds:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not a.no_show:
            cv2.destroyAllWindows()

    el = time.time() - t_start
    print("\n%d frames in %.1fs = %.1f fps tracking" % (frames, el, frames / max(el, 1e-9)))
    if live.grading_ms:
        print("a full grading pass took %.2fs on %s" % (live.grading_ms / 1000, g.device))
    if live.last_result is not None:
        print()
        print(live.last_result.summary())
    print("\nLive grades are the same instrument seen through a moving camera. "
          "Every accuracy figure this project reports was measured on still "
          "photographs; motion blur and changing angle are not in those numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
