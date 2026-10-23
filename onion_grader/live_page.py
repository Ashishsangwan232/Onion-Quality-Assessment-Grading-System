"""
The "Live camera" tab of the Streamlit dashboard.

    streamlit run onion_grader/app.py      -> pick "Live camera" at the top

How it works (same two-speed design as live.py):

    EVERY FRAME    YOLO tracking -> boxes + a stable id per onion   (fast)
    EVERY N SEC    full grading pass (gate, SAM, classifier, grade)  (slow)

The camera is opened ON THE MACHINE RUNNING STREAMLIT. That is what you want
when you run the app on your own PC: the laptop webcam, a USB camera, or a
phone on the same Wi-Fi through an IP-camera app. (If you ever host the app on
a server, the server has no webcam -- that would need a browser-side stream.)

Every accuracy figure in this project was measured on still photographs. A
moving camera adds blur and changing angle, so live grades are a preview; use
"Grade now" with the camera held still, or a snapshot, for anything you keep.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import time
from collections import deque

import numpy as np
import streamlit as st
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(os.path.dirname(HERE), "live_snapshots")

LIVE_CSS = """
<style>
.og-live-frame{border-radius:15px;background:#10150f;overflow:hidden;margin-top:12px;min-height:300px;display:grid;place-items:center}
.og-live-off{height:360px;border-radius:15px;background:#edf2ec;margin-top:12px;display:grid;place-items:center;text-align:center;color:var(--muted);font-size:12.5px;padding:20px}
.og-live-off b{display:block;font-size:40px;margin-bottom:10px}
.og-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.og-dot.on{background:#d9453d;box-shadow:0 0 0 3px rgba(217,69,61,.18);animation:ogp 1.4s infinite}
.og-dot.off{background:#9aa59d}
.og-dot.pause{background:#c58a27}
@keyframes ogp{50%{opacity:.35}}
.og-live-stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:4px 0 6px}
@media(max-width:1050px){.og-live-stats{grid-template-columns:repeat(3,1fr)}}
</style>
"""


# ------------------------------------------------------------------ helpers
def _state():
    ss = st.session_state
    ss.setdefault("cam_on", False)
    ss.setdefault("cam_paused", False)
    ss.setdefault("grade_req", False)
    ss.setdefault("live_log", [])
    ss.setdefault("live_last_frame", None)     # BGR, what was on screen
    ss.setdefault("live_last_raw", None)       # BGR, un-annotated
    ss.setdefault("live_last_graded", None)    # (jpg bytes, result json, summary dict)
    ss.setdefault("live_snaps", [])
    return ss


def _open(source):
    import cv2
    cap = cv2.VideoCapture(source)
    if isinstance(source, int) and not cap.isOpened():
        cap.release()                          # Windows: DirectShow often works
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:                          # noqa: BLE001
        pass
    return cap


class LatestReader:
    """Reads a live camera in its own thread and keeps ONLY the newest frame.

    Without this, OpenCV queues every frame the phone sends. The grader
    processes ~5-10 frames a second while the phone sends 25-30, so the queue
    grows and the picture falls further and further behind -- the "lag" you
    see. Dropping stale frames keeps the view at most one frame behind."""

    def __init__(self, cap):
        import threading
        self.cap = cap
        self.frame = None
        self.ok = True
        self.seq = 0
        self.lock = threading.Lock()
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        fails = 0
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > 100:
                    self.ok = False
                    return
                time.sleep(0.02)
                continue
            fails = 0
            with self.lock:
                self.frame, self.seq = f, self.seq + 1

    def read(self, last_seq=-1, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.frame is not None and self.seq != last_seq:
                    return True, self.frame.copy(), self.seq
            if not self.ok:
                break
            time.sleep(0.005)
        return False, None, last_seq

    def release(self):
        self.stop = True
        self.t.join(timeout=1.0)
        self.cap.release()


_POOL = None


def _pool():
    """One background worker for grading passes (shared across reruns)."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onion-grade")
    return _POOL


def _grade_work(g, live, frame, tracked):
    """Runs in the worker thread: grade + render. Touches no Streamlit state."""
    t = time.perf_counter()
    res = live.grade_now(frame, tracked, tempfile.gettempdir())
    secs = time.perf_counter() - t
    src = os.path.join(tempfile.gettempdir(), "_live_frame_%d.jpg" % os.getpid())
    out = os.path.join(tempfile.gettempdir(), "_live_graded_bg.jpg")
    g.render(res, out, src)
    with open(out, "rb") as fh:
        return res, secs, fh.read()


def _fix_url(url: str):
    """Forgive the two usual mistakes with phone-camera apps:
    https:// (the app's self-signed certificate makes OpenCV fail) and a bare
    address without the stream path (IP Webcam and DroidCam both serve /video)."""
    u = (url or "").strip()
    if not u:
        return None
    if "://" not in u:
        u = "http://" + u
    if u.startswith("https://"):
        u = "http://" + u[len("https://"):]
    rest = u.split("://", 1)[1]
    if "/" not in rest.rstrip("/"):
        u = u.rstrip("/") + "/video"
    return u


def _summary(res, secs):
    return {"time": time.strftime("%H:%M:%S"),
            "onions": res.n_onions, "healthy": res.n_healthy,
            "unhealthy": res.n_unhealthy, "borderline": res.n_borderline,
            "% unhealthy": round(res.pct_unhealthy, 1) if res.n_onions else 0.0,
            "need review": sum(1 for o in res.onions if o.needs_review),
            "ignored": len(res.rejected),
            "in domain": "yes" if res.domain_ok else "NO",
            "grade secs": round(secs, 2)}


def _stats_html(tracked, n_h, n_u, fps, age, passes, state):
    dot = {"on": "on", "paused": "pause", "off": "off"}[state]
    lab = {"on": "LIVE", "paused": "PAUSED", "off": "OFF"}[state]
    age_s = "—" if age is None else ("%.0fs" % age)
    return ('<div class="og-live-stats">'
            '<div class="og-stat"><div class="l">camera</div><div class="n sm">'
            '<span class="og-dot %s"></span>%s</div><small>%.1f fps tracking</small></div>'
            '<div class="og-stat"><div class="l">in view</div><div class="n">%d</div><small>tracked objects</small></div>'
            '<div class="og-stat green"><div class="l">healthy</div><div class="n">%d</div><small>in view, graded</small></div>'
            '<div class="og-stat red"><div class="l">unhealthy</div><div class="n">%d</div><small>in view, graded</small></div>'
            '<div class="og-stat amber"><div class="l">last grade</div><div class="n sm">%s</div><small>ago</small></div>'
            '<div class="og-stat"><div class="l">grading passes</div><div class="n">%d</div><small>this session</small></div>'
            '</div>' % (dot, lab, fps, tracked, n_h, n_u, age_s, passes))


def _counts(live, tracked):
    n_h = n_u = 0
    for tid, b in tracked:
        g, ign = live.lookup(tid, b)
        if g is not None and ign is None:
            n_h += g["healthy"]
            n_u += not g["healthy"]
    return n_h, n_u


def _log_csv(log):
    buf = io.StringIO()
    if log:
        w = csv.DictWriter(buf, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    return buf.getvalue()


# ------------------------------------------------------------------ page
def render_live(load, md, card):
    import cv2
    from onion_grader.live import LiveGrader

    ss = _state()
    st.markdown(LIVE_CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------- topbar
    md('<div class="og-eyebrow">Live camera</div>'
       '<div class="og-h1">Live Onion Grading</div>'
       '<div class="og-caption">Tracks every onion in the camera view and re-grades them '
       'every few seconds. Every accuracy figure here was measured on still photographs — '
       'hold the camera still (or use a stand) and press <b>Grade now</b> for results you keep.</div>')
    md('<div class="og-spacer"></div>')

    col_ctl, col_vid = st.columns([0.9, 2.1], gap="medium")

    # ------------------------------------------------------------- controls
    with col_ctl:
        with card("live_ctl"):
            md('<div class="og-card-h">Camera</div>'
               '<div class="og-sub">Pick a source, then switch the camera on.</div>')
            kind = st.radio("source", ["Webcam", "Phone / IP camera", "Video file"],
                            horizontal=False, key="live_kind", disabled=ss.cam_on)
            source = None
            if kind == "Webcam":
                idx = st.selectbox("camera number", [0, 1, 2, 3], key="live_idx",
                                   disabled=ss.cam_on,
                                   help="0 is usually the built-in camera; try 1 or 2 for a USB camera.")
                source = int(idx)
            elif kind == "Phone / IP camera":
                url = st.text_input("stream URL", key="live_url", disabled=ss.cam_on,
                                    placeholder="http://192.168.1.7:8080/video",
                                    help="Install an IP-camera app on the phone (e.g. 'IP Webcam' "
                                         "on Android), start it, and paste its /video URL. "
                                         "Phone and PC must be on the same Wi-Fi.")
                source = _fix_url(url)
                if source and source != url.strip():
                    st.caption("Using `%s`" % source)
            else:
                vf = st.file_uploader("video", type=["mp4", "avi", "mov", "mkv"],
                                      key="live_vid", disabled=ss.cam_on,
                                      label_visibility="collapsed")
                if vf is not None:
                    vp = os.path.join(tempfile.gettempdir(), "onion_live_" + vf.name)
                    if not os.path.exists(vp):
                        with open(vp, "wb") as fh:
                            fh.write(vf.getvalue())
                    source = vp

            if not ss.cam_on:
                if st.button("▶  Camera ON", type="primary", use_container_width=True,
                             disabled=source is None, key="cam_on_btn"):
                    ss.cam_on, ss.cam_paused = True, False
                    ss.pop("live_grader", None)       # fresh tracker for a fresh stream
                    st.rerun()
            else:
                if st.button("■  Camera OFF", type="primary", use_container_width=True,
                             key="cam_off_btn"):
                    ss.cam_on = False
                    st.rerun()

            b1, b2 = st.columns(2)
            if b1.button("⏸ Pause" if not ss.cam_paused else "▶ Resume",
                         use_container_width=True, disabled=not ss.cam_on, key="pause_btn"):
                ss.cam_paused = not ss.cam_paused
                st.rerun()
            if b2.button("⚡ Grade now", use_container_width=True,
                         disabled=not ss.cam_on, key="grade_btn"):
                ss.grade_req = True
                st.rerun()
            snap_clicked = st.button("📸 Snapshot", use_container_width=True,
                                     disabled=ss.live_last_frame is None, key="snap_btn")

            st.divider()
            md('<div class="og-card-h" style="font-size:13px">Settings</div>')
            auto = st.toggle("auto-grade", True, key="live_auto",
                             help="Run a full grading pass automatically every N seconds.")
            every = st.slider("grade every (seconds)", 2, 30, 5, key="live_every",
                              disabled=not auto)
            imgsz = st.select_slider("tracking resolution", [320, 480, 640, 960], 480,
                                     key="live_imgsz",
                                     help="Lower is faster. Grading always runs at full resolution.")
            show_boxes = st.toggle("show boxes & labels", True, key="live_boxes")
            mirror = st.toggle("mirror (selfie view)", False, key="live_mirror",
                               disabled=kind != "Webcam")

    # ------------------------------------------------------------- video + outputs
    with col_vid:
        with card("live_video"):
            md('<div class="og-card-h">Camera view</div>'
               '<div class="og-sub">Coloured box = graded onion · grey = not graded yet · '
               'thin grey = ignored (not an onion / duplicate).</div>')
            frame_slot = st.empty()
            note_slot = st.empty()
    stats_slot = st.empty()

    col_last, col_log = st.columns([1.2, 1], gap="medium")
    with col_last:
        with card("live_last"):
            md('<div class="og-card-h">Last graded frame</div>'
               '<div class="og-sub">The full result of the most recent grading pass.</div>')
            last_slot = st.empty()
            last_info = st.empty()
            if ss.live_last_graded is not None:
                jpg, rjson, _ = ss.live_last_graded
                d1, d2 = st.columns(2)
                d1.download_button("↓ graded image", jpg, file_name="live_graded.jpg",
                                   mime="image/jpeg", use_container_width=True, key="dl_lastimg")
                d2.download_button("↓ result JSON", rjson, file_name="live_grades.json",
                                   mime="application/json", use_container_width=True,
                                   key="dl_lastjson")
    with col_log:
        with card("live_log"):
            md('<div class="og-card-h">Session log</div>'
               '<div class="og-sub">One row per grading pass. The same onions are re-graded '
               'each pass, so rows are not summed.</div>')
            log_slot = st.empty()
            l1, l2 = st.columns(2)
            l1.download_button("↓ log CSV", _log_csv(ss.live_log), file_name="live_log.csv",
                               mime="text/csv", use_container_width=True,
                               disabled=not ss.live_log, key="dl_log")
            if l2.button("Clear log", use_container_width=True, disabled=not ss.live_log,
                         key="clear_log"):
                ss.live_log = []
                st.rerun()
            snap_slot = st.empty()

    def show_log():
        if ss.live_log:
            log_slot.dataframe(list(reversed(ss.live_log[-200:])), hide_index=True,
                               use_container_width=True, height=300)
        else:
            log_slot.markdown('<div class="og-sub" style="padding:40px 0;text-align:center">'
                              'No grading passes yet.</div>', unsafe_allow_html=True)

    def show_last():
        if ss.live_last_graded is None:
            last_slot.markdown('<div class="og-empty" style="height:220px"><div><b>🧅</b>'
                               'Nothing graded yet.</div></div>', unsafe_allow_html=True)
            return
        jpg, _, summ = ss.live_last_graded
        last_slot.image(jpg, use_container_width=True)
        msg = ("%(onions)d onions · %(healthy)d healthy · %(unhealthy)d unhealthy · "
               "%(need review)d need review · graded at %(time)s in %(grade secs)ss" % summ)
        if summ["in domain"] == "NO":
            last_info.warning(msg + "\n\nOut of the calibrated domain — counts hold, "
                              "grades may be systematically off for this camera/light.")
        else:
            last_info.caption(msg)

    show_log()
    show_last()

    # ------------------------------------------------------------- snapshot
    if snap_clicked and ss.live_last_frame is not None:
        os.makedirs(SNAP_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shown_p = os.path.join(SNAP_DIR, "live_%s_view.jpg" % stamp)
        cv2.imwrite(shown_p, ss.live_last_frame)
        raw_p = os.path.join(SNAP_DIR, "live_%s_raw.jpg" % stamp)
        if ss.live_last_raw is not None:
            cv2.imwrite(raw_p, ss.live_last_raw)
        ss.live_snaps.append(shown_p)
        with open(shown_p, "rb") as fh:
            snap_bytes = fh.read()
        snap_slot.success("Saved to `live_snapshots/%s` (and the raw frame beside it — "
                          "upload that on the Photo tab for a full still-photo grade)."
                          % os.path.basename(shown_p))
        st.download_button("↓ download snapshot", snap_bytes,
                           file_name=os.path.basename(shown_p), mime="image/jpeg",
                           key="dl_snap_%s" % stamp)

    # ------------------------------------------------------------- idle / paused
    g = load()
    if not ss.cam_on or ss.cam_paused:
        if ss.cam_on and ss.live_last_frame is not None:
            frame_slot.image(ss.live_last_frame, channels="BGR", use_container_width=True)
            note_slot.info("Paused — press Resume to continue, or Grade now to grade this view.")
        else:
            frame_slot.markdown('<div class="og-live-off"><div><b>📷</b>Camera is off.<br>'
                                'Pick a source and press <b>Camera ON</b>.</div></div>',
                                unsafe_allow_html=True)
        lr = ss.get("live_grader")
        stats_slot.markdown(_stats_html(0, 0, 0, 0.0,
                                        (time.time() - lr.last_graded) if lr and lr.last_graded else None,
                                        len(ss.live_log), "paused" if ss.cam_on else "off"),
                            unsafe_allow_html=True)
        if ss.cam_on and ss.grade_req and ss.live_last_raw is not None:
            ss.grade_req = False
            _grade(ss, g, ss.live_last_raw, [], note_slot)
            show_log()
            show_last()
        return

    # ------------------------------------------------------------- live loop
    with st.spinner("loading models…"):
        g._load()
    live = ss.get("live_grader")
    if live is None:
        live = LiveGrader(g, every=every, imgsz=imgsz)
        ss.live_grader = live
    live.every, live.imgsz = every, imgsz

    src = source
    if src is None:
        ss.cam_on = False
        frame_slot.error("No source selected.")
        return
    cap = _open(src)
    if not cap.isOpened():
        ss.cam_on = False
        frame_slot.error(
            "Could not open %r.\n\n"
            "- **Webcam:** close other apps using the camera (Teams, Zoom, the Camera app), "
            "then try camera number 1 or 2.\n"
            "- **Phone:** check the URL in the phone's browser first; both devices must be "
            "on the same Wi-Fi.\n"
            "- **Video:** the file may be in a codec OpenCV cannot read — try MP4 (H.264)." % (src,))
        return

    fps_hist = deque(maxlen=20)
    last_stats = 0.0
    is_file = isinstance(src, str) and os.path.exists(src)
    reader = None if is_file else LatestReader(cap)   # files play every frame, in order
    seq = -1
    job = None
    try:
        while ss.cam_on:
            if reader is None:
                ok, frame = cap.read()
            else:
                ok, frame, seq = reader.read(seq)
            if not ok:
                if is_file:
                    note_slot.info("End of video.")
                    ss.cam_on = False
                    break
                note_slot.warning("No frame from the camera — retrying…")
                time.sleep(0.3)
                continue
            if mirror and not is_file:
                frame = cv2.flip(frame, 1)
            t0 = time.perf_counter()
            tracked = live.track(frame)
            fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))

            # grading runs in the background; the video keeps moving meanwhile
            if job is not None and job.done():
                try:
                    res, secs, jpg = job.result()
                    _store(ss, res, secs, jpg, note_slot)
                    show_log()
                    show_last()
                except Exception as e:      # noqa: BLE001
                    note_slot.error("Grading failed: %s" % e)
                job = None
            due = auto and (time.time() - live.last_graded > every)
            if (ss.grade_req or due) and job is None:
                ss.grade_req = False
                live.last_graded = time.time()          # don't queue another meanwhile
                job = _pool().submit(_grade_work, g, live, frame.copy(), list(tracked))
                note_slot.caption("⚡ grading this frame in the background…")

            shown = live.draw(frame, tracked, float(np.mean(fps_hist))) if show_boxes else frame
            ss.live_last_frame = shown
            ss.live_last_raw = frame
            frame_slot.image(shown, channels="BGR", use_container_width=True)

            if not live.tracking and live.track_error:
                note_slot.caption("Tracking unavailable (%s) — boxes are matched to grades by "
                                  "overlap instead. `pip install lap` enables tracking."
                                  % live.track_error[:120])
            if time.time() - last_stats > 0.5:
                last_stats = time.time()
                n_h, n_u = _counts(live, tracked)
                stats_slot.markdown(_stats_html(
                    len(tracked), n_h, n_u, float(np.mean(fps_hist)),
                    (time.time() - live.last_graded) if live.last_graded else None,
                    len(ss.live_log), "on"), unsafe_allow_html=True)
    finally:
        if job is not None:
            try:
                res, secs, jpg = job.result(timeout=60)
                _store(ss, res, secs, jpg, note_slot)
            except Exception:               # noqa: BLE001
                pass
        if reader is not None:
            reader.release()
        else:
            cap.release()


def _store(ss, res, secs, jpg, note_slot):
    summ = _summary(res, secs)
    if not res.is_onion_photo:
        summ["onions"] = 0
        note_slot.warning("No onions in this view — the gate rejected everything it saw.")
    ss.live_log.append(summ)
    ss.live_last_graded = (jpg, res.to_json(), summ)


def _grade(ss, g, frame, tracked, note_slot, live=None):
    """One full grading pass on `frame`; updates the log and 'last graded'."""
    import cv2
    t = time.perf_counter()
    if live is not None:
        res = live.grade_now(frame, tracked, tempfile.gettempdir())
    else:
        p = os.path.join(tempfile.gettempdir(), "_live_frame.jpg")
        cv2.imwrite(p, frame)
        res = g.grade(p)
        lr = ss.get("live_grader")
        if lr is not None:
            lr.last_result, lr.last_graded = res, time.time()
    secs = time.perf_counter() - t
    src = os.path.join(tempfile.gettempdir(), "_live_frame.jpg")
    out = os.path.join(tempfile.gettempdir(), "_live_graded.jpg")
    g.render(res, out, src)
    with open(out, "rb") as fh:
        jpg = fh.read()
    summ = _summary(res, secs)
    if not res.is_onion_photo:
        summ["onions"] = 0
        note_slot.warning("No onions in this view — the gate rejected everything it saw.")
    ss.live_log.append(summ)
    ss.live_last_graded = (jpg, res.to_json(), summ)
