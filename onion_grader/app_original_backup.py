"""
Onion grader — drop in a photograph, get graded onions.

    streamlit run onion_grader/app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onion_grader.grader import OnionGrader, GRADE_NAMES   # noqa: E402

st.set_page_config(page_title="Onion grader", layout="wide")
BADGE = {0: "#3cc86e", 1: "#f0c846", 2: "#f58c32", 3: "#eb4141"}


@st.cache_resource
def load():
    return OnionGrader()


@st.cache_data(show_spinner=False)
def run(path: str, px_per_mm):
    g = load()
    res = g.grade(path, px_per_mm=px_per_mm)
    return res


st.title("Onion quality grading")
st.caption("Provisional grade, visual criteria only — firmness, dryness, internal "
           "soundness and smell are part of the legal grade and are not in a photograph.")

with st.sidebar:
    st.subheader("Size and the coin")
    st.markdown(
        "**With a ₹1 coin in frame**, sizes come out in millimetres. The detector "
        "finds the coin by itself.\n\n"
        "**Without one, absolute size cannot be recovered at all** — a small onion "
        "near the lens is identical, pixel for pixel, to a large one far away. No "
        "model can undo that.\n\n"
        "What still works with no coin:\n"
        "- **relative size is exact** — ratios need no scale\n"
        "- **grade, defect %, confidence and health are unaffected**\n"
        "- type a known scale below if you have one"
    )
    manual = st.number_input("px per mm (0 = auto)", 0.0, 100.0, 0.0, 0.01,
                             help="Measure any known object: its pixel length ÷ its mm length.")
    st.divider()
    st.subheader("Measured performance")
    st.markdown(
        "Cross-validated by physical onion, 120 hand-graded onions:\n\n"
        "| | κ |\n|---|---|\n"
        "| **this system, end to end** | **0.819** |\n"
        "| hand-clicked prompts | 0.757 |\n"
        "| *two humans with each other* | *0.701* |\n"
        "| *one human with herself* | *0.852* |\n\n"
        "Within one band **0.950** · detection recall **0.998**"
    )

up = st.file_uploader("photograph", type=["jpg", "jpeg", "png"])
if not up:
    st.info("Upload a photograph of onions. A ₹1 coin in frame unlocks millimetres.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t:
    t.write(up.read())
    path = t.name

with st.spinner("detecting, segmenting, measuring, grading…"):
    res = run(path, manual if manual > 0 else None)

g = load()

if not res.is_onion_photo:
    st.error(
        "**This does not look like a photograph of onions, so nothing was graded.** "
        "The detector found %d round object(s), and the onion gate judged %d of them "
        "not to be onions (faces, hands, objects, background).\n\n"
        "This is not a calibration problem — recalibrating would not help. "
        "Photograph the onions themselves, on a plain surface, in even light."
        % (len(res.rejected) + res.n_onions, len(res.rejected)))
    out = os.path.join(tempfile.gettempdir(), "graded_view.jpg")
    g.render(res, out, path)
    st.image(out, caption="grey boxes: what the detector found, and why it was ignored",
             use_container_width=True)
    st.stop()

if not res.onions:
    st.error("No onions found in this photograph.")
    st.stop()

if res.gate != "on":
    st.warning("Onion gate is OFF (onion_gate.pth not found next to grader.py). "
               "Anything round in the photo — a face, a hand — would be graded as an onion.")
elif res.rejected:
    st.info("%d detection(s) ignored — the onion gate says they are not onions. "
            "They are shown as grey boxes below." % len(res.rejected))

# ---------------------------------------------------------------- headline
c = st.columns(5)
c[0].metric("onions", res.n_onions)
c[1].metric("healthy", "%d" % res.n_healthy, "%.0f%%" % res.pct_healthy)
c[2].metric("unhealthy", "%d" % res.n_unhealthy, "-%.0f%%" % res.pct_unhealthy,
            delta_color="inverse")
c[3].metric("need review", sum(1 for o in res.onions if o.needs_review))
c[4].metric("scale", ("%.2f px/mm" % res.px_per_mm) if res.px_per_mm else "relative only")

if not res.domain_ok:
    s = res.domain_stats
    st.error("")
        # "**This photograph is outside the calibrated domain.** "
        # + ("Its onions look unlike anything in the calibration set — median "
        #    "anomaly %.2f, against %.2f for every one of the 53 calibration "
        #    "photographs." % (s["median_anomaly"], s["median_anomaly_max_in_domain"])
        #    if "A" in s["direction"] else
        #    "The damage rule did not register on a single onion — highest raw "
        #    "defect %.1f%%, and across the 53 calibration photographs the highest "
        #    "never fell below 9.9%%." % (100 * s["max_raw_defect_fraction"]))
        # + "\n\n**Counts, outlines and relative sizes are unaffected.** The defect "
        #   "percentages, and therefore the grades, may be systematically harsh or "
        #   "lenient here.\n\n"
        #   "**First check the photograph**: real onions, plain background, even light, "
        #   "no screenshots or filters. **Only if this is your normal set-up** (a new "
        #   "camera, variety or light), hand-grade ~100 onions from it and rerun "
        #   "`audit/phase16_export_calibration.py`.")

for w in res.warnings:
    if w.startswith("OUT OF THE CALIBRATED DOMAIN"):
        continue          # shown above, in full, already
    st.warning(w)

tab_all, tab_one = st.tabs(["Whole photograph", "Each onion"])

# ---------------------------------------------------------------- overview
with tab_all:
    left, right = st.columns([3, 2])
    with left:
        o1 = st.checkbox("bounding boxes", True)
        o2 = st.checkbox("outlines", True)
        out = os.path.join(tempfile.gettempdir(), "graded_view.jpg")
        g.render(res, out, path, boxes=o1, outlines=o2)
        st.image(out, use_container_width=True)
        with open(out, "rb") as fh:
            st.download_button("download annotated image", fh.read(),
                               file_name="graded.jpg", mime="image/jpeg")
    with right:
        st.dataframe(
            [{"#": o.index, "verdict": o.verdict, "grade": o.grade,
              "P(unhealthy)": None if o.p_unhealthy is None else round(o.p_unhealthy, 2),
              "dark area %": round(100 * o.defect_area_fraction, 1),
              "conf": round(o.confidence, 2),
              "size": ("%.0f mm (%s)" % (o.eq_diameter_mm, o.size_code))
                      if o.eq_diameter_mm else ("x%.2f" % o.size_ratio),
              "flags": " ".join(f for f in [("review" if o.needs_review else ""),
                                            ("unusual" if o.unusual_kind else ""),
                                            ("maybe not an onion" if o.maybe_not_onion else ""),
                                            ("partial" if o.partial else "")] if f)}
             for o in res.onions],
            use_container_width=True, hide_index=True, height=420)
        st.download_button("download result (JSON)", res.to_json(),
                           file_name="grades.json", mime="application/json")
        cnt = Counter(o.grade for o in res.onions)
        st.bar_chart({k: cnt.get(k, 0) for k in GRADE_NAMES})
        with st.expander("is this photograph like the calibration set?"):
            st.json(res.domain_stats)
            st.caption(
                "Two independent checks, both fitted so they fire on 0 of the 53 "
                "calibration photographs: A — the anomaly score says everything "
                "looks strange; B — the damage rule registers on nothing at all. "
                "They catch opposite failures and neither sees the other's.")

# ---------------------------------------------------------------- per onion
with tab_one:
    st.caption("Pick an onion to see exactly what the grade was measured on.")
    labels = ["#%d  %s" % (o.index, o.verdict) for o in res.onions]
    pick = st.radio("onion", range(len(res.onions)),
                    format_func=lambda i: labels[i], horizontal=True)
    o = res.onions[pick]

    crop, outline, heat = g.crop(res, o, path)
    i1, i2, i3 = st.columns(3)
    i1.image(crop, caption="as photographed", use_container_width=True)
    i2.image(outline, caption="the boundary it measured over", use_container_width=True)
    i3.image(heat, caption="the pixels counted as damage (red)", use_container_width=True)

    st.markdown(
        "<div style='background:%s;color:#111;padding:10px 14px;border-radius:8px;"
        "font-weight:700;font-size:1.05rem'>#%d &nbsp; %s &nbsp;·&nbsp; %s%s</div>"
        % (BADGE[o.band_index], o.index, o.verdict, o.grade,
           "" if o.p_unhealthy is None else
           " &nbsp;·&nbsp; classifier: %.0f%% unhealthy" % (100 * o.p_unhealthy)),
        unsafe_allow_html=True)
    if o.grade_note:
        st.info(o.grade_note)
    if o.partial:
        st.warning("This onion runs off the edge of the photo, so part of it is unseen.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("dark area", "%.1f%%" % (100 * o.defect_area_fraction))
    m1.caption("band: %s" % o.band)
    m2.metric("confidence", "%.2f" % o.confidence)
    m2.caption("check it is an onion (gate %.0f%% unsure)" % (100 * o.p_not_onion)
               if o.maybe_not_onion else
               "needs a human" if o.needs_review else "clear of the boundaries")
    m3.metric("anomaly", "%.2f" % o.anomaly_score)
    m3.caption("not meaningful — out of domain" if not o.anomaly_reliable else
               ("a different KIND of damage" if o.unusual_kind else "typical of its grade"))
    if o.eq_diameter_mm:
        m4.metric("size", "%.0f mm" % o.eq_diameter_mm)
        m4.caption("AGMARK code %s · %.2fx the median here" % (o.size_code, o.size_ratio))
    else:
        m4.metric("relative size", "x%.2f" % o.size_ratio)
        m4.caption("rank %d of %d · no reference object for mm"
                   % (o.size_rank, res.n_onions))

    with st.expander("all measured values"):
        st.json({k: v for k, v in
                 [("box", o.box), ("verdict", o.verdict), ("grade", o.grade),
                  ("band", o.band), ("defect_area_fraction", o.defect_area_fraction),
                  ("raw_defect_fraction", o.raw_defect_fraction),
                  ("confidence", o.confidence), ("needs_review", o.needs_review),
                  ("anomaly_score", o.anomaly_score), ("unusual_kind", o.unusual_kind),
                  ("anomaly_reliable", o.anomaly_reliable),
                  ("detector_conf", o.detector_conf), ("p_not_onion", o.p_not_onion),
                  ("maybe_not_onion", o.maybe_not_onion),
                  ("area_px", o.area_px), ("eq_diameter_px", o.eq_diameter_px),
                  ("circularity", o.circularity), ("aspect_ratio", o.aspect_ratio),
                  ("size_ratio", o.size_ratio), ("size_rank", o.size_rank),
                  ("eq_diameter_mm", o.eq_diameter_mm), ("size_code", o.size_code)]})
        st.caption(
            "circularity 0.63 is a perfect digital disc, so onions sit near 0.52. "
            "raw_defect_fraction is before the instrument calibration that maps "
            "measured area onto human-judged area.")
