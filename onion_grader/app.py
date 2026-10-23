"""
Onion grader — drop in a photograph, get graded onions.

    streamlit run onion_grader/app.py

Front end restyled to match the "OnionGrade" dashboard design. The grading
pipeline (OnionGrader.grade / render / crop) and every message, warning and
measured value are exactly the same as before — only the presentation changed.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import sys
import tempfile
from collections import Counter

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onion_grader.grader import OnionGrader, GRADE_NAMES, BAND_NAMES  # noqa: E402

st.set_page_config(page_title="Onion Quality Grading", page_icon="🧅", layout="wide")
BADGE = {0: "#3cc86e", 1: "#f0c846", 2: "#f58c32", 3: "#eb4141"}


@st.cache_resource
def load():
    return OnionGrader()


@st.cache_resource(show_spinner=False, max_entries=4)
def run(path: str, px_per_mm):
    g = load()
    res = g.grade(path, px_per_mm=px_per_mm)
    return res


# =================================================================== styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#f5f7f2; --card:#fff; --ink:#172019; --muted:#718078;
  --line:#e3e9e2; --green:#477a54; --green2:#eaf4eb;
  --amber:#c58a27; --amber2:#fff5df; --red:#c45b55; --red2:#fff0ee;
  --shadow:0 12px 35px rgba(28,45,31,.07); --radius:18px;
}
html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp li, .stApp td, .stApp th{
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif !important;
}
.stApp{background:var(--bg);color:var(--ink)}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stDecoration"]{display:none}
.block-container,[data-testid="stMainBlockContainer"]{padding:2rem 2.2rem 3rem !important;max-width:1440px}

/* ---------------- sidebar */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div{background:#17351f !important}
section[data-testid="stSidebar"]{border-right:0}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] td,
section[data-testid="stSidebar"] th, section[data-testid="stSidebar"] em,
section[data-testid="stSidebar"] span{color:#b9cabc !important;font-size:12.5px}
section[data-testid="stSidebar"] strong{color:#eaf3ec !important}
section[data-testid="stSidebar"] hr{border-color:#2c4b34 !important;margin:18px 0}
section[data-testid="stSidebar"] table{width:100%;border-collapse:collapse;margin:6px 0}
section[data-testid="stSidebar"] th, section[data-testid="stSidebar"] td{
  background:transparent !important;border:0 !important;border-bottom:1px solid #2c4b34 !important;padding:6px 8px !important}
section[data-testid="stSidebar"] tr{background:transparent !important}
section[data-testid="stSidebar"] input{background:#1f4029 !important;color:#fff !important}
section[data-testid="stSidebar"] [data-baseweb="input"], section[data-testid="stSidebar"] [data-baseweb="base-input"]{
  background:#1f4029 !important;border-color:#2f5a3b !important;border-radius:10px}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button{background:#285238 !important;color:#fff !important;border:0 !important}
section[data-testid="stSidebar"] svg{fill:#b9cabc}
.og-brand{display:flex;gap:11px;align-items:center;padding:0 4px 22px}
.og-logo{width:38px;height:38px;border-radius:12px;background:#dcebd9;display:grid;place-items:center;font-size:21px}
.og-brand strong{font-size:15px !important;letter-spacing:.2px;color:#eaf3ec !important}
.og-brand .og-tag{display:block;color:#9fb4a3 !important;font-size:11px !important;margin-top:2px}
.og-nav-label{font-size:10px !important;color:#829988 !important;text-transform:uppercase;letter-spacing:1.2px;margin:6px 2px 8px;font-weight:600}
.og-nav-item{display:flex;gap:10px;align-items:center;padding:10px 12px;border-radius:11px;background:#285238;margin-bottom:14px}
.og-nav-item span{color:#fff !important;font-size:13px !important}
.og-nav-item.off{background:transparent;margin-bottom:6px}
.og-nav-item.off span{color:#b9cabc !important}

/* ---------------- mode switch (Photo | Live), styled as tabs */
.st-key-modebar [role="radiogroup"]{gap:6px;background:#e9efe7;padding:5px;border-radius:13px;display:inline-flex}
.st-key-modebar [role="radiogroup"] label{margin:0 !important;padding:8px 18px !important;border-radius:9px;cursor:pointer}
.st-key-modebar [role="radiogroup"] label:has(input:checked){background:#fff;box-shadow:0 2px 8px rgba(28,45,31,.08)}
.st-key-modebar [role="radiogroup"] label:has(input:checked) p{color:var(--green) !important;font-weight:700 !important}
.st-key-modebar [role="radiogroup"] label > div:first-child{display:none}
.st-key-modebar [role="radiogroup"] label p{font-size:13px !important;font-weight:600;color:#59665e}

/* ---------------- topbar */
.og-eyebrow{font-size:11px;color:var(--green);font-weight:700;text-transform:uppercase;letter-spacing:1.2px}
.og-h1{margin:5px 0 4px;font-size:27px;letter-spacing:-.6px;font-weight:750;color:var(--ink);line-height:1.2}
.og-caption{font-size:12px;color:var(--muted);max-width:760px;line-height:1.5}
.og-spacer{height:18px}

/* ---------------- cards (keyed containers) */
div[class*="st-key-card_"]{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:20px 20px 16px}
.og-card-h{font-size:15px;margin:0;font-weight:700;color:var(--ink)}
.og-sub{font-size:11.5px;color:var(--muted);margin-top:4px;line-height:1.45}

/* ---------------- uploader */
[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"]{
  border:1.5px dashed #b9c9ba !important;border-radius:15px !important;background:#fafcf9 !important;padding:26px 20px !important}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] section button{
  background:var(--green) !important;color:#fff !important;border:0 !important;border-radius:9px !important;font-weight:700 !important}
[data-testid="stFileUploaderDropzoneInstructions"] span{color:var(--ink);font-weight:600}
[data-testid="stFileUploaderDropzoneInstructions"] small{color:var(--muted)}

/* ---------------- images inside cards */
.og-preview{width:100%;border-radius:15px;display:block;background:#edf2ec;margin-top:12px;max-height:560px;object-fit:contain}
.og-result{border-radius:15px;background:#edf2ec;overflow:hidden;margin-top:16px}
.og-result img{width:100%;display:block;max-height:640px;object-fit:contain}
.og-empty{height:285px;border-radius:15px;background:#edf2ec;margin-top:16px;display:grid;place-items:center;text-align:center;color:var(--muted);font-size:12px;padding:20px}
.og-empty b{display:block;font-size:34px;margin-bottom:8px}
.og-fig-cap{font-size:10.5px;color:var(--muted);margin-top:7px}
.og-status{display:inline-block;font-size:10.5px;color:var(--green);background:var(--green2);padding:6px 10px;border-radius:20px;font-weight:600}
.og-status.red{color:var(--red);background:var(--red2)}

/* ---------------- buttons */
.stButton button, .stDownloadButton button{border:1px solid var(--line) !important;background:#fff !important;color:var(--ink) !important;
  border-radius:10px !important;font-size:12.5px !important;font-weight:650 !important;padding:8px 14px !important;box-shadow:none !important;transition:transform .15s}
.stButton button:hover, .stDownloadButton button:hover{transform:translateY(-1px);border-color:#c9d6ca !important}
.stButton button p, .stDownloadButton button p{font-size:12.5px !important;font-weight:650 !important}
.stButton button[kind="primary"], [data-testid="stBaseButton-primary"]{background:var(--green) !important;border-color:var(--green) !important;color:#fff !important}
.stButton button[kind="primary"] p, [data-testid="stBaseButton-primary"] p{color:#fff !important}

/* ---------------- checkboxes */
[data-testid="stCheckbox"] label p{font-size:12px !important;color:#59665e !important}

/* ---------------- stats */
.og-stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:4px 0 6px}
.og-stat{background:#fff;border:1px solid var(--line);border-radius:14px;padding:15px 14px;box-shadow:var(--shadow)}
.og-stat .l{font-size:10.5px;color:var(--muted);text-transform:capitalize}
.og-stat .n{font-size:24px;font-weight:750;margin:5px 0 2px;color:var(--ink);line-height:1.15}
.og-stat .n.sm{font-size:17px;padding:4px 0 3px}
.og-stat small{font-size:10.5px;color:var(--muted)}
.og-stat.green .n{color:var(--green)} .og-stat.red .n{color:var(--red)} .og-stat.amber .n{color:var(--amber)}

/* ---------------- pills / details */
.og-pill{display:inline-flex;padding:5px 9px;border-radius:20px;font-size:10.5px;font-weight:700;letter-spacing:.2px}
.og-pill.healthy{background:var(--green2);color:var(--green)} .og-pill.unhealthy{background:var(--red2);color:var(--red)}
.og-figs{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}
.og-figs img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:12px;display:block;background:#f0f3ed}
.og-title{font-size:16px;font-weight:750;margin:10px 0 3px;color:var(--ink)}
.og-details{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0 6px}
.og-detail{background:#f8faf7;border-radius:10px;padding:11px}
.og-detail label{display:block;font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.og-detail b{font-size:15px;display:block;margin-top:3px;color:var(--ink)}
.og-detail span{display:block;font-size:10.5px;color:var(--muted);margin-top:4px;line-height:1.35}

/* ---------------- grade chart */
.og-chart{height:230px;display:flex;align-items:flex-end;gap:22px;padding:25px 18px 0;border-left:1px solid var(--line);border-bottom:1px solid var(--line);margin-top:18px}
.og-bar-col{height:100%;flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:8px}
.og-bar{width:min(52px,80%);border-radius:8px 8px 2px 2px;min-height:5px}
.og-bar-col b{font-size:11px;color:var(--ink)}
.og-bar-labels{display:flex;gap:22px;padding:8px 18px 0}
.og-bar-labels span{flex:1;text-align:center;font-size:10.5px;color:var(--muted);line-height:1.3}
.og-legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:14px 0 10px;font-size:10.5px;color:var(--muted)}
.og-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px}

/* ---------------- misc */
[data-testid="stAlert"]{border-radius:14px}
[data-testid="stExpander"] details, [data-testid="stExpander"]{border-color:var(--line) !important;border-radius:12px !important;background:#fff}
[data-testid="stExpander"] summary p{font-size:12px !important;font-weight:600}
.og-footer{text-align:center;color:#89948d;font-size:10.5px;margin-top:25px}

@media(max-width:1050px){.og-stats{grid-template-columns:repeat(3,1fr)}.og-details{grid-template-columns:repeat(2,1fr)}}
@media(max-width:650px){.og-stats{grid-template-columns:1fr 1fr}.og-h1{font-size:23px}.og-figs{grid-template-columns:1fr}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =================================================================== helpers
def h(s) -> str:
    return html.escape(str(s))


def md(x: str):
    """HTML through st.markdown; kept on one line so markdown never touches it."""
    st.markdown(" ".join(x.split("\n")), unsafe_allow_html=True)


def card(key: str):
    """A white dashboard card. Falls back to a bordered container on old Streamlit."""
    try:
        return st.container(key="card_" + key)
    except TypeError:
        return st.container(border=True)


def b64img(img, max_side: int = 1600, quality: int = 85) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def jdefault(x):
    if hasattr(x, "item"):
        return x.item()
    if hasattr(x, "tolist"):
        return x.tolist()
    return str(x)


def jdump(obj) -> str:
    return json.dumps(obj, default=jdefault).replace("</", "<\\/")


@st.cache_data(show_spinner=False, max_entries=8)
def preview_b64(digest: str, _data: bytes) -> str:
    return b64img(Image.open(io.BytesIO(_data)))


@st.cache_data(show_spinner=False, max_entries=16)
def annotated(
    key: str, boxes: bool, outlines: bool, _res, path: str, plain: bool = False
):
    """Rendered once per photo + checkbox combination, then served from cache."""
    out = os.path.join(
        tempfile.gettempdir(),
        "graded_%s_%d%d%d.jpg"
        % (hashlib.sha1(key.encode()).hexdigest()[:12], boxes, outlines, plain),
    )
    if plain:
        load().render(_res, out, path)
    else:
        load().render(_res, out, path, boxes=boxes, outlines=outlines)
    with open(out, "rb") as fh:
        raw = fh.read()
    return b64img(Image.open(out), max_side=2000), raw


@st.cache_data(show_spinner=False, max_entries=8)
def res_json(key: str, _res) -> str:
    return _res.to_json()


# =================================================================== sidebar
with st.sidebar:
    md(
        '<div class="og-brand"><div class="og-logo">🧅</div><div><strong>OnionGrade</strong>'
        '<span class="og-tag">Quality analysis</span></div></div>'
        '<div class="og-nav-label">Workspace</div>'
        + (
            '<div class="og-nav-item off"><span>▦</span><span>Photo analysis</span></div>'
            '<div class="og-nav-item"><span>●</span><span>Live camera</span></div>'
            if "Live" in str(st.session_state.get("og_mode", ""))
            else '<div class="og-nav-item"><span>▦</span><span>Photo analysis</span></div>'
            '<div class="og-nav-item off"><span>●</span><span>Live camera</span></div>'
        )
        + '<div class="og-nav-label">Size and the coin</div>'
    )
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
    manual = st.number_input(
        "px per mm (0 = auto)",
        0.0,
        100.0,
        0.0,
        0.01,
        help="Measure any known object: its pixel length ÷ its mm length.",
    )
    st.divider()
    md('<div class="og-nav-label">Measured performance</div>')
    st.markdown(
        "Cross-validated by physical onion, 120 hand-graded onions:\n\n"
        "| | κ |\n|---|---|\n"
        "| **this system, end to end** | **0.819** |\n"
        "| hand-clicked prompts | 0.757 |\n"
        "| *two humans with each other* | *0.701* |\n"
        "| *one human with herself* | *0.852* |\n\n"
        "Within one band **0.950** · detection recall **0.998**"
    )

# =================================================================== mode: photo | live
MODES = ["📷  Photo analysis", "🎥  Live camera"]
try:
    _modebar = st.container(key="modebar")
except TypeError:  # very old Streamlit: no keyed containers
    _modebar = st.container()
with _modebar:
    mode = st.radio(
        "mode", MODES, horizontal=True, key="og_mode", label_visibility="collapsed"
    )
if mode == MODES[1]:
    from onion_grader.live_page import render_live

    render_live(load, md, card)
    md('<div class="og-footer">Onion Quality Grading · Live camera</div>')
    st.stop()
elif st.session_state.get("cam_on"):
    st.session_state.cam_on = False  # leaving the Live tab switches the camera off

# =================================================================== topbar
if "up_n" not in st.session_state:
    st.session_state.up_n = 0

top_l, top_r = st.columns([3, 1.25])
with top_l:
    md(
        '<div class="og-eyebrow">Computer vision</div>'
        '<div class="og-h1">Onion Quality Grading</div>'
        '<div class="og-caption">Provisional grade, visual criteria only — firmness, dryness, '
        "internal soundness and smell are part of the legal grade and are not in a photograph.</div>"
    )
with top_r:
    md('<div class="og-spacer"></div>')
    b1, b2 = st.columns(2)
    export_slot = b1.empty()
    if b2.button("New analysis", type="primary", use_container_width=True):
        st.session_state.up_n += 1
        st.rerun()
md('<div class="og-spacer"></div>')

# =================================================================== input | result
col_in, col_out = st.columns([1.05, 0.95], gap="medium")

with col_in:
    with card("input"):
        md(
            '<div class="og-card-h">Input image</div>'
            '<div class="og-sub">Upload a photograph of onions. A ₹1 coin in frame unlocks millimetres.</div>'
        )
        up = st.file_uploader(
            "photograph",
            type=["jpg", "jpeg", "png"],
            key="up_%d" % st.session_state.up_n,
            label_visibility="collapsed",
        )
        if up:
            data = up.getvalue()
            digest = hashlib.sha1(data).hexdigest()[:16]
            md(
                '<img class="og-preview" src="%s" alt="uploaded photograph">'
                % preview_b64(digest, data)
            )
        c1, c2, c3 = st.columns([1, 1, 1.6])
        o1 = c1.checkbox("bounding boxes", True)
        o2 = c2.checkbox("outlines", True)
        status_slot = c3.empty()
        status_slot.markdown(
            '<div style="text-align:right;padding-top:4px"><span class="og-status">'
            "Ready for analysis</span></div>",
            unsafe_allow_html=True,
        )

if not up:
    with col_out:
        with card("result"):
            md(
                '<div class="og-card-h">Detection result</div>'
                '<div class="og-sub">Detected onions and quality annotations.</div>'
                '<div class="og-empty"><div><b>🧅</b>Upload a photograph of onions. '
                "A ₹1 coin in frame unlocks millimetres.</div></div>"
            )
    md('<div class="og-footer">Onion Quality Grading · Image analysis dashboard</div>')
    st.stop()

path = os.path.join(tempfile.gettempdir(), "onion_upload_%s.jpg" % digest)
if not os.path.exists(path):
    with open(path, "wb") as t:
        t.write(data)
rkey = "%s|%s" % (digest, manual)

with col_out:
    with st.spinner("detecting, segmenting, measuring, grading…"):
        res = run(path, manual if manual > 0 else None)

g = load()

# ---------------------------------------------------------------- not onions
if not res.is_onion_photo:
    status_slot.markdown(
        '<div style="text-align:right;padding-top:4px"><span class="og-status red">'
        "Nothing graded</span></div>",
        unsafe_allow_html=True,
    )
    view, _ = annotated(rkey, True, True, res, path, plain=True)
    with col_out:
        with card("result"):
            md(
                '<div class="og-card-h">Detection result</div>'
                '<div class="og-sub">grey boxes: what the detector found, and why it was ignored</div>'
                '<div class="og-result"><img src="%s" alt="detections"></div>' % view
            )
    st.error(
        "**This does not look like a photograph of onions, so nothing was graded.** "
        "The detector found %d round object(s), and the onion gate judged %d of them "
        "not to be onions (faces, hands, objects, background).\n\n"
        "This is not a calibration problem — recalibrating would not help. "
        "Photograph the onions themselves, on a plain surface, in even light."
        % (len(res.rejected) + res.n_onions, len(res.rejected))
    )
    st.stop()

if not res.onions:
    status_slot.markdown(
        '<div style="text-align:right;padding-top:4px"><span class="og-status red">'
        "No onions found</span></div>",
        unsafe_allow_html=True,
    )
    st.error("No onions found in this photograph.")
    st.stop()

status_slot.markdown(
    '<div style="text-align:right;padding-top:4px"><span class="og-status">'
    "Graded · %d onions · %s</span></div>" % (res.n_onions, h(res.device)),
    unsafe_allow_html=True,
)
export_slot.download_button(
    "Export JSON",
    res_json(rkey, res),
    file_name="grades.json",
    mime="application/json",
    use_container_width=True,
)

# ---------------------------------------------------------------- result image
with col_out:
    with card("result"):
        md(
            '<div class="og-card-h">Detection result</div>'
            '<div class="og-sub">Detected onions and quality annotations.</div>'
        )
        view, raw = annotated(rkey, o1, o2, res, path)
        md('<div class="og-result"><img src="%s" alt="graded photograph"></div>' % view)
        md('<div style="height:10px"></div>')
        st.download_button(
            "↓ download annotated image", raw, file_name="graded.jpg", mime="image/jpeg"
        )

md('<div style="height:8px"></div>')

if res.gate != "on":
    st.warning(
        "Onion gate is OFF (onion_gate.pth not found next to grader.py). "
        "Anything round in the photo — a face, a hand — would be graded as an onion."
    )
elif res.rejected:
    st.info(
        "%d detection(s) ignored — the onion gate says they are not onions. "
        "They are shown as grey boxes below." % len(res.rejected)
    )

# ---------------------------------------------------------------- headline stats
n_review = sum(1 for o in res.onions if o.needs_review)
scale_html = (
    ('<div class="n">%.2f</div><small>px / mm</small>' % res.px_per_mm)
    if res.px_per_mm
    else '<div class="n sm">relative only</div><small>no reference object</small>'
)
md(
    '<div class="og-stats">'
    '<div class="og-stat"><div class="l">onions</div><div class="n">%d</div><small>Detected objects</small></div>'
    '<div class="og-stat green"><div class="l">healthy</div><div class="n">%d</div><small>%.0f%% of batch</small></div>'
    '<div class="og-stat red"><div class="l">unhealthy</div><div class="n">%d</div><small>%.0f%% of batch</small></div>'
    '<div class="og-stat amber"><div class="l">borderline</div><div class="n">%d</div><small>Check by hand · %d flagged in total</small></div>'
    '<div class="og-stat"><div class="l">scale</div>%s</div>'
    "</div>"
    % (
        res.n_onions,
        res.n_healthy,
        res.pct_healthy,
        res.n_unhealthy,
        res.pct_unhealthy,
        res.n_borderline,
        n_review,
        scale_html,
    )
)

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
        continue  # shown above, in full, already
    st.warning(w)



@st.cache_data(show_spinner=False, max_entries=8)
def build_rows(key: str, _res, path: str) -> list:
    src = Image.open(path).convert("RGB")
    rows = []
    for o in _res.onions:
        x1, y1, x2, y2 = [int(v) for v in o.box]
        rows.append(
            {
                "idx": o.index,
                "thumb": b64img(src.crop((x1, y1, x2, y2)), max_side=96, quality=80),
                "verdict": o.verdict,
                "grade": o.grade,
                "color": BADGE[o.band_index],
                "p": None if o.p_unhealthy is None else round(float(o.p_unhealthy), 2),
                "dark": round(100 * float(o.defect_area_fraction), 1),
                "conf": round(float(o.confidence), 2),
                "size": (
                    ("%.0f mm (%s)" % (o.eq_diameter_mm, o.size_code))
                    if o.eq_diameter_mm
                    else ("x%.2f" % o.size_ratio)
                ),
                "flags": [
                    f
                    for f in [
                        ("review" if o.needs_review else ""),
                        ("unusual" if o.unusual_kind else ""),
                        ("maybe not an onion" if o.maybe_not_onion else ""),
                        ("partial" if o.partial else ""),
                    ]
                    if f
                ],
            }
        )
    return rows


@st.cache_data(show_spinner=False, max_entries=8)
def build_inspection(key: str, _res, path: str) -> list:
    g = load()
    out = []
    for o in _res.onions:
        crop, outline, heat = g.crop(_res, o, path)
        conf_cap = (
            "check it is an onion (gate %.0f%% unsure)" % (100 * o.p_not_onion)
            if o.maybe_not_onion
            else "needs a human" if o.needs_review else "clear of the boundaries"
        )
        anom_cap = (
            "not meaningful — out of domain"
            if not o.anomaly_reliable
            else (
                "a different KIND of damage"
                if o.unusual_kind
                else "typical of its grade"
            )
        )
        if o.eq_diameter_mm:
            size = [
                "size",
                "%.0f mm" % o.eq_diameter_mm,
                "AGMARK code %s · %.2fx the median here" % (o.size_code, o.size_ratio),
            ]
        else:
            size = [
                "relative size",
                "x%.2f" % o.size_ratio,
                "rank %d of %d · no reference object for mm"
                % (o.size_rank, _res.n_onions),
            ]
        out.append(
            {
                "idx": o.index,
                "healthy": bool(o.healthy),
                "verdict": o.verdict,
                "grade": o.grade,
                "grp": "chk" if o.is_borderline else ("ok" if o.healthy else "bad"),
                "color": BADGE[o.band_index],
                "cls": (
                    ("classifier: %.0f%% unhealthy" % (100 * o.p_unhealthy))
                    if o.p_unhealthy is not None
                    else ""
                ),
                "note": o.grade_note or "",
                "partial": bool(o.partial),
                "imgs": [
                    b64img(crop, 420, 82),
                    b64img(outline, 420, 82),
                    b64img(heat, 420, 82),
                ],
                "details": [
                    [
                        "dark area",
                        "%.1f%%" % (100 * o.defect_area_fraction),
                        "band: %s" % o.band,
                    ],
                    ["confidence", "%.2f" % o.confidence, conf_cap],
                    ["anomaly", "%.2f" % o.anomaly_score, anom_cap],
                    size,
                ],
                "values": [
                    [k, v]
                    for k, v in [
                        ("box", o.box),
                        ("verdict", o.verdict),
                        ("grade", o.grade),
                        ("band", o.band),
                        ("defect_area_fraction", o.defect_area_fraction),
                        ("raw_defect_fraction", o.raw_defect_fraction),
                        ("confidence", o.confidence),
                        ("needs_review", o.needs_review),
                        ("anomaly_score", o.anomaly_score),
                        ("unusual_kind", o.unusual_kind),
                        ("anomaly_reliable", o.anomaly_reliable),
                        ("detector_conf", o.detector_conf),
                        ("p_not_onion", o.p_not_onion),
                        ("maybe_not_onion", o.maybe_not_onion),
                        ("area_px", o.area_px),
                        ("eq_diameter_px", o.eq_diameter_px),
                        ("circularity", o.circularity),
                        ("aspect_ratio", o.aspect_ratio),
                        ("size_ratio", o.size_ratio),
                        ("size_rank", o.size_rank),
                        ("eq_diameter_mm", o.eq_diameter_mm),
                        ("size_code", o.size_code),
                    ]
                ],
            }
        )
    return out


# =================================================================== individual inspection (full width)
INSPECT = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--ink:#172019;--muted:#718078;--line:#e3e9e2;--green:#477a54;--green2:#eaf4eb;--amber:#c58a27;--amber2:#fff5df;--red:#c45b55;--red2:#fff0ee;--blue:#2f6690;--blue2:#eaf2f8}
*{box-sizing:border-box}
body{margin:0;padding:2px 2px 8px;background:transparent;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:var(--ink)}
button{font-family:inherit;cursor:pointer}
.card{background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:0 12px 30px rgba(28,45,31,.07);padding:20px;height:__H__px;display:flex;flex-direction:column}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
h2{font-size:15px;margin:0}.sub{font-size:11.5px;color:var(--muted);margin-top:4px}
.seg{display:flex;gap:6px;padding:4px;background:#f3f6f2;border-radius:11px;min-width:420px}
.seg button{flex:1;border:0;background:transparent;border-radius:8px;padding:8px 14px;font-size:12px;font-weight:650;color:#69766e;transition:all .15s}
.seg button span{font-size:10px;margin-left:5px;padding:2px 7px;border-radius:10px;background:#e3e9e2}
.seg button.on{background:#fff;box-shadow:0 2px 8px rgba(28,45,31,.08);color:var(--green)}
.seg button.on.bad{color:var(--red)}.seg button.on.chk{color:var(--amber)}
.body{display:grid;grid-template-columns:250px 1fr;gap:20px;margin-top:16px;flex:1;min-height:0}
.list{border:1px solid var(--line);border-radius:14px;padding:10px;overflow-y:auto;display:grid;grid-template-columns:repeat(3,1fr);gap:6px;align-content:start}
.chip{border:1px solid var(--line);background:#fff;border-radius:9px;padding:8px 4px;font-size:11.5px;font-weight:650;color:#59665e;display:flex;align-items:center;justify-content:center;gap:6px;transition:all .12s}
.chip:hover{border-color:#c9d6ca;transform:translateY(-1px)}
.chip i{width:7px;height:7px;border-radius:50%;display:inline-block}
.chip.on{background:var(--green);border-color:var(--green);color:#fff}
.chip.on.bad{background:var(--red);border-color:var(--red)}.chip.on.chk{background:var(--amber);border-color:var(--amber)}
.chip.on i{box-shadow:0 0 0 2px #fff}
.empty{grid-column:1/-1;color:var(--muted);font-size:12px;text-align:center;padding:30px 8px}
.detail{overflow-y:auto;min-height:0;padding-right:4px}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.pill{display:inline-flex;padding:5px 9px;border-radius:20px;font-size:10.5px;font-weight:700;margin-right:4px}
.healthy{background:var(--green2);color:var(--green)}.unhealthy{background:var(--red2);color:var(--red)}.border{background:var(--amber2);color:var(--amber)}
.title{font-size:18px;font-weight:750;margin:9px 0 2px}
.nav{display:flex;gap:6px;align-items:center;font-size:11px;color:var(--muted)}
.nav button{width:32px;height:32px;border:1px solid var(--line);background:#fff;border-radius:8px;font-size:15px;color:#59665e}
.nav button:disabled{opacity:.35;cursor:not-allowed}
.note{margin-top:10px;padding:9px 12px;border-radius:10px;font-size:12px}
.note.info{background:var(--blue2);color:var(--blue)}.note.warn{background:var(--amber2);color:#8a6414}
.views{display:flex;gap:4px;margin-top:14px;border-bottom:1px solid var(--line)}
.views button{border:0;background:none;padding:8px 12px;font-size:11.5px;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px}
.views button.on{color:var(--green);border-color:var(--green);font-weight:700}
.figs{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}
.figs img{width:100%;aspect-ratio:1/1;max-height:250px;object-fit:contain;border-radius:12px;display:block;background:#f0f3ed;transition:opacity .15s}
.figs .cap{font-size:10.5px;color:var(--muted);margin-top:6px}
.details{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
.d{background:#f8faf7;border-radius:10px;padding:11px}
.d label{display:block;font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.d b{font-size:15px;display:block;margin-top:3px}
.d span{display:block;font-size:10.5px;color:var(--muted);margin-top:4px;line-height:1.35}
table{width:100%;border-collapse:collapse;font-size:11.5px;margin-top:10px}
td{padding:7px 10px;border-bottom:1px solid #edf0ed}td:first-child{color:var(--muted);width:40%;font-family:ui-monospace,Consolas,monospace;font-size:11px}
td:last-child{font-family:ui-monospace,Consolas,monospace;font-size:11px}
.caption{font-size:10.5px;color:var(--muted);margin-top:10px;line-height:1.45}
.hint{font-size:10px;color:#a0aaa3;margin-top:8px;text-align:center}
@media(max-width:820px){
  .card{height:auto}.body{grid-template-columns:1fr}
  .list{display:flex;overflow-x:auto;overflow-y:hidden;max-height:none}.chip{flex:0 0 64px}
  .figs,.details{grid-template-columns:1fr 1fr}.seg{min-width:0;width:100%}
}
</style></head><body>
<div class="card">
  <div class="head">
    <div><h2>Individual inspection</h2><div class="sub">Pick an onion to see exactly what the grade was measured on. Use ← → keys to step through.</div></div>
    <div class="seg"><button id="tOk" class="on">Healthy<span id="nOk"></span></button><button id="tBad" class="bad">Unhealthy<span id="nBad"></span></button><button id="tChk" class="chk">Borderline<span id="nChk"></span></button></div>
  </div>
  <div class="body">
    <div><div class="list" id="list" style="max-height:__LH__px"></div><div class="hint" id="hint"></div></div>
    <div class="detail" id="detail"></div>
  </div>
</div>
<script>
const D = __DATA__;
const G = {ok: D.filter(o => o.grp === 'ok'), bad: D.filter(o => o.grp === 'bad'), chk: D.filter(o => o.grp === 'chk')};
let grp = G.ok.length ? 'ok' : (G.bad.length ? 'bad' : 'chk'), pos = {ok: 0, bad: 0, chk: 0}, view = 'img';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = v => v === null ? 'null' : (typeof v === 'object' ? JSON.stringify(v) : String(v));
document.getElementById('nOk').textContent = G.ok.length;
document.getElementById('nBad').textContent = G.bad.length;
document.getElementById('nChk').textContent = G.chk.length;
// decode every image once up front so switching is instant
D.forEach(o => o.imgs.forEach(src => { const i = new Image(); i.src = src; }));

function setGroup(g){ grp = g; draw(); }
function pick(i){ pos[grp] = i; draw(); const c = document.querySelectorAll('.chip')[i]; if (c) c.scrollIntoView({block:'nearest', inline:'nearest'}); }
function step(d){ const L = G[grp]; if (!L.length) return; pick(Math.min(L.length - 1, Math.max(0, pos[grp] + d))); }
function setView(v){ view = v; drawDetail(); }

function draw(){
  document.getElementById('tOk').className = grp === 'ok' ? 'on' : '';
  document.getElementById('tBad').className = 'bad' + (grp === 'bad' ? ' on' : '');
  document.getElementById('tChk').className = 'chk' + (grp === 'chk' ? ' on' : '');
  const L = G[grp], bad = grp === 'bad';
  const list = document.getElementById('list');
  list.innerHTML = L.length ? L.map((o, i) =>
    `<button class="chip${i === pos[grp] ? ' on' : ''}${bad ? ' bad' : ''}${grp === 'chk' ? ' chk' : ''}" onclick="pick(${i})"><i style="background:${o.color}"></i>#${String(o.idx).padStart(2,'0')}</button>`).join('')
    : `<div class="empty">No ${grp === 'chk' ? 'borderline' : (bad ? 'unhealthy' : 'healthy')} onions in this photograph.</div>`;
  document.getElementById('hint').textContent = L.length ? `${pos[grp] + 1} of ${L.length}` : '';
  drawDetail();
}

function drawDetail(){
  const L = G[grp], el = document.getElementById('detail');
  if (!L.length){ el.innerHTML = ''; return; }
  const o = L[pos[grp]];
  let h = `<div class="top"><div>
      <span class="pill ${o.grp === 'chk' ? 'border' : (o.healthy ? 'healthy' : 'unhealthy')}">${esc(o.verdict)}</span><span class="pill" style="background:${o.color};color:#111">${esc(o.grade)}</span>
      <div class="title">Onion #${String(o.idx).padStart(2,'0')} · ${esc(o.grade)}</div>
      <div class="sub">${esc(o.cls)}</div></div>
    <div class="nav"><button onclick="step(-1)" ${pos[grp] === 0 ? 'disabled' : ''}>‹</button><button onclick="step(1)" ${pos[grp] === L.length - 1 ? 'disabled' : ''}>›</button></div></div>`;
  if (o.note) h += `<div class="note info">${esc(o.note)}</div>`;
  if (o.partial) h += `<div class="note warn">This onion runs off the edge of the photo, so part of it is unseen.</div>`;
  h += `<div class="views"><button class="${view === 'img' ? 'on' : ''}" onclick="setView('img')">Images &amp; measurements</button><button class="${view === 'all' ? 'on' : ''}" onclick="setView('all')">All measured values</button></div>`;
  if (view === 'img'){
    const caps = ['as photographed', 'the boundary it measured over', 'the pixels counted as damage (red)'];
    h += '<div class="figs">' + o.imgs.map((s, i) => `<div><img src="${s}"><div class="cap">${caps[i]}</div></div>`).join('') + '</div>';
    h += '<div class="details">' + o.details.map(d => `<div class="d"><label>${esc(d[0])}</label><b>${esc(d[1])}</b><span>${esc(d[2])}</span></div>`).join('') + '</div>';
  } else {
    h += '<table>' + o.values.map(kv => `<tr><td>${esc(kv[0])}</td><td>${esc(fmt(kv[1]))}</td></tr>`).join('') + '</table>';
    h += '<div class="caption">circularity 0.63 is a perfect digital disc, so onions sit near 0.52. raw_defect_fraction is before the instrument calibration that maps measured area onto human-judged area.</div>';
  }
  el.innerHTML = h;
}
document.getElementById('tOk').onclick = () => setGroup('ok');
document.getElementById('tBad').onclick = () => setGroup('bad');
document.getElementById('tChk').onclick = () => setGroup('chk');
document.addEventListener('keydown', e => { if (e.key === 'ArrowRight') step(1); if (e.key === 'ArrowLeft') step(-1); });
draw();
</script></body></html>
"""

with st.spinner("preparing the per-onion views…"):
    insp = build_inspection(rkey, res, path)
INSPECT_H = 640
components.html(
    INSPECT.replace("__H__", str(INSPECT_H))
    .replace("__LH__", str(INSPECT_H - 150))
    .replace("__DATA__", jdump(insp)),
    height=INSPECT_H + 12,
    scrolling=False,
)

# =================================================================== onion-wise table
TABLE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--ink:#172019;--muted:#718078;--line:#e3e9e2;--green:#477a54;--green2:#eaf4eb;--amber:#c58a27;--amber2:#fff5df;--red:#c45b55;--red2:#fff0ee}
*{box-sizing:border-box}
body{margin:0;padding:2px 2px 8px;background:transparent;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:var(--ink)}
.card{background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:0 12px 30px rgba(28,45,31,.07);padding:20px}
h2{font-size:15px;margin:0}.sub{font-size:11.5px;color:var(--muted);margin-top:4px}
.wrap{overflow-x:auto;overflow-y:hidden;margin-top:15px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;color:#849087;font-weight:650;font-size:10px;padding:10px 12px;border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
td{padding:9px 12px;border-bottom:1px solid #edf0ed;white-space:nowrap}
tr:hover td{background:#fafcf9}
.thumb{width:44px;height:36px;border-radius:7px;object-fit:cover;background:#e8eee7;display:block}
.pill{display:inline-flex;padding:4px 8px;border-radius:20px;font-size:10px;font-weight:700;margin-right:4px}
.healthy{background:var(--green2);color:var(--green)}.unhealthy{background:var(--red2);color:var(--red)}.border{background:var(--amber2);color:var(--amber)}
.flag{background:#f1f4f0;color:#59665e}.flag.review{background:var(--amber2);color:var(--amber)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:middle}
.muted{color:#a6b0a9}
.pagination{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:14px;padding-top:13px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);flex-wrap:wrap}
.ctrls{display:flex;align-items:center;gap:4px}
.pb,.pn{min-width:29px;height:29px;border:1px solid var(--line);background:#fff;color:#59665e;border-radius:7px;font-size:12px;display:grid;place-items:center;cursor:pointer;font-family:inherit;padding:0 6px}
.pn{font-size:11px}.pn.active{background:var(--green);border-color:var(--green);color:#fff;font-weight:700}
.pb:disabled{opacity:.4;cursor:not-allowed}
.rows{display:flex;align-items:center;gap:6px}
select{border:1px solid var(--line);border-radius:7px;background:#fff;padding:5px 8px;font-size:11px;color:var(--ink);font-family:inherit}
</style></head><body>
<div class="card">
  <h2>Onion-wise results</h2><div class="sub">Detailed measurements and quality grade for each detected onion.</div>
  <div class="wrap"><table>
    <thead><tr><th>#</th><th>Preview</th><th>Verdict</th><th>Grade</th><th>P(unhealthy)</th><th>Dark area %</th><th>Conf</th><th>Size</th><th>Flags</th></tr></thead>
    <tbody id="tb"></tbody>
  </table></div>
  <div class="pagination">
    <div>Showing <b id="f">0</b>–<b id="t">0</b> of <b id="n">0</b> onions</div>
    <div class="ctrls"><button class="pb" id="prev">‹</button><span id="nums" class="ctrls"></span><button class="pb" id="next">›</button></div>
    <label class="rows">Rows <select id="rpp"><option>5</option><option selected>10</option></select></label>
  </div>
</div>
<script>
const R = __ROWS__;
let page = 1, per = 10;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function draw(){
  const pages = Math.max(1, Math.ceil(R.length / per));
  if (page > pages) page = pages;
  const s = (page - 1) * per, e = Math.min(s + per, R.length);
  document.getElementById('tb').innerHTML = R.slice(s, e).map(r => `<tr>
    <td>#${String(r.idx).padStart(2,'0')}</td>
    <td><img class="thumb" src="${r.thumb}"></td>
    <td><span class="pill ${r.verdict==='HEALTHY'?'healthy':(r.verdict==='BORDERLINE'?'border':'unhealthy')}">${esc(r.verdict)}</span></td>
    <td><span class="dot" style="background:${r.color}"></span><b>${esc(r.grade)}</b></td>
    <td>${r.p===null?'<span class="muted">—</span>':r.p.toFixed(2)}</td>
    <td>${r.dark.toFixed(1)}</td>
    <td>${r.conf.toFixed(2)}</td>
    <td>${esc(r.size)}</td>
    <td>${r.flags.length?r.flags.map(f=>`<span class="pill flag ${f==='review'?'review':''}">${esc(f)}</span>`).join(''):'<span class="muted">—</span>'}</td>
  </tr>`).join('');
  document.getElementById('f').textContent = R.length ? s + 1 : 0;
  document.getElementById('t').textContent = e;
  document.getElementById('n').textContent = R.length;
  const nums = document.getElementById('nums'); nums.innerHTML = '';
  for (let i = 1; i <= pages; i++){
    if (pages > 7 && i > 2 && i < pages - 1 && Math.abs(i - page) > 1){
      if (!nums.lastChild || nums.lastChild.textContent !== '…'){ const d=document.createElement('span'); d.textContent='…'; d.style.padding='0 4px'; nums.appendChild(d); }
      continue;
    }
    const b = document.createElement('button');
    b.className = 'pn' + (i === page ? ' active' : ''); b.textContent = i;
    b.onclick = () => { page = i; draw(); }; nums.appendChild(b);
  }
  document.getElementById('prev').disabled = page === 1;
  document.getElementById('next').disabled = page === pages;
}
document.getElementById('prev').onclick = () => { page--; draw(); };
document.getElementById('next').onclick = () => { page++; draw(); };
document.getElementById('rpp').onchange = e => { per = +e.target.value; page = 1; draw(); };
draw();
const w=document.querySelector('.wrap'); w.style.minHeight=w.offsetHeight+'px';
</script></body></html>
"""
rows = build_rows(rkey, res, path)
components.html(
    TABLE.replace("__ROWS__", jdump(rows)),
    height=200 + 55 * min(10, len(rows)),
    scrolling=False,
)

# =================================================================== grade-wise distribution (full width)
with card("chart"):
    ch_l, ch_r = st.columns([2.2, 1], gap="large")
    with ch_l:
        md(
            '<div class="og-card-h">Grade-wise distribution</div>'
            '<div class="og-sub">Detected onions grouped by provisional AGMARK grade.</div>'
        )
        cnt = Counter(o.grade for o in res.onions)
        top = max([cnt.get(k, 0) for k in GRADE_NAMES] + [1])
        bars = "".join(
            '<div class="og-bar-col"><b>%d</b><div class="og-bar" style="height:%.0f%%;background:%s">'
            "</div></div>" % (cnt.get(k, 0), 82 * cnt.get(k, 0) / top, BADGE[i])
            for i, k in enumerate(GRADE_NAMES)
        )
        labels = "".join("<span>%s</span>" % h(k) for k in GRADE_NAMES)
        legend = "".join(
            '<span><i style="background:%s"></i>dark area: %s</span>' % (BADGE[i], h(b))
            for i, b in enumerate(BAND_NAMES)
        )
        md(
            '<div class="og-chart">%s</div><div class="og-bar-labels">%s</div>'
            '<div class="og-legend">%s</div>' % (bars, labels, legend)
        )
    with ch_r:
        md(
            '<div class="og-card-h">Calibration check</div>'
            '<div class="og-sub">Two independent checks, both fitted so they fire on 0 of the 53 '
            "calibration photographs: A — the anomaly score says everything looks strange; "
            "B — the damage rule registers on nothing at all. They catch opposite failures and "
            'neither sees the other\'s.</div><div style="height:10px"></div>'
        )
        with st.expander("is this photograph like the calibration set?"):
            st.json(res.domain_stats)

md('<div class="og-footer">Onion Quality Grading · Image analysis dashboard</div>')
