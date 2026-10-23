# Onion grader — the working system

Give it a photograph. It finds every onion, measures how much of each is
damaged, assigns a statutory grade with a confidence, flags the ones that need a
human, and reports size in millimetres if a coin is in frame.

**No clicking.** Everything in Phases 9–13 started from a hand-placed point per
onion. This does not.

```
cd "G:\Quality test\onion-quality-assessment\onion-quality-assessment"

python -m onion_grader.grader Onion_mix/IMG-20260915-WA0017.jpg ^
       --out graded.jpg --json grades.json
```

or the app:

```
streamlit run onion_grader/app.py
```

or **live from a camera**:

```
python -m onion_grader.live                                    # laptop camera
python -m onion_grader.live --source http://192.168.1.7:8080/video   # phone
python -m onion_grader.live --source clip.mp4 --out graded.mp4 --no-show
```

`q` quit · `g` grade now · `s` snapshot · `space` pause.
For a phone, install any IP-camera app and pass its `http://.../video` URL, with
the phone and the computer on the same wi-fi. Needs `pip install lap`.

**It runs two loops, because grading cannot run per frame.** YOLO tracks every
frame and gives each onion a stable id; the full grade runs every few seconds and
attaches to those ids, so a grade follows the right onion as it moves. The header
says how many seconds old the grades are. Ungraded onions are grey.

**Live grading is an interface onto this instrument, not a validated mode of it.**
Every accuracy figure below was measured on still photographs; motion blur and
changing angle are in none of them.

or from Python:

```python
from onion_grader.grader import OnionGrader
g = OnionGrader()
res = g.grade("tray.jpg")
print(res.summary())
for o in res.onions:
    print(o.index, o.grade, o.defect_area_fraction, o.confidence, o.size_code)
g.render(res, "graded.jpg", "tray.jpg")
```

---

## What it outputs, per onion

| field | |
|---|---|
| `grade` | Extra Class eligible · Class I · Class II · Below minimum |
| `band` | none · up to ⅕ · ⅕ to ½ · over ½ |
| `defect_area_fraction` | 0–1, calibrated to human judgement |
| `confidence` | 0–1, distance to the nearest grade boundary |
| `needs_review` | confidence < 0.5 — the onion sits near a line |
| `anomaly_score`, `unusual_kind` | a *different kind* of damage, not just more of it |
| `anomaly_reliable` | false when the photograph is out of the calibrated domain |
| `verdict` | **HEALTHY** / **UNHEALTHY** (any defect above the detection limit) |
| `size_ratio`, `size_rank` | relative size — **always available, no coin needed** |
| `eq_diameter_mm`, `size_code` | A/B/C/D, only with a reference object |
| `box`, `area_px`, `circularity`, `aspect_ratio` | geometry |

Plus, for the photograph as a whole: `n_onions`, `n_healthy`, `n_unhealthy`,
`pct_healthy`, `pct_unhealthy`, `px_per_mm`, `scale_source`, `warnings`,
`domain_ok` and `domain_stats`.

The annotated image carries a header bar with the count and the healthy /
unhealthy split, a **bounding box plus the measured outline** on every onion, and
a per-onion label reading `#3 UNHEALTHY 34%`.

The app has an **"Each onion" tab**: pick any onion and see it three ways — as
photographed, the boundary the grade was measured over, and the exact pixels
counted as damage — beside every measured value.

---

## How well it works

Measured on 120 hand-graded onions, **calibration cross-validated by physical
onion** so no onion is graded by a curve that saw it:

| | quadratic κ | exact band | within one band |
|---|---|---|---|
| **this system, end to end** | **0.819** *[0.724, 0.894]* | 0.617 | **0.950** |
| hand-clicked prompts (Phase 12) | 0.757 | 0.617 | 0.925 |
| *two humans with each other* | *0.701* | *0.542* | *0.858* |
| *one human with herself* | *0.852* | *0.600* | — |

**It grades closer to a human than a second human does.** The end-to-end system
beats the hand-clicked pipeline because the detector gives SAM a *box*, and a box
is a less ambiguous prompt than a point.

**Detection:** recall **0.998** (537 of 538 onions found), precision 0.968 after
the coin rule.

**Counting** is the weakest link and always over-counts, never under-counts —
`within ±1` on 72% of photographs before the coin rule.

---

## It checks that each box is an onion before grading it

The detector was trained only on onion photographs, so it never learned what an
onion is *not*. Given a screenshot of people it drew boxes on faces and a lamp,
and the rest of the pipeline graded them ("#3 UNHEALTHY, Class II, 24.3%"). The
domain guard below cannot catch that: it asks whether the onions look like the
calibration onions, not whether they are onions at all.

So every detection now passes through an **onion gate** (`gate.py`) before SAM
and grading. It is the project's own EfficientNet-B0 v3 classifier
(`onion_gate.pth`), trained on 50k crops that included a *Not Onion* class
(people, hands, chairs, bottles, fruit, background). Only its *not onion*
probability is used; grading is unchanged.

| p(not onion) | what happens |
|---|---|
| **≥ 0.8** | box ignored, drawn grey as "not an onion" |
| 0.5 – 0.8 | graded, but flagged `maybe_not_onion` and sent for review; dropped if the photo is also out of domain |
| box covers a **human face** | box ignored, drawn grey as "face - not an onion" (OpenCV YuNet, `face_yunet.onnx`) |
| more than half the boxes ignored | **whole photo refused**: "not a photograph of onions", nothing graded |

Measured here: on 50 tray photographs, **436 of 438 grades are identical** with
and without the gate. The 2 it removed were patches of bare floor that used to be
graded "Extra Class". No real onion scored above 0.55. The screenshot from the
bug report is refused (all 4 boxes scored 0.98–0.99).

Faces get their own detector because the classifier misses them: in a market
photo it scored a vendor's face only 12% "not onion". YuNet scores real faces
0.88–0.92 and never went above 0.86 on an onion in the 50 trays, so faces at 0.9+
are trusted outright; 0.8–0.9 only when the classifier agrees. The tray grades
were unchanged with it switched on.

It is not perfect: on 50 generic not-onion images, 21 produced detections and
**8 still got through** (fingernails, a pepper, a cucumber, a few people). The
permanent fix for those is to teach the detector itself — add them as
background images with `tools/add_background_images.py` and fine-tune. Measure
both with `python -m onion_grader.gatecheck` on your own photographs.

Thresholds live in `calibration.json` under `onion_gate`. Set
`"enabled": false` to switch the gate off.

---

## HEALTHY / UNHEALTHY comes from a learned classifier, not the dark-area rule

The dark-area rule (a pixel is damage if it is greyish AND dark) was tuned on
one session of red onions. It misses light-brown rot, and on white onions it
counts shadows as damage. Measured:

| | dark-area rule | learned classifier (v3) |
|---|---|---|
| 120 hand-graded onions (v3 never saw them; the rule was **fitted** on them) | 83% | **93%** |
| 96 labelled photos, `2. Bulb`: healthy photos wrongly flagged | 53% | 4% |
| same, unhealthy photos caught | 78% | 100% |
| single-onion photos, accuracy | 71% | 96% |

(The `2. Bulb` numbers flatter v3, which was trained on that dataset; the
120-onion row is the honest one.)

So the verdict now comes from `onion_gate.pth`'s Healthy/Unhealthy output, and
the AGMARK grade is kept consistent with it: an unhealthy onion is at least
Class I; an onion the classifier is sure is sound (p < 0.2) is Extra Class; in
between, the dark-area band stands and the onion is flagged for review. The
dark-area % is still shown, labelled as a measurement. Set
`"verdict": {"source": "area"}` in `calibration.json` for the old behaviour.

Also new: boxes lying inside another box (a bulb without its stalk inside the
box with it, skin fragments) are dropped as duplicates -- 7 on the 50 trays,
all genuine duplicates -- and onions cut off by the photo edge are marked
`partial`. Re-check on your own photos with `python -m onion_grader.benchmark`.

---

## It tells you when it is out of its depth

The constants were fitted on one camera, one light, one market's produce. This
project's most repeated finding is that **thresholds do not transfer**. So the
system checks, on every photograph, whether it is being asked a question outside
the range it was fitted on — two checks, in opposite directions:

| | it fires when | across the 53 calibration photographs |
|---|---|---|
| **A** | *everything looks strange* — median anomaly > 0.30 | never above 0.287 |
| **B** | *nothing registers as damage* — no onion clears 5% | never below 9.9% |

Both are needed, and the second one is not decoration. Brighten a tray and every
raw defect fraction goes to **exactly zero** while the anomaly score goes *down* —
the photograph grades 100% healthy, a visibly rotten onion reads 2.0%, and
direction A never notices. That was the first thing this guard was tested on and
it failed. See `results/phase16b/findings.md`.

When it fires, the system **keeps answering** and stops claiming what it can no
longer support:

- counts, boxes, outlines, relative sizes — **unaffected**, all scale-free
- defect % and grade — **still reported**, with the warning attached
- `unusual_kind` — **withdrawn**, because out of domain it only detects
  unfamiliarity

If the input really is onions from your normal set-up, the fix is
recalibration, not a better threshold: re-grade ~100 onions from your own source
and rerun `audit/phase16_export_calibration.py`. If it is not (a screenshot, a
face, a filtered photo), recalibrating would teach the grader nonsense: fix the
photo instead. The onion gate above handles the "not onions at all" case.

**Two honest caveats, both measured in Phase 17.** The guard fired on only
**20%** of a genuinely foreign dataset (different country, camera and cultivar),
so treat it as a tripwire for gross shifts, not a domain detector. And **the
recalibration path above has never been validated on a second domain** — the one
free labelled dataset available turned out to be unable to test it, because 98%
of its photographs carry a single label and its *background* predicts that label
better than our instrument does. Validating it needs photographs that mix sound
and damaged onions in the same frame, labelled per onion.

---

## Size, and what happens with no coin

**With a ₹1 coin in frame** the detector finds it by itself — a coin is the small
grey round thing, and the rule separating it from onions removed 54 of 72
spurious boxes and **zero** real onions. Sizes come out in millimetres with an
AGMARK size code.

**Without a reference object, absolute size cannot be recovered at all.** This is
geometry, not a gap in the model: a small onion near the lens produces *exactly*
the same pixels as a large onion far away. One photograph does not contain the
information. Any system that reports millimetres without a reference is guessing,
and this one will not.

What still works with no coin, unchanged:

| | |
|---|---|
| grade, defect %, confidence, healthy/unhealthy | **unaffected** — all scale-free |
| **relative size** | **exact** — `size_ratio` (÷ the photo's median) and `size_rank` |
| shape: circularity, aspect ratio | **unaffected** — ratios need no scale |
| absolute mm, AGMARK size code | **not reported**, and a warning says why |

Three ways to get millimetres back:

1. **Put any known object in frame.** It does not have to be a ₹1 coin — a ₹5
   coin, a ruler, an A4 sheet, a standard crate edge. Tell me which and I will
   add it to the coin rule.
2. **Pass the scale yourself** — `--px-per-mm 3.2`, or the sidebar box in the
   app. Measure any object you know: its pixel length ÷ its real length in mm.
3. **Fix the camera height** and measure the scale once; it then holds for every
   photograph taken at that height.

**The coin is assumed to be the 21.93 mm (2011-series) ₹1.** If yours is the
25 mm variant, every millimetre figure scales by 1.14× — grades, percentages and
confidences are unaffected.

## It reads dark skin as damage

The grade comes from one rule: a pixel counts as damaged if it is **dull and
dark**. Dark onion skin is also dull and dark, so the rule over-reads on
dark-skinned cultivars — and under-reads on pale ones.

Measured on the 50 onions a person graded as having **no visible defect**:

| | |
|---|---|
| correlation between skin darkness and measured defect | **+0.54** |
| called damaged anyway (past the 0.10 detection limit) | **17 of 50 — 34%** |
| pushed past the ⅕ line, into Class II or worse | **7 of 50 — 14%** |
| the single worst | **0.500** — read as half damaged |

The six worst are the six darkest. **A third of sound dark onions lose their top
grade to their own skin colour.**

Three corrections were tried — thresholds scaled to each onion's own skin
brightness, to its saturation, and a de-bias fitted on sound onions only. **None
works.** The first two cost 0.090 and 0.287 κ; the third is nearly free but
barely de-biases. Every measured bias reduction has an interval wide enough to
contain no change at all, while the accuracy loss is unambiguous. Details and
data in `results/phase17c/findings.md`.

If you grade a dark cultivar, recalibrate on it. If you grade a mix, expect the
dark ones to be judged harshly.

---

## What it does not do

- **Firmness, dryness, internal soundness, smell.** These are part of the legal
  grade and are invisible in a photograph. Every output says *provisional*.
- **Disease species.** Not determinable from RGB; dropped early and deliberately.
- **Defect type beyond "unusual".** Only two kinds are separable in our labels
  (dry skin damage, and sunken rot), and only 13 onions of the latter.
- **Video.** Cut honestly as future work.

---

## Speed and hardware

It picks its own device: **CUDA if there is one, then Apple `mps`, otherwise
CPU**. Override with `--device cuda|cpu|mps` or `ONION_DEVICE`. Half precision
turns on automatically for CUDA and stays off elsewhere, where it is slower.
Every result reports which device ran it and how long each stage took.

Measured warm, 960×640 photograph, same code both machines:

| | 16-core CPU | 2-core CPU |
|---|---|---|
| YOLO detection at 960 | 0.34 | 1.35 |
| MobileSAM encode + decode | **0.86** | **3.2** |
| everything else | 0.10 | 0.16 |
| **total** | **1.30 s** | 4.7 s |

**The two model passes are ~92% of it**, and MobileSAM's encoder is the single
biggest item. Cores help a lot — 16 of them give **3.6×** over 2.

**The CUDA path is written and plumbed but has never run on an actual GPU.** Run
`python -m onion_grader.selftest` to see what your machine is really doing: it
reports where the weights landed, times each stage, and on a GPU machine also
runs the same photograph on CPU so the speedup is measured rather than claimed.

If it reports `torch 2.x.x+cpu` / `built with CUDA None`, **your PyTorch cannot
use a GPU at all**, whatever hardware is present. Check `nvidia-smi` first; if
there is an NVIDIA card, reinstall inside the venv:

```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Live tracking defaults to imgsz 480, which on this box is **4× faster than 960
and finds 100% of the boxes 960 finds** on synthetic trays. That is agreement
with 960, not ground-truth recall; grading always runs at 960.

## Install

```
pip install ultralytics numpy pillow scipy streamlit
pip install lap          # only for live camera tracking
```

Weights are found automatically in `onion_grader/`, the project root,
`new V3 file/` or `weights/`, or set `ONION_YOLO` / `ONION_SAM`:

| file | where from |
|---|---|
| `onion_yolo.pt` | your trained detector, in `new V3 file/` |
| `mobile_sam.pt` | downloaded by ultralytics on first run if absent |
| `onion_gate.pth` | the v3 classifier, copied from `extra stuff/unused-weights/new V3 file/best_onion_model_v3.pth` (or set `ONION_GATE`) |

---

## A note on `calibration.json`

It holds every fitted constant: the two dull-pixel thresholds, the isotonic map
from measured to human-judged area, the detection limit, the anomaly Gaussian,
and the resolution correction.

**Those constants were fitted on all 120 labelled onions**, which is right for
*using* the system and wrong for *scoring* it. The numbers in the table above
come from cross-validated fits that never saw the onion they graded. The file
says so in its own `_README` field.

**The AGMARK ⅕ and ½ boundaries are in there too and are never fitted.** Only the
detection limit — where "free from defects" ends, which the standard leaves
undefined — is.

### Recalibrating for your own inspector or camera

The project's most repeated finding is that **thresholds do not transfer** — not
across cameras, not across lighting, and not across people. The system reproduces
the rubric it was calibrated on and loses ≈0.13 κ against a different annotator.

If grades feel systematically harsh or lenient, that is expected and fixable:
re-grade ~100 onions yourself and rerun

```
python audit/phase16_export_calibration.py
```

against your labels. Nothing else needs to change.
