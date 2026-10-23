# Technical reference — what this project is built from

Everything below describes the shipped system in `onion_grader/`. It is a
build-and-maintain document: what each piece is, what it does to the pixels, and
where to change it.

---

## 1. The stack

| | | |
|---|---|---|
| **ultralytics** | 8.4.x | loads and runs both neural models |
| **PyTorch** | 2.x | the runtime underneath ultralytics |
| **NumPy** | 2.x | all pixel maths |
| **SciPy** (`scipy.ndimage`) | 1.x | connected components, erosion, dilation, local texture |
| **Pillow** | 12.x | image I/O and every drawing operation |
| **Streamlit** | 1.x | the app |
| **OpenCV** | 4.x | live camera only; already required by ultralytics |
| **lap** | 0.5.x | live camera only; ByteTrack's assignment solver |

```
pip install ultralytics numpy pillow scipy streamlit
```

No OpenCV, no scikit-learn, no TensorFlow at inference time. `scipy.ndimage` and
Pillow cover the morphology and the drawing, which keeps the install small and
avoids the OpenCV/Pillow colour-order confusion entirely — **everything in this
codebase is RGB, never BGR.**

OpenCV enters only for **live camera mode** (`live.py`), and it is already an
ultralytics dependency, so nothing new is installed. `lap` is needed there too,
for ByteTrack. Everything in `grader.py` stays RGB and OpenCV-free.

Measured warm on a 2-core CPU, 960×640: **≈4.7 s per photograph** — MobileSAM's
encoder 67%, YOLO 28%, all post-processing 3%. The device is chosen
automatically (§2.3).

---

## 2. The two neural models

### 2.1 Detector — `onion_yolo.pt`

| | |
|---|---|
| architecture | **YOLO26-m**, fine-tuned |
| parameters | **21.77 M** |
| classes | **1** — `onion` |
| training | 100 epochs, `imgsz=960`, batch 8, `optimizer=auto`, `lr0=0.01` |
| trained on | Kaggle |
| file size | 42 MB |

Inference settings live in `calibration.json → detector` (`conf: 0.25`,
`imgsz: 960`) so they can be changed without touching code.

It has **one class**. It does not know what a coin is, what a defect is, or what
grade anything should be. It finds onion-shaped things and returns boxes.

### 2.2 Segmenter — `mobile_sam.pt`

| | |
|---|---|
| architecture | **MobileSAM** (Segment Anything with a TinyViT image encoder) |
| parameters | **10.14 M** total — image encoder 6.08 M, plus prompt encoder and mask decoder |
| file size | 39 MB |
| input resolution | **1024 × 1024**, fixed |
| how it is prompted | **bounding boxes** from the detector |

MobileSAM is *promptable*: it does not decide what to segment, it segments what
you point at. Here it is prompted with boxes rather than points, because a box
states both where the object is and roughly how big it is, so there is no
ambiguity about whether "this pixel" means the root plate, a patch of skin, or
the whole bulb.

It is **not fine-tuned**. It is used as shipped. Only the prompt is ours.

Ultralytics downloads `mobile_sam.pt` automatically on first run if it is absent.

### 2.3 Device

`pick_device()` returns **cuda → mps → cpu**, overridable with the `device=`
argument or `ONION_DEVICE`. `half` defaults to True on CUDA only. On CPU the
thread count is set to all cores **after** the models load — ultralytics resets
it during setup, so setting it earlier does nothing.

### 2.4 Where the weights are found

`_find()` in `grader.py` looks, in order:

1. an explicit path passed to `OnionGrader(detector=..., sam=...)`
2. the environment variables `ONION_YOLO` / `ONION_SAM`
3. `onion_grader/`, the project root, `new V3 file/`, `weights/`

---

## 3. The pipeline, step by step

```
photograph (RGB)
   │
   ├─▶ YOLO26-m ─────────────▶ boxes
   │        └─ coin rule ────▶ scale (px per mm)   [optional]
   │
   ├─▶ MobileSAM (box-prompted) ──▶ one binary mask per onion
   │
   ├─▶ per mask:
   │      dull-pixel fraction  ──▶ raw defect fraction
   │      isotonic map         ──▶ calibrated defect fraction
   │      thresholds           ──▶ band + grade + confidence
   │      Mahalanobis          ──▶ anomaly score
   │      geometry             ──▶ area, diameter, circularity, aspect ratio
   │
   ├─▶ whole photograph: relative sizes, domain guard
   │
   └─▶ render: outlines, boxes, labels, header bar
```

Everything after MobileSAM is deterministic arithmetic. There is no third model,
no classifier on top, and nothing random — the same photograph always produces
byte-identical output.

---

## 4. Detection and the coin rule

`_detect()` runs YOLO, then separates coins from onions using two numbers:

```python
if a < rule["max_area_ratio"] * med and s < rule["max_saturation"]:
    coins.append((b, a))
else:
    onions.append(b)
```

- `med` — the median box area in *this* photograph
- `a` — this box's area
- `s` — the mean **saturation** inside the box

A coin is *small relative to the onions in the same frame* and *grey*. Both
constants (`max_area_ratio: 0.45`, `max_saturation: 0.25`) live in
`calibration.json → detector.coin_rule`.

This is two comparisons, not a classifier. It is scale-free because it compares
against the median box in the same image, so it works whether the photo is close
up or far away.

**Saturation** here and everywhere else in the project is HSV *S*, computed
directly rather than via a colour-space conversion:

```python
v = rgb.max(2)                        # value
s = (v - rgb.min(2)) / max(v, 1e-6)   # saturation
```

---

## 5. Segmentation, and the two things that bite

```python
masks = self._masks(path, boxes.copy(), (H, W))
```

### 5.1 `boxes.copy()` is mandatory

Ultralytics' `prompt_inference` **rescales the bbox array in place** to SAM's
1024 px input. Pass the detector's own array and every box you later report comes
back multiplied by 1024/W. The masks are unaffected, so the drawn overlay looks
perfect while every reported coordinate is wrong. Always hand it a copy.

### 5.2 `torch.inference_mode()` is mandatory

Without it the image encoder retains its activation graph — roughly 3 GB per
photograph. Both the encode and the prompt step are wrapped:

```python
with torch.inference_mode():
    pred.setup_source(path)
    for batch in pred.dataset:
        im = pred.preprocess(batch[1])
        pred.features = pred.get_im_features(im)
        break
    low, _ = pred.prompt_inference(im, bboxes=boxes, multimask_output=False)
```

The image is encoded **once** and all boxes are decoded against those cached
features, which is why N onions cost barely more than one.

Low-resolution masks come back at SAM's internal size and are lifted to the
photograph's size with `ops.scale_masks(..., padding=False)`, then thresholded at
logit > 0.

### 5.3 One mask, one blob

A mask can come back with stray islands. `_measure` keeps only the connected
component containing the box centre:

```python
lab, n = ndimage.label(mask)
if n > 1:
    want = lab[cy, cx]                       # the blob under the box centre
    if want == 0:                            # centre landed on a hole
        want = argmax(component sizes) + 1   # fall back to the largest
    mask = lab == want
```

Anything under 50 px is dropped as noise.

---

## 6. How the outline is drawn

This is the part that makes the output look like a measurement rather than a
guess, and it is four lines.

The mask is a boolean array the size of the photograph. Its **edge** is the set
of pixels that are inside the mask but would fall out if the mask shrank by one
pixel:

```python
eroded = ndimage.binary_erosion(mask, np.ones((3, 3), bool), border_value=0)
edge   = mask & ~eroded                                   # a 1-px shell
thick  = ndimage.binary_dilation(edge, np.ones((3, 3), bool))
arr[thick] = PALETTE[onion.band_index]                    # paint it
```

| step | what it does |
|---|---|
| `binary_erosion` with a 3×3 structuring element | shrinks the mask by one pixel in every direction |
| `mask & ~eroded` | the one-pixel ring that was lost — the boundary |
| `binary_dilation` | fattens the ring to ~3 px so it survives being viewed small |
| assignment into the array | paints those pixels the band colour |

`border_value=0` matters: it makes the erosion treat the image border as
*outside* the mask, so an onion touching the edge of the photograph still gets an
outline along that edge instead of silently losing it.

The outline is **the actual measured boundary**, not a smoothed or fitted curve.
No contour tracing, no polygon approximation, no convex hull. What you see
outlined is exactly the pixel set the defect fraction was computed over — which
is the point: if the outline looks wrong, the number is wrong, and you can see
that without reading any code.

### The colour is the grade

```python
PALETTE = [(60, 200, 110),    # band 0  — no visible defect      green
           (240, 200, 70),    # band 1  — up to 1/5              yellow
           (245, 140, 50),    # band 2  — 1/5 to 1/2             orange
           (235, 65, 65)]     # band 3  — over 1/2               red
```

Indexed by `band_index`, so outline, box, and label chip always agree with each
other and with the grade. Changing a band changes all three at once.

---

## 7. How the defect pixels are found

The same rule that produces the grade also produces the red overlay, so the
picture cannot disagree with the number:

```python
dull = (saturation < s_thr) & (value < v_thr) & mask
raw  = dull.sum() / mask.sum()
```

with `s_thr = 0.500` and `v_thr = 0.450` from `calibration.json → defect`.
Plain English: **a damaged pixel is one that is both dull and dark, and inside
the onion.**

`raw` is then passed through an **isotonic map** — a 27-point monotone lookup
stored as two arrays, applied with `np.interp` — which converts "fraction of
pixels my rule flagged" into "fraction of the surface a person would call
damaged". Monotone means it can stretch and compress the scale but can never
reorder two onions.

The band comes from three cut points:

```python
cuts = [detection_limit] + agmark_cuts      # [0.10, 0.20, 0.50]
band = int(np.digitize(frac, cuts))         # 0, 1, 2 or 3
```

Three cuts, four bands. **Two cuts cannot produce four bands** — `np.digitize`
with two cut points emits at most three values, which is a silent, plausible-
looking failure if you ever edit this.

**Confidence** is distance to the nearest cut, normalised and capped:

```python
conf = min(1.0, min(abs(frac - t) for t in cuts) / 0.10)
needs_review = conf < 0.5
```

An onion sitting on a boundary gets a low confidence and a review flag; one in
the middle of a band gets 1.0.

### The red overlay

In the per-onion view, flagged pixels are blended rather than replaced, so the
texture underneath stays visible and you can judge whether the rule was right:

```python
heat[dmap] = (0.4 * heat[dmap] + 0.6 * np.array([255, 60, 50])).astype(np.uint8)
```

---

## 8. Anomaly score

A 3-D Gaussian over **(saturation, value, texture)**, where texture is the local
standard deviation in a 5×5 window computed the fast way:

```python
tex = sqrt(uniform_filter(g**2, 5) - uniform_filter(g, 5)**2)
```

Every pixel in the mask gets a squared Mahalanobis distance to the mean:

```python
z    = stack([s, v, tex]) - mu
dist = einsum("ij,jk,ik->i", z, inv_cov, z)
anom = (dist > threshold).mean()
```

So `anomaly_score` is **the fraction of the onion's pixels that do not look like
ordinary onion skin**. It is a different question from "how much is damaged" —
it answers "is this a different *kind* of thing" — and it is reported separately
rather than folded into the grade.

`mu`, `inv_cov` and `threshold` are in `calibration.json → anomaly`.

---

## 9. Geometry

All from the mask, no model involved:

| output | how |
|---|---|
| `area_px` | `mask.sum()` |
| `eq_diameter_px` | `2 * sqrt(area / π)` — diameter of a circle of the same area |
| `circularity` | `4π · area / perimeter²`, perimeter = the same 1-px edge shell |
| `aspect_ratio` | ratio of the square roots of the eigenvalues of the pixel-coordinate covariance matrix — the long axis over the short axis, orientation-free |

A note for anyone reading the numbers: **a perfect digital disc has circularity
≈ 0.63, not 1.0**, because a pixel-stepped boundary is longer than a smooth one.
Onions land near 0.52. This is a property of counting boundary pixels, not a bug.

---

## 10. Size: relative always, millimetres only with a reference

### Relative — needs nothing

```python
med = median(eq_diameter_px for all onions)
size_ratio = eq_diameter_px / med
size_rank  = position when sorted largest first
```

Ratios cancel the unknown scale, so these are exact in every photograph.

### Millimetres — needs a reference object

With a coin detected (or `px_per_mm` supplied):

```python
corr = (px_per_mm / 3.33) ** (1 - alpha / 2)      # alpha = 1.525
d_mm = eq_diameter_px / px_per_mm * corr
```

The correction exists because **SAM's mask area does not scale as
resolution²** — measured at resolution^1.525. Without correcting for it, the same
onion photographed at two resolutions measures two different sizes. `3.33` is the
reference px/mm the constant was established at.

The size code is then a lookup on the major axis:

| code | major axis (mm) |
|---|---|
| A | 0 – 30 |
| B | 30 – 45 |
| C | 45 – 60 |
| D | 60 + |

**Without a reference object, no millimetre figure is reported at all** and a
warning says why. One photograph does not contain the information: a small onion
near the lens produces the same pixels as a large one far away. Grade, defect %,
confidence, health and relative size are all unaffected, because all of them are
ratios.

---

## 11. The domain guard

Two photograph-level checks, in opposite directions, both from outputs already
computed:

```python
fail_a = median_anomaly > 0.30 or flagged_fraction > 0.40     # everything looks strange
fail_b = n_onions >= 3 and max_raw < 0.05                     # nothing registers at all
```

Direction B exists because a brightness shift can push every pixel above `v_thr`,
driving every raw fraction to exactly 0.0 — while the anomaly score goes *down*,
not up. Direction A cannot see that, and it is the dangerous case, because the
photograph then grades as entirely healthy.

When it fires the system keeps reporting counts, outlines and relative sizes
(none of them depend on the calibration), attaches a warning to the grades, and
sets `anomaly_reliable = False` — because out of domain that score measures
unfamiliarity rather than a kind of damage.

Constants: `calibration.json → domain_guard`.

---

## 12. Drawing the annotated image

All Pillow. `ImageDraw` for shapes and text, `ImageFont.truetype` for DejaVu with
a `load_default()` fallback so it never crashes on a machine without fonts.

Everything scales with image width so output looks the same on a 640 px crop and
a 4000 px photograph:

```python
scale = max(1, int(round(W / 900)))
```

Draw order matters, and it is:

1. **Outlines** — painted into the NumPy array *before* it becomes a PIL image,
   because they are per-pixel assignments, not shapes
2. **Boxes** — `d.rectangle(..., outline=col, width=2*scale)`
3. **Label chips** — a filled rectangle the width of the text (`d.textlength`)
   with `#3 UNHEALTHY 34%` on top in near-black
4. **Header bar** — counts and percentages, each coloured to match its meaning,
   plus the scale line and, when the guard fires, an amber strip

The label sits above its box, except when that would put it under the header bar
or off the top of the image, in which case it moves inside or below:

```python
bar = (74 if res.domain_ok else 100) * scale
ty  = y1 - th if y1 - th >= bar else (y1 + 2*scale if y1 >= bar else y2)
```

`render()` takes `boxes=`, `outlines=` and `header=` flags so the app's
checkboxes can turn each layer off independently.

---

## 13. The per-onion view

`crop(res, onion, source, pad=0.12)` returns **three images of the same region**:

| | |
|---|---|
| `crop` | the pixels as photographed |
| `outline` | the same, with the measured boundary drawn |
| `heat` | the same, with the pixels counted as damage in red |

The box is padded 12% on each side so the onion is not cut flush to its own
boundary. The defect map is recomputed inside the crop using the identical rule,
so the three images are guaranteed to be consistent with each other and with the
reported number.

This is what makes the grade auditable by eye: if the boundary is wrong you see
it, and if the red pixels are shadow rather than rot you see that too.

---

## 14. `calibration.json`

Every fitted constant in one file. Nothing numeric is hard-coded in `grader.py`
except the palette and the review threshold.

```
defect       s_thr, v_thr, isotonic_x, isotonic_y, detection_limit, agmark_cuts
anomaly      mu (3), inv_cov (3×3), threshold
scale        coin_mm, resolution_exponent
detector     conf, imgsz, coin_rule { max_area_ratio, max_saturation }
domain_guard median_anom_max, flagged_fraction_max, max_raw_floor, min_onions…
size_codes   [name, lo, hi] × 4
```

`agmark_cuts` are the statutory ⅕ and ½ and are **never fitted**. Only
`detection_limit` — where "free from defects" ends — is.

Regenerate the whole file with:

```
python audit/phase16_export_calibration.py
```

---

## 15. Files

```
onion_grader/
  grader.py          the pipeline, the drawing, and the CLI      (~500 lines)
  live.py            camera mode: track every frame, grade on a cadence
  app.py             the Streamlit interface
  calibration.json   every fitted constant
  onion_yolo.pt      the detector
  mobile_sam.pt      the segmenter (auto-downloaded if absent)
  README.md          how to use it
  TECH.md            this file
```

### Three ways to run it

```bash
# command line
python -m onion_grader.grader photo.jpg --out graded.jpg --json out.json
python -m onion_grader.grader photo.jpg --px-per-mm 3.2

# app
streamlit run onion_grader/app.py
```

```python
# python
from onion_grader.grader import OnionGrader
g   = OnionGrader()
res = g.grade("tray.jpg")
print(res.summary())
g.render(res, "graded.jpg", "tray.jpg")
crop, outline, heat = g.crop(res, res.onions[0], "tray.jpg")
```

### Data model

`Result` — `image, width, height, n_onions, px_per_mm, scale_source, onions[],
warnings[], domain_ok, domain_stats` plus computed `n_healthy`, `n_unhealthy`,
`pct_healthy`, `pct_unhealthy`.

`Onion` — `index, box, grade, band, band_index, defect_area_fraction,
raw_defect_fraction, confidence, needs_review, anomaly_score, unusual_kind,
anomaly_reliable, area_px, eq_diameter_px, circularity, aspect_ratio,
size_ratio, size_rank, eq_diameter_mm, size_code`, plus `healthy` and `verdict`.

The mask is attached as `onion._mask` for rendering and is deliberately **not**
serialised — `to_json()` would otherwise emit a full-resolution boolean array per
onion.

---

## 16. Things that will bite you

| | |
|---|---|
| `boxes.copy()` | ultralytics mutates the bbox array in place; without the copy every reported box is wrong by 1024/W while the image still looks right. **CPU-float32-only** — `torch.as_tensor` shares memory there, but must copy for CUDA or fp16, so a GPU developer never sees this |
| `pred.batch = batch` | `prompt_inference` scales prompts by `pred.src_shape`, which reads `pred.batch[1][0]`. Omit it and prompts are scaled against whatever image the predictor last fully processed |
| timing on the first call | includes model load and a cold predictor. Always measure warm, median of repeats |
| `torch.inference_mode()` | ~3 GB per photograph leaks without it |
| RGB everywhere | no OpenCV in this codebase, so never assume BGR |
| three cuts, four bands | `np.digitize` with two cut points can only emit three |
| `border_value=0` | omit it and onions touching the image edge lose part of their outline |
| `_mask` is not serialisable | keep it out of `asdict()` |
| circularity ≈ 0.63 is perfect | pixel boundaries are longer than smooth ones |
| the coin is assumed 21.93 mm | the other ₹1 variant is 25 mm; every mm figure then scales by 1.14× |
