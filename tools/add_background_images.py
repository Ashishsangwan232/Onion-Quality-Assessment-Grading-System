"""Teach the detector what is NOT an onion (the permanent fix, layer 2).

The onion gate removes faces and objects AFTER the detector has fired. This
script fixes the detector itself: it adds "background images" -- pictures with
no onions and an EMPTY label file -- to your YOLO training set. Ultralytics
treats those as pure negatives, so the retrained onion_yolo.pt stops drawing
boxes on faces, hands, lamps and screenshots in the first place.

    python tools/add_background_images.py --data path/to/data.yaml \
        --negatives "extra stuff/datasets/data/onion_quality/3. Not Onion" \
        --also gatecheck_out/graded_but_not_onion --fraction 0.10

then retrain from the current weights (fine-tune, ~30 epochs is enough):

    yolo detect train model=onion_grader/onion_yolo.pt data=path/to/data.yaml \
         imgsz=960 epochs=30 patience=10

Ultralytics recommends 0-10% background images; more starts to cost recall.
After retraining, check recall has not dropped (it was 0.998) and rerun
python -m onion_grader.gatecheck before swapping the new weights in.
"""
import argparse, glob, os, random, shutil
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="the YOLO data.yaml used to train onion_yolo.pt")
ap.add_argument("--negatives", required=True, help="folder of images with no onions")
ap.add_argument("--also", nargs="*", default=[], help="extra negative folders (e.g. gatecheck failures)")
ap.add_argument("--fraction", type=float, default=0.10, help="negatives as a fraction of train images")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.data))
root = cfg.get("path") or os.path.dirname(os.path.abspath(a.data))
if not os.path.isabs(root):
    root = os.path.join(os.path.dirname(os.path.abspath(a.data)), root)
train = cfg["train"]
if isinstance(train, list) or str(train).endswith(".txt"):
    raise SystemExit("train: must be a single image folder for this script")
img_dir = train if os.path.isabs(train) else os.path.join(root, train)
lbl_dir = img_dir.replace(os.sep + "images", os.sep + "labels")
n_train = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

pool = []
for d in [a.negatives] + a.also:
    pool += [p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
random.Random(a.seed).shuffle(pool)
want = min(len(pool), int(round(a.fraction * n_train)))
os.makedirs(lbl_dir, exist_ok=True)
for i, p in enumerate(pool[:want]):
    name = "bg_%05d_%s" % (i, os.path.basename(p))
    shutil.copy2(p, os.path.join(img_dir, name))
    open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), "w").close()  # empty = no onions
print("train images: %d  ->  added %d background images (%.0f%%)" % (n_train, want, 100 * want / max(n_train, 1)))
print("images: %s\nlabels: %s" % (img_dir, lbl_dir))



# python -m onion_grader.benchmark --healthy "G:\Quality test\onion-quality-assessment\onion-quality-assessment\extra stuff\datasets\data\onion_quality\2. Bulb\1. Healthy\2. White Onion\2. Multiple" --unhealthy "G:\Quality test\onion-quality-assessment\onion-quality-assessment\extra stuff\datasets\data\onion_quality\2. Bulb\2. Unhealthy\2. White Onion\2. Multiple" 