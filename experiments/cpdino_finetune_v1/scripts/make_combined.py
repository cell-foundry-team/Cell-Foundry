"""Combined view: annotation and prediction on the SAME image, so matches, misses and extras show at once.

Colours (outline of every cell):
  green    IMES annotation (inner edge)
  blue     predicted cell that overlaps an annotated cell (agrees)
  orange   predicted cell with (almost) no annotated cell under it: an EXTRA, either a false detection
           or a real cell nobody annotated; this is the list to check with IMES
  magenta  annotated cell the model missed (almost nothing predicted on it)
"agrees / extra / missed" use one threshold (--thr, default 0.2): a predicted cell is an extra when less than
this share of its area lies on annotated cells; an annotated cell is missed when less than this share of its
area is covered by predictions.

  ~/cellfoundry_push/cellpose-env/.venv/bin/python make_combined.py \
      --data ~/porcine_data_local/Training_porcine_bf --pred-root ~/cpdino_results_3seed/predict \
      --split ~/cellfoundry_push/splits/porcine_bf_v1.json --subset test \
      --labels cpdino-raw cpdino-ft-clahe-v1-s0 --out ~/cpdino_overlays/combined --zoom
Writes <name>_combined.png (whole image), <name>_combined_zoom.png (with --zoom) and counts.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from skimage.io import imread
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_overlays import find_pred, grid2, panel, to_rgb, worst_window  # noqa: E402

GREEN, BLUE, ORANGE, MAGENTA = (40, 200, 90), (40, 110, 235), (255, 140, 0), (225, 40, 200)


def share_inside(a, b):
    """For every label in a: share of its pixels where b > 0. Index = label; 0 and absent labels -> nan."""
    n = int(a.max()) + 1
    tot = np.bincount(a.ravel(), minlength=n).astype(np.float64)
    ins = np.bincount(a[b > 0].ravel(), minlength=n).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = ins / tot
    s[0] = np.nan
    s[tot == 0] = np.nan
    return s


def combine(rgb, gt, pm, thr):
    gt, pm = gt.astype(np.int64), pm.astype(np.int64)
    pred_share = share_inside(pm, gt)      # per predicted cell: share lying on annotated cells
    gt_share = share_inside(gt, pm)        # per annotated cell: share covered by predictions
    pred_cls = np.zeros(len(pred_share), np.uint8)
    pred_cls[1:] = np.where(pred_share[1:] < thr, 2, 1)       # 1 agrees, 2 extra
    pred_cls[np.isnan(pred_share)] = 0                         # label ids that do not occur
    gt_missed = np.zeros(len(gt_share), bool)
    gt_missed[1:] = gt_share[1:] < thr
    out = rgb.copy()
    b_gt = find_boundaries(gt, mode="inner")
    out[b_gt] = GREEN
    out[b_gt & gt_missed[gt]] = MAGENTA
    # outer edge of each predicted cell: background pixels next to it take the class of the neighbour
    edge = find_boundaries(pm, mode="outer") & (pm == 0)
    from scipy.ndimage import grey_dilation
    near = grey_dilation(np.where(pm > 0, pm, 0), size=(3, 3))
    ecls = pred_cls[near]
    out[edge & (ecls == 1)] = BLUE
    out[edge & (ecls == 2)] = ORANGE
    n_pred = int((~np.isnan(pred_share)).sum())
    n_gt = int((~np.isnan(gt_share)).sum())
    counts = dict(n_pred=n_pred, n_agree=int((pred_cls == 1).sum()), n_extra=int((pred_cls == 2).sum()),
                  n_gt=n_gt, n_missed=int(gt_missed.sum()))
    return out, counts


def legend(width):
    bar = Image.new("RGB", (width, 64), (246, 243, 236))
    d = ImageDraw.Draw(bar)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    x = 12
    for col, txt in [(GREEN, "IMES annotation"), (BLUE, "prediction agrees"),
                     (ORANGE, "extra (no annotation under it)"), (MAGENTA, "annotated, missed")]:
        d.rectangle([x, 20, x + 28, 44], fill=col)
        d.text((x + 38, 18), txt, fill=(27, 42, 46), font=font)
        x += 38 + int(d.textlength(txt, font=font)) + 36
    return np.asarray(bar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--zoom", action="store_true")
    ap.add_argument("--crop", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    a = ap.parse_args()

    data, pred_root, out = Path(a.data).expanduser(), Path(a.pred_root).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    names = json.load(open(Path(a.split).expanduser()))["splits"][a.subset]
    rows = []
    for name in names:
        img_file = next(data.glob(f"{name}_image.*"), None)
        gt_file = next(data.glob(f"{name}_instances.*"), None)
        if img_file is None or gt_file is None:
            print("missing image or ground truth:", name)
            continue
        rgb = to_rgb(imread(img_file))
        gt = tifffile.imread(gt_file) if gt_file.suffix.lower() in (".tif", ".tiff") else imread(gt_file)
        if gt.ndim == 3:
            gt = gt[..., 0]
        full, preds = [], {}
        for label in a.labels:
            f = find_pred(pred_root, a.subset, label, name)
            if f is None:
                print(f"no prediction for {label}: {name}")
                continue
            pm = tifffile.imread(f)
            img, c = combine(rgb, gt, pm, a.thr)
            preds[label] = (pm, img)
            rows.append({"image": name, "label": label, **c})
            title = f"{label}: {c['n_agree']} agree, {c['n_extra']} extra | {c['n_missed']} of {c['n_gt']} annotated missed"
            full.append(panel(img, title, 1100))
        if not full:
            continue
        stem = name.replace(" ", "_")
        grid = grid2(full)
        lg = legend(grid.shape[1])
        Image.fromarray(np.concatenate([lg, grid], axis=0)).save(out / f"{stem}_combined.png")
        print("wrote", out / f"{stem}_combined.png")
        if a.zoom:
            H, W = gt.shape
            last = list(preds)[-1]
            if a.crop:
                x0, y0, x1, y1 = int(a.crop[0] * W), int(a.crop[1] * H), int(a.crop[2] * W), int(a.crop[3] * H)
            else:
                x0, y0, ww, wh = worst_window(gt, preds[last][0])
                x1, y1 = x0 + ww, y0 + wh
            sl = (slice(y0, y1), slice(x0, x1))
            tiles = [panel(img[sl], label, 1000) for label, (_, img) in preds.items()]
            zgrid = grid2(tiles)
            Image.fromarray(np.concatenate([legend(zgrid.shape[1]), zgrid], axis=0)).save(out / f"{stem}_combined_zoom.png")
            print("wrote", out / f"{stem}_combined_zoom.png")
    with open(out / "counts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "label", "n_pred", "n_agree", "n_extra", "n_gt", "n_missed"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / "counts.csv")


if __name__ == "__main__":
    main()
