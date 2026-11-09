"""Side-by-side overlay figures: image | ground truth | one panel per predicted label.

Outlines are drawn on the original image: ground truth in green, predictions in red.
Each panel title shows the fat area as % of the image.

Run with the project's Python (it has numpy, tifffile, scikit-image, pillow):
  ~/cellfoundry_push/cellpose-env/.venv/bin/python make_overlays.py \
      --data ~/porcine_data_local/Training_porcine_bf \
      --pred-root ~/cpdino_results_3seed/predict \
      --split ~/cellfoundry_push/splits/porcine_bf_v1.json --subset test \
      --labels cpdino-raw cpdino-ft-clahe-v1-s0 --out ~/cpdino_overlays/test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from skimage.io import imread
from skimage.segmentation import find_boundaries

PANEL_W = 760
GREEN, RED = (40, 200, 90), (230, 40, 40)


def to_rgb(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3].astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def outline(rgb, mask, color):
    out = rgb.copy()
    out[find_boundaries(mask, mode="thick")] = color
    return out


def panel(rgb, title, width=PANEL_W):
    im = Image.fromarray(rgb)
    h = round(im.height * width / im.width)
    im = im.resize((width, h), Image.LANCZOS)
    bar = Image.new("RGB", (width, 56), (246, 243, 236))
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(bar).text((10, 14), title, fill=(27, 42, 46), font=font)
    canvas = Image.new("RGB", (width, h + 56))
    canvas.paste(bar, (0, 0))
    canvas.paste(im, (0, 56))
    return np.asarray(canvas)


def worst_window(gt, pm, frac=0.4, step=0.05):
    """Top-left corner (fractions) of the window where fat/no-fat disagrees most."""
    diff = ((gt > 0) != (pm > 0)).astype(np.float64)
    H, W = diff.shape
    ii = np.pad(diff.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    wh, ww = int(H * frac), int(W * frac)
    best, pos = -1, (0, 0)
    for y in range(0, H - wh + 1, max(1, int(H * step))):
        for x in range(0, W - ww + 1, max(1, int(W * step))):
            v = ii[y + wh, x + ww] - ii[y, x + ww] - ii[y + wh, x] + ii[y, x]
            if v > best:
                best, pos = v, (x, y)
    return pos[0], pos[1], ww, wh


def grid2(tiles, ncols=2):
    rows = []
    for i in range(0, len(tiles), ncols):
        row = tiles[i:i + ncols]
        row += [np.zeros_like(row[0])] * (ncols - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def find_pred(pred_root, subset, label, name):
    hits = sorted(pred_root.glob(f"*_{subset}_{label}/masks/{label}/{name}_{label}_instances.tif"))
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-root", required=True, help="folder that contains the <time>_<subset>_<label> run folders")
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", action="store_true",
                    help="also write <name>_zoom.png: a 2x2 grid on one enlarged region")
    ap.add_argument("--crop", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                    help="zoom region as fractions of the image (default: region where the LAST label "
                         "disagrees most with the ground truth, so it shows the worst area, not the best)")
    a = ap.parse_args()

    data, pred_root, out = Path(a.data).expanduser(), Path(a.pred_root).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    names = json.load(open(Path(a.split).expanduser()))["splits"][a.subset]

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
        panels = [panel(rgb, "image"),
                  panel(outline(rgb, gt, GREEN), f"ground truth  fat {100 * (gt > 0).mean():.1f}%")]
        preds = {}
        for label in a.labels:
            f = find_pred(pred_root, a.subset, label, name)
            if f is None:
                print(f"no prediction for {label}: {name}")
                continue
            pm = tifffile.imread(f)
            preds[label] = pm
            panels.append(panel(outline(rgb, pm, RED), f"{label}  fat {100 * (pm > 0).mean():.1f}%"))
        h = max(p.shape[0] for p in panels)
        panels = [np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0))) for p in panels]
        dest = out / f"{name.replace(' ', '_')}.png"
        Image.fromarray(np.concatenate(panels, axis=1)).save(dest)
        print("wrote", dest)
        if a.zoom and preds:
            H, W = gt.shape
            if a.crop:
                x0, y0 = int(a.crop[0] * W), int(a.crop[1] * H)
                x1, y1 = int(a.crop[2] * W), int(a.crop[3] * H)
                note = "chosen region"
            else:
                x0, y0, ww, wh = worst_window(gt, preds[a.labels[-1]] if a.labels[-1] in preds else list(preds.values())[-1])
                x1, y1 = x0 + ww, y0 + wh
                note = "region of largest disagreement"
            sl = (slice(y0, y1), slice(x0, x1))
            tiles = [panel(rgb[sl], f"image ({note})", 1000),
                     panel(outline(rgb, gt, GREEN)[sl], "ground truth", 1000)]
            for label, pm in preds.items():
                tiles.append(panel(outline(rgb, pm, RED)[sl], label, 1000))
            zdest = out / f"{name.replace(' ', '_')}_zoom.png"
            Image.fromarray(grid2(tiles)).save(zdest)
            print("wrote", zdest)


if __name__ == "__main__":
    main()
