"""Turn the expert's answers (decisions.csv + edits.json from the review page) into full-size masks of accepted cells.

For every source image one label image  <image>_accepted_cells.tif  is written (0 = background):
  * APPROVED, yellow outline reshaped      -> the reshaped outline REPLACES the model mask (label = candidate number, C007 -> 7);
                                              any extra outlines drawn on the same picture are extra cells (label 1000 + number)
  * APPROVED, outline(s) drawn, no reshape -> the drawn outline(s) REPLACE the model mask (same label)
  * APPROVED, nothing changed              -> the model's mask of that candidate
  * REJECTED, outline(s) drawn             -> the outlines are extra fat cells nobody had marked (label = 1000 + candidate number)
  * REJECTED, nothing drawn                -> nothing is written (a hard negative)
A summary.csv lists every accepted cell and where it came from. Paste the cells into the training annotations at the
same pixel coordinates (the label images have exactly the size of the original images).

    python export_accepted_masks.py --edits edits.json --cycle-dir . --out accepted_cycle0
"""
import argparse, csv, json, re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--edits", required=True, help="edits.json downloaded from the review page")
ap.add_argument("--cycle-dir", default=".")
ap.add_argument("--out", required=True)
a = ap.parse_args()

cyc, out = Path(a.cycle_dir), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
review = {r["candidate_id"]: r for r in csv.DictReader(open(cyc / "review.csv", encoding="utf-8"))}
edits = json.load(open(a.edits, encoding="utf-8"))["items"]

safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s)
canvases, summary = {}, []


def canvas_for(image):
    if image not in canvases:
        src = cyc / "candidate_masks" / f"{image}_candidates.tif"
        shape = tifffile.imread(src).shape
        canvases[image] = np.zeros(shape, np.uint16)
    return canvases[image]


def raster(poly, shape):
    im = Image.new("I", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=1)
    return np.array(im) > 0


for e in edits:
    cid, image, dec, outlines = e["candidate_id"], e["source_image"], e["decision"], e.get("outlines", [])
    adjusted = e.get("adjusted_outline")
    if cid not in review:
        continue
    num = int(cid[1:])
    can = canvas_for(image)
    if dec == "approve":
        if adjusted:
            can[raster(adjusted, can.shape)] = num
            summary.append(dict(candidate_id=cid, image=image, source="expert_reshaped_outline", n_outlines=len(outlines), label=num))
            for poly in outlines:
                can[raster(poly, can.shape)] = 1000 + num
            if outlines:
                summary.append(dict(candidate_id=cid, image=image, source="extra_cell_on_approved", n_outlines=len(outlines), label=1000 + num))
        elif outlines:
            for poly in outlines:
                m = raster(poly, can.shape)
                can[m] = num
            summary.append(dict(candidate_id=cid, image=image, source="expert_outline", n_outlines=len(outlines), label=num))
        else:
            cand = tifffile.imread(cyc / "candidate_masks" / f"{image}_candidates.tif") == num
            can[cand] = num
            summary.append(dict(candidate_id=cid, image=image, source="model_mask", n_outlines=0, label=num))
    elif dec == "reject" and outlines:
        for poly in outlines:
            can[raster(poly, can.shape)] = 1000 + num
        summary.append(dict(candidate_id=cid, image=image, source="extra_cell_on_rejected", n_outlines=len(outlines), label=1000 + num))

for image, can in canvases.items():
    tifffile.imwrite(out / f"{image}_accepted_cells.tif", can, compression="zlib")
with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["candidate_id", "image", "source", "n_outlines", "label"])
    w.writeheader(); w.writerows(summary)

n_model = sum(s["source"] == "model_mask" for s in summary)
n_exp = sum(s["source"] in ("expert_outline", "expert_reshaped_outline") for s in summary)
n_extra = sum(s["source"] in ("extra_cell_on_rejected", "extra_cell_on_approved") for s in summary)
n_reshaped = sum(s["source"] == "expert_reshaped_outline" for s in summary)
print(f"accepted with model mask: {n_model} | accepted with expert outline: {n_exp} (of which reshaped yellow outline: {n_reshaped}) | extra cells drawn: {n_extra}")
print(f"wrote {len(canvases)} label image(s) and summary.csv to {out}")
