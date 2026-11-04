"""Active-learning cycle 0 for porcine bright-field: find fat cells the models see but the annotations do not.

Adapted from the H. pylori "union approach" (ActiveLearning_Union_Approach.ipynb):
  1. run two zero-shot models (cpdino-vitb, cpdino) on whole images;
  2. a predicted cell is a CANDIDATE if it lies in unannotated territory (<10% of it overlaps any annotation);
  3. union of both models; cells proposed by BOTH are ranked first (likely missed annotations);
  4. top-N candidates, capped per image, exported as thumbnails + review.csv for the expert (decision column blank);
  5. candidate masks are saved at full resolution so confirmed cells can be pasted back exactly (coordinates in the CSV).

Held-out test images (split porcine_bf_v1) are NOT put in the review package; their pool is reported only as a diagnostic.
Nothing here is verified by an expert yet: candidates are proposals, not new ground truth.
"""
import argparse, csv, json, os, sys, time, platform
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries, relabel_sequential

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path.home() / "cf_work/data/Training_porcine_bf"))
ap.add_argument("--split", default=str(Path.home() / "cellfoundry_push/splits/porcine_bf_v1.json"))
ap.add_argument("--out", default=str(Path.home() / "cf_work/al_cycle0"))
ap.add_argument("--models", nargs="+", default=["cpdino-vitb", "cpdino"])
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--top-n", type=int, default=100)
ap.add_argument("--per-image-cap", type=int, default=6)
ap.add_argument("--new-frac", type=float, default=0.10, help="candidate if < this fraction of the cell overlaps an annotation")
ap.add_argument("--agree-iou", type=float, default=0.30)
ap.add_argument("--min-area", type=int, default=150)
ap.add_argument("--layers", action="store_true", help="also write base/gt/cand layers per candidate (for the drawing review page)")
ap.add_argument("--no-caption", action="store_true", help="do not burn the candidate id/model/confidence into the thumbnail (use for expert review)")
a = ap.parse_args()

OUT = Path(a.out)
for d in ("masks", "candidate_masks", "thumbnails", "reports", "layers"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

split = json.load(open(a.split))["splits"]
split_of = {n: s for s, names in split.items() for n in names}


def condition(name):
    low = name.lower()
    return "control" if ("negativ" in low or "ctrl" in low) else "differentiated"


def load_labels(p):
    arr = tifffile.imread(p) if p.suffix.lower() in (".tif", ".tiff") else np.array(Image.open(p))
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr.max(axis=0)
    return relabel_sequential(arr.astype(np.int64))[0].astype(np.int32)


def find_pairs(d):
    d = Path(d)
    out = []
    for img in sorted(d.glob("*_image.*")):
        base = img.stem[: -len("_image")]
        gt = next((d / f"{base}_instances{e}" for e in (".tif", ".tiff", ".png") if (d / f"{base}_instances{e}").exists()), None)
        out.append((base, img, gt))
    return out[: a.limit] if a.limit else out


pairs = find_pairs(a.data)
print(f"images: {len(pairs)} | models: {a.models}", flush=True)

# ------------------------------------------------------------------ run models (cached)
from cellpose import models as cpm, io as cpio
import torch
use_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


preds = {}  # (model, base) -> (masks, mean_prob per label)
for mname in a.models:
    t0 = time.time()
    model = cpm.CellposeModel(gpu=use_gpu, pretrained_model=mname)
    print(f"[{mname}] loaded, dtype {next(model.net.parameters()).dtype}, gpu={use_gpu} ({time.time()-t0:.0f}s)", flush=True)
    (OUT / "masks" / mname).mkdir(exist_ok=True)
    for i, (base, imgp, gtp) in enumerate(pairs, 1):
        cache = OUT / "masks" / mname / f"{base}_{mname}.npz"
        if cache.exists():
            z = np.load(cache)
            preds[(mname, base)] = (z["masks"], z["prob"])
            continue
        img = cpio.imread(str(imgp))
        t1 = time.time()
        masks, flows, _ = model.eval(img, batch_size=8, flow_threshold=0.4, cellprob_threshold=0.0,
                                     channel_axis=-1 if img.ndim == 3 else None)
        masks = np.asarray(masks).astype(np.uint16)
        cp = sigmoid(np.asarray(flows[2], dtype=np.float32))
        n = int(masks.max())
        prob = np.array(ndi.mean(cp, labels=masks, index=np.arange(1, n + 1))) if n else np.zeros(0)
        np.savez_compressed(cache, masks=masks, prob=prob.astype(np.float32))
        preds[(mname, base)] = (masks, prob.astype(np.float32))
        print(f"  [{mname}] {i}/{len(pairs)} {base[-18:]}: {n} cells ({time.time()-t1:.1f}s)", flush=True)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ------------------------------------------------------------------ candidates
gt_areas = []
for base, imgp, gtp in pairs:
    if gtp:
        g = load_labels(gtp)
        gt_areas += list(np.bincount(g.ravel())[1:])
gt_areas = np.array([x for x in gt_areas if x > 0])
print(f"GT cell area px: median {np.median(gt_areas):.0f}, 5th pct {np.percentile(gt_areas,5):.0f}; min candidate area = {a.min_area}", flush=True)

cat_stats = {m: {"all": 0, "inside_annotation": 0, "partial": 0, "new": 0, "candidates": 0} for m in a.models}
rows, per_image = [], []
cand_label_imgs = {}
for base, imgp, gtp in pairs:
    gt = load_labels(gtp) if gtp else np.zeros(preds[(a.models[0], base)][0].shape, np.int32)
    gtb = gt > 0
    H, W = gt.shape
    per_model = {}
    for m in a.models:
        masks, prob = preds[(m, base)]
        n = int(masks.max())
        areas = np.bincount(masks.ravel(), minlength=n + 1)
        inter = np.bincount(masks[gtb].ravel(), minlength=n + 1)
        frac = np.zeros(n + 1)
        frac[1:] = inter[1:] / np.maximum(areas[1:], 1)
        cands = []
        for lab in range(1, n + 1):
            if areas[lab] == 0:
                continue
            cat_stats[m]["all"] += 1
            f = frac[lab]
            key = "inside_annotation" if f >= 0.5 else ("partial" if f >= a.new_frac else "new")
            cat_stats[m][key] += 1
            if f < a.new_frac and areas[lab] >= a.min_area:
                cands.append(dict(label=lab, area=int(areas[lab]), frac=float(f), conf=float(prob[lab - 1]), model=m))
                cat_stats[m]["candidates"] += 1
        per_model[m] = (masks, areas, cands)
    # union with agreement: match every candidate of model 0 to a candidate of the other model
    first, others = a.models[0], a.models[1:]
    merged = []
    used_other = set()
    m0_masks, m0_areas, m0_c = per_model[first]
    for c in m0_c:
        region = m0_masks == c["label"]
        entry = dict(c, models=first, masks=m0_masks, conf_mean=c["conf"])
        for om in others:
            om_masks, om_areas, om_c = per_model[om]
            ov = np.bincount(om_masks[region].ravel(), minlength=om_masks.max() + 1)
            ov[0] = 0
            if ov.sum():
                j = int(ov.argmax())
                iou = ov[j] / (c["area"] + om_areas[j] - ov[j])
                oc = next((x for x in om_c if x["label"] == j), None)
                if oc is not None and iou >= a.agree_iou:
                    entry["models"] = f"both"
                    entry["conf_mean"] = (c["conf"] + oc["conf"]) / 2
                    used_other.add((om, j))
        merged.append(entry)
    for om in others:
        om_masks, om_areas, om_c = per_model[om]
        for oc in om_c:
            if (om, oc["label"]) not in used_other:
                merged.append(dict(oc, models=om, masks=om_masks, conf_mean=oc["conf"]))
    union_area = sum(e["area"] for e in merged)
    per_image.append(dict(image=base, split=split_of.get(base, "unlisted"), condition=condition(base),
                          gt_cells=int(gt.max()), gt_area_pct=100 * gtb.mean(), n_candidates=len(merged),
                          n_both=sum(e["models"] == "both" for e in merged), cand_area_pct=100 * union_area / (H * W)))
    for e in merged:
        e.update(image=base, imgp=imgp, split=split_of.get(base, "unlisted"), condition=condition(base), gt=gt, H=H, W=W)
        rows.append(e)

# ------------------------------------------------------------------ selection (train+val only, diversity cap)
eligible = [r for r in rows if r["split"] in ("train", "val")]
eligible.sort(key=lambda r: ((r["models"] == "both"), r["conf_mean"]), reverse=True)
chosen, per_img = [], {}
for r in eligible:
    if per_img.get(r["image"], 0) >= a.per_image_cap:
        continue
    chosen.append(r)
    per_img[r["image"]] = per_img.get(r["image"], 0) + 1
    if len(chosen) >= a.top_n:
        break

COL = {"both": (255, 215, 0), "cpdino-vitb": (255, 40, 40), "cpdino": (40, 90, 255)}
CROP = 256
csv_rows = []
cand_imgs = {}
for k, r in enumerate(chosen, 1):
    cid = f"C{k:03d}"
    masks = r["masks"]
    region = masks == r["label"]
    ys, xs = np.nonzero(region)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cl = int(np.clip(cx - CROP // 2, 0, r["W"] - CROP)); ct = int(np.clip(cy - CROP // 2, 0, r["H"] - CROP))
    img = np.array(Image.open(r["imgp"]).convert("RGB"))
    tile = img[ct:ct + CROP, cl:cl + CROP].copy()
    gtb = find_boundaries(r["gt"][ct:ct + CROP, cl:cl + CROP], mode="inner")
    cb = find_boundaries(region[ct:ct + CROP, cl:cl + CROP], mode="inner")
    cb = ndi.binary_dilation(cb, iterations=1)
    tile[gtb] = (0, 255, 0)
    tile[cb] = COL[r["models"]]
    pil = Image.fromarray(tile).resize((CROP * 2, CROP * 2), Image.NEAREST)
    if not a.no_caption:
        ImageDraw.Draw(pil).text((6, 4), f"{cid} {r['models']} p={r['conf_mean']:.2f}", fill=(255, 255, 255))
    if a.layers:
        up = lambda m: np.repeat(np.repeat(m, 2, 0), 2, 1)
        base = Image.fromarray(img[ct:ct + CROP, cl:cl + CROP]).resize((CROP * 2, CROP * 2), Image.BICUBIC)
        base.save(OUT / "layers" / f"{cid}_base.png")
        def layer(lab, colour, grow):
            b = find_boundaries(up(lab), mode="inner")
            if grow:
                b = ndi.binary_dilation(b, iterations=1)
            rgba = np.zeros(b.shape + (4,), np.uint8)
            rgba[b] = colour + (255,)
            return Image.fromarray(rgba, "RGBA")
        layer(r["gt"][ct:ct + CROP, cl:cl + CROP].astype(np.int32), (0, 255, 0), False).save(OUT / "layers" / f"{cid}_gt.png")
        import cv2
        m512 = up(region[ct:ct + CROP, cl:cl + CROP]).astype(np.uint8)
        cs, _ = cv2.findContours(m512, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs:  # editable outline: 64 evenly spaced points, in 512-px canvas coordinates
            c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
            cc = np.vstack([c, c[:1]])
            cum = np.concatenate([[0], np.cumsum(np.sqrt((np.diff(cc, axis=0) ** 2).sum(1)))])
            t = np.linspace(0, cum[-1], 65)[:-1]
            pts = [[round(float(x), 1), round(float(y), 1)] for x, y in zip(np.interp(t, cum, cc[:, 0]), np.interp(t, cum, cc[:, 1]))]
            json.dump(pts, open(OUT / "layers" / f"{cid}_poly.json", "w"))
        layer(region[ct:ct + CROP, cl:cl + CROP].astype(np.int32), (255, 215, 0), True).save(OUT / "layers" / f"{cid}_cand.png")
    tn = f"{cid}.png"
    pil.save(OUT / "thumbnails" / tn)
    lab_img = cand_imgs.setdefault(r["image"], np.zeros((r["H"], r["W"]), np.uint16))
    lab_img[region] = k
    csv_rows.append(dict(candidate_id=cid, image=r["image"], split=r["split"], condition=r["condition"], models=r["models"],
                         conf_mean=round(r["conf_mean"], 3), area_px=r["area"], frac_overlap_gt=round(r["frac"], 3),
                         bbox_x=int(x0), bbox_y=int(y0), bbox_w=int(x1 - x0), bbox_h=int(y1 - y0),
                         crop_x=cl, crop_y=ct, crop_w=CROP, crop_h=CROP, thumbnail=tn, decision="", notes=""))
for base, lab in cand_imgs.items():
    tifffile.imwrite(OUT / "candidate_masks" / f"{base}_candidates.tif", lab)

with open(OUT / "review.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["candidate_id"])
    w.writeheader(); w.writerows(csv_rows)
with open(OUT / "reports" / "per_image.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(per_image[0].keys())); w.writeheader()
    for p in per_image:
        w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()})

# contact sheet of the first 20 for a quick look
if csv_rows:
    n = min(20, len(csv_rows)); cols = 5; rowsn = (n + cols - 1) // cols; s = 256
    sheet = Image.new("RGB", (cols * s, rowsn * s))
    for i in range(n):
        sheet.paste(Image.open(OUT / "thumbnails" / csv_rows[i]["thumbnail"]).resize((s, s)), ((i % cols) * s, (i // cols) * s))
    sheet.save(OUT / "reports" / "contact_sheet_top20.png")

summary = dict(
    n_images=len(pairs), models=a.models, min_area=a.min_area, new_frac=a.new_frac, agree_iou=a.agree_iou,
    predicted_cells_by_category=cat_stats,
    candidates_total=len(rows), candidates_both=sum(r["models"] == "both" for r in rows),
    candidates_in_review_package=len(chosen), review_package_images=len(per_img),
    review_package_both=sum(r["models"] == "both" for r in chosen),
    held_out_test_candidates_not_exported=sum(r["split"] == "test" for r in rows),
    versions=dict(python=platform.python_version(), torch=torch.__version__),
)
json.dump(summary, open(OUT / "reports" / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
