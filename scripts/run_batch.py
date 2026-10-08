"""Batch zero-shot evaluation of Cellpose models on a whole group folder.

For every model and every ``*_image.*`` file in the group folder:
segment -> save the label image -> score against ``<name>_instances.*`` with the
same metric code as ``score_masks.py`` / ``run_cpsam.py`` (Cellpose
``metrics.average_precision``: one-to-one IoU matching, a match counts at IoU >= t).

Outputs go to a NEW folder per run (nothing is overwritten):

    Cloud_results\\batch\\<YYYYmmdd-HHMMSS>_<group>\\
        per_image.csv     one row per (model, image)
        summary.csv       one row per model x condition: mean +- std over images, pooled F1, counts, timing
        run_config.json   all settings + versions (for the decision log / reproducibility)
        masks\\<model>\\<name>_<model>_instances.tif
        overlays\\<model>\\<name>_<model>_overlay.png   (green = GT, red = prediction, yellow = both)

per_image.csv columns
    model, image, series (experiment batch), condition (differentiated / control)
    n_true, n_pred, count_diff = n_pred - n_true, count_error = count_diff / n_true
    TP@0.5, FP@0.5, FN@0.5   (one-to-one IoU matching, match if IoU >= 0.5)
    precision@0.5 = TP / n_pred, recall@0.5 = TP / n_true
    F1@0.5  = 2TP / (2TP + FP + FN)            headline: are the cells found?
    F1@0.75 = same with IoU >= 0.75            outline accuracy (matters for cell size)
    gt_median_diam_px (typical annotated cell diameter), time_s
    (Cellpose's "AP"/SA = TP/(TP+FP+FN) is not stored: SA = F1 / (2 - F1).)
    Images with 0 annotated cells: F1 is undefined (left empty); n_pred = false detections.

summary.csv: one row per model x condition (differentiated = Positiv/_Diff,
control = Negativ/_Ctrl) plus 'all'
    F1@0.5 mean / std / min / max over images, pooled F1@0.5 (from summed TP/FP/FN, so
    every cell counts equally), F1@0.75 mean, precision / recall mean, median |count error|
    (robust to a few disastrous images), pooled count error, mean time, peak GPU memory.
    Timing excludes a warm-up run.

Usage (env ``cf-core``, from the repo folder):
    python scripts\\run_batch.py                                  # all 4 Cellpose models, porcine BF
    python scripts\\run_batch.py --models cpdino cpdino-vitb
    python scripts\\run_batch.py --group-dir "C:\\...\\Cloud\\Training_porcine_ph"
    python scripts\\run_batch.py --limit 2                        # quick test on 2 images
    python scripts\\run_batch.py --pattern "*Positiv_2[1-6]_image.*" --diameter 60   # subset + resize test
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_masks import GT_EXTS, load_labels, score  # noqa: E402  (same metric code)

DEFAULT_GROUP_DIR = Path(r"C:\Users\Priyam\Desktop\Project\Cloud\Training_porcine_bf")
DEFAULT_OUT_ROOT = Path(r"C:\Users\Priyam\Desktop\Project\Cloud_results")
DEFAULT_MODELS = ["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"]
IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

PER_IMAGE_FIELDS = [
    "model", "image", "series", "condition",
    "n_true", "n_pred", "count_diff", "count_error",
    "TP@0.5", "FP@0.5", "FN@0.5", "precision@0.5", "recall@0.5", "F1@0.5", "F1@0.75",
    "gt_median_diam_px", "time_s",
]


# ----------------------------------------------------------------------------- data
def find_pairs(group_dir: Path, limit: int | None, pattern: str = "*_image.*") -> list[tuple[Path, Path | None]]:
    """List (image, gt) pairs: ``<name>_image.<ext>`` -> ``<name>_instances.{tif,tiff,png}``.
    ``pattern`` is a glob on file names, e.g. "*Positiv_2[1-6]_image.*" for a subset."""
    images = sorted(
        p for p in group_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.endswith("_image")
    )
    pairs = []
    for img in images:
        base = img.stem[: -len("_image")]
        gt = next((group_dir / f"{base}_instances{e}" for e in GT_EXTS
                   if (group_dir / f"{base}_instances{e}").exists()), None)
        pairs.append((img, gt))
    return pairs[:limit] if limit else pairs


def parse_condition(name: str) -> str:
    """Culture condition from the filename (IMES naming):
    'Positiv' / '_Diff' -> 'differentiated'; 'Negativ' / '_Ctrl' -> 'control'; else 'other'.
    NOTE: control images can still contain a few annotated cells (Negativ_2: 5, Ctrl: 2-5),
    so they are scored like any image and summarised separately."""
    low = name.lower()
    if "negativ" in low or re.search(r"(^|[_\s])ctrl($|[_\s])", low):
        return "control"
    if "positiv" in low or re.search(r"(^|[_\s])diff($|[_\s])", low):
        return "differentiated"
    return "other"


def parse_series(name: str) -> str:
    """Experiment series = first token of the filename (e.g. 'CLD-AIM-04', 'PAD11a').
    Needed later for leakage-free splits (keep a series together or stratify by it)."""
    return re.split(r"[\s_]", name.strip())[0]


def save_overlay(path: Path, img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    """Image with GT outlines in green and predicted outlines in red (yellow = both)."""
    from skimage.segmentation import find_boundaries

    im = np.asarray(img)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    im = im[..., :3].astype(np.float32)
    lo, hi = np.percentile(im, [1, 99])
    im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
    gray = im.mean(-1, keepdims=True).repeat(3, -1) * 0.8
    gb, pb = find_boundaries(gt, mode="inner"), find_boundaries(pred, mode="inner")
    gray[gb] = [0, 1, 0]
    gray[pb] = [1, 0, 0]
    gray[gb & pb] = [1, 1, 0]
    from skimage import io as skio
    skio.imsave(path, (gray * 255).astype(np.uint8), check_contrast=False)


# ---------------------------------------------------------------------------- model
def build_model(name: str, use_gpu: bool):
    """Load a built-in Cellpose 4 model. Refuses unknown names (Cellpose would silently
    fall back to cpsam_v2, which would mislabel every result)."""
    from cellpose import models

    if name not in models.MODEL_NAMES:
        raise SystemExit(f"Unknown model '{name}'. Choose from {models.MODEL_NAMES}.")
    return models.CellposeModel(gpu=use_gpu, pretrained_model=name)


def segment(model, img: np.ndarray, a: argparse.Namespace) -> np.ndarray:
    masks, _flows, _styles = model.eval(
        img, batch_size=a.batch_size, diameter=a.diameter,
        channel_axis=-1 if np.asarray(img).ndim == 3 else None,
        flow_threshold=a.flow_threshold, cellprob_threshold=a.cellprob_threshold,
    )
    return np.asarray(masks)


# -------------------------------------------------------------------------- helpers
def _cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover
        return None


def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (round(x, nd) if isinstance(x, float) else x)


def _stats(vals: list[float]) -> dict:
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    if v.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "max", "median")}
    return {"mean": v.mean(), "std": v.std(ddof=1) if v.size > 1 else 0.0,
            "min": v.min(), "max": v.max(), "median": float(np.median(v))}


def versions(use_gpu: bool) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import cellpose, torch
        info.update(cellpose=cellpose.version, torch=torch.__version__,
                    cuda_available=torch.cuda.is_available())
        if use_gpu and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # pragma: no cover
        info["version_error"] = repr(e)
    try:
        info["repo_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, timeout=10).stdout.strip() or "unknown"
    except Exception:
        info["repo_commit"] = "unknown"
    return info


# ----------------------------------------------------------------------------- main
def evaluate_model(name, pairs, a, mask_dir: Path, overlay_dir: Path | None) -> list[dict]:
    from cellpose import io as cpio

    torch = _cuda() if not a.cpu else None
    print(f"\n=== {name}: loading model")
    model = build_model(name, use_gpu=not a.cpu)

    first = next((img for img, gt in pairs if gt is not None), None)
    if first is not None:  # warm-up: not timed, not recorded
        t0 = time.time()
        segment(model, cpio.imread(str(first)), a)
        print(f"    warm-up on {first.name}: {time.time() - t0:.1f} s (not recorded)")

    rows = []
    mask_dir.mkdir(parents=True, exist_ok=True)
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    for i, (img_path, gt_path) in enumerate(pairs, 1):
        base = img_path.stem[: -len("_image")]
        if gt_path is None:
            print(f"    [{i}/{len(pairs)}] {img_path.name}: no GT mask found -> skipped")
            continue
        img = cpio.imread(str(img_path))
        if torch:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        pred = segment(model, img, a)
        if torch:
            torch.cuda.synchronize()
        dt_s = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch else float("nan")

        tifffile.imwrite(mask_dir / f"{base}_{name}_instances.tif", pred.astype(np.uint16))
        gt = load_labels(gt_path)
        pred_l = load_labels_array(pred)
        res = score(gt, pred_l)
        areas = np.bincount(gt.ravel())[1:]
        areas = areas[areas > 0]
        gt_diam = float(np.median(2 * np.sqrt(areas / np.pi))) if areas.size else float("nan")
        if overlay_dir:
            save_overlay(overlay_dir / f"{base}_{name}_overlay.png", img, gt, pred_l)

        row = {k: "" for k in PER_IMAGE_FIELDS}
        row.update(model=name, image=base, series=parse_series(base),
                   condition=parse_condition(base), n_true=res["n_true"],
                   n_pred=res["n_pred"], count_diff=res["n_pred"] - res["n_true"],
                   gt_median_diam_px=_fmt(gt_diam, 1), time_s=round(dt_s, 2),
                   _peak_gpu_gb=_fmt(peak, 2))  # internal only (summary), not written per image
        if res["negative_image"]:  # 0 annotated cells: F1 undefined, n_pred = false detections
            msg = f"no annotated cells, false detections={res['n_pred']}"
        else:
            m5, m75 = res["per_t"][0.5], res["per_t"][0.75]
            n_t, n_p = res["n_true"], res["n_pred"]
            row.update({
                "count_error": _fmt(res["count_error"]),
                "TP@0.5": m5["TP"], "FP@0.5": m5["FP"], "FN@0.5": m5["FN"],
                "precision@0.5": _fmt(m5["TP"] / n_p if n_p else float("nan")),
                "recall@0.5": _fmt(m5["TP"] / n_t), "F1@0.5": _fmt(m5["F1"]),
                "F1@0.75": _fmt(m75["F1"]),
            })
            msg = (f"F1@0.5={m5['F1']:.3f} F1@0.75={m75['F1']:.3f} "
                   f"n_pred/n_true={n_p}/{n_t} count={res['count_error']:+.1%}")
        print(f"    [{i}/{len(pairs)}] {base}: {msg}  ({dt_s:.1f} s)")
        rows.append(row)

    del model
    if torch:
        torch.cuda.empty_cache()
    return rows


def load_labels_array(arr: np.ndarray) -> np.ndarray:
    """Same normalisation as score_masks.load_labels, for an in-memory prediction."""
    from skimage.segmentation import relabel_sequential
    arr = np.squeeze(np.asarray(arr))
    return relabel_sequential(arr.astype(np.int64))[0].astype(np.int32)


def _summarise_subset(name: str, condition: str, rows: list[dict]) -> dict:
    scored = [r for r in rows if r["n_true"] != 0]  # images with >= 1 annotated cell
    g = lambda key, rs=scored: [float(r[key]) for r in rs if r[key] != ""]  # noqa: E731
    tp, fp, fn = sum(g("TP@0.5")), sum(g("FP@0.5")), sum(g("FN@0.5"))
    f1, f175 = _stats(g("F1@0.5")), _stats(g("F1@0.75"))
    s = {"model": name, "condition": condition, "n_images": len(scored),
         "n_true_total": int(sum(g("n_true"))), "n_pred_total": int(sum(g("n_pred"))),
         "F1@0.5_mean": _fmt(f1["mean"]), "F1@0.5_std": _fmt(f1["std"]),
         "F1@0.5_min": _fmt(f1["min"]), "F1@0.5_max": _fmt(f1["max"]),
         "pooled_F1@0.5": _fmt(2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else float("nan")),
         "F1@0.75_mean": _fmt(f175["mean"]),
         "precision@0.5_mean": _fmt(_stats(g("precision@0.5"))["mean"]),
         "recall@0.5_mean": _fmt(_stats(g("recall@0.5"))["mean"]),
         "abs_count_error_median": _fmt(_stats([abs(x) for x in g("count_error")])["median"])}
    s["pooled_count_error"] = _fmt((s["n_pred_total"] - s["n_true_total"]) / s["n_true_total"]
                                   if s["n_true_total"] else float("nan"))
    s["time_s_mean"] = _fmt(_stats(g("time_s", rows))["mean"], 2)
    s["peak_gpu_gb_max"] = _fmt(max(g("_peak_gpu_gb", rows), default=float("nan")), 2)
    return s


def summarise(rows: list[dict]) -> list[dict]:
    """One row per model x condition (differentiated / control / other) plus one 'all' row per model."""
    out = []
    for name in dict.fromkeys(r["model"] for r in rows):
        mrows = [r for r in rows if r["model"] == name]
        for cond in dict.fromkeys(r["condition"] for r in mrows):
            out.append(_summarise_subset(name, cond, [r for r in mrows if r["condition"] == cond]))
        out.append(_summarise_subset(name, "all", mrows))
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    p.add_argument("--pattern", default="*_image.*",
                   help='glob for a subset of images, e.g. "*Positiv_2[1-6]_image.*"')
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--diameter", type=float, default=None,
                   help="expected cell diameter in px; Cellpose resizes by 30/diameter (None = no resizing). "
                        "NOTE: eval(rescale=...) is ignored by Cellpose 4.2.1, so there is no --rescale.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-overlays", action="store_true", help="don't save overlay PNGs")
    p.add_argument("--note", default="", help="free text stored in run_config.json")
    a = p.parse_args(argv)

    pairs = find_pairs(a.group_dir, a.limit, a.pattern)
    if not pairs:
        raise SystemExit(f"No '*_image.*' files in {a.group_dir}")
    n_gt = sum(gt is not None for _, gt in pairs)
    print(f"group: {a.group_dir.name} | images: {len(pairs)} | with GT: {n_gt} | models: {a.models}")

    tag = "" if a.diameter is None else f"_diam{a.diameter:g}"
    tag += "" if (a.flow_threshold, a.cellprob_threshold) == (0.4, 0.0) else f"_ft{a.flow_threshold:g}_cp{a.cellprob_threshold:g}"
    run_dir = a.out_root / "batch" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.group_dir.name}{tag}"
    run_dir.mkdir(parents=True)
    config = {"args": {k: str(v) for k, v in vars(a).items()}, "versions": versions(not a.cpu),
              "images": [img.name for img, _ in pairs], "started": dt.datetime.now().isoformat()}
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    for name in a.models:
        all_rows += evaluate_model(name, pairs, a, run_dir / "masks" / name,
                                   None if a.no_overlays else run_dir / "overlays" / name)
        write_csv(run_dir / "per_image.csv", all_rows, PER_IMAGE_FIELDS)  # saved after each model

    summary = summarise(all_rows)
    write_csv(run_dir / "summary.csv", summary)
    config["finished"] = dt.datetime.now().isoformat()
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n=== summary per model and condition (mean +- std over images; pooled = all cells together)")
    print(f"{'model':<12} {'condition':<14} {'n':>3} {'F1@0.5':>13} {'pooled':>7} {'F1@0.75':>8} "
          f"{'cells pred/GT':>14} {'med |count err|':>15} {'s/img':>6}")
    for s in summary:
        print(f"{s['model']:<12} {s['condition']:<14} {s['n_images']:>3} "
              f"{s['F1@0.5_mean']:>6} ±{s['F1@0.5_std']:<5} {s['pooled_F1@0.5']:>7} {s['F1@0.75_mean']:>8} "
              f"{str(s['n_pred_total']) + '/' + str(s['n_true_total']):>14} {s['abs_count_error_median']:>15} "
              f"{s['time_s_mean']:>6}")
    print(f"\nsaved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
