"""Segment a whole group folder with chosen Cellpose models and score every image
with the full metric set (fat %, fat % error, foreground Dice, count, F1 / SA).

Like scripts/run_batch.py, but:
  * --models is REQUIRED (no default): you always say which models run;
  * scoring uses ``score`` / ``summarise`` from ``score_masks_v2.py`` (must be in
    the SAME folder as this file), so the batch run, the re-scoring and the single-image
    tool share one metric definition. Each image is scored from its SAVED mask file, so
    re-scoring that folder later gives identical numbers by construction;
  * W&B (optional) logs one run per model, whose summary is that model's averages
    over the dataset, not the last image.

For every model and every ``*_image.*`` file in --group-dir:
    segment -> save mask (+ overlay) -> score against ``<name>_instances.*``

Outputs go to a NEW folder per run (nothing is overwritten):
    <out-root>\\batch_v2\\<YYYYmmdd-HHMMSS>_<group>\\
        per_image.csv     one row per (model, image): all metrics + time + GPU memory
        summary.csv       one row per model x condition (differentiated / control / all)
        run_config.json   settings, image list, versions, git commit
        masks\\<model>\\<name>_<model>_instances.tif
        overlays\\<model>\\<name>_<model>_overlay.png   (green = GT, red = prediction, yellow = both)

summary.csv: two kinds of "average" over the dataset
    mean over images  every IMAGE counts equally (fat % error, Dice, F1 mean / std)
    pooled            every CELL counts equally (pooled F1@0.5, pooled count error:
                      TP/FP/FN and cell counts are summed over all images first)
    Area metrics use every image (also controls); count and F1 use images with >= 1 GT cell.

Usage (uv, cellpose-env; from cellpose-env\\):
    uv run python <path>\\run_batch_v2.py `
        --group-dir "<data>\\Training_porcine_bf" --out-root "<...>\\Cloud_results" `
        --models cpsam_v2 cpsam cpdino cpdino-vitb
    quick test:  add --limit 2
    W&B:         add --wandb --wandb-project <project> [--wandb-group <group>] [--wandb-entity <team>]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_masks_v2 import (  # noqa: E402  one shared metric definition
    GT_EXTS, PER_IMAGE_FIELDS, THRESHOLDS, fmt, load_labels, log_wandb, score_row, stats,
    summarise, versions, write_csv,
)

IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
BATCH_FIELDS = PER_IMAGE_FIELDS + ["time_s", "peak_gpu_gb"]


# ----------------------------------------------------------------------------- data
def find_pairs(group_dir: Path, pattern: str, limit: int | None) -> list[tuple[Path, Path | None]]:
    """(image, GT) pairs: <name>_image.<ext> -> <name>_instances.{tif,tiff,png} (top level)."""
    images = sorted(p for p in group_dir.glob(pattern)
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.endswith("_image"))
    pairs = []
    for img in images:
        base = img.stem[: -len("_image")]
        gt = next((group_dir / f"{base}_instances{e}" for e in GT_EXTS
                   if (group_dir / f"{base}_instances{e}").exists()), None)
        pairs.append((img, gt))
    return pairs[:limit] if limit else pairs


def save_overlay(path: Path, img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    """GT outlines green, predicted outlines red, both yellow (same as run_batch.py)."""
    from skimage import io as skio
    from skimage.segmentation import find_boundaries

    im = np.asarray(img)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    im = im[..., :3].astype(np.float32)
    lo, hi = np.percentile(im, [1, 99])
    im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
    out = im.mean(-1, keepdims=True).repeat(3, -1) * 0.8
    gb, pb = find_boundaries(gt, mode="inner"), find_boundaries(pred, mode="inner")
    out[gb], out[pb], out[gb & pb] = [0, 1, 0], [1, 0, 0], [1, 1, 0]
    skio.imsave(path, (out * 255).astype(np.uint8), check_contrast=False)


# ---------------------------------------------------------------------------- model
def build_model(name: str, use_gpu: bool):
    """Built-in Cellpose 4 model. Refuses unknown names (Cellpose would silently fall back
    to another model, which would mislabel every result)."""
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


def run_model(name: str, pairs, a, run_dir: Path) -> tuple[list[dict], str]:
    """Segment + score every image with one model. Returns rows and the model's dtype."""
    from cellpose import io as cpio

    torch = None
    if not a.cpu:
        try:
            import torch as _t
            torch = _t if _t.cuda.is_available() else None
        except Exception:
            torch = None

    print(f"\n=== {name}: loading model")
    model = build_model(name, use_gpu=torch is not None)
    dtype = str(next(model.net.parameters()).dtype)  # Cellpose 4 defaults to bfloat16
    print(f"    loaded as {dtype}")

    first = next((img for img, gt in pairs if gt is not None), None)
    if first is not None:  # warm-up: not timed, not recorded
        t0 = time.time()
        segment(model, cpio.imread(str(first)), a)
        print(f"    warm-up: {time.time() - t0:.1f} s (not recorded)")

    mask_dir = run_dir / "masks" / name
    ov_dir = None if a.no_overlays else run_dir / "overlays" / name
    mask_dir.mkdir(parents=True, exist_ok=True)
    if ov_dir:
        ov_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (img_path, gt_path) in enumerate(pairs, 1):
        base = img_path.stem[: -len("_image")]
        if gt_path is None:
            print(f"    [{i}/{len(pairs)}] {base}: no GT mask -> skipped")
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

        pred_path = mask_dir / f"{base}_{name}_instances.tif"
        tifffile.imwrite(pred_path, pred.astype(np.uint16))
        row = score_row(name, run_dir.name, pred_path, gt_path)  # scored from the saved file
        row.update(time_s=round(dt_s, 2), peak_gpu_gb=fmt(peak, 2))
        rows.append(row)
        if ov_dir:
            save_overlay(ov_dir / f"{base}_{name}_overlay.png", img, load_labels(gt_path), load_labels(pred_path))

        msg = f"fat% err={row['fat_pct_error']:+} pp  Dice={row['fg_dice']}"
        if row["F1@0.5"] != "":
            msg += f"  F1@0.5={row['F1@0.5']}  cells {row['n_pred']}/{row['n_true']}"
        else:
            msg += f"  (no annotated cells, false detections={row['n_pred']})"
        print(f"    [{i}/{len(pairs)}] {base}: {msg}  ({dt_s:.1f} s)")

    del model
    if torch:
        torch.cuda.empty_cache()
    return rows, dtype


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="Cellpose models to run, e.g. cpsam_v2 cpsam cpdino cpdino-vitb (no default)")
    p.add_argument("--group-dir", type=Path, required=True, help="folder with *_image.* and *_instances.* files")
    p.add_argument("--out-root", type=Path, required=True, help="results root, e.g. ...\\Cloud_results")
    p.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    p.add_argument("--pattern", default="*_image.*", help='glob for a subset, e.g. "*Positiv_1*_image.*"')
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--diameter", type=float, default=None,
                   help="expected cell diameter in px; Cellpose resizes by 30/diameter (None = no resizing)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-overlays", action="store_true", help="don't save overlay PNGs")
    p.add_argument("--note", default="", help="free text stored in run_config.json (and W&B config)")
    p.add_argument("--wandb", action="store_true", help="log one W&B run per model (metrics only, no images)")
    p.add_argument("--wandb-project", default=None, help="W&B project (required with --wandb)")
    p.add_argument("--wandb-group", default="zero-shot-porcine-bf", help="W&B group for these runs")
    p.add_argument("--wandb-entity", default=None, help="W&B team name (or set WANDB_ENTITY)")
    a = p.parse_args(argv)
    if a.wandb and not a.wandb_project:
        p.error("--wandb needs --wandb-project <name>")
    if len(set(a.models)) != len(a.models):
        p.error("a model is listed twice in --models")

    pairs = find_pairs(a.group_dir, a.pattern, a.limit)
    if not pairs:
        raise SystemExit(f"No '*_image.*' files in {a.group_dir}")
    n_gt = sum(gt is not None for _, gt in pairs)
    print(f"group: {a.group_dir.name} | images: {len(pairs)} | with GT: {n_gt} | models: {a.models}")

    tag = "" if a.diameter is None else f"_diam{a.diameter:g}"
    tag += "" if (a.flow_threshold, a.cellprob_threshold) == (0.4, 0.0) else \
        f"_ft{a.flow_threshold:g}_cp{a.cellprob_threshold:g}"
    run_dir = a.out_root / "batch_v2" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.group_dir.name}{tag}"
    run_dir.mkdir(parents=True)  # fails if it exists: never overwrite
    config = {"args": {k: str(v) for k, v in vars(a).items()}, "group": a.group_dir.name,
              "images": [img.name for img, _ in pairs], "iou_thresholds": THRESHOLDS,
              "versions": versions(), "started": dt.datetime.now().isoformat()}
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict] = []
    dtypes: dict[str, str] = {}
    for name in a.models:
        model_rows, dtypes[name] = run_model(name, pairs, a, run_dir)
        rows += model_rows
        write_csv(run_dir / "per_image.csv", rows, BATCH_FIELDS)  # saved after each model
    if not rows:
        raise SystemExit("Nothing scored (no image had a GT mask).")

    summary = summarise(rows)
    for s in summary:  # timing per model x condition
        sub = [r for r in rows if r["model"] == s["model"] and (s["condition"] == "all" or r["condition"] == s["condition"])]
        s["time_s_mean"] = fmt(stats([float(r["time_s"]) for r in sub])["mean"], 2)
        s["peak_gpu_gb_max"] = fmt(max((float(r["peak_gpu_gb"]) for r in sub if r["peak_gpu_gb"] != ""),
                                       default=float("nan")), 2)
    write_csv(run_dir / "summary.csv", summary)
    config.update(model_dtypes=dtypes, finished=dt.datetime.now().isoformat())
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if a.wandb:
        for name in a.models:
            mrows = [r for r in rows if r["model"] == name]
            if mrows:
                log_wandb(a, name, mrows, summary, {**config, "versions": {**config["versions"],
                                                                         "model_dtype": dtypes[name]}})

    print("\n=== summary per model and condition")
    print("    mean = average over images (each image counts equally); pooled = all cells summed first")
    print(f"{'model':<12} {'condition':<15} {'n':>3} {'fat% err':>9} {'|fat% err|':>10} {'Dice':>6} "
          f"{'cnt err(pool)':>13} {'F1@0.5':>7} {'F1 pool':>7} {'F1@0.7':>7} {'s/img':>6}")
    for s in summary:
        print(f"{s['model']:<12} {s['condition']:<15} {s['n_images']:>3} {s['fat_pct_error_mean']:>9} "
              f"{s['fat_pct_abs_error_mean']:>10} {s['fg_dice_mean']:>6} {s['pooled_count_error']:>13} "
              f"{s['F1@0.5_mean']:>7} {s['pooled_F1@0.5']:>7} {s['F1@0.7_mean']:>7} {s['time_s_mean']:>6}")
    v = config["versions"]
    print(f"\ncode: {Path(__file__).name} + {v['script']} @ {v['repo_commit']} ({v['repo_branch']})")
    if v["scripts_uncommitted_changes"]:
        print("WARNING: the scripts that ran have uncommitted changes -> results do not match the recorded commit.")
    print(f"saved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
