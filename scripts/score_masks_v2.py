"""Re-score saved model predictions with fat-area metrics (stand-alone, no repo changes).

Reads the label images that ``run_batch.py`` already saved
(``<batch run>\\masks\\<model>\\<name>_<model>_instances.tif``), finds the matching
ground truth (``<group-dir>\\<name>_instances.{tif,tiff,png}``) and scores them.
No model is re-run.

Metrics (same matching code as score_masks.py: Cellpose ``metrics.average_precision``,
one-to-one IoU matching, a match counts at IoU >= t):

  1. Area (IMES's primary readout)
       fat %            = (pixels with label > 0) / (H * W) * 100   (GT and prediction)
       fat % error      = fat %(pred) - fat %(GT), percentage points (+ = over-estimate)
       foreground Dice  = 2 |P n T| / (|P| + |T|)   (NaN if both are empty)
  2. Count
       count error      = (n_pred - n_true) / n_true
  3. Instances
       F1@t             = 2TP / (2TP + FP + FN), t = 0.5, 0.7, 0.75, 0.9
       SA@t             = TP / (TP + FP + FN)  (Cellpose calls this "AP"; NOT COCO AP)

Area metrics use every image (also controls). Count and F1 use images with >= 1 GT cell.

Outputs go to a NEW folder (nothing is overwritten):
    <out-root>\\rescore\\<YYYYmmdd-HHMMSS>_<group>\\
        per_image.csv, summary.csv, rescore_config.json

Usage (uv, cellpose-env; run from cellpose-env\\):
    uv run python <path>\\score_masks_v2.py `
        --batch-dirs "<run folder 1>" "<run folder 2>" ... `
        --group-dir "<data>\\Training_porcine_bf"
    Add --wandb --wandb-project <name> [--wandb-group <group>] [--wandb-entity <team>]
    to log one W&B run per model (metrics only, no images).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage import io as skio
from skimage.segmentation import relabel_sequential

THRESHOLDS = [0.5, 0.7, 0.75, 0.9]
GT_EXTS = (".tif", ".tiff", ".png")

PER_IMAGE_FIELDS = [
    "model", "source_run", "image", "series", "condition",
    "fat_pct_true", "fat_pct_pred", "fat_pct_error", "fat_pct_abs_error", "fg_dice",
    "n_true", "n_pred", "count_diff", "count_error",
    "TP@0.5", "FP@0.5", "FN@0.5", "precision@0.5", "recall@0.5",
    "F1@0.5", "F1@0.7", "F1@0.75", "F1@0.9",
]


# ------------------------------------------------------------------------- helpers
def load_labels(path: Path) -> np.ndarray:
    """Label image as 2D int32 with labels 1..N (0 = background). Same as score_masks.py."""
    if path.suffix.lower() in (".jpg", ".jpeg"):
        raise ValueError(f"Refusing lossy JPEG mask: {path}")
    arr = tifffile.imread(path) if path.suffix.lower() in (".tif", ".tiff") else skio.imread(path)
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D label image, got shape {arr.shape}: {path}")
    return relabel_sequential(arr.astype(np.int64))[0].astype(np.int32)


def parse_condition(name: str) -> str:
    """Same rule as run_batch.py: Positiv/_Diff -> differentiated; Negativ/_Ctrl -> control."""
    low = name.lower()
    if "negativ" in low or re.search(r"(^|[_\s])ctrl($|[_\s])", low):
        return "control"
    if "positiv" in low or re.search(r"(^|[_\s])diff($|[_\s])", low):
        return "differentiated"
    return "other"


def parse_series(name: str) -> str:
    return re.split(r"[\s_]", name.strip())[0]


def fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (round(x, nd) if isinstance(x, float) else x)


def stats(vals: list[float]) -> dict:
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    if v.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "max", "median")}
    return {"mean": v.mean(), "std": v.std(ddof=1) if v.size > 1 else 0.0,
            "min": v.min(), "max": v.max(), "median": float(np.median(v))}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _git(*args: str) -> str | None:
    """Run a read-only git command in this script's folder; None if git or the repo is missing."""
    import subprocess
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                           cwd=Path(__file__).resolve().parent)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def versions() -> dict:
    """Software versions + the git commit of this script, so every number can be traced to its code."""
    info = {"python": platform.python_version(), "script": Path(__file__).name}
    try:
        import cellpose, torch
        info.update(cellpose=cellpose.version, torch=torch.__version__)
    except Exception as e:  # pragma: no cover
        info["version_error"] = repr(e)
    # BUGFIX 2: check only the files that produced the numbers (this scorer + the script that was
    # started), not the whole folder; an unrelated untracked file in scripts/ gave a false warning.
    here = Path(__file__).resolve().parent
    used = {Path(__file__).name}
    started = Path(sys.argv[0]).resolve()
    if started.parent == here:
        used.add(started.name)
    status = _git("status", "--porcelain", "--", *sorted(used))
    info["repo_commit"] = _git("rev-parse", "--short", "HEAD") or "not in a git repo"
    info["repo_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    info["scripts_uncommitted_changes"] = None if status is None else bool(status)
    return info


# ------------------------------------------------------------------------- metrics
def area_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """fat % of GT and prediction, signed fat % error (pp), foreground Dice."""
    t, p = gt > 0, pred > 0
    n_t, n_p, n_tp = int(t.sum()), int(p.sum()), int(np.logical_and(t, p).sum())
    fat_true, fat_pred = 100.0 * n_t / t.size, 100.0 * n_p / t.size
    dice = 2.0 * n_tp / (n_t + n_p) if (n_t + n_p) else float("nan")
    return {"fat_pct_true": fat_true, "fat_pct_pred": fat_pred,
            "fat_pct_error": fat_pred - fat_true, "fg_dice": dice}


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Area readouts for every image; instance metrics when the GT has >= 1 cell."""
    from cellpose import metrics

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs prediction {pred.shape}")
    n_true, n_pred = int(gt.max()), int(pred.max())
    out = {"n_true": n_true, "n_pred": n_pred, **area_metrics(gt, pred)}
    out["negative_image"] = n_true == 0
    if n_true == 0:
        return out
    sa, tp, fp, fn = metrics.average_precision([gt], [pred], threshold=THRESHOLDS)
    out["count_error"] = (n_pred - n_true) / n_true
    out["per_t"] = {}
    for i, t in enumerate(THRESHOLDS):
        TP, FP, FN = int(tp[0, i]), int(fp[0, i]), int(fn[0, i])
        out["per_t"][t] = {"F1": 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else float("nan"),
                           "SA": float(sa[0, i]), "TP": TP, "FP": FP, "FN": FN}
    return out


def score_row(model: str, source_run: str, pred_path: Path, gt_path: Path) -> dict:
    base = pred_path.stem[: -len(f"_{model}_instances")]
    res = score(load_labels(gt_path), load_labels(pred_path))
    row = {k: "" for k in PER_IMAGE_FIELDS}
    row.update(
        model=model, source_run=source_run, image=base, series=parse_series(base),
        condition=parse_condition(base),
        fat_pct_true=fmt(res["fat_pct_true"]), fat_pct_pred=fmt(res["fat_pct_pred"]),
        fat_pct_error=fmt(res["fat_pct_error"]), fat_pct_abs_error=fmt(abs(res["fat_pct_error"])),
        fg_dice=fmt(res["fg_dice"]),
        n_true=res["n_true"], n_pred=res["n_pred"], count_diff=res["n_pred"] - res["n_true"],
    )
    if not res["negative_image"]:
        m5, n_t, n_p = res["per_t"][0.5], res["n_true"], res["n_pred"]
        row.update({
            "count_error": fmt(res["count_error"]),
            "TP@0.5": m5["TP"], "FP@0.5": m5["FP"], "FN@0.5": m5["FN"],
            "precision@0.5": fmt(m5["TP"] / n_p if n_p else float("nan")),
            "recall@0.5": fmt(m5["TP"] / n_t),
            **{f"F1@{t:g}": fmt(res["per_t"][t]["F1"]) for t in THRESHOLDS},
        })
    return row


def _vals(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r[key] != ""]


def summarise_subset(model: str, condition: str, rows: list[dict]) -> dict:
    scored = [r for r in rows if r["n_true"] != 0]
    fe, fae, dice = stats(_vals(rows, "fat_pct_error")), stats(_vals(rows, "fat_pct_abs_error")), \
        stats(_vals(rows, "fg_dice"))
    tp, fp, fn = sum(_vals(scored, "TP@0.5")), sum(_vals(scored, "FP@0.5")), sum(_vals(scored, "FN@0.5"))
    n_true, n_pred = int(sum(_vals(scored, "n_true"))), int(sum(_vals(scored, "n_pred")))
    f1 = stats(_vals(scored, "F1@0.5"))
    return {
        "model": model, "condition": condition, "n_images": len(rows), "n_scored": len(scored),
        "fat_pct_true_mean": fmt(stats(_vals(rows, "fat_pct_true"))["mean"], 2),
        "fat_pct_pred_mean": fmt(stats(_vals(rows, "fat_pct_pred"))["mean"], 2),
        "fat_pct_error_mean": fmt(fe["mean"], 2), "fat_pct_error_std": fmt(fe["std"], 2),
        "fat_pct_abs_error_mean": fmt(fae["mean"], 2), "fat_pct_abs_error_max": fmt(fae["max"], 2),
        "fg_dice_mean": fmt(dice["mean"]), "fg_dice_std": fmt(dice["std"]),
        "n_true_total": n_true, "n_pred_total": n_pred,
        "pooled_count_error": fmt((n_pred - n_true) / n_true if n_true else float("nan")),
        "abs_count_error_median": fmt(stats([abs(x) for x in _vals(scored, "count_error")])["median"]),
        "F1@0.5_mean": fmt(f1["mean"]), "F1@0.5_std": fmt(f1["std"]),
        "pooled_F1@0.5": fmt(2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else float("nan")),
        "F1@0.7_mean": fmt(stats(_vals(scored, "F1@0.7"))["mean"]),
        "F1@0.75_mean": fmt(stats(_vals(scored, "F1@0.75"))["mean"]),
    }


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        mrows = [r for r in rows if r["model"] == model]
        for cond in dict.fromkeys(r["condition"] for r in mrows):
            out.append(summarise_subset(model, cond, [r for r in mrows if r["condition"] == cond]))
        out.append(summarise_subset(model, "all", mrows))
    return out


# ------------------------------------------------------------------------------ W&B
def log_wandb(a, model: str, rows: list[dict], summary: list[dict], config: dict) -> None:
    """One W&B run per model: config, per-image table, summary values. Metrics only."""
    import wandb

    def num(v):
        return None if v == "" else v

    run = wandb.init(
        project=a.wandb_project, entity=a.wandb_entity, group=a.wandb_group, name=model,
        job_type="rescore",
        config={"model": model, "group": config["group"], "source_runs": sorted({r["source_run"] for r in rows}),
                "iou_thresholds": THRESHOLDS, "note": a.note, **config["versions"]},
        settings=wandb.Settings(x_disable_meta=True, x_disable_machine_info=True),
    )
    try:
        run.log({"per_image": wandb.Table(columns=PER_IMAGE_FIELDS,
                                          data=[[num(r[k]) for k in PER_IMAGE_FIELDS] for r in rows])})
        msum = [s for s in summary if s["model"] == model]
        run.log({"summary_table": wandb.Table(columns=list(msum[0].keys()),
                                              data=[[num(v) for v in s.values()] for s in msum])})
        for s in msum:  # run summary = this model's averages, not the last image
            run.summary.update({f"{s['condition']}/{k}": num(v) for k, v in s.items()
                                if k not in ("model", "condition")})
    finally:
        run.finish()


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-dirs", nargs="+", required=True, type=Path,
                   help="run folders made by run_batch.py (each has a 'masks' folder)")
    p.add_argument("--group-dir", required=True, type=Path, help="folder with the GT *_instances files")
    p.add_argument("--out-root", type=Path, default=None,
                   help="results root; default: two levels above the first batch dir (Cloud_results)")
    p.add_argument("--note", default="", help="free text stored in the config")
    p.add_argument("--wandb", action="store_true", help="log one W&B run per model (metrics only)")
    p.add_argument("--wandb-project", default=None, help="W&B project (required with --wandb)")
    p.add_argument("--wandb-group", default="zero-shot-porcine-bf", help="W&B group for these runs")
    p.add_argument("--wandb-entity", default=None, help="W&B team name (or set WANDB_ENTITY)")
    a = p.parse_args(argv)
    if a.wandb and not a.wandb_project:
        p.error("--wandb needs --wandb-project <name>")

    rows: list[dict] = []
    missing_gt = 0
    for batch_dir in a.batch_dirs:
        mask_root = batch_dir / "masks"
        if not mask_root.is_dir():
            raise SystemExit(f"No 'masks' folder in {batch_dir}")
        for model_dir in sorted(d for d in mask_root.iterdir() if d.is_dir()):
            model = model_dir.name
            preds = sorted(model_dir.glob(f"*_{model}_instances.tif"))
            print(f"{batch_dir.name} / {model}: {len(preds)} saved predictions")
            for pred_path in preds:
                base = pred_path.stem[: -len(f"_{model}_instances")]
                gt_path = next((a.group_dir / f"{base}_instances{e}" for e in GT_EXTS
                                if (a.group_dir / f"{base}_instances{e}").exists()), None)
                if gt_path is None:
                    missing_gt += 1
                    print(f"    {base}: no GT found -> skipped")
                    continue
                rows.append(score_row(model, batch_dir.name, pred_path, gt_path))
    if not rows:
        raise SystemExit("Nothing scored.")

    out_root = a.out_root or a.batch_dirs[0].resolve().parent.parent
    run_dir = out_root / "rescore" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.group_dir.name}"
    run_dir.mkdir(parents=True)  # fails if it exists: never overwrite

    summary = summarise(rows)
    config = {"args": {k: str(v) for k, v in vars(a).items()},
              "source_runs": [b.name for b in a.batch_dirs], "group": a.group_dir.name,
              "iou_thresholds": THRESHOLDS, "n_rows": len(rows), "missing_gt": missing_gt,
              "versions": versions(), "created": dt.datetime.now().isoformat()}
    write_csv(run_dir / "per_image.csv", rows, PER_IMAGE_FIELDS)
    write_csv(run_dir / "summary.csv", summary)
    (run_dir / "rescore_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if a.wandb:
        for model in dict.fromkeys(r["model"] for r in rows):
            log_wandb(a, model, [r for r in rows if r["model"] == model], summary, config)

    print("\n=== summary (fat % error in percentage points; + = over-estimate)")
    print(f"{'model':<12} {'condition':<15} {'n':>3} {'fat% err':>9} {'|fat% err|':>10} {'Dice':>6} "
          f"{'count err':>9} {'F1@0.5':>7} {'F1@0.7':>7}")
    for s in summary:
        print(f"{s['model']:<12} {s['condition']:<15} {s['n_images']:>3} {s['fat_pct_error_mean']:>9} "
              f"{s['fat_pct_abs_error_mean']:>10} {s['fg_dice_mean']:>6} {s['pooled_count_error']:>9} "
              f"{s['F1@0.5_mean']:>7} {s['F1@0.7_mean']:>7}")
    v = config["versions"]
    print(f"\ncode: {v['script']} @ {v['repo_commit']} ({v['repo_branch']})")
    if v["scripts_uncommitted_changes"]:
        print("WARNING: the scripts that ran have uncommitted changes -> results do not match the recorded commit.")
    print(f"saved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
