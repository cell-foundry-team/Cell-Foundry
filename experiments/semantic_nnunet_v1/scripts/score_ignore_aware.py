"""Fat % error that does not punish a model for finding cells the annotators left out.

Why: the val/test annotations are partial too. A model trained with the ignore label should find
more of the unoutlined fat cells, and score_masks_v2.py counts every such pixel as a false
positive, so the plain |fat % error| is biased against exactly the change being tested.

What: for each image, pixels marked in <ignore-dir>/<name>_ignore.png (make_ignore_masks.py
--subsets val) are left out of BOTH the prediction and the annotation, and fat % is computed on
the remaining pixels. Report this NEXT TO the plain score_masks_v2.py result, never instead of it:
the ignore regions come from zero-shot cpsam, so they are an estimate, not ground truth.

Input: one or more prediction run folders in the shared format
    <run>/masks/<label>/<name>_<label>_instances.tif   (as written by export_predictions.py)

Usage (environment: experiments/semantic_nnunet_v1, which has tifffile + imageio):
    uv run --no-sync --project <exp> python score_ignore_aware.py --group-dir <data> \\
        --ignore-dir <outside repo>/ignore_masks_v1 --runs <run1> <run2> ... --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpdino_vitb_finetune_v1" / "scripts"))
from split_io import find_gt, load_labels  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--ignore-dir", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True, help="CSV to write (outside the repo is fine)")
    a = p.parse_args(argv)

    import tifffile
    import imageio.v3 as iio

    rows = []
    for run in a.runs:
        files = sorted(run.glob("masks/*/*_instances.tif"))
        if not files:
            raise SystemExit(f"{run}: no masks/<label>/*_instances.tif found")
        for f in files:
            label = f.parent.name
            suffix = f"_{label}_instances.tif"
            if not f.name.endswith(suffix):
                raise SystemExit(f"{f.name}: does not end with {suffix}")
            name = f.name[: -len(suffix)]
            pred = np.squeeze(tifffile.imread(f)) > 0
            gt = load_labels(find_gt(a.group_dir, name)) > 0
            ign_path = a.ignore_dir / f"{name}_ignore.png"
            if not ign_path.exists():
                raise SystemExit(f"{name}: no ignore mask {ign_path} (run make_ignore_masks.py --subsets val)")
            ign = np.asarray(iio.imread(ign_path)) > 0
            if not (pred.shape == gt.shape == ign.shape):
                raise SystemExit(f"{name}: shapes differ pred {pred.shape} gt {gt.shape} ignore {ign.shape}")
            keep = ~ign
            n = int(keep.sum())
            fat_gt = 100.0 * (gt & keep).sum() / n
            fat_pred = 100.0 * (pred & keep).sum() / n
            fat_gt_all = 100.0 * gt.mean()
            fat_pred_all = 100.0 * pred.mean()
            rows.append({"label": label, "name": name, "ignore_pct": round(100.0 * ign.mean(), 2),
                         "fat_gt_all": round(fat_gt_all, 2), "fat_pred_all": round(fat_pred_all, 2),
                         "abs_err_all": round(abs(fat_pred_all - fat_gt_all), 2),
                         "fat_gt_kept": round(fat_gt, 2), "fat_pred_kept": round(fat_pred, 2),
                         "abs_err_kept": round(abs(fat_pred - fat_gt), 2)})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'label':40s} {'n':>2s} {'|err| all':>10s} {'|err| ignore-aware':>19s}")
    for label in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == label]
        print(f"{label:40s} {len(rs):2d} {np.mean([r['abs_err_all'] for r in rs]):10.2f} "
              f"{np.mean([r['abs_err_kept'] for r in rs]):19.2f}")
    print(f"\nper-image rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
