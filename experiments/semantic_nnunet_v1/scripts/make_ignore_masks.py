"""Ignore masks for the partial annotations of split porcine_bf_v1.

Problem: the biologists outlined only the fat cells they were sure of. Every other pixel is
labelled background, so real but unoutlined fat cells are taught as "not fat".

Fix (nnU-Net ignore label): pixels that probably belong to an unoutlined cell get label "ignore"
and do not count in the loss. Which pixels: zero-shot Cellpose-SAM (cpsam) is run on the image;
a predicted cell is a CANDIDATE if less than --new-frac of it overlaps an annotation and it has
at least --min-area pixels (the same rule as active_learning/al_cycle0.py). The candidates,
grown by --dilate pixels, become ignore, but never on top of an annotated cell. All other
unannotated pixels stay background, so the model still sees real background.

Control images (name contains "negativ" or "ctrl") get an EMPTY ignore mask: their few annotated
cells are close to the truth, and zero-shot models invent cells on them, so ignoring those
would throw away the "almost no fat here" signal.

Writes, for every image of the chosen subsets (default: train only):
    <out-dir>/<name>_ignore.png      uint8, 1 = ignore, 0 = keep the annotation as it is
    <out-dir>/summary.csv            per image: candidates, ignore %, annotated fat %
    <out-dir>/config.json            settings, split fingerprint, versions
    <out-dir>/cache/<model>/<name>.npz   the cpsam masks, so a re-run with other settings is fast
--out-dir holds images derived from IMES data: keep it OUTSIDE the repo.

Val/test ignore masks (--subsets val) are ONLY for the ignore-aware score
(score_ignore_aware.py). prepare_nnunet_dataset.py uses ignore masks for TRAIN cases only.

Usage (environment: cellpose-env, which has cellpose 4.2.1.1):
    uv run --no-sync --project <repo>/cellpose-env python make_ignore_masks.py \\
        --split-file <repo>/splits/porcine_bf_v1.json --group-dir <data>/Training_porcine_bf \\
        --out-dir <outside repo>/ignore_masks_v1 --subsets train val
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpdino_vitb_finetune_v1" / "scripts"))
from split_io import find_gt, find_image, load_cellpose, load_labels, load_split  # noqa: E402


def is_control(name: str) -> bool:
    low = name.lower()
    return "negativ" in low or "ctrl" in low


def candidates_to_ignore(pred: np.ndarray, gt: np.ndarray, new_frac: float, min_area: int,
                         dilate: int) -> tuple[np.ndarray, int]:
    """Boolean ignore mask and the number of candidate cells (see module docstring)."""
    from scipy import ndimage as ndi

    n = int(pred.max())
    if n == 0:
        return np.zeros(gt.shape, bool), 0
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(pred), labels=pred, index=idx)
    inside = ndi.sum(gt > 0, labels=pred, index=idx)
    keep = idx[(area >= min_area) & (inside < new_frac * area)]
    cand = np.isin(pred, keep)
    if dilate > 0 and cand.any():
        cand = ndi.binary_dilation(cand, structure=ndi.generate_binary_structure(2, 1), iterations=dilate)
    return cand & (gt == 0), int(len(keep))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="OUTSIDE the repo")
    p.add_argument("--subsets", nargs="+", choices=("train", "val", "test"), default=["train"])
    p.add_argument("--model", default="cpsam", help="Cellpose built-in model name (default cpsam)")
    p.add_argument("--new-frac", type=float, default=0.10)
    p.add_argument("--min-area", type=int, default=150)
    p.add_argument("--dilate", type=int, default=3, help="grow candidates by this many pixels")
    p.add_argument("--max-ignore-pct", type=float, default=40.0,
                   help="stop if any image would be more than this %% ignore (sanity check)")
    a = p.parse_args(argv)

    import imageio.v3 as iio
    import torch
    from importlib.metadata import version as pkg_version

    split = load_split(a.split_file)
    if split["group_dir_name"] != a.group_dir.name:
        raise SystemExit(f"Split was made for '{split['group_dir_name']}', not '{a.group_dir.name}'.")
    cache_dir = a.out_dir / "cache" / a.model.replace("/", "_")   # per model: no stale reuse
    cache_dir.mkdir(parents=True, exist_ok=True)

    gpu = torch.cuda.is_available()
    if not gpu:
        print("WARNING: no CUDA GPU found, running cpsam on CPU (slow).", flush=True)
    model = None

    rows = []
    for subset in a.subsets:
        for name in split["splits"][subset]:
            gt = load_labels(find_gt(a.group_dir, name))
            out_png = a.out_dir / f"{name}_ignore.png"
            if is_control(name):
                ign, n_cand, note = np.zeros(gt.shape, bool), 0, "control: no ignore"
            else:
                cache = cache_dir / f"{name}.npz"
                if cache.exists():
                    pred = np.load(cache)["masks"]
                else:
                    if model is None:
                        model = load_cellpose(a.model, gpu)   # stops on unknown names (no silent cpsam_v2)
                        print(f"loaded '{a.model}' (cellpose {pkg_version('cellpose')}, gpu={gpu})", flush=True)
                    img = np.asarray(iio.imread(find_image(a.group_dir, name)))
                    if img.ndim == 3 and img.shape[-1] == 4:
                        img = img[..., :3]
                    pred, _, _ = model.eval(img, batch_size=8, flow_threshold=0.4, cellprob_threshold=0.0,
                                            channel_axis=-1 if img.ndim == 3 else None)
                    pred = np.asarray(pred).astype(np.int32)
                    np.savez_compressed(cache, masks=pred)
                if pred.shape != gt.shape:
                    raise SystemExit(f"{name}: prediction {pred.shape} and mask {gt.shape} differ in size")
                ign, n_cand = candidates_to_ignore(pred, gt, a.new_frac, a.min_area, a.dilate)
                note = ""
            pct = 100.0 * ign.mean()
            if pct > a.max_ignore_pct:
                raise SystemExit(f"{name}: {pct:.1f} % would be ignore (> {a.max_ignore_pct} %). "
                                 "Check the cpsam output before training on this.")
            iio.imwrite(out_png, ign.astype(np.uint8))
            row = {"subset": subset, "name": name, "candidates": n_cand,
                   "ignore_pct": round(pct, 2), "annotated_fat_pct": round(100.0 * (gt > 0).mean(), 2),
                   "note": note}
            rows.append(row)
            print(f"  {subset:5s} {name}: {n_cand} candidates, ignore {pct:.1f} %, "
                  f"annotated fat {row['annotated_fat_pct']:.1f} % {note}", flush=True)

    with open(a.out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out_dir / "config.json").write_text(json.dumps({
        "split": {"name": split["name"], "fingerprint": split["fingerprint"]},
        "subsets": a.subsets, "model": a.model, "new_frac": a.new_frac, "min_area": a.min_area,
        "dilate": a.dilate, "controls": "empty ignore mask",
        "cellpose": pkg_version("cellpose"), "torch": torch.__version__,
        "made": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    tr = [r["ignore_pct"] for r in rows if r["subset"] == "train"]
    if tr:
        print(f"\ntrain: mean ignore {np.mean(tr):.1f} % of pixels (max {max(tr):.1f} %)")
    print(f"{len(rows)} masks -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
