"""Robustness check: does the predicted fat % change when the SAME image is flipped, rotated,
made brighter or darker? A stable model should give (almost) the same fat %.

What it is and is not:
  * It measures STABILITY of one model on the images of one subset.
  * It does NOT add images: the variants of one image are not independent samples, so it says
    nothing about how many images we tested on (the noise of a 6-image test stays the same).

Variants (fixed list):
    identity, hflip, vflip, rot90, bright_0.8, bright_1.2, gamma_0.8, gamma_1.2
Geometric variants (hflip, vflip, rot90) are undone on the predicted mask, so every variant mask
lines up with the ground truth again and can be scored with score_masks_v2.py as usual.
Photometric variants (brightness x factor, gamma) leave the geometry alone.

Output (new folder, nothing is overwritten):
    <out-root>/robust/<YYYYmmdd-HHMMSS>_<subset>_<label>/
        masks/<label>-<variant>/<name>_<label>-<variant>_instances.tif   (shared format)
        robustness_by_image.csv     image, variant, objects, fat_pct, delta_pp_vs_identity, dice_vs_identity
        robustness_summary.csv      per variant: mean / max |delta| over images, mean Dice vs identity
        robust_config.json
Then (optional) score everything against the ground truth:
    uv run --project <repo>/cellpose-env python <repo>/scripts/score_masks_v2.py \\
        --batch-dirs <that run folder> --group-dir <data>/Training_porcine_bf

Default subset is val. --subset test needs --confirm-test: the test set of porcine_bf_v1 was used
once already (cpsam, 4 Nov); this check chooses nothing, but report it as a diagnostic only.

Usage (cellpose-env):
    python robustness_check.py --split-file <repo>/splits/porcine_bf_v1.json --subset val \\
        --group-dir <data>/Training_porcine_bf --out-root <outside repo> \\
        --model <weights file from finetune_cpdino.py> --preprocess none --label cpsam-ft-raw-v1-s0
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
from imes_preprocess import grey_clahe, imes_clahe  # noqa: E402
from split_io import (  # noqa: E402
    SUBSETS, check_label, find_image, load_cellpose, load_split, save_mask, sha256_file,
)

PREPROCESS = {"none": lambda img: img, "imes_clahe": imes_clahe, "grey_clahe": grey_clahe}


def _scale(img: np.ndarray, fn) -> np.ndarray:
    """Apply fn to the image as float in 0..1 and return it in the original dtype."""
    top = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    out = np.clip(fn(img.astype(np.float64) / top), 0.0, 1.0) * top
    return np.rint(out).astype(img.dtype) if np.issubdtype(img.dtype, np.integer) else out.astype(img.dtype)


# name -> (forward on the image, inverse on the mask)
VARIANTS = {
    "identity": (lambda x: x, lambda m: m),
    "hflip": (lambda x: x[:, ::-1], lambda m: m[:, ::-1]),
    "vflip": (lambda x: x[::-1], lambda m: m[::-1]),
    "rot90": (lambda x: np.rot90(x, 1), lambda m: np.rot90(m, -1)),
    "bright_0.8": (lambda x: _scale(x, lambda v: v * 0.8), lambda m: m),
    "bright_1.2": (lambda x: _scale(x, lambda v: v * 1.2), lambda m: m),
    "gamma_0.8": (lambda x: _scale(x, lambda v: v ** 0.8), lambda m: m),
    "gamma_1.2": (lambda x: _scale(x, lambda v: v ** 1.2), lambda m: m),
}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else float(2 * (a & b).sum() / s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--subset", choices=SUBSETS, default="val")
    p.add_argument("--confirm-test", action="store_true", help="required for --subset test")
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--model", required=True, help="built-in name or a weights file")
    p.add_argument("--label", required=True)
    p.add_argument("--preprocess", choices=sorted(PREPROCESS), required=True)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--diameter", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args(argv)

    if a.subset == "test" and not a.confirm_test:
        p.error("--subset test needs --confirm-test (diagnostic only; the test set was already used once)")
    label = check_label(a.label)
    split = load_split(a.split_file)
    if split["group_dir_name"] != a.group_dir.name:
        raise SystemExit(f"Split was made for '{split['group_dir_name']}', not '{a.group_dir.name}'.")
    names = split["splits"][a.subset]
    images = {n: find_image(a.group_dir, n) for n in names}

    import torch
    from cellpose import io as cpio

    use_gpu = (not a.cpu) and torch.cuda.is_available()
    print(f"split {split['name']} | subset {a.subset} | {len(names)} images x {len(VARIANTS)} variants | "
          f"model {a.model} | preprocess {a.preprocess} | GPU {use_gpu}")
    model = load_cellpose(a.model, use_gpu)
    run_dir = a.out_root / "robust" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.subset}_{label}"
    run_dir.mkdir(parents=True)
    config = {"args": {k: str(v) for k, v in vars(a).items()}, "variants": list(VARIANTS), "images": names,
              "split": {"name": split["name"], "fingerprint": split["fingerprint"], "subset": a.subset},
              "weights_sha256": sha256_file(Path(model.pretrained_model)),
              "started": dt.datetime.now().isoformat(timespec="seconds")}
    (run_dir / "robust_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    for i, name in enumerate(names, 1):
        raw = cpio.imread(str(images[name]))
        base_mask = None
        for vname, (fwd, inv) in VARIANTS.items():
            img = PREPROCESS[a.preprocess](np.ascontiguousarray(fwd(raw)))
            masks = np.asarray(model.eval(img, batch_size=a.batch_size, diameter=a.diameter,
                                          channel_axis=-1 if img.ndim == 3 else None,
                                          flow_threshold=a.flow_threshold,
                                          cellprob_threshold=a.cellprob_threshold)[0])
            masks = np.ascontiguousarray(inv(masks))
            if masks.shape != raw.shape[:2]:
                raise SystemExit(f"{name}/{vname}: mask {masks.shape} != image {raw.shape[:2]} after undoing")
            if vname == "identity":
                base_mask = masks
            save_mask(run_dir, f"{label}-{vname}", name, masks, raw.shape)
            fat, fat0 = 100 * (masks > 0).mean(), 100 * (base_mask > 0).mean()
            rows.append({"image": name, "variant": vname, "objects": int(masks.max()),
                         "fat_pct": round(fat, 3), "delta_pp_vs_identity": round(fat - fat0, 3),
                         "dice_vs_identity": round(dice(masks, base_mask), 4)})
        print(f"    [{i}/{len(names)}] {name}: " + ", ".join(
            f"{r['variant']} {r['delta_pp_vs_identity']:+.2f}" for r in rows[-len(VARIANTS):] if r["variant"] != "identity"))

    with open(run_dir / "robustness_by_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for vname in VARIANTS:
        vr = [r for r in rows if r["variant"] == vname]
        d = [abs(r["delta_pp_vs_identity"]) for r in vr]
        dc = [r["dice_vs_identity"] for r in vr if r["dice_vs_identity"] == r["dice_vs_identity"]]
        summary.append({"variant": vname, "n_images": len(vr), "mean_abs_delta_pp": round(float(np.mean(d)), 3),
                        "max_abs_delta_pp": round(float(np.max(d)), 3),
                        "mean_dice_vs_identity": round(float(np.mean(dc)), 4) if dc else ""})
    with open(run_dir / "robustness_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\n{'variant':<12} {'mean |delta| pp':>16} {'max |delta| pp':>15} {'Dice vs identity':>17}")
    for s in summary:
        print(f"{s['variant']:<12} {s['mean_abs_delta_pp']:>16} {s['max_abs_delta_pp']:>15} {s['mean_dice_vs_identity']:>17}")
    print(f"\n-> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
