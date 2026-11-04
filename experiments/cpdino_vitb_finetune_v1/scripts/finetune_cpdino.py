"""Fine-tune a CellposeDINO model (default cpdino-vitb) on the TRAIN images of a split,
with Cellpose's own training routine (cellpose.train.train_seg, Cellpose 4.2.1.1).

Input images get the same preprocessing at training and at prediction time
(--preprocess imes_clahe = IMES's per-channel CLAHE; see imes_preprocess.py).
The VAL images are passed to Cellpose only to print a validation LOSS every 10 epochs;
model quality is measured afterwards with predict_split.py + score_masks_v2.py.
The TEST images are never read by this script.

Baseline settings = Cellpose defaults (checked in train.py of 4.2.1.1):
    AdamW, learning rate 1e-5 (10-epoch warm-up, step-down at the end), weight decay 0.1,
    100 epochs, batch size 1, 384 x 384 tiles (DINO default), random rotation, scale 0.75-1.25,
    horizontal flip. Training runs in float32 (Cellpose converts the bf16 model itself);
    saved weights are float32.
    One "epoch" = one random 384 x 384 crop from EACH training image (18 crops).
    cpsam / cpsam_v2 (Cellpose-SAM, ViT-L) also work via --base-model; they train on 256 x 256
    tiles (Cellpose requires it), picked automatically. Run on a 32 GB V100 (CAIRO): about 8.5 GB
    peak and about 10 min for 100 epochs on a free GPU. Not tested on an 8 GB laptop GPU, so do
    the 2-epoch memory test first there.
Changed on purpose:
    --min-train-masks 0 (Cellpose default 5 would silently DROP images with < 5 cells,
    i.e. our training control image, which is what teaches "no fat here").
Seeds: training has two random sources. (1) torch: random layer-skipping inside the network
(stochastic depth) -> torch.manual_seed(seed). (2) numpy: which crop, rotation, scale and flip
each image gets. Cellpose re-seeds numpy with the epoch number every epoch (np.random.seed(epoch),
and seed 42 for the val loss), so on its own every run would draw the same crops. This script
shifts those numpy seeds by 100000 * seed while train_seg runs: --seed 0 = unchanged Cellpose
behaviour; seeds 1, 2, ... give different crops/augmentations without overlapping epochs.

Output (new folder per run, nothing is overwritten):
    <out-root>\\train\\<YYYYmmdd-HHMMSS>_<label>\\
        models\\<label>                 fine-tuned weights (Cellpose format, no extension)
        losses.csv                      train loss per epoch, val loss where computed
        train_config.json               all settings, split, image lists, versions, timing,
                                        peak GPU memory, SHA-256 of the saved weights
        train.log                       Cellpose's training log

Quick memory/speed test first (2 epochs; with so few epochs the learning rate stays near 0,
so this run is NOT a real model):
    uv run --project "<repo>\\cellpose-env" python "<exp>\\scripts\\finetune_cpdino.py" `
        --split-file "<repo>\\splits\\porcine_bf_v1.json" --group-dir "<data>\\Training_porcine_bf" `
        --out-root "<...>\\Cloud_results\\cpdino_ft" --label test-2ep --n-epochs 2
Baseline:
    ... --label cpdino-vitb-ft-clahe-v1          (all other settings = defaults above)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imes_preprocess import grey_clahe, imes_clahe  # noqa: E402
from split_io import (  # noqa: E402
    check_label, find_gt, find_image, load_cellpose, load_labels, load_split, sha256_file, versions,
)

PREPROCESS = {"none": lambda img: img, "imes_clahe": imes_clahe, "grey_clahe": grey_clahe}
SEED_STRIDE = 100_000   # > number of epochs, so runs with different seeds never share an epoch seed


@contextmanager
def numpy_seed_offset(offset: int):
    """While active, np.random.seed(s) becomes np.random.seed(s + offset).

    cellpose.train calls np.random.seed(iepoch) (and 42 for the val loss) through the shared
    numpy module, so this shifts Cellpose's seeds without changing its code. offset 0 = no change.
    """
    original = np.random.seed

    def shifted(seed=None):
        return original(None if seed is None else int(seed) + offset)

    np.random.seed = shifted
    try:
        yield
    finally:
        np.random.seed = original


def load_subset(names, group_dir: Path, preprocess: str):
    """Images (preprocessed, H x W x 3) and GT labels for a list of names."""
    from cellpose import io as cpio

    imgs, lbls, n_cells = [], [], []
    for n in names:
        raw = cpio.imread(str(find_image(group_dir, n)))
        lab = load_labels(find_gt(group_dir, n))
        if raw.shape[:2] != lab.shape:
            raise SystemExit(f"{n}: image {raw.shape[:2]} and mask {lab.shape} differ in size")
        imgs.append(PREPROCESS[preprocess](raw))
        lbls.append(lab)
        n_cells.append(int(lab.max()))
    return imgs, lbls, n_cells


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--label", required=True, help="name of the fine-tuned model, e.g. cpdino-vitb-ft-clahe-v1")
    p.add_argument("--base-model", default="cpdino-vitb", choices=["cpdino-vitb", "cpdino", "cpsam", "cpsam_v2"])
    p.add_argument("--preprocess", choices=sorted(PREPROCESS), default="imes_clahe")
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--bsize", type=int, default=None,
                   help="training tile size (default: 384 for the DINO models, 256 for cpsam/cpsam_v2, "
                        "which Cellpose requires)")
    p.add_argument("--min-train-masks", type=int, default=0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="0 = Cellpose's own behaviour; 1, 2, ... = different crops, augmentation and layer drops")
    p.add_argument("--note", default="")
    a = p.parse_args(argv)

    label = check_label(a.label)
    split = load_split(a.split_file)
    if split["group_dir_name"] != a.group_dir.name:
        raise SystemExit(f"Split was made for '{split['group_dir_name']}', not '{a.group_dir.name}'.")
    train_names, val_names = split["splits"]["train"], split["splits"]["val"]

    import torch
    from cellpose import io as cpio, train

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found: training on CPU would take far too long.")
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    run_dir = a.out_root / "train" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{label}"
    run_dir.mkdir(parents=True)   # fails if it exists: never overwrite
    cpio.logger_setup()           # Cellpose prints its training log to the console
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    print(f"split {split['name']} ({split['fingerprint']}) | train {len(train_names)} | val {len(val_names)} "
          f"| base {a.base_model} | preprocess {a.preprocess} | epochs {a.n_epochs}")
    t_load = time.time()
    tr_imgs, tr_lbls, tr_cells = load_subset(train_names, a.group_dir, a.preprocess)
    va_imgs, va_lbls, va_cells = load_subset(val_names, a.group_dir, a.preprocess)
    print(f"    loaded + preprocessed {len(tr_imgs) + len(va_imgs)} images in {time.time() - t_load:.1f} s")
    low = [n for n, c in zip(train_names, tr_cells) if c < 5]
    if low:
        print(f"    note: {len(low)} training image(s) with < 5 cells kept (min_train_masks={a.min_train_masks}): {low}")

    model = load_cellpose(a.base_model, use_gpu=True)
    base_weights = Path(model.pretrained_model)
    # Cellpose reads the backbone from the weights: "sam_vitl" (cpsam, cpsam_v2) only trains on
    # 256 x 256 tiles (train_seg raises otherwise); the DINO backbones default to 384.
    if a.bsize is None:
        bsize = 256 if model.backbone == "sam_vitl" else 384
    elif model.backbone == "sam_vitl" and a.bsize != 256:
        raise SystemExit(f"--bsize {a.bsize} is not supported for {a.base_model}: Cellpose requires 256.")
    else:
        bsize = a.bsize
    print(f"    backbone {model.backbone} | training tile size {bsize}")

    config = {"args": {k: str(v) for k, v in vars(a).items()},
              "split": {"name": split["name"], "fingerprint": split["fingerprint"]},
              "train_images": train_names, "val_images": val_names,
              "train_cells": dict(zip(train_names, tr_cells)), "val_cells": dict(zip(val_names, va_cells)),
              "base_weights": str(base_weights), "base_weights_sha256": sha256_file(base_weights),
              "backbone": model.backbone, "bsize_used": bsize, "versions": versions(),
              "cellpose_defaults_used": {"optimizer": "AdamW", "scale_range": 0.5, "rescale": False,
                                         "normalize": "default (1-99 percentile)"},
              "numpy_seed_offset": SEED_STRIDE * a.seed,
              "started": dt.datetime.now().isoformat(timespec="seconds")}
    (run_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with numpy_seed_offset(SEED_STRIDE * a.seed):
        weights_path, train_losses, val_losses = train.train_seg(
            model.net, train_data=tr_imgs, train_labels=tr_lbls,
            test_data=va_imgs, test_labels=va_lbls, channel_axis=-1,
            learning_rate=a.learning_rate, weight_decay=a.weight_decay, n_epochs=a.n_epochs,
            batch_size=a.batch_size, bsize=bsize, min_train_masks=a.min_train_masks,
            save_path=run_dir, save_every=a.save_every, model_name=label,
        )
    seconds = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    with (run_dir / "losses.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for e, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            w.writerow([e, f"{tl:.5f}", f"{vl:.5f}" if (e == 5 or e % 10 == 0) else ""])

    weights_path = Path(weights_path)
    load_cellpose(str(weights_path), use_gpu=True)   # re-load check: every key must load
    config.update(weights_file=str(weights_path), weights_sha256=sha256_file(weights_path),
                  train_seconds=round(seconds, 1), seconds_per_epoch=round(seconds / a.n_epochs, 2),
                  peak_gpu_gb=round(peak_gb, 2),
                  final_train_loss=float(train_losses[-1]),
                  finished=dt.datetime.now().isoformat(timespec="seconds"))
    (run_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\ntrained {a.n_epochs} epochs in {seconds / 60:.1f} min ({seconds / a.n_epochs:.1f} s/epoch), "
          f"peak GPU memory {peak_gb:.2f} GB")
    print(f"final train loss {train_losses[-1]:.4f}")
    print(f"weights: {weights_path}  (re-loaded and checked)")
    print(f"run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
