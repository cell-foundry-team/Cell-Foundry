"""Segment the images of ONE subset of a split (train / val / test) with one Cellpose model
and save the masks in our shared format, ready for score_masks_v2.py.

The model is either a built-in name (e.g. cpdino, zero-shot) or a fine-tuned weights
file made by finetune_cpdino.py. Optional preprocessing: imes_clahe (IMES's per-channel CLAHE).

Output (new folder per run, nothing is overwritten):
    <out-root>\\predict\\<YYYYmmdd-HHMMSS>_<subset>_<label>\\
        masks\\<label>\\<name>_<label>_instances.tif   uint16 labels, original image size
        predict_config.json                            settings, split, image list, versions

Safety:
  * --subset test needs --confirm-test (the test set is used once, at the end);
  * the run stops with an error unless EVERY image of the subset got a mask of the right size.

Scoring afterwards (unchanged scorer, from the repo):
    uv run --project "<repo>\\cellpose-env" python "<repo>\\scripts\\score_masks_v2.py" `
        --batch-dirs "<this run folder>" --group-dir "<data>\\Training_porcine_bf"

Usage:
    uv run --project "<repo>\\cellpose-env" python "<exp>\\scripts\\predict_split.py" `
        --split-file "<repo>\\splits\\porcine_bf_v1.json" --subset val `
        --group-dir "<data>\\Training_porcine_bf" --out-root "<...>\\Cloud_results\\cpdino_ft" `
        --model cpdino --preprocess imes_clahe --label cpdino-clahe
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imes_preprocess import imes_clahe  # noqa: E402
from split_io import (  # noqa: E402
    SUBSETS, check_label, find_image, load_cellpose, load_split, sha256_file, versions,
)

PREPROCESS = {"none": lambda img: img, "imes_clahe": imes_clahe}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-file", type=Path, required=True, help="JSON made by make_split.py")
    p.add_argument("--subset", choices=SUBSETS, required=True)
    p.add_argument("--confirm-test", action="store_true", help="required for --subset test")
    p.add_argument("--group-dir", type=Path, required=True, help="folder with *_image.* files")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--model", required=True, help="built-in name (cpdino, cpdino-vitb, ...) or a weights file")
    p.add_argument("--label", required=True,
                   help="name for this model+setting in folders and results, e.g. cpdino-clahe")
    p.add_argument("--preprocess", choices=sorted(PREPROCESS), required=True)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--diameter", type=float, default=None, help="None = no resizing (as in zero-shot runs)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--note", default="")
    a = p.parse_args(argv)

    if a.subset == "test" and not a.confirm_test:
        p.error("--subset test needs --confirm-test (the test set is used ONCE, at the end)")
    label = check_label(a.label)
    split = load_split(a.split_file)
    names = split["splits"][a.subset]
    if split["group_dir_name"] != a.group_dir.name:
        raise SystemExit(f"Split was made for '{split['group_dir_name']}', not '{a.group_dir.name}'.")
    images = {n: find_image(a.group_dir, n) for n in names}   # stops early if one is missing

    import tifffile
    import torch
    from cellpose import io as cpio

    use_gpu = (not a.cpu) and torch.cuda.is_available()
    print(f"split {split['name']} ({split['fingerprint']}) | subset {a.subset} | {len(names)} images | "
          f"model {a.model} | preprocess {a.preprocess} | GPU {use_gpu}")
    model = load_cellpose(a.model, use_gpu)
    print(f"    loaded {model.backbone}, weights {Path(model.pretrained_model).name}, "
          f"dtype {next(model.net.parameters()).dtype} (all keys checked)")

    run_dir = a.out_root / "predict" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.subset}_{label}"
    mask_dir = run_dir / "masks" / label
    mask_dir.mkdir(parents=True)   # fails if it exists: never overwrite

    config = {"args": {k: str(v) for k, v in vars(a).items()},
              "split": {"name": split["name"], "fingerprint": split["fingerprint"], "subset": a.subset},
              "images": names, "weights_file": str(model.pretrained_model),
              "weights_sha256": sha256_file(Path(model.pretrained_model)),
              "backbone": model.backbone, "versions": versions(),
              "started": dt.datetime.now().isoformat(timespec="seconds")}
    (run_dir / "predict_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    written, times = [], []
    for i, name in enumerate(names, 1):
        raw = cpio.imread(str(images[name]))
        img = PREPROCESS[a.preprocess](raw)
        t0 = time.time()
        masks = np.asarray(model.eval(img, batch_size=a.batch_size, diameter=a.diameter,
                                      channel_axis=-1 if img.ndim == 3 else None,
                                      flow_threshold=a.flow_threshold,
                                      cellprob_threshold=a.cellprob_threshold)[0])
        times.append(time.time() - t0)
        if masks.shape != raw.shape[:2]:
            raise SystemExit(f"{name}: mask shape {masks.shape} != image shape {raw.shape[:2]}")
        if masks.max() > np.iinfo(np.uint16).max:
            raise SystemExit(f"{name}: more than 65535 objects, cannot store as uint16")
        out = mask_dir / f"{name}_{label}_instances.tif"
        tifffile.imwrite(out, masks.astype(np.uint16))
        written.append(name)
        print(f"    [{i}/{len(names)}] {name}: {int(masks.max())} objects ({times[-1]:.1f} s)")

    missing = sorted(set(names) - set(written))
    if missing:
        raise SystemExit(f"Missing masks for {missing}")
    config.update(finished=dt.datetime.now().isoformat(timespec="seconds"),
                  seconds_per_image=round(float(np.mean(times)), 2))
    (run_dir / "predict_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nall {len(names)} masks written -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
