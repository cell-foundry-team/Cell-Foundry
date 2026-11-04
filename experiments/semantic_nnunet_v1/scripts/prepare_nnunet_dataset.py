"""Turn the porcine_bf_v1 split into an nnU-Net v2 dataset for SEMANTIC segmentation
(every pixel: fat = 1, not fat = 0), so nnU-Net can be compared with the instance models on
the same split and with the same scorer.

What it writes (nothing in the repo; --nnunet-raw must point outside it):
    <nnUNet_raw>/Dataset<ID>_PorcineBFFat/
        imagesTr/case_XXX_0000.png     train + val images (RGB, one file with 3 channels)
        labelsTr/case_XXX.png          binary masks: instance mask > 0 -> 1, background 0
        imagesTs/case_XXX_0000.png     the 6 TEST images (nnU-Net never trains on imagesTs)
        dataset.json                   channel_names R/G/B, labels {background 0, fat 1}
        case_map.json                  case_XXX <-> original image name, and its subset
        splits_final.json              one fold: train = split train, val = split val
Image names contain spaces, so cases get neutral ids (case_000 ...); case_map.json maps back.

Format follows nnU-Net's own docs (documentation/dataset_format.md) and its 2D RGB PNG example
(nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py): RGB PNG kept as one file,
channel_names {0: R, 1: G, 2: B}, labels consecutive from 0.

splits_final.json must end up in <nnUNet_preprocessed>/Dataset<ID>_PorcineBFFat/ (nnU-Net's
documentation/manual_data_splits.md). That folder is created by nnUNetv2_plan_and_preprocess,
so run this script again with --copy-splits-to after preprocessing (see README.md).

Preprocessing: --preprocess none | imes_clahe | grey_clahe, the same functions as the
cpdino-vitb scripts (imported, not copied). nnU-Net then applies its own per-channel z-score.

Usage (environment: experiments/semantic_nnunet_v1):
    uv run --project <exp> python <exp>/scripts/prepare_nnunet_dataset.py \\
        --split-file <repo>/splits/porcine_bf_v1.json --group-dir <data>/Training_porcine_bf \\
        --nnunet-raw <outside repo>/nnUNet_raw --preprocess none
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpdino_vitb_finetune_v1" / "scripts"))
from imes_preprocess import grey_clahe, imes_clahe, to_three_channels  # noqa: E402
from split_io import find_gt, find_image, load_labels, load_split  # noqa: E402

PREPROCESS = {"none": lambda img: img, "imes_clahe": imes_clahe, "grey_clahe": grey_clahe}
DATASET_NAME = "PorcineBFFat"


def dataset_dir(root: Path, dataset_id: int) -> Path:
    return root / f"Dataset{dataset_id:03d}_{DATASET_NAME}"


def read_rgb(path: Path) -> np.ndarray:
    import imageio.v3 as iio

    img = to_three_channels(np.asarray(iio.imread(path)))
    if img.dtype != np.uint8:
        raise SystemExit(f"{path.name}: expected an 8-bit image, got {img.dtype}")
    return img


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--group-dir", type=Path, help="folder with *_image.* / *_instances.* (not needed with --copy-splits-to)")
    p.add_argument("--nnunet-raw", type=Path, required=True, help="nnUNet_raw folder, OUTSIDE the repo")
    p.add_argument("--dataset-id", type=int, default=501)
    p.add_argument("--preprocess", choices=sorted(PREPROCESS), default="none")
    p.add_argument("--copy-splits-to", type=Path, default=None,
                   help="after nnUNetv2_plan_and_preprocess: the nnUNet_preprocessed folder; copies "
                        "splits_final.json there and exits")
    a = p.parse_args(argv)

    out = dataset_dir(a.nnunet_raw, a.dataset_id)

    if a.copy_splits_to is not None:
        target = dataset_dir(a.copy_splits_to, a.dataset_id)
        if not target.is_dir():
            raise SystemExit(f"{target} does not exist: run nnUNetv2_plan_and_preprocess first.")
        shutil.copy(out / "splits_final.json", target / "splits_final.json")
        print(f"copied splits_final.json -> {target}")
        return 0

    if a.group_dir is None:
        p.error("--group-dir is required when preparing the dataset")
    split = load_split(a.split_file)
    if split["group_dir_name"] != a.group_dir.name:
        raise SystemExit(f"Split was made for '{split['group_dir_name']}', not '{a.group_dir.name}'.")
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting (use a new --dataset-id or delete it).")
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (out / sub).mkdir(parents=True)

    import imageio.v3 as iio

    case_map, splits = {}, {"train": [], "val": []}
    index = 0
    for subset in ("train", "val", "test"):
        for name in split["splits"][subset]:
            case = f"case_{index:03d}"
            index += 1
            img = PREPROCESS[a.preprocess](read_rgb(find_image(a.group_dir, name)))
            folder = "imagesTs" if subset == "test" else "imagesTr"
            iio.imwrite(out / folder / f"{case}_0000.png", img)
            if subset != "test":
                labels = load_labels(find_gt(a.group_dir, name))
                if labels.shape != img.shape[:2]:
                    raise SystemExit(f"{name}: mask {labels.shape} and image {img.shape[:2]} differ in size")
                iio.imwrite(out / "labelsTr" / f"{case}.png", (labels > 0).astype(np.uint8))
                splits[subset].append(case)
            case_map[case] = {"name": name, "subset": subset}
            print(f"    {subset:5s} {case}: {name}")

    n_train = len(splits["train"]) + len(splits["val"])
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "fat": 1},
        "numTraining": n_train,
        "file_ending": ".png",
        "name": f"Dataset{a.dataset_id:03d}_{DATASET_NAME}",
        "description": f"porcine_bf_v1 ({split['fingerprint']}), binary fat mask, preprocess={a.preprocess}",
    }, indent=2), encoding="utf-8")
    (out / "case_map.json").write_text(json.dumps({
        "split": {"name": split["name"], "fingerprint": split["fingerprint"]},
        "preprocess": a.preprocess, "cases": case_map,
    }, indent=2), encoding="utf-8")
    (out / "splits_final.json").write_text(json.dumps([splits], indent=2), encoding="utf-8")

    print(f"\n{n_train} training cases ({len(splits['train'])} train / {len(splits['val'])} val), "
          f"{len(split['splits']['test'])} test images -> {out}")
    print("next: nnUNetv2_plan_and_preprocess, then re-run with --copy-splits-to (see README.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
