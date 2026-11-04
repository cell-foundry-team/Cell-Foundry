# nnU-Net semantic baseline on porcine brightfield (v1)

Semantic segmentation: every pixel is fat (1) or not fat (0), no separate cells. This matches the
project's primary metric (fat proportion) directly, as the mentor suggested on 22 Oct ("semantic
segmentation should be enough and is easier"). Same split (`splits/porcine_bf_v1.json`) and same
scorer (`scripts/score_masks_v2.py`, unchanged) as the cpdino-vitb experiment, so the area numbers
are directly comparable.

**Status: written, not run yet.** Nothing here has been executed on real data.

Why nnU-Net: it configures patch size, network depth and batch size itself from the dataset,
has published small-data results (e.g. 46 training images), Apache-2.0 licence, maintained
(nnunetv2 2.8.1, July 2026).

## What is different from the instance models

- Only the **area columns** are meaningful: fat %, fat % error, |fat % error|, foreground Dice.
  The fat area is saved as one label, so count error and F1 are not meaningful for this model.
- nnU-Net normalises each channel itself (z-score), so `--preprocess none` is the natural default.
- Training images = split train (18) + val (4) in `imagesTr`; a custom `splits_final.json` makes
  fold 0 use exactly those 18 / 4. The 6 test images go to `imagesTs`, which nnU-Net never uses
  for training.

## Folder contents

| Path | Content |
|---|---|
| `pyproject.toml` | Own uv environment (nnunetv2 2.8.1, torch 2.14.1; Windows gets the CUDA build) |
| `scripts/prepare_nnunet_dataset.py` | Split -> nnU-Net dataset (binary masks, case ids, dataset.json, splits_final.json) |
| `scripts/export_predictions.py` | nnU-Net predictions -> shared mask format for `score_masks_v2.py` |

Reused, not copied: `split_io.py` and `imes_preprocess.py` from `experiments/cpdino_vitb_finetune_v1/scripts/`.

## Steps (bash; PowerShell: use `$env:NAME = "..."` instead of `export`)

```bash
exp="<repo>/experiments/semantic_nnunet_v1"
data="<data>/Training_porcine_bf"
work="<folder OUTSIDE the repo>/nnunet"          # holds real IMES images: never inside the repo
export nnUNet_raw="$work/nnUNet_raw"
export nnUNet_preprocessed="$work/nnUNet_preprocessed"
export nnUNet_results="$work/nnUNet_results_s0"  # one results folder per repeat (see step 4)

# 0. environment (once)
cd "$exp" && uv sync

# 1. build the dataset from the split
uv run python scripts/prepare_nnunet_dataset.py --split-file <repo>/splits/porcine_bf_v1.json \
    --group-dir "$data" --nnunet-raw "$nnUNet_raw" --preprocess none

# 2. let nnU-Net plan and preprocess (2D only)
uv run nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity

# 3. put the custom split in place (must come AFTER step 2)
uv run python scripts/prepare_nnunet_dataset.py --split-file <repo>/splits/porcine_bf_v1.json \
    --nnunet-raw "$nnUNet_raw" --copy-splits-to "$nnUNet_preprocessed"

# 4. train fold 0. First a 1-epoch sanity run, then a real one.
uv run nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_1epoch        # does it run? memory?
uv run nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_100epochs     # first real run
```

Trainer choice: nnU-Net's default is much longer; ready-made shorter trainers exist
(`nnUNetTrainer_1epoch`, `_5epochs`, `_10epochs`, `_20epochs`, `_50epochs`, `_100epochs`, `_250epochs`, ...).
Fix the trainer before looking at results (protocol rule 3).

Repeats instead of seeds: nnU-Net writes to a fixed folder per dataset/trainer/fold, so a second
run would overwrite the first. For 3 repeats (`-s0`, `-s1`, `-s2`), set `nnUNet_results` to a new
folder each time (`..._s0`, `..._s1`, `..._s2`). The runs differ through nnU-Net's own randomness;
they are not seeded like `finetune_cpdino.py`.

```bash
# 5. VAL: nnU-Net already predicted the 4 val images (whole images) at the end of training
pred_val="$nnUNet_results/Dataset501_PorcineBFFat/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_0/validation"
uv run python scripts/export_predictions.py --pred-dir "$pred_val" \
    --case-map "$nnUNet_raw/Dataset501_PorcineBFFat/case_map.json" --group-dir "$data" \
    --subset val --out-root "$work/scored" --label nnunet-2d-raw-v1-s0

# 6. score with the shared scorer (cellpose-env has its dependencies)
uv run --project <repo>/cellpose-env python <repo>/scripts/score_masks_v2.py \
    --batch-dirs "$work/scored/predict/<timestamp>_val_nnunet-2d-raw-v1-s0" --group-dir "$data"

# 7. TEST, once, only after all choices are fixed on val
uv run nnUNetv2_predict -i "$nnUNet_raw/Dataset501_PorcineBFFat/imagesTs" -o "$work/pred_test_s0" \
    -d 501 -c 2d -f 0 -tr nnUNetTrainer_100epochs
uv run python scripts/export_predictions.py --pred-dir "$work/pred_test_s0" \
    --case-map "$nnUNet_raw/Dataset501_PorcineBFFat/case_map.json" --group-dir "$data" \
    --subset test --out-root "$work/scored" --label nnunet-2d-raw-v1-s0
```

## Open points

- Not run yet: memory and time per epoch on the 8 GB laptop and on CAIRO are unknown (step 4's
  1-epoch run answers this).
- `uv sync` not run yet either: there is no `uv.lock` for this environment until someone runs it.
- Preprocessing variants (`imes_clahe`, `grey_clahe`) need a separate dataset id each
  (e.g. `--dataset-id 502`), because the images themselves change.
