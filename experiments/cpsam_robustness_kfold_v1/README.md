# cpsam: robustness check and 4-fold cross-validation (v1)

Two small experiments on the cpsam fine-tune (settings locked on 4 Nov: raw images, 100 epochs,
Cellpose defaults). Nothing here chooses a model or a setting.

**Status: written and unit-tested on fake data (fold files, undo of flips/rotation, pooling).
Not run on a model yet.**

## 1. Robustness check (`scripts/robustness_check.py`)
Question: does the fat % change if the same image is flipped, rotated, brighter or darker?
Variants: identity, hflip, vflip, rot90, bright_0.8, bright_1.2, gamma_0.8, gamma_1.2. Geometric variants
are undone on the predicted mask so it lines up with the ground truth again.
Output: fat %, change vs the identity run (in pp) and Dice vs the identity mask, per image and per variant.

What it does not do: it does not add images. Variants of one image are not independent samples, so the
noise of a 6-image test stays the same. It measures stability of one model, nothing else.
Default subset is `val` (4 images, ~2 min on a V100). `--subset test` needs `--confirm-test`: the porcine_bf_v1
test set was already used once for cpsam, so a test run is a diagnostic only.

```bash
uv run --no-sync --project <repo>/cellpose-env python experiments/cpsam_robustness_kfold_v1/scripts/robustness_check.py \
    --split-file splits/porcine_bf_v1.json --subset val --group-dir <data>/Training_porcine_bf \
    --out-root <outside repo>/cpsam_robust --model <train folder>/models/cpsam-ft-raw-v1-s0 \
    --preprocess none --label cpsam-ft-raw-v1-s0
```

## 2. 4-fold cross-validation (`make_kfold_splits.py`, `run_kfold_cpsam.sh`, `summarise_kfold.py`)
Question: how much does the score depend on WHICH images are held out? Every one of the 28 images is the
test image of exactly one fold (7 images per fold; train 18, val 3 for loss watching, test 7). Fold files have
the same format as `splits/porcine_bf_v1.json`, so the existing scripts read them unchanged. Stratified like
porcine_bf_v1; with 3 control images, one fold has no control.

Limits: the 28 images include the 6 test images already used, so this is a new experiment, reported separately.
It gives a score on all 28 images instead of 6, but no new images.

Cost (estimate from the measured 5.7 s per epoch when the GPU is free): about 11 min per fold-seed.
4 folds x 1 seed = about 45 min; 4 folds x 3 seeds = about 2.3 h.

```bash
export DATA=~/Training_porcine_bf OUT=~/cellfoundry_results/cpsam_kfold FOLDS="0 1 2 3" SEEDS="0"
bash experiments/cpsam_robustness_kfold_v1/scripts/run_kfold_cpsam.sh     # in tmux; prints the scoring commands at the end
```
Scoring and pooling: `score_masks_v2.py --batch-dirs <OUT>/predict/*_test_cpsam-kf* --group-dir <data>`, then
`summarise_kfold.py --per-image <rescore folder>/per_image.csv` (per seed over all 28 images, per fold, sd over
images and sd over seeds named separately).

## Folder
| Path | Content |
|---|---|
| `scripts/make_kfold_splits.py` | writes `porcine_bf_kfold4_fold{0..3}.json` |
| `scripts/run_kfold_cpsam.sh` | train + predict each fold, one job at a time, `--no-sync` |
| `scripts/summarise_kfold.py` | pools per-image scores over folds and seeds |
| `scripts/robustness_check.py` | the robustness check above |

Reused, not copied: `split_io.py`, `imes_preprocess.py`, `finetune_cpdino.py`, `predict_split.py`
from `experiments/cpdino_vitb_finetune_v1/scripts/` and the scorer in `scripts/`.
Needs the cpsam support of PR #41 (`--base-model cpsam`); this branch is built on top of it.
