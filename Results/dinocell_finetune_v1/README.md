# DINOCell fine-tuning on porcine brightfield, split `porcine_bf_v1`

Run on CAIRO on 5 Nov 2026, raw images, 1000 steps, best-validation-Dice checkpoint, seeds 0, 1, 2, LoRA, base weights
`DINOCell_demo_model.pt` (Hugging Face KadenStillwagon/DINOCell). The scripts are on branch `pavin/training-day`
(`experiments/dinocell_finetune_v1/`), not yet on main. Weights, masks and images are not committed.

## A bug in the first run, and the fix
The first run (`*_BUGGY_loss`) collapsed: the model marked every pixel as fat (validation precision 0.18, recall 1.0, Dice 0.29,
whole-image fat % error +59 pp). Cause: the background term of the cell-probability loss was -log(2 - p) instead of -log(1 - p) as in
IMES's notebook, so "all fat" had the lowest loss. Fixed in commit f15046c on `pavin/training-day`. The buggy runs are kept here only
as a record and must not be used for results.

## Validation after the fix (3 fat images + 1 control, 3 seeds)
|Fat % error| 3.17 ± 0.42 pp, signed +3.0, Dice 0.831, F1@0.5 0.386; control image about +1.1 pp with 18 to 20 times too many objects.

## Test (5 fat images + 1 control, used once, 3 seeds)
|Fat % error| 4.36 ± 0.55 pp (3.82, 4.92, 4.35), signed +3.95 ± 0.80, Dice 0.795, F1@0.5 0.481, F1@0.7 0.249, count error about +89%;
control image +0.8 to +0.9 pp with about 50 objects (2 annotated). Do not re-run test predictions.

Side by side (not a clean same-setting ranking: other models used raw or CLAHE and different seeds): cpsam 1.82 ± 0.36,
cpdino ViT-L 2.04 ± 0.08, cpdino-vitb 2.79 ± 0.24, DINOCell 4.36 ± 0.55 pp. One split, 5 fat test images.

## Files
`scores/<set>/per_image.csv`, `summary.csv` (computed by `scripts/score_masks_v2.py`; the mean ± sd over seeds above is my own calculation
from the three per-seed rows), `train/<label>/train_config.json` and `metrics.jsonl` (training and validation crop metrics every 50 steps).
