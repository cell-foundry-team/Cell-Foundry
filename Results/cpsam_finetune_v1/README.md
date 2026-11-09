# cpsam fine-tuning on porcine brightfield, split `porcine_bf_v1`

Run on CAIRO (Tesla V100 32 GB, torch 2.14.1+cu126, cellpose 4.2.1.1), 4 to 5 Nov 2026. Script:
`experiments/cpdino_vitb_finetune_v1/scripts/finetune_cpdino.py --base-model cpsam` (added in PR #41).
Split `porcine_bf_v1` (fingerprint `22d6fd1c60e2`): 18 train, 4 val, 6 test whole images.
100 epochs, 256 px tiles, Cellpose default augmentation, seeds 0, 1, 2. Weights, masks and images are not committed.

## Validation (3 fat images + 1 control)
| setting | \|fat % error\| (pp), mean ± sd over 3 seeds | Dice |
|---|---|---|
| raw (chosen) | 1.58 ± 0.31 | 0.879 |
| CLAHE (side check, not used to choose) | 1.90 ± 0.31 | 0.876 |
Difference 0.31 pp is under the 0.5 pp tie margin, so raw stays. Raw was locked on 4 Nov before seeds 1 and 2 were seen.

## Test (5 fat images + 1 control, used once, raw, 3 seeds)
| | \|fat % error\| (pp) | signed (pp) | Dice | count error | control image |
|---|---|---|---|---|---|
| zero-shot cpsam | 6.32 | -6.32 | 0.573 | +127% | 57 objects |
| fine-tuned raw | 1.82 ± 0.36 | +1.00 ± 0.65 | 0.853 | +2.9% | 2.3 objects (2 annotated) |

Per image the signs differ (Positiv_12 and 23 over-estimated in every seed, Positiv_13 under-estimated), so the signed mean hides the
per-image error. The IMES mask is incomplete, so a positive difference is not proof of an error.

## Files
`scores/<set>/per_image.csv`, `summary.csv`, `seeds_summary.csv` (from `scripts/score_masks_v2.py`), `train/<label>/train_config.json`
and `losses.csv`. Test predictions for cpsam were made once; do not re-run them.
