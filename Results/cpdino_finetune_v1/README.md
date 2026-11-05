# cpdino (ViT-L) fine-tuning on porcine brightfield, split `porcine_bf_v1`

Same procedure as cpdino-vitb (`experiments/cpdino_vitb_finetune_v1/`), with `--base-model cpdino`.
Scripts and runbook: `experiments/cpdino_finetune_v1/`. Run on the Cairo cluster (Tesla V100 32 GB,
torch 2.14.1+cu126, cellpose 4.2.1.1). Seed 0 was trained on 2026-11-04, seeds 1 and 2 on 2026-11-05.

## Setup

- Split `porcine_bf_v1` (fingerprint `22d6fd1c60e2`): 18 train / 4 val / 6 test whole images.
- Cellpose `train_seg` defaults: AdamW, lr 1e-5, weight decay 0.1, 100 epochs, batch 1, 384 x 384 crops,
  `min_train_masks 0`. Peak GPU memory 9.51 GB in every run; 13.3 to 20.8 s per epoch (22 to 35 min per run;
  the slowest run, raw seed 0, shared the GPU with another job).
- Settings compared on val: raw (`none`) vs IMES CLAHE (`imes_clahe`).
- **Seeds: 3 per setting (0, 1, 2)**; numbers are mean ± SD over seeds.
- Decision rule (fixed before the results): lower seed-mean |fat % error| on the differentiated val images wins;
  a difference < 0.5 pp is a tie and goes to CLAHE.

## Validation (4 images: 3 differentiated, 1 control; used for the decision)

| Differentiated val (3 images) | \|fat % error\| (pp) | Dice | F1@0.5 | F1@0.7 | count error |
|---|---|---|---|---|---|
| zero-shot cpdino | 4.51 | 0.758 | 0.536 | 0.407 | +1.846 |
| fine-tuned raw (3 seeds) | 2.07 ± 0.15 | 0.878 ± 0.002 | 0.823 ± 0.002 | 0.663 ± 0.008 | -0.058 ± 0.010 |
| fine-tuned CLAHE (3 seeds) | 2.17 ± 0.23 | 0.878 ± 0.002 | 0.819 ± 0.006 | 0.662 ± 0.009 | -0.062 ± 0.005 |

Control val image, false objects: zero-shot 88, raw 7.7 ± 6.5 (range 1 to 14), CLAHE 8.7 ± 5.0 (range 4 to 14).

Rule: 2.07 vs 2.17 pp, difference 0.10 pp < 0.5 pp, tie, **CLAHE chosen** (`cpdino-ft-clahe-v1`).
Paired over seeds (same crops and augmentation): raw is lower in 3 of 3 seed pairs, mean difference -0.10 ± 0.08 pp,
paired t-test p = 0.17. With three pairs this is not a significant difference.

## Test (6 images: 5 differentiated, 1 control; used once, after the choice)

| Differentiated test (5 images) | \|fat % error\| (pp) | signed fat % error | Dice | F1@0.5 | F1@0.7 | count error |
|---|---|---|---|---|---|---|
| zero-shot cpdino | 3.72 | -0.53 | 0.737 | 0.562 | 0.408 | +0.387 |
| fine-tuned CLAHE (3 seeds) | 2.04 ± 0.08 | +0.94 ± 0.16 | 0.854 ± 0.001 | 0.796 ± 0.003 | 0.618 ± 0.007 | -0.030 ± 0.018 |

Per seed, |fat % error|: 2.13, 1.98, 2.01 (range 1.98 to 2.13).
Control test image: false objects 59 (zero-shot) vs 7.0 ± 3.6 (seeds: 3, 8, 10); |fat % error| 1.17 vs 0.10 ± 0.09.

## Comparison with cpdino-vitb (same split and scorer; ViT-B figures from `experiments/cpdino_vitb_finetune_v1/results/`)

| | ViT-B | ViT-L |
|---|---|---|
| Test \|fat % error\|, zero-shot to fine-tuned CLAHE | 5.80 to 2.79 ± 0.24 | 3.72 to 2.04 ± 0.08 |
| Test Dice | 0.68 to 0.84 | 0.74 to 0.85 |
| Test F1@0.7 | 0.38 to 0.60 | 0.41 to 0.62 |
| Val \|fat % error\|, fine-tuned raw / CLAHE | 1.52 ± 0.17 / 1.65 ± 0.18 | 2.07 ± 0.15 / 2.17 ± 0.23 |

ViT-L is lower on test (0.75 pp, larger than either SD) and higher on val (about 0.5 pp). The two small sets
(3 and 5 differentiated images) point in different directions, so no consistent winner is claimed. ViT-L needs
about three times the GPU memory (9.51 vs 3.22 GB) and two to three times the time per run.

## Visual check on the test images

`test_overlay_counts.csv` compares the model outlines with the IMES annotation per image
(`scripts/make_combined.py`: a predicted cell is an *extra* when less than 20% of it lies on annotated cells; an
annotated cell is *missed* when less than 20% of it is covered by predictions). Over the 5 differentiated test
images (1,815 annotated cells), zero-shot vs fine-tuned CLAHE seed 0:

| | zero-shot | fine-tuned (seed 0) |
|---|---|---|
| predicted cells | 2,517 | 1,731 |
| extra cells | 582 | 108 |
| annotated cells missed | 378 | 159 |

On the control image: 59 vs 3 predicted objects (IMES annotated 2). Extras are either false detections or real cells
that were not annotated (IMES annotates only cells it is sure about); this was not checked with IMES. On
PAD11a_B1 the improvement is smallest (54 to 32 extras, 34 to 31 missed). `scripts/make_overlays.py` draws
side-by-side overlays.

## Notes

- Test-set use: zero-shot cpdino and CLAHE seed 0 were scored on test on 2026-11-04, before seeds 1 and 2 existed.
  CLAHE seeds 1 and 2 were scored on 2026-11-05 after the 3-seed validation rule also chose CLAHE, so no choice
  depended on a test number. Raw fine-tuned models were never scored on test.
- Raw and CLAHE are indistinguishable on val with three seeds; raw is slightly ahead on |fat % error| (3 of 3
  seeds, p = 0.17). The choice follows the pre-set rule.
- `cellpose-env` installs a CUDA 13 torch by default, which does not run on the V100 / driver 570 (CUDA 12.8).
  `torch==2.14.1+cu126` and `torchvision==0.29.1+cu126` were installed into the venv by hand, with `UV_NO_SYNC=1`.
  A Linux cu126 pin in `cellpose-env` is still to be discussed with the team.
- Files: `val3_*.csv` and `test3_*.csv` (`summary`, `per_image`, `seeds_summary`) are the 3-seed results;
  `val_*.csv` and `test_*.csv` are the earlier seed-0-only results (2026-11-04), kept for the record.
  `train_config_*.json` for all six runs. Weights, masks and images are not committed.
