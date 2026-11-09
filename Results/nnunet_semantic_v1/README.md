# nnU-Net 2D (semantic fat / not fat) on porcine brightfield, split `porcine_bf_v1`

Run on CAIRO on 9 Nov 2026. Scripts: `experiments/semantic_nnunet_v1/` on branch `pavin/training-day` (not yet on main).
Raw images, our split as fold 0 (18 train, 4 val), `nnUNetTrainer_10epochs`, 3 repeats (separate results folders; nnU-Net has no seed flag).
Weights, masks and images are not committed.

## Why 10 epochs, not 100
The 1-epoch timing test showed 204 s for the first epoch. Extrapolated, 100 epochs looked like 5.7 hours per repeat, so the trainer was changed
to 10 epochs before any nnU-Net score was seen. The real runs then took about 12 minutes each, so the extrapolation was wrong; a longer trainer
is possible but would be a new experiment decided after these numbers.

## Which columns count
The output is one label (fat or not), so only fat % error and Dice mean anything. Count error (-99.7%) and F1 (0) are artefacts of a one-object output.

## Validation (3 fat images + 1 control)
|Fat % error| 3.14 ± 0.22 pp (3.20, 2.89, 3.32), signed about -0.4, Dice 0.869; control image +2.9, +1.9, +5.8 pp.

## Test (5 fat images + 1 control, used once)
|Fat % error| 2.96 ± 0.11 pp (2.90, 3.08, 2.89), signed +2.83 ± 0.04, Dice 0.846 (0.847, 0.847, 0.844); control +0.28, +0.29, +0.31 pp.
Mean ± sd over repeats are my calculation from the per-repeat rows of `scores/test_raw_3repeats/summary.csv`. Do not re-run the test predictions.

Side by side (not a clean ranking; other models used raw or CLAHE and different settings): cpsam 1.82 ± 0.36, cpdino ViT-L 2.04 ± 0.08,
cpdino-vitb 2.79 ± 0.24, nnU-Net 2.96 ± 0.11, DINOCell 4.36 ± 0.55 pp. nnU-Net over-estimates in every repeat (+2.8 pp) with very little spread.
Licence: Apache-2.0, so it can be delivered commercially.

## Notes
- nnU-Net hangs after "Training done" (the process idles, the checkpoint is already saved). The run script stops it and predicts with `nnUNetv2_predict -chk checkpoint_final.pth`.
- The shared CAIRO account hit its 50 GB disk quota once; the log of one repeat stopped silently at that point and the repeat was re-run.
- `logs/` holds a short summary of each training log (epoch time, pseudo Dice).
