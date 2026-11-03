# cpdino-vitb fine-tuning on porcine brightfield (v1)

**Data:** porcine brightfield (`Training_porcine_bf`) · **Status:** final for this setting (test set used once)
**Split:** `porcine_bf_v1` – *proposed, pending team approval*

## Summary

- Fine-tuning cpdino-vitb on 18 porcine BF images **halves the typical fat-% error on unseen test images**
  (5.80 → **2.79 ± 0.24 pp**), raises fat-area Dice from 0.68 to **0.84** and F1@0.7 from 0.38 to **0.60** (3 training seeds).
- **IMES's CLAHE preprocessing gives no benefit for cpdino-vitb**, neither zero-shot nor after fine-tuning.
  After fine-tuning, raw input is slightly but consistently better (0.13 pp, 3 of 3 paired seeds).
  **Recommendation: raw input (no CLAHE) as the cpdino-vitb default from now on.**
- One run takes ~10 min on an 8 GB laptop GPU (peak memory 3.22 GB); CLAHE does not change training time or memory.

## 1. Data and split

28 images: 23 `CLD-AIM-04 … Positiv_<n>`, `Negativ_2`, `PAD11a/c …_Diff` (2), `PAD11a/c …_Ctrl` (2).

`porcine_bf_v1`: 18 / 4 / 6 **whole images** (no crops shared between subsets), stratified, random within
each stratum (seed 42), fingerprint `22d6fd1c60e2` (SHA-256 of the image lists; every script re-checks it).

| Stratum | Train (18) | Val (4) | Test (6) |
|---|---|---|---|
| Controls (`Negativ`, `_Ctrl`) | PAD11a_D1_Ctrl | Negativ_2 | PAD11c_D5_Ctrl |
| PAD11 `_Diff` | PAD11c_B5_Diff | – | PAD11a_B1_Diff |
| Positiv_21–26 (all zero-shot models failed on these) | 21, 24, 25 | 22 | 23, 26 |
| Other Positiv | 7, 8, 9, 10, 11, 14, 15, 17, 18, 27, 28, 29, 30 | 16, 19 | 12, 13 |

Val = development check set. Test = used **once**, after all choices were fixed on val.

## 2. Method

| Item | Setting |
|---|---|
| Model | `cpdino-vitb` (Cellpose 4.2.1.1, DINOv3 ViT-B), full fine-tuning |
| Environment | `cellpose-env` (Python 3.12, torch 2.14.1+cu132) |
| Preprocessing | (a) none (raw) or (b) IMES per-channel CLAHE: per channel min–max → uint8 → CLAHE (clipLimit 1.0, 8×8 tiles) → min–max. Same settings as IMES's notebook. Cellpose's own 1–99 percentile normalisation is applied on top in both cases |
| Training | Cellpose `train_seg` defaults: AdamW, lr 1e-5 (warm-up epochs 0–9, constant 10–49, halved every 5 epochs 50–99), weight decay 0.1, 100 epochs, batch 1, 384×384 crops, rotation + scale 0.75–1.25 + flip, FP32. Changed: `min_train_masks = 0`. Final epoch used (no checkpoint selection) |
| Seeds | Each setting trained 3 times (seeds 0, 1, 2) to measure run-to-run variation. A seed fixes the random choices in training (which crops, rotations, scales and flips, and which layers are randomly skipped). Seed 0 = Cellpose's default behaviour. Seed k of raw and seed k of CLAHE make identical random choices → **paired** comparison |
| Prediction | `flow_threshold 0.4`, `cellprob_threshold 0.0`, no diameter (same as all zero-shot runs), bf16 |
| Scoring | `scripts/score_masks_v2.py` (unchanged): fat %, fat % error, fat-area Dice, count error, F1 / SA at IoU 0.5–0.9 |

**Pre-set decision rule** (fixed before the raw fine-tune results were seen): the setting with the lower
seed-mean |fat % error| on the differentiated val images wins; a difference < 0.5 pp counts as a tie → CLAHE.

## 3. Results on val (4 images)

Differentiated val images (3: Positiv_16, 19, 22), mean ± SD over 3 seeds [min–max]:

| | Zero-shot raw | Zero-shot CLAHE | FT CLAHE | FT raw |
|---|---|---|---|---|
| \|fat % error\| (pp) | 7.90 | 7.62 | 1.65 ± 0.18 [1.45–1.81] | **1.52 ± 0.17** [1.33–1.66] |
| fat % error (pp) | −2.34 | −2.34 | −0.63 ± 0.46 | −0.67 ± 0.45 |
| fat-area Dice | 0.687 | 0.699 | 0.874 ± 0.003 | 0.875 ± 0.003 |
| count error (pooled) | −11.9 % | +14.7 % | −5.5 ± 1.7 % | −5.1 ± 1.6 % |
| F1@0.5 | 0.574 | 0.551 | 0.817 ± 0.006 | 0.819 ± 0.003 |
| F1@0.7 | 0.388 | 0.389 | 0.662 ± 0.003 | 0.668 ± 0.007 |

Control val image (Negativ_2, 5 annotated cells): zero-shot raw +8.16 pp fat (370 objects), zero-shot CLAHE
+9.88 pp (414); FT CLAHE −0.05 ± 0.12 pp (4–10 objects); FT raw −0.05 ± 0.04 pp (5–7 objects).

**Paired comparison, FT raw vs FT CLAHE** (same seed = same crops; 3 pairs; paired t-test):

| Metric | raw − CLAHE | raw better | p |
|---|---|---|---|
| \|fat % error\| (pp), pre-specified | −0.127 ± 0.021 | 3 / 3 | 0.009 |
| fat-area Dice | +0.001 | 3 / 3 | 0.057 |
| F1@0.7 | +0.006 | 3 / 3 | 0.17 |
| F1@0.5 | +0.002 | 2 / 3 | 0.40 |
| \|count error\| | −0.5 pp | 3 / 3 | 0.034 |
| control \|fat % error\| (pp) | −0.06 | 3 / 3 | 0.049 |

**Zero-shot raw vs CLAHE:** no consistent effect. Fat area slightly better with CLAHE, but F1@0.5, count error
and the control get worse. CLAHE mainly adds objects (Positiv_22: 274 → 526, GT 394), likely large lipid
droplets split off as cells.

**Rule outcome:** 1.52 vs 1.65 pp → difference 0.13 pp < 0.5 pp → tie → CLAHE was used for the test run.
The paired analysis (done afterwards) shows the difference is small but consistent in favour of raw; the 0.5 pp
margin was meant as practical relevance, not as a significance test.

## 4. Results on test (6 images, used once) – FT CLAHE

Differentiated test images (5: Positiv_12, 13, 23, 26, PAD11a_B1_Diff):

| | Zero-shot cpdino-vitb (raw) | FT CLAHE (mean ± SD, 3 seeds) |
|---|---|---|
| \|fat % error\| (pp) | 5.80 | **2.79 ± 0.24** [2.62–3.06] |
| fat % error (pp) | −0.53 | +1.84 ± 0.35 [1.54–2.23] |
| fat-area Dice | 0.678 | **0.843 ± 0.001** |
| count error (pooled) | −13.7 % | **+2.3 ± 1.0 %** |
| F1@0.5 | 0.569 | **0.787 ± 0.002** |
| F1@0.7 | 0.383 | **0.603 ± 0.005** |

Control test image (PAD11c_D5_Ctrl): zero-shot +0.88 pp (53 objects); FT +0.71 ± 0.02 pp (41–44 objects).

FT raw models were not run on test.

## 5. Findings

1. **Fine-tuning is what helps.** All improvements come from training; the 3 seeds agree closely.
2. **CLAHE does not help cpdino-vitb** (zero-shot: mixed; fine-tuned: raw slightly but consistently better).
   CLAHE is part of IMES's DINOCell recipe; cpdino-vitb does not need it. It also costs nothing measurable:
   training time 10.0–10.2 min and peak GPU memory 3.22 GB with and without CLAHE.
3. **Signed vs absolute error:** zero-shot's small signed test error (−0.53 pp) comes from large over- and
   under-estimates cancelling. |fat % error| is the meaningful measure. The fine-tuned model has a consistent
   positive bias on test (+1.8 pp) – predictable, but its cause is not yet examined (per-image analysis open).
4. **Test is harder than val** (2.8 vs 1.65 pp): it contains two large-droplet images (Positiv_23, 26) and the PAD11 Diff image.
5. **PAD11 control barely improves** (53 → 43 false objects), unlike Negativ_2 on val (370 → ~5).
   The PAD11 series has only 2 training images.

## 6. Limits

- Porcine brightfield only; one split; 6 test images.
- Spread is over 3 training seeds, not over splits or new images.
- Ground truth is partially annotated (likely biases fat % low).
- Paired p-values come from 3 seeds and several metrics; |fat % error| was the pre-specified metric.
- Cellpose's printed val loss is not comparable across seeds (the seed shift also changes its val crops).

## 7. Cellpose 4.2.1.1 behaviours found in its source (relevant for anyone training Cellpose models)

1. An unknown model name or missing weights path only logs a warning and then **loads `cpsam_v2` instead**.
2. Weights are loaded with `strict=False`: non-matching keys are skipped silently.
3. The built-in cpdino / cpdino-vitb files contain no `diam_labels` / `diam_mean` (default 30; never read at prediction).
4. `min_train_masks` (default 5) silently removes training images with fewer than 5 cells.
5. `train_seg` re-seeds numpy with the epoch number every epoch, so repeated runs draw identical crops unless the seed is shifted.
6. Training converts the bf16 model to FP32; saved weights are FP32.
7. The network type (ViT-B / ViT-L) is read from the weights, not from the file name.

The scripts guard against 1–2 (stop with an error), allow 3 only at the default value, set 4 explicitly
and shift the numpy seeds for 5 (`--seed 0` = unchanged Cellpose behaviour).

## 8. Files and reproduction

**In this folder:**

| File | Content |
|---|---|
| `README.md` | This report |
| `val_summary.csv`, `val_per_image.csv` | `score_masks_v2.py` output for the 8 val runs (2 zero-shot, 6 fine-tuned) |
| `val_seeds_summary.csv` | Mean ± SD over seeds for the val runs (`summarise_seeds.py`; the paired table in §3 is printed by the same script) |
| `test_summary.csv`, `test_per_image.csv` | `score_masks_v2.py` output for the 4 test runs (zero-shot + 3 FT CLAHE seeds) |
| `test_seeds_summary.csv` | Mean ± SD over seeds for the test runs |

**Elsewhere:**

| Path | Purpose |
|---|---|
| `../README.md` | **Step-by-step: how these results were produced** (all commands in order) |
| `../PROTOCOL_porcine_bf_v1.md` | How to evaluate any other model on the same split |
| `../scripts/` | The scripts used for this experiment (see `../README.md`) |
| `splits/porcine_bf_v1.json` (repo root) | The shared split (fingerprint `22d6fd1c60e2`) |
| `scripts/score_masks_v2.py` (repo root) | Scoring (unchanged) |

`<exp>` below = `<repo>\experiments\cpdino_vitb_finetune_v1`.

**Reproduce one run** (PowerShell, environment `cellpose-env`; `<repo>`, `<data>`, `<results>` are placeholders):

```powershell
# fine-tune (one seed; repeat with --seed 1 and 2)
uv run --project "<repo>\cellpose-env" python "<exp>\scripts\finetune_cpdino.py" `
    --split-file "<repo>\splits\porcine_bf_v1.json" --group-dir "<data>\Training_porcine_bf" `
    --out-root "<results>\cpdino_ft" --preprocess none --seed 0 --label cpdino-vitb-ft-raw-v1-s0

# predict the val subset
uv run --project "<repo>\cellpose-env" python "<exp>\scripts\predict_split.py" `
    --split-file "<repo>\splits\porcine_bf_v1.json" --subset val --group-dir "<data>\Training_porcine_bf" `
    --out-root "<results>\cpdino_ft" --model "<train run>\models\cpdino-vitb-ft-raw-v1-s0" `
    --preprocess none --label cpdino-vitb-ft-raw-v1-s0

# score (one or more prediction folders) and summarise over seeds
uv run --project "<repo>\cellpose-env" python "<repo>\scripts\score_masks_v2.py" `
    --batch-dirs <prediction folders> --group-dir "<data>\Training_porcine_bf"
uv run --project "<repo>\cellpose-env" python "<exp>\scripts\summarise_seeds.py" `
    --summary "<rescore folder>\summary.csv"
```

Zero-shot runs: `predict_split.py --model cpdino-vitb` (built-in name). Test: add `--subset test --confirm-test`.
Run outputs are not in git: `<results>\cpdino_ft\train\` (weights, losses, config), `predict\` (masks), `rescore\` (CSVs).
