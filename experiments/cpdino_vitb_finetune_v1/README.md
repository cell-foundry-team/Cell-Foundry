# cpdino-vitb fine-tuning on porcine brightfield – how it was run

This folder contains the scripts, the step-by-step procedure and the results of fine-tuning **cpdino-vitb**
(Cellpose 4.2.1.1, DINOv3 ViT-B) on the porcine brightfield images with the shared split `porcine_bf_v1`.

**Result in one line:** on unseen test images, fine-tuning halves the typical fat-% error
(5.80 → 2.79 ± 0.24 pp, 3 seeds); IMES's CLAHE preprocessing does not help cpdino-vitb.
Full numbers: [`results/README.md`](results/README.md).

## Folder contents

| Path | Content |
|---|---|
| `README.md` | This file: the procedure, step by step |
| `scripts/make_split.py` | Creates the stratified 18 / 4 / 6 whole-image split |
| `scripts/imes_preprocess.py` | IMES per-channel CLAHE (same settings as IMES's notebook) |
| `scripts/split_io.py` | Shared helpers (split loading + fingerprint check, safe model loading, `save_mask`) |
| `scripts/predict_split.py` | Runs one Cellpose model on one subset and saves masks in the shared format |
| `scripts/finetune_cpdino.py` | Fine-tunes cpdino-vitb (or cpdino) on the train subset |
| `scripts/summarise_seeds.py` | Mean ± SD over seeds, pre-set decision rule, paired seed-by-seed comparison |
| `PROTOCOL_porcine_bf_v1.md` | How to evaluate **another** model the same way |
| `results/` | Report (`README.md`) and the score CSVs (val and test) |

Also used (repo root): `splits/porcine_bf_v1.json` (the split), `scripts/score_masks_v2.py` (scoring, unchanged),
`cellpose-env/` (uv environment).

## Setup

- Environment: `cellpose-env` (Python 3.12, cellpose 4.2.1.1, torch 2.14.1+cu132, dinov3). All commands run with
  `uv run --project "<repo>\cellpose-env" python …`.
- Hardware: one 8 GB laptop GPU (training peak 3.22 GB, ~10 min per 100-epoch run).
- Data: `Training_porcine_bf` (28 image/mask pairs; not in git).
- Placeholders in the commands below (PowerShell):

```powershell
$repo = "<path to the repo>"
$exp  = "$repo\experiments\cpdino_vitb_finetune_v1"
$cp   = "$repo\cellpose-env"
$data = "<path to the data>\Training_porcine_bf"
$out  = "<path to a results folder outside the repo>\cpdino_ft"
$split = "$repo\splits\porcine_bf_v1.json"
```

## Procedure (in the order it was run)

### Step 1 – Make the split

```powershell
python "$exp\scripts\make_split.py" --group-dir $data                       # prints only
python "$exp\scripts\make_split.py" --group-dir $data --write --out $split  # saves the JSON
```

Result: 18 train / 4 val / 6 test whole images, stratified (controls, PAD11 Diff, Positiv_21–26, other Positiv),
random within each stratum (seed 42), fingerprint `22d6fd1c60e2`. The script refuses to overwrite and stops if the
folder content changes.

### Step 2 – Check the preprocessing on one image

```powershell
uv run --project $cp python "$exp\scripts\imes_preprocess.py" `
    --image "$data\CLD-AIM-04 PIG 10x Training Positiv_7_image.png" --out "$out\preview\Positiv_7_clahe.png"
```

Result: 1080 × 1920 × 3 uint8 in and out; visual check of the CLAHE image. (The CLAHE function was also checked to be
bit-identical to the image step of IMES's notebook on synthetic test images.)

### Step 3 – Zero-shot references on val (raw and CLAHE)

```powershell
uv run --project $cp python "$exp\scripts\predict_split.py" --split-file $split --subset val `
    --group-dir $data --out-root $out --model cpdino-vitb --preprocess none --label cpdino-vitb-raw
uv run --project $cp python "$exp\scripts\predict_split.py" --split-file $split --subset val `
    --group-dir $data --out-root $out --model cpdino-vitb --preprocess imes_clahe --label cpdino-vitb-clahe
```

Check: the raw run reproduced the earlier zero-shot object counts exactly (Negativ_2 370, Positiv_16 304,
Positiv_19 279, Positiv_22 274).

### Step 4 – Score the zero-shot runs

```powershell
uv run --project $cp python "$repo\scripts\score_masks_v2.py" `
    --batch-dirs "<predict folder raw>" "<predict folder clahe>" --group-dir $data
```

Result: CLAHE alone gives no consistent improvement (details in `results/README.md` §3).

### Step 5 – Short training test (memory and speed)

```powershell
uv run --project $cp python "$exp\scripts\finetune_cpdino.py" --split-file $split `
    --group-dir $data --out-root $out --label test-2ep --n-epochs 2
```

Result: training works on 8 GB; peak 3.22 GB; ~6 s per epoch (18 crops). Not a usable model (learning rate ≈ 0 in
the first epochs).

### Step 6 – Baseline fine-tune (CLAHE, seed 0), predict and score val

```powershell
uv run --project $cp python "$exp\scripts\finetune_cpdino.py" --split-file $split `
    --group-dir $data --out-root $out --label cpdino-vitb-ft-clahe-v1
uv run --project $cp python "$exp\scripts\predict_split.py" --split-file $split --subset val `
    --group-dir $data --out-root $out `
    --model "<train folder>\models\cpdino-vitb-ft-clahe-v1" --preprocess imes_clahe --label cpdino-vitb-ft-clahe-v1
```

Training settings = Cellpose `train_seg` defaults (AdamW, lr 1e-5, weight decay 0.1, 100 epochs, batch 1,
384 × 384 crops) with `min_train_masks 0`; the final epoch is used. Result: large improvement on val
(|fat % error| 7.9 → 1.45 pp; control 370 → 4 false objects).

### Step 7 – Decision rule, fixed before the next results

> Train both preprocessings with 3 seeds. The setting with the lower seed-mean |fat % error| on the differentiated
> val images wins; a difference < 0.5 pp is a tie → CLAHE (IMES's preprocessing).

### Step 8 – Five more training runs (raw seeds 0–2, CLAHE seeds 1–2), ~50 min

```powershell
foreach ($r in "none:0", "none:1", "none:2", "imes_clahe:1", "imes_clahe:2") {
    $pre, $seed = $r.Split(":")
    $tag = if ($pre -eq "none") { "raw" } else { "clahe" }
    uv run --project $cp python "$exp\scripts\finetune_cpdino.py" --split-file $split `
        --group-dir $data --out-root $out --preprocess $pre --seed $seed --label "cpdino-vitb-ft-$tag-v1-s$seed"
}
```

The seed shifts Cellpose's per-epoch numpy seeds (`--seed 0` = unchanged Cellpose behaviour), so seed k of raw and
seed k of CLAHE draw the same crops and augmentations → paired comparison.

### Step 9 – Predict val with the five new models

```powershell
foreach ($label in "cpdino-vitb-ft-raw-v1-s0", "cpdino-vitb-ft-raw-v1-s1", "cpdino-vitb-ft-raw-v1-s2",
                   "cpdino-vitb-ft-clahe-v1-s1", "cpdino-vitb-ft-clahe-v1-s2") {
    $dir = Get-ChildItem "$out\train" -Directory -Filter "*_$label" | Select-Object -First 1
    $pre = if ($label -like "*-raw-*") { "none" } else { "imes_clahe" }
    uv run --project $cp python "$exp\scripts\predict_split.py" --split-file $split --subset val `
        --group-dir $data --out-root $out --model "$($dir.FullName)\models\$label" --preprocess $pre --label $label
}
```

### Step 10 – Score all 8 val runs and apply the rule

```powershell
$runs = (Get-ChildItem "$out\predict" -Directory -Filter "*_val_*").FullName
uv run --project $cp python "$repo\scripts\score_masks_v2.py" --batch-dirs $runs --group-dir $data
$latest = Get-ChildItem "$out\rescore" -Directory | Sort-Object Name | Select-Object -Last 1
uv run --project $cp python "$exp\scripts\summarise_seeds.py" --summary "$($latest.FullName)\summary.csv"
```

Result: raw 1.52 vs CLAHE 1.65 pp → difference 0.13 pp < 0.5 → tie → CLAHE used for the test run.

### Step 11 – Test set, used once (zero-shot reference + 3 CLAHE seeds)

```powershell
uv run --project $cp python "$exp\scripts\predict_split.py" --split-file $split --subset test --confirm-test `
    --group-dir $data --out-root $out --model cpdino-vitb --preprocess none --label cpdino-vitb-raw
foreach ($label in "cpdino-vitb-ft-clahe-v1", "cpdino-vitb-ft-clahe-v1-s1", "cpdino-vitb-ft-clahe-v1-s2") {
    $dir = Get-ChildItem "$out\train" -Directory -Filter "*_$label" | Select-Object -First 1
    uv run --project $cp python "$exp\scripts\predict_split.py" --split-file $split --subset test --confirm-test `
        --group-dir $data --out-root $out --model "$($dir.FullName)\models\$label" --preprocess imes_clahe --label $label
}
$runs = (Get-ChildItem "$out\predict" -Directory -Filter "*_test_*").FullName
uv run --project $cp python "$repo\scripts\score_masks_v2.py" --batch-dirs $runs --group-dir $data
$latest = Get-ChildItem "$out\rescore" -Directory | Sort-Object Name | Select-Object -Last 1
uv run --project $cp python "$exp\scripts\summarise_seeds.py" --summary "$($latest.FullName)\summary.csv"
```

Result (5 differentiated test images): |fat % error| 5.80 → **2.79 ± 0.24 pp**, Dice 0.68 → **0.84**,
F1@0.7 0.38 → **0.60**.

### Step 12 – Paired analysis (afterwards, on val)

`summarise_seeds.py` compares raw and CLAHE seed by seed: raw is better in 3 of 3 pairs on the pre-specified
metric (−0.13 pp, paired p = 0.009) and on most other metrics. Conclusion: **CLAHE gives no benefit for
cpdino-vitb; raw input is recommended as the default from now on** (decided on val; test numbers stay as run).

## Safety features of the scripts

- The split JSON is checked by its fingerprint; hand-edited splits are refused.
- `--subset test` requires `--confirm-test`; `finetune_cpdino.py` never reads the test images.
- No silent fallbacks: an unknown model name or missing weights file stops the run (Cellpose itself would load
  `cpsam_v2` instead), and every weight is checked after loading.
- Nothing is overwritten: every run gets a new time-stamped folder.
- Every run saves its settings, image lists, software versions and the SHA-256 of the weights.
