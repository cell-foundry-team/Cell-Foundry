# Evaluation protocol: porcine brightfield, split `porcine_bf_v1`

How to train and evaluate **any** model (Cellpose models, DINOCell, nnU-Net, µSAM, SAM3, …) so that its numbers
are directly comparable with the cpdino-vitb results in `results/` of this folder.
The tools referred to below are in `scripts/` of this folder (`<exp>` = `<repo>\experiments\cpdino_vitb_finetune_v1`).

## Rules

1. **Same split for everyone:** `splits/porcine_bf_v1.json` (18 train / 4 val / 6 test whole images,
   fingerprint `22d6fd1c60e2`). Never edit the JSON by hand; `split_io.load_split()` re-checks the fingerprint.
2. **Train only on `train`.** `val` may be used to watch training and to make choices. `test` is never used for choices.
3. **Fix the decision rule before looking at the results** (e.g. "lower seed-mean |fat % error| on the
   differentiated val images wins").
4. **3 seeds per setting** (`-s0`, `-s1`, `-s2` at the end of the label). Report mean ± SD over seeds.
   If two settings share seeds (same random choices), compare them **paired** (`summarise_seeds.py`).
5. **Test once:** after all choices are made on val, predict the 6 test images once per final model and report the
   numbers as they are. Test numbers of several models may be reported side by side, but never used to pick one.
6. **Score with `scripts/score_masks_v2.py`**, unchanged, on whole images at the original size.
7. Do not commit images, masks, weights or virtual environments. Small CSVs (`summary.csv`, `per_image.csv`,
   `seeds_summary.csv`) and a results `README.md` go into `Results/<experiment>/`.

## Shared mask format (what `score_masks_v2.py` reads)

```
<run folder>/masks/<label>/<image name>_<label>_instances.tif
```

- `<image name>` = the image file name without `_image.<ext>`, e.g. `CLD-AIM-04 PIG 10x Training Positiv_16`.
- `<label>` = model + setting + seed, letters/digits/`.`/`_`/`-` only, e.g. `dinocell-ft-imes-v1-s0`.
- 2D `uint16` label image, 0 = background, 1..N = objects, **same height × width as the original image**.
- One run folder per model × subset. `score_masks_v2.py --batch-dirs` takes one or more run folders.

## Cellpose models (cpdino, cpdino-vitb, cpsam, cpsam_v2)

Use the ready-made scripts (environment: `cellpose-env`):

```powershell
# fine-tune on train (one seed); --base-model cpdino-vitb | cpdino; --preprocess none | imes_clahe
uv run --project "<repo>\cellpose-env" python "<exp>\scripts\finetune_cpdino.py" `
    --split-file "<repo>\splits\porcine_bf_v1.json" --group-dir "<data>\Training_porcine_bf" `
    --out-root "<results>\cpdino_ft" --base-model cpdino --preprocess none --seed 0 --label cpdino-ft-raw-v1-s0

# predict val (built-in name for zero-shot, or the fine-tuned weights file)
uv run --project "<repo>\cellpose-env" python "<exp>\scripts\predict_split.py" `
    --split-file "<repo>\splits\porcine_bf_v1.json" --subset val --group-dir "<data>\Training_porcine_bf" `
    --out-root "<results>\cpdino_ft" --model "<train run>\models\cpdino-ft-raw-v1-s0" `
    --preprocess none --label cpdino-ft-raw-v1-s0
# test: same command with --subset test --confirm-test (once, at the end)
```

Notes: `finetune_cpdino.py` supports the DINO backbones (cpdino-vitb, cpdino). cpdino (ViT-L) has not been tested
on an 8 GB GPU; run a short memory test first (`--n-epochs 2 --label test-2ep`). Use the **same preprocessing for
training and prediction**.

## Any other model (DINOCell, nnU-Net, µSAM, SAM3, …)

Use your model's own environment and training code, but read the split and write masks with the shared helpers
(`split_io.py` needs only numpy, scikit-image and tifffile):

```python
import sys; sys.path.insert(0, r"<repo>\experiments\cpdino_vitb_finetune_v1\scripts")
from pathlib import Path
from split_io import load_split, find_image, save_mask

split = load_split(Path(r"<repo>\splits\porcine_bf_v1.json"))
train_names = split["splits"]["train"]      # train ONLY on these
val_names   = split["splits"]["val"]        # choices are made here
# test_names = split["splits"]["test"]      # once, at the end

run_dir = Path(r"<results>\my_model\predict\<timestamp>_val_my-model-v1-s0")
for name in val_names:
    image = read_image(find_image(Path(r"<data>\Training_porcine_bf"), name))   # your reader
    mask  = my_model_predict(image)                                            # 2D labels, original size
    save_mask(run_dir, "my-model-v1-s0", name, mask, image.shape)              # shared format, never overwrites
```

Then score and summarise (in `cellpose-env`, which has the scorer's dependencies):

```powershell
uv run --project "<repo>\cellpose-env" python "<repo>\scripts\score_masks_v2.py" `
    --batch-dirs "<run folder 1>" "<run folder 2>" --group-dir "<data>\Training_porcine_bf"
uv run --project "<repo>\cellpose-env" python "<exp>\scripts\summarise_seeds.py" `
    --summary "<rescore folder>\summary.csv" --compare my-model-v1 my-model-v2
```

`summarise_seeds.py` groups labels ending in `-s<k>` into one setting (a label without `-s<k>` counts as seed 0)
and prints mean ± SD over seeds; with `--compare A B` it also prints the paired seed-by-seed comparison.

## Reporting

Report per image type (`differentiated`, `control`; do not mix them for F1, which is 0 on controls by definition):
|fat % error| (main), signed fat % error, fat-area Dice, pooled count error, F1@0.5, F1@0.7.
State the number of seeds, the decision rule and whether the test set was used.
