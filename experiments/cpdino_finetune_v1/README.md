# Run on the Cairo GPU: fine-tune cpdino (ViT-L) with Priyam's cpdino-vitb procedure

Goal: repeat exactly what Priyam did for **cpdino-vitb** (`experiments/cpdino_vitb_finetune_v1/README.md`), but with
**cpdino** (the larger ViT-L DINOv3 backbone). Same split, same scripts, same settings; only `--base-model cpdino`
differs. Everything is driven by one script, `run_cpdino_pipeline.sh`, in this folder.

| Path | Content |
|---|---|
| `run_cpdino_pipeline.sh` | One command per stage: check, memtest, train, predict-val, score-val, test |
| `scripts/finetune_cpdino.py` | Priyam's fine-tuning script; default `--base-model cpdino` (ViT-L) |
| `scripts/predict_split.py` | Predicts one subset with a built-in or fine-tuned model, shared mask format |
| `scripts/split_io.py` | Split loading + fingerprint check, safe model loading, `save_mask` |
| `scripts/imes_preprocess.py` | IMES per-channel CLAHE (`--preprocess imes_clahe`) |
| `scripts/summarise_seeds.py` | Mean ± SD over seeds, pre-set decision rule (defaults name the cpdino settings) |

`scripts/` are copies of `experiments/cpdino_vitb_finetune_v1/scripts/` with the same logic; only defaults and
examples changed from cpdino-vitb to cpdino. The split (`splits/porcine_bf_v1.json`) and the scorer
(`scripts/score_masks_v2.py` at the repo root) are the shared ones.

## Rules (from `PROTOCOL_porcine_bf_v1.md`; they keep the numbers comparable with cpdino-vitb)

1. Train only on the 18 train images; never edit `splits/porcine_bf_v1.json` (fingerprint `22d6fd1c60e2`).
2. Use the 4 val images for every choice (here: raw vs CLAHE preprocessing).
3. Decision rule, fixed **before** looking at results: lower seed-mean |fat % error| on the differentiated val images
   wins; a difference < 0.5 pp is a tie, and a tie goes to CLAHE (as for cpdino-vitb).
4. 3 seeds per setting (0, 1, 2); report mean ± SD.
5. Touch the 6 test images exactly once, at the very end.
6. Do not commit images, masks or weights. Small CSVs and a results README are fine.

## 0. Get the data to the server

The image data is **not** in git. Your copy is
`/Users/prethebhaneelaveni/Documents/porcine_data/Training_porcine_bf` (56 files = 28 image/mask pairs, ~114 MB):

```bash
scp -r "$HOME/Documents/porcine_data/Training_porcine_bf" <user>@<cairo-host>:~/
```

The folder must keep the name `Training_porcine_bf`; the scripts refuse any other name.

## 1. Set up on Cairo (once)

Needs Linux with an NVIDIA GPU and driver, `git`, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cell-foundry-team/Cell-Foundry.git     # or: cd Cell-Foundry && git fetch
cd Cell-Foundry
git checkout cpdino-prethebha                 # this branch has the scripts; or use main once it is merged
(cd cellpose-env && uv sync)                  # cellpose 4.2.1.1, torch 2.14.1, dinov3 (downloads a few GB)

export DATA=~/Training_porcine_bf             # the image folder you copied
export OUT=~/results/cpdino_ft                # outside the repo, plenty of disk
S=./experiments/cpdino_finetune_v1/run_cpdino_pipeline.sh
chmod +x $S
```

Run long steps inside `tmux` (or `nohup ... &`) so an SSH disconnect does not kill them; repeat the `export` lines
inside the new tmux session. Re-running a stage is safe: finished trainings and predictions are skipped, and nothing
is ever overwritten.

If Cairo is a **Windows** machine, use Priyam's PowerShell commands in
`experiments/cpdino_vitb_finetune_v1/README.md` (steps 5-11), but call the scripts in this folder's `scripts/`
(default `--base-model cpdino`) and use `--model cpdino` for zero-shot predictions; the bash script will not run there.

## 2. Run, in this order

```bash
$S check          # CUDA present? split fingerprint 22d6fd1c60e2? all 28 images found?
$S memtest        # 2-epoch run, NOT a real model; cpdino ViT-L is untested on 8 GB GPUs
```

After `memtest`, open the newest `$OUT/train/*_cpdino-test-2ep/train_config.json` and note `peak_gpu_gb` and
`seconds_per_epoch`. cpdino-vitb used 3.22 GB and ~6 s/epoch (about 10 min per 100-epoch run); ViT-L will be larger.
If it runs out of memory, stop and ask the team before changing `--bsize` or `--batch-size`, because that changes
the comparison with cpdino-vitb.

```bash
$S train          # raw + CLAHE x seeds 0,1,2 = 6 runs (100 epochs each)
$S predict-val    # zero-shot cpdino (reference) + the 6 fine-tuned models on the 4 val images
$S score-val      # scores val, prints mean ± SD per setting and applies the pre-set decision rule
```

`score-val` ends with a line like `PRE-SET RULE: ... -> CHOSEN: cpdino-ft-clahe-v1` (or `-raw-v1`).
Write the choice down; it decides `FINAL` below.

```bash
# Only now, once, with the setting chosen on val (none = raw, imes_clahe = CLAHE):
FINAL=imes_clahe CONFIRM_TEST=yes $S test
```

`test` predicts the 6 test images with the zero-shot cpdino and the 3 seeds of the chosen setting, scores them, and
prints the seed table. Do not run it twice or use it to pick a model.

## 3. Where things end up (under `$OUT`)

| Folder | Content |
|---|---|
| `train/<time>_<label>/` | `models/<label>` weights, `losses.csv`, `train_config.json` (settings, peak GPU memory, SHA-256), `train.log` |
| `predict/<time>_<subset>_<label>/` | predicted masks in the shared format |
| `rescore/<time>_Training_porcine_bf/` | `summary.csv`, `per_image.csv`, `seeds_summary.csv` |

Labels: `cpdino-ft-raw-v1-s0..s2`, `cpdino-ft-clahe-v1-s0..s2`, zero-shot reference `cpdino-raw`.

## 4. Bring back

Copy to your Mac or the repo: the `summary.csv`, `per_image.csv` and `seeds_summary.csv` from the val and the test
`rescore/` folders, plus the `train_config.json` files (for memory and time). Do not commit weights, masks or images.
Report per image type (differentiated, control): |fat % error|, signed fat % error, fat-area Dice, pooled count error,
F1@0.5, F1@0.7; state 3 seeds, the decision rule, and that the test set was used once.

Reference to compare with (cpdino-vitb, test, 5 differentiated images): |fat % error| 5.80 → 2.79 ± 0.24 pp,
Dice 0.68 → 0.84, F1@0.7 0.38 → 0.60. Do not assume cpdino gives the same result.

## 5. The same steps without the script (what it runs for each step)

```bash
CP=cellpose-env; EXP=experiments/cpdino_finetune_v1; SPLIT=splits/porcine_bf_v1.json
# train one model (preprocess none|imes_clahe, seed 0|1|2)
uv run --project $CP python $EXP/scripts/finetune_cpdino.py --split-file $SPLIT --group-dir $DATA --out-root $OUT \
    --base-model cpdino --preprocess none --seed 0 --label cpdino-ft-raw-v1-s0
# predict val (or: --subset test --confirm-test, once)
uv run --project $CP python $EXP/scripts/predict_split.py --split-file $SPLIT --subset val --group-dir $DATA \
    --out-root $OUT --model "$OUT/train/<time>_cpdino-ft-raw-v1-s0/models/cpdino-ft-raw-v1-s0" \
    --preprocess none --label cpdino-ft-raw-v1-s0
# zero-shot reference: --model cpdino --preprocess none --label cpdino-raw
# score and summarise
uv run --project $CP python scripts/score_masks_v2.py --batch-dirs <predict folders...> --group-dir $DATA --out-root $OUT
uv run --project $CP python $EXP/scripts/summarise_seeds.py --summary <rescore folder>/summary.csv \
    --compare cpdino-ft-raw-v1 cpdino-ft-clahe-v1 --tie-goes-to cpdino-ft-clahe-v1
```

## 6. If something goes wrong

- `No CUDA GPU found`: the driver or the torch build is wrong; run `$S check` and look at the `cuda` line.
- `Split was made for 'Training_porcine_bf'...`: the data folder was renamed.
- Fingerprint error: someone edited `splits/porcine_bf_v1.json`; restore it with `git checkout splits/porcine_bf_v1.json`.
- Out of GPU memory in `memtest`: see step 2; ask before changing settings.
- Run folder already exists: the scripts never overwrite; pick a new `--label` or delete the partial folder.
