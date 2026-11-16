# DINOCell fine-tuning on porcine brightfield (v1)

LoRA fine-tuning of IMES's DINOCell base checkpoint on the shared split `splits/porcine_bf_v1.json`,
ported from IMES's Colab notebook, with the same preprocessing choices as the cpdino-vitb
experiment so numbers are comparable. Scoring uses `scripts/score_masks_v2.py`, unchanged.

**Status: written, not run end to end.** A 1-step run on a Mac got through data loading, flow
generation and model + LoRA construction (2,912,387 trainable of 100,528,771 parameters, the same
count as IMES's own run) and then ran out of memory in the first forward pass on Apple's GPU (MPS).
The training step, the validation loop and the export have never run. The prediction script's input
conversion was checked against the training one (identical) and its sliding window on a stand-in model.

## Folder contents

| Path | Content |
|---|---|
| `scripts/finetune_dinocell.py` | LoRA fine-tuning on the split's train images (port of IMES's notebook) |
| `scripts/predict_dinocell_split.py` | Segment one subset with a DINOCell checkpoint, save masks for `score_masks_v2.py` |

Reused from `experiments/cpdino_vitb_finetune_v1/scripts/`: `split_io.py`, `imes_preprocess.py`.

## Important: DINOCell's own pipeline preprocesses by itself

`dinocell/pipeline.py` (0.74) always converts the image to grey and applies CLAHE on each 512 x 512
**crop** at inference; it cannot be switched off. IMES's training applies CLAHE per colour channel on
the **whole image**. Consequences:

- `predict_dinocell_split.py` has two modes. Default: preprocess the whole image with `--preprocess`
  (same as in training) and run DINOCell's sliding window with no internal CLAHE. Use this for every
  model trained here, with the SAME `--preprocess` as in training.
- `--native-pipeline` runs DINOCell exactly as shipped. Use it only for the zero-shot base checkpoint
  (that is what `scripts/run_dinocell_batch.py` does).
- Never score a fine-tuned model through `run_dinocell_batch.py` or `pipeline.run`: its input would
  differ from what it was trained on.

## Environment

The repo's MAIN environment (`dinocell==0.74`, root `pyproject.toml`), not `cellpose-env`.
Scoring runs in `cellpose-env`. Training needs a CUDA GPU (a Mac falls back to MPS for a code
check only: fp32, no AMP, and it ran out of memory).

## Steps (bash; placeholders in <>)

```bash
repo=<repo>; exp=$repo/experiments/dinocell_finetune_v1
data=<data>/Training_porcine_bf
out=<results folder OUTSIDE the repo>/dinocell_ft
base=<path>/DINOCell_demo_model.pt          # the 401 MB base checkpoint (not in git)

# 1. 1-step check of the real training step (CUDA): does it run, how much memory?
uv run python $exp/scripts/finetune_dinocell.py --split-file $repo/splits/porcine_bf_v1.json \
    --group-dir $data --out-root $out --weights $base --label test-1step --steps 1 --validate-every 0

# 2. real run (choose --preprocess BEFORE looking at results; imes_clahe = IMES's own recipe)
uv run python $exp/scripts/finetune_dinocell.py --split-file $repo/splits/porcine_bf_v1.json \
    --group-dir $data --out-root $out --weights $base --preprocess imes_clahe --seed 0 \
    --label dinocell-ft-clahe-v1-s0

# 3. predict VAL with the fine-tuned model (same --preprocess as in step 2)
uv run python $exp/scripts/predict_dinocell_split.py --split-file $repo/splits/porcine_bf_v1.json \
    --subset val --group-dir $data --out-root $out \
    --weights $out/train/<timestamp>_dinocell-ft-clahe-v1-s0/models/dinocell-ft-clahe-v1-s0 \
    --preprocess imes_clahe --label dinocell-ft-clahe-v1-s0

# 3b. zero-shot base checkpoint on the same val images, for comparison
uv run python $exp/scripts/predict_dinocell_split.py --split-file $repo/splits/porcine_bf_v1.json \
    --subset val --group-dir $data --out-root $out --weights $base --preprocess none \
    --native-pipeline --label dinocell-zeroshot

# 4. score (cellpose-env)
uv run --project $repo/cellpose-env python $repo/scripts/score_masks_v2.py \
    --batch-dirs $out/predict/<timestamp>_val_dinocell-ft-clahe-v1-s0 \
                 $out/predict/<timestamp>_val_dinocell-zeroshot --group-dir $data
```
TEST: same as step 3 with `--subset test --confirm-test`, once, only after all choices are fixed on val.
Protocol rules (3 seeds per setting, decision rule fixed in advance, test once): see
`experiments/cpdino_vitb_finetune_v1/PROTOCOL_porcine_bf_v1.md`.

## Known differences from IMES's notebook

- Split: `porcine_bf_v1` (18 train / 4 val), not the notebook's own 22 / 6.
- Validation during training is still the notebook's 12 fixed patches (about 25 % of each val image,
  mostly near cells). It only picks the checkpoint (best validation Dice, as in the notebook).
  Real model quality comes from steps 3 and 4 on whole images.
- Preprocessing is a flag; the notebook hard-codes per-channel CLAHE.
- Only the exported merged weights matter for prediction; no resume checkpoints are written.
