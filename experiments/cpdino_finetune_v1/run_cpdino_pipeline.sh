#!/usr/bin/env bash
# Fine-tune cpdino (ViT-L DINOv3) on split porcine_bf_v1, the same way cpdino-vitb was done
# (experiments/cpdino_vitb_finetune_v1/README.md, steps 3-11). Uses that folder's scripts unchanged.
#
# Needs a Linux/CUDA machine for the train/predict stages (finetune_cpdino.py stops without CUDA).
# `check` also runs on a Mac (no GPU needed).
#
#   export DATA=/path/to/Training_porcine_bf     # image + mask pairs, not in git
#   export OUT=/path/to/results/cpdino_ft        # outside the repo; nothing here is committed
#   ./run_cpdino_pipeline.sh check               # environment, split fingerprint, data folder
#   ./run_cpdino_pipeline.sh memtest             # 2-epoch memory/speed test (not a real model)
#   ./run_cpdino_pipeline.sh train               # raw + CLAHE x seeds 0,1,2 (6 runs); skips finished ones
#   ./run_cpdino_pipeline.sh predict-val         # zero-shot reference + the 6 fine-tuned models on val
#   ./run_cpdino_pipeline.sh score-val           # score val, seed table, pre-set decision rule
#   # --- decide, then touch the test set exactly once ---
#   FINAL=none|imes_clahe CONFIRM_TEST=yes ./run_cpdino_pipeline.sh test
#
# Decision rule (same as cpdino-vitb; fix it BEFORE looking at results): lower seed-mean |fat % error| on the
# differentiated val images wins; a difference < 0.5 pp is a tie and goes to $TIE_GOES_TO.
# Override with e.g. TIE_GOES_TO=cpdino-ft-raw-v1 if the team agrees on that before running.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$REPO/experiments/cpdino_vitb_finetune_v1"
CP="$REPO/cellpose-env"
SPLIT="$REPO/splits/porcine_bf_v1.json"

: "${DATA:?set DATA to the Training_porcine_bf folder}"
: "${OUT:?set OUT to a results folder outside the repo}"

BASE=cpdino
P=cpdino-ft                                  # label prefix, e.g. cpdino-ft-raw-v1-s0
SEEDS=(0 1 2)
SETTINGS=("none:raw" "imes_clahe:clahe")     # preprocess:tag
TIE_GOES_TO="${TIE_GOES_TO:-$P-clahe-v1}"    # as in the cpdino-vitb protocol

py() { uv run --project "$CP" python "$@"; }
latest() { ls -d "$1"/*/ 2>/dev/null | sort | tail -1; }
train_dir() { ls -d "$OUT"/train/*_"$1" 2>/dev/null | head -1 || true; }
has_predict() { compgen -G "$OUT/predict/*_$1_$2" >/dev/null; }

predict() {   # predict <subset> <model> <preprocess> <label> [extra args]
    local subset=$1 model=$2 pre=$3 label=$4; shift 4
    if has_predict "$subset" "$label"; then echo "skip: $subset $label already predicted"; return; fi
    py "$EXP/scripts/predict_split.py" --split-file "$SPLIT" --subset "$subset" \
        --group-dir "$DATA" --out-root "$OUT" --model "$model" --preprocess "$pre" --label "$label" "$@"
}

stage_check() {
    [ -d "$DATA" ] || { echo "DATA folder not found: $DATA"; exit 1; }
    [ -f "$SPLIT" ] || { echo "split not found: $SPLIT"; exit 1; }
    py -c "
import sys; sys.path.insert(0, r'$EXP/scripts')
from pathlib import Path
from split_io import load_split
s = load_split(Path(r'$SPLIT'))
print('split', s['name'], s['fingerprint'], {k: len(v) for k, v in s['splits'].items()})
import torch, cellpose
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| cellpose', cellpose.version if hasattr(cellpose, 'version') else '?')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0); print('GPU', p.name, round(p.total_memory / 1e9, 1), 'GB')
"
}

stage_memtest() {
    py "$EXP/scripts/finetune_cpdino.py" --split-file "$SPLIT" --group-dir "$DATA" --out-root "$OUT" \
        --base-model "$BASE" --preprocess none --n-epochs 2 --label cpdino-test-2ep
    echo "Check train_config.json in the new train folder: peak_gpu_gb and seconds_per_epoch."
}

stage_train() {
    for s in "${SETTINGS[@]}"; do
        pre=${s%%:*}; tag=${s##*:}
        for seed in "${SEEDS[@]}"; do
            label="$P-$tag-v1-s$seed"
            if [ -n "$(train_dir "$label")" ]; then echo "skip: $label already trained"; continue; fi
            py "$EXP/scripts/finetune_cpdino.py" --split-file "$SPLIT" --group-dir "$DATA" --out-root "$OUT" \
                --base-model "$BASE" --preprocess "$pre" --seed "$seed" --label "$label"
        done
    done
}

stage_predict_val() {
    predict val "$BASE" none cpdino-raw     # zero-shot reference
    for s in "${SETTINGS[@]}"; do
        pre=${s%%:*}; tag=${s##*:}
        for seed in "${SEEDS[@]}"; do
            label="$P-$tag-v1-s$seed"
            d="$(train_dir "$label")"
            [ -n "$d" ] || { echo "no trained model for $label (run: train)"; exit 1; }
            predict val "$d/models/$label" "$pre" "$label"
        done
    done
}

stage_score_val() {
    shopt -s nullglob
    runs=("$OUT"/predict/*_val_cpdino-raw "$OUT"/predict/*_val_"$P"-*)
    [ ${#runs[@]} -gt 0 ] || { echo "no val predictions (run: predict-val)"; exit 1; }
    py "$REPO/scripts/score_masks_v2.py" --batch-dirs "${runs[@]}" --group-dir "$DATA" --out-root "$OUT"
    d="$(latest "$OUT/rescore")"
    py "$EXP/scripts/summarise_seeds.py" --summary "${d}summary.csv" \
        --compare "$P-raw-v1" "$P-clahe-v1" --tie-goes-to "$TIE_GOES_TO"
}

stage_test() {
    [ "${CONFIRM_TEST:-}" = "yes" ] || { echo "Test set is used once. Set CONFIRM_TEST=yes to proceed."; exit 1; }
    case "${FINAL:-}" in
        none) tag=raw ;;
        imes_clahe) tag=clahe ;;
        *) echo "Set FINAL=none or FINAL=imes_clahe (the setting chosen on val)."; exit 1 ;;
    esac
    predict test "$BASE" none cpdino-raw --confirm-test      # zero-shot reference
    for seed in "${SEEDS[@]}"; do
        label="$P-$tag-v1-s$seed"
        d="$(train_dir "$label")"
        [ -n "$d" ] || { echo "no trained model for $label"; exit 1; }
        predict test "$d/models/$label" "$FINAL" "$label" --confirm-test
    done
    shopt -s nullglob
    runs=("$OUT"/predict/*_test_*)
    py "$REPO/scripts/score_masks_v2.py" --batch-dirs "${runs[@]}" --group-dir "$DATA" --out-root "$OUT"
    d="$(latest "$OUT/rescore")"
    py "$EXP/scripts/summarise_seeds.py" --summary "${d}summary.csv"   # no val rule here: settings are already chosen
}

case "${1:-}" in
    check) stage_check ;;
    memtest) stage_memtest ;;
    train) stage_train ;;
    predict-val) stage_predict_val ;;
    score-val) stage_score_val ;;
    test) stage_test ;;
    *) sed -n '2,22p' "$0"; exit 1 ;;
esac
