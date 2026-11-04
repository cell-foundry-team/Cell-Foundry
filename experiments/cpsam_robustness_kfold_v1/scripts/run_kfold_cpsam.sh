#!/usr/bin/env bash
# Train and test cpsam on every k-fold fold, one run after another (GPU: one job at a time).
# For each fold F and seed S:
#   1. finetune_cpdino.py  (cpsam, raw images, 100 epochs, the fold's split file)
#   2. predict_split.py    on the fold's TEST images (the held-out fold)
# Afterwards score each fold's masks with score_masks_v2.py and pool with summarise_kfold.py
# (the last lines of this script print the exact commands).
#
# Settings are the ones locked on 4 Nov for cpsam: raw images, 100 epochs, defaults otherwise.
# Nothing is chosen with these results; this only measures how much the score depends on
# WHICH images are held out.
#
# Usage (from the repo root, inside tmux on CAIRO; asks nothing, so read it first):
#   export DATA=~/Training_porcine_bf        OUT=~/cellfoundry_results/cpsam_kfold
#   export FOLDS="0 1 2 3"                   SEEDS="0"        # first pass: 1 seed per fold
#   bash experiments/cpsam_robustness_kfold_v1/scripts/run_kfold_cpsam.sh
# Resume after a stop: set FOLDS / SEEDS to what is left (finished runs are skipped).
set -euo pipefail

: "${DATA:?set DATA to the Training_porcine_bf folder}"
: "${OUT:?set OUT to a results folder OUTSIDE the repo}"
FOLDS="${FOLDS:-0 1 2 3}"
SEEDS="${SEEDS:-0}"
EXP="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
SPLITS="$OUT/splits"
PY="uv run --no-sync --project $REPO/cellpose-env python"   # --no-sync: never restore the cu130 torch build

mkdir -p "$OUT"
if [ ! -d "$SPLITS" ]; then
    $PY "$EXP/scripts/make_kfold_splits.py" --group-dir "$DATA" --out-dir "$SPLITS" --k 4 --seed 42
fi

for F in $FOLDS; do
  for S in $SEEDS; do
    LABEL="cpsam-kf${F}-raw-s${S}"
    SPLIT="$SPLITS/porcine_bf_kfold4_fold${F}.json"
    if ls "$OUT"/predict/*_test_"$LABEL" >/dev/null 2>&1; then
        echo "== $LABEL: already predicted, skipping"; continue
    fi
    echo "== $(date +%H:%M:%S) train $LABEL"
    $PY "$REPO/experiments/cpdino_vitb_finetune_v1/scripts/finetune_cpdino.py" \
        --split-file "$SPLIT" --group-dir "$DATA" --out-root "$OUT" --label "$LABEL" \
        --base-model cpsam --preprocess none --seed "$S"
    WEIGHTS="$(ls -d "$OUT"/train/*_"$LABEL" | tail -1)/models/$LABEL"
    echo "== $(date +%H:%M:%S) predict held-out fold $F with $WEIGHTS"
    $PY "$REPO/experiments/cpdino_vitb_finetune_v1/scripts/predict_split.py" \
        --split-file "$SPLIT" --subset test --confirm-test --group-dir "$DATA" --out-root "$OUT" \
        --model "$WEIGHTS" --preprocess none --label "$LABEL"
  done
done

echo
echo "== done $(date +%H:%M:%S). Score (one scorer run over all prediction folders), then pool:"
echo "   $PY $REPO/scripts/score_masks_v2.py --batch-dirs $OUT/predict/*_test_cpsam-kf* --group-dir $DATA"
echo "   $PY $EXP/scripts/summarise_kfold.py --per-image <the rescore folder>/per_image.csv"
