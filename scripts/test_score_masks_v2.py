"""Synthetic test for score_masks_v2.py (hand-computed answers; no real data).

Run from cellpose-env\\ with the script in the same folder as this test:
    uv run python <path>\\test_score_masks_v2.py
Creates fake 100x100 masks in a temporary folder that is deleted afterwards.
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from score_masks_v2 import score  # noqa: E402

H = W = 100


def sq(lbl, k, r0, r1, c0, c1):
    lbl[r0:r1, c0:c1] = k


# Image A: GT 2 cells (400 px each); prediction: cell 1 exact, cell 2 shifted 5 px (IoU 0.6), 1 extra 10x10
gtA = np.zeros((H, W), np.uint16); sq(gtA, 1, 10, 30, 10, 30); sq(gtA, 2, 50, 70, 50, 70)
prA = np.zeros((H, W), np.uint16); sq(prA, 1, 10, 30, 10, 30); sq(prA, 2, 50, 70, 55, 75); sq(prA, 3, 80, 90, 80, 90)
# Image B: control, GT empty, one false 10x10 detection
gtB = np.zeros((H, W), np.uint16)
prB = np.zeros((H, W), np.uint16); sq(prB, 1, 0, 10, 0, 10)
# Image C: control, both empty
gtC = np.zeros((H, W), np.uint16); prC = np.zeros((H, W), np.uint16)

# --- 1. score() on image A, against hand-computed values
r = score(gtA.astype(np.int32), prA.astype(np.int32))
checks = {
    "fat_pct_true = 800/10000 = 8 %": (r["fat_pct_true"], 8.0),
    "fat_pct_pred = 900/10000 = 9 %": (r["fat_pct_pred"], 9.0),
    "fat_pct_error = +1 pp": (r["fat_pct_error"], 1.0),
    "fg_dice = 2*700/(800+900) = 0.8235": (r["fg_dice"], 1400 / 1700),
    "F1@0.5 = 2*2/(2*2+1+0) = 0.8": (r["per_t"][0.5]["F1"], 0.8),
    "F1@0.7 = 2*1/(2*1+2+1) = 0.4": (r["per_t"][0.7]["F1"], 0.4),
    "F1@0.75 = 0.4 (same as 0.7)": (r["per_t"][0.75]["F1"], 0.4),
    "count_error = (3-2)/2 = +0.5": (r["count_error"], 0.5),
    "SA@0.5 = 2/(2+1+0) = 0.667": (r["per_t"][0.5]["SA"], 2 / 3),
    "SA@0.7 = 1/(1+2+1) = 0.25": (r["per_t"][0.7]["SA"], 0.25),
}
ok = True
for name, (got, want) in checks.items():
    # BUGFIX 1: Cellpose stores SA as float32 (metrics.py: ap = np.zeros(..., np.float32)), so 2/3
    # arrives as 0.66666669 (error 2e-8). A 1e-9 tolerance wrongly failed it; 1e-6 is float32-safe.
    good = abs(got - want) < 1e-6
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {name}: got {got:.4f}")
rB = score(gtB.astype(np.int32), prB.astype(np.int32))
rC = score(gtC.astype(np.int32), prC.astype(np.int32))
for name, good in {
    "B negative image flagged": rB["negative_image"] is True,
    "B fat_pct_error = +1 pp": abs(rB["fat_pct_error"] - 1.0) < 1e-9,
    "B fg_dice = 0 (pred only)": rB["fg_dice"] == 0.0,
    "C fg_dice = NaN (both empty)": np.isnan(rC["fg_dice"]),
    "old keys kept (per_t 0.5/0.75/0.9)": all(t in r["per_t"] for t in (0.5, 0.75, 0.9)),
}.items():
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {name}")

# --- 2. rescore_batch.py end to end on a fake results tree (2 models, 2 runs)
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    group = tmp / "data" / "Training_porcine_bf"; group.mkdir(parents=True)
    for base, gt in (("PAD1_Positiv_1", gtA), ("PAD1_Negativ_1", gtB), ("PAD1_Negativ_2", gtC)):
        tifffile.imwrite(group / f"{base}_instances.tif", gt)
    runs = []
    for run_name, model in (("20261018-131049_Training_porcine_bf", "cpsam"),
                            ("20261018-133215_Training_porcine_bf", "cpdino-vitb")):
        d = tmp / "Cloud_results" / "batch" / run_name / "masks" / model; d.mkdir(parents=True)
        for base, pr in (("PAD1_Positiv_1", prA), ("PAD1_Negativ_1", prB), ("PAD1_Negativ_2", prC)):
            tifffile.imwrite(d / f"{base}_{model}_instances.tif", pr)
        runs.append(str(tmp / "Cloud_results" / "batch" / run_name))
    cmd = [sys.executable, str(REPO / "score_masks_v2.py"), "--batch-dirs", *runs, "--group-dir", str(group)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout[-1500:]); print(p.stderr[-1500:])
    out = next((tmp / "Cloud_results" / "rescore").iterdir())
    with (out / "summary.csv").open(newline="", encoding="utf-8") as f:
        summ = list(csv.DictReader(f))
    with (out / "per_image.csv").open(newline="", encoding="utf-8") as f:
        per = list(csv.DictReader(f))
    s_all = next(s for s in summ if s["model"] == "cpsam" and s["condition"] == "all")
    for name, good in {
        "per_image rows = 2 models x 3 images = 6": len(per) == 6,
        "summary rows = 2 models x (differentiated, control, all) = 6": len(summ) == 6,
        "all: fat error mean over 3 images = (1+1+0)/3 = 0.67": abs(float(s_all["fat_pct_error_mean"]) - 2 / 3) < 0.006,
        "all: Dice mean ignores NaN = (0.8235+0)/2 = 0.412": abs(float(s_all["fg_dice_mean"]) - 0.8235 / 2) < 0.001,
        "all: F1 only from image A (n_scored = 1)": s_all["n_scored"] == "1" and s_all["F1@0.5_mean"] == "0.8",
    }.items():
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name}")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
