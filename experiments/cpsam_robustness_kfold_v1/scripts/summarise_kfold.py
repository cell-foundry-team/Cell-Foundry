"""Pool k-fold results: every image was tested once per seed, so the 4 folds together give a
score for ALL 28 images instead of 6.

Input: per_image.csv from score_masks_v2.py run over the fold prediction folders. Model labels
must look like "<anything>-kf<fold>-<anything>-s<seed>" (as made by run_kfold_cpsam.sh).

Output (printed + kfold_summary.csv beside the input, never overwritten):
  * per seed: mean and sd over the pooled images of fat % difference (signed), |fat % difference|
    and foreground Dice; differentiated images and control images are kept apart;
  * mean +/- sd over seeds of those per-seed means (if more than one seed);
  * per fold: the same numbers on that fold's images only (how much it depends on the held-out set).
The sd over images (spread between images) is NOT the sd over seeds (spread between training runs):
both are printed and named.

Usage (any Python 3.9+):
    python summarise_kfold.py --per-image <rescore folder>/per_image.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

LABEL = re.compile(r"kf(\d+).*-s(\d+)$")


def num(x: str) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def agg(rows: list[dict], key: str) -> tuple[float, float, int]:
    v = [num(r[key]) for r in rows]
    v = [x for x in v if x is not None]
    if not v:
        return float("nan"), float("nan"), 0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-image", type=Path, required=True)
    a = p.parse_args(argv)

    rows = []
    with open(a.per_image, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = LABEL.search(r["model"])
            if m:
                r["fold"], r["seed"] = int(m.group(1)), int(m.group(2))
                rows.append(r)
    if not rows:
        raise SystemExit("no rows with a label like ...-kf<fold>-...-s<seed> in " + str(a.per_image))

    out = []
    seeds = sorted({r["seed"] for r in rows})
    print(f"{'level':<22} {'cond':<15} {'n':>3} {'signed pp':>10} {'|diff| pp':>10} {'sd over imgs':>13} {'Dice':>7}")
    for cond in ("differentiated", "control"):
        per_seed = []
        for s in seeds:
            sub = [r for r in rows if r["seed"] == s and r["condition"] == cond]
            if not sub:
                continue
            sg, _, n = agg(sub, "fat_pct_error")
            ab, sd, _ = agg(sub, "fat_pct_abs_error")
            dc, _, _ = agg(sub, "fg_dice")
            per_seed.append((sg, ab, dc))
            out.append({"level": f"seed {s}, all folds", "condition": cond, "n_images": n,
                        "signed_mean": round(sg, 3), "abs_mean": round(ab, 3), "sd_over_images": round(sd, 3),
                        "dice_mean": round(dc, 4) if dc == dc else ""})
            print(f"{'seed ' + str(s) + ', all folds':<22} {cond:<15} {n:>3} {sg:>10.2f} {ab:>10.2f} {sd:>13.2f} {dc:>7.3f}")
        if len(per_seed) > 1:
            for i, nm in enumerate(("signed", "abs", "dice")):
                vals = [t[i] for t in per_seed]
                out.append({"level": f"mean over {len(per_seed)} seeds ({nm})", "condition": cond,
                            "signed_mean" if nm == "signed" else ("abs_mean" if nm == "abs" else "dice_mean"):
                            round(st.mean(vals), 4), "sd_over_seeds": round(st.stdev(vals), 4)})
            ab = [t[1] for t in per_seed]
            print(f"{'mean +/- sd over seeds':<22} {cond:<15} {'':>3} {'':>10} {st.mean(ab):>10.2f}  (sd over seeds {st.stdev(ab):.2f})")
    print()
    print(f"{'fold (all seeds)':<22} {'cond':<15} {'n':>3} {'signed pp':>10} {'|diff| pp':>10}")
    for fold in sorted({r["fold"] for r in rows}):
        for cond in ("differentiated", "control"):
            sub = [r for r in rows if r["fold"] == fold and r["condition"] == cond]
            if not sub:
                continue
            sg, _, n = agg(sub, "fat_pct_error")
            ab, _, _ = agg(sub, "fat_pct_abs_error")
            out.append({"level": f"fold {fold}", "condition": cond, "n_images": n,
                        "signed_mean": round(sg, 3), "abs_mean": round(ab, 3)})
            print(f"{'fold ' + str(fold):<22} {cond:<15} {n:>3} {sg:>10.2f} {ab:>10.2f}")

    dest = a.per_image.with_name("kfold_summary.csv")
    if dest.exists():
        raise SystemExit(f"\n{dest} exists; not overwriting (printed table above is the result)")
    fields = ["level", "condition", "n_images", "signed_mean", "abs_mean", "sd_over_images", "sd_over_seeds", "dice_mean"]
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"\n-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
