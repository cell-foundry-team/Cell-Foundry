"""Average the score_masks_v2.py results over training seeds, and apply the pre-set rule
for choosing the preprocessing (decided BEFORE the raw results were seen).

Input: summary.csv from a score_masks_v2.py run (one row per label x condition).
Labels "<setting>-s<k>" are grouped into "<setting>"; a label without "-s<k>" counts as seed 0
(e.g. cpdino-vitb-ft-clahe-v1 = seed 0 of setting cpdino-vitb-ft-clahe-v1).

Output: printed table + seeds_summary.csv in the same folder (never overwritten).
    mean = average over seeds, sd = standard deviation over seeds (spread between training runs),
    min / max over seeds.

Pre-set rule (fixed on 3 Nov 2026, before the raw fine-tune results were known):
    compare the two settings on the DIFFERENTIATED check images by the seed-mean of the
    typical miss |fat % error|; if they differ by less than --margin (0.5 pp), choose
    --tie-goes-to (CLAHE = IMES's preprocessing); otherwise choose the lower one.
    (The 0.5 pp margin means "large enough to matter in practice"; it is not a significance test.)

PAIRED comparison (added 3 Nov 2026): seed k of setting A and seed k of setting B use the same
crops, augmentation and layer drops (same seed), so they are compared seed by seed:
difference per seed, how often each setting wins, and a paired t-test (p-value only if scipy is
installed; otherwise only t). With 3 seeds this measures stability across TRAINING RUNS on the
same images, not generalisation to other images.

Usage (any Python 3.9+; scipy optional for the p-value):
    uv run --project "<repo>\\cellpose-env" python "<exp>\\scripts\\summarise_seeds.py" `
        --summary "<results>\\rescore\\<run>\\summary.csv" [--compare <setting A> <setting B>]
    (--compare defaults to the cpdino-vitb raw vs CLAHE settings; pass other setting names for other models)
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

METRICS = {
    "differentiated": [("fat_pct_error_mean", "fat% err (pp)"), ("fat_pct_abs_error_mean", "|fat% err| (pp)"),
                       ("fg_dice_mean", "Dice"), ("pooled_count_error", "count err"),
                       ("F1@0.5_mean", "F1@0.5"), ("F1@0.7_mean", "F1@0.7")],
    "control": [("fat_pct_error_mean", "fat% err (pp)"), ("n_pred_total", "objects")],
}


# metrics for the paired comparison: (condition, column, label, "lower" or "higher" is better, use |value|)
PAIRED = [
    ("differentiated", "fat_pct_abs_error_mean", "|fat% err| (pp)", "lower", False),
    ("differentiated", "fg_dice_mean", "Dice", "higher", False),
    ("differentiated", "F1@0.5_mean", "F1@0.5", "higher", False),
    ("differentiated", "F1@0.7_mean", "F1@0.7", "higher", False),
    ("differentiated", "pooled_count_error", "|count err|", "lower", True),
    ("control", "fat_pct_error_mean", "control |fat% err|", "lower", True),
]


def paired_report(values: dict, A: str, B: str) -> None:
    """Seed-by-seed comparison of settings A and B on the seeds both have."""
    try:
        from scipy import stats as sps
    except Exception:
        sps = None
    print(f"\nPAIRED (same seed = same crops/augmentation): A = {A}, B = {B}")
    print(f"{'metric':<20} {'seeds':<7} {'A per seed':<22} {'B per seed':<22} "
          f"{'A-B mean ± sd':<18} {'better':<12} {'paired t':>9} {'p':>7}")
    for cond, key, name, better, use_abs in PAIRED:
        a, b = values.get((A, cond, key), {}), values.get((B, cond, key), {})
        seeds = sorted(set(a) & set(b))
        if len(seeds) < 2:
            continue
        av = [abs(a[s]) if use_abs else a[s] for s in seeds]
        bv = [abs(b[s]) if use_abs else b[s] for s in seeds]
        d = [x - y for x, y in zip(av, bv)]
        b_wins = sum((x > y) if better == "lower" else (x < y) for x, y in zip(av, bv))
        a_wins = sum((x < y) if better == "lower" else (x > y) for x, y in zip(av, bv))
        sd = st.stdev(d)
        t = st.mean(d) / (sd / len(d) ** 0.5) if sd > 0 else float("inf") if st.mean(d) else 0.0
        p = f"{sps.ttest_rel(av, bv).pvalue:.3f}" if (sps is not None and sd > 0) else "n/a"
        fmt3 = lambda v: " ".join(f"{x:.3f}" for x in v)  # noqa: E731
        win = f"B {b_wins}/{len(d)}" if b_wins >= a_wins else f"A {a_wins}/{len(d)}"
        print(f"{name:<20} {','.join(map(str, seeds)):<7} {fmt3(av):<22} {fmt3(bv):<22} "
              f"{st.mean(d):+.3f} ± {sd:.3f}{'':<3} {win:<12} {t:>9.2f} {p:>7}")
    print("('better' = which setting wins in how many seed pairs; lower |error| / higher Dice and F1 is better)")


def setting_and_seed(label: str) -> tuple[str, int]:
    m = re.fullmatch(r"(.+?)-s(\d+)", label)
    return (m.group(1), int(m.group(2))) if m else (label, 0)


def agg(values: list[float]) -> dict:
    return {"n_seeds": len(values), "mean": st.mean(values),
            "sd": st.stdev(values) if len(values) > 1 else float("nan"),
            "min": min(values), "max": max(values)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", type=Path, required=True, help="summary.csv from score_masks_v2.py")
    p.add_argument("--compare", nargs=2, default=["cpdino-vitb-ft-raw-v1", "cpdino-vitb-ft-clahe-v1"],
                   metavar=("A", "B"), help="two settings for the pre-set rule")
    p.add_argument("--tie-goes-to", default="cpdino-vitb-ft-clahe-v1")
    p.add_argument("--margin", type=float, default=0.5, help="pp; smaller differences count as a tie")
    a = p.parse_args(argv)

    with a.summary.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # values[(setting, condition, metric)] = {seed: value}
    values: dict[tuple[str, str, str], dict[int, float]] = {}
    for r in rows:
        if r["condition"] not in METRICS:
            continue
        setting, seed = setting_and_seed(r["model"])
        for key, _ in METRICS[r["condition"]]:
            if r.get(key, "") != "":
                cell = values.setdefault((setting, r["condition"], key), {})
                if seed in cell:
                    raise SystemExit(f"Seed {seed} of '{setting}' appears twice in {a.summary}.")
                cell[seed] = float(r[key])

    out_rows = []
    settings = list(dict.fromkeys(k[0] for k in values))
    for cond, metrics in METRICS.items():
        print(f"\n=== {cond} images: mean ± sd over seeds [min .. max]")
        print(f"{'setting':<26} {'seeds':<8}" + "".join(f"{name:>24}" for _, name in metrics))
        for s in settings:
            cells, seeds = [], set()
            for key, name in metrics:
                d = values.get((s, cond, key))
                if not d:
                    cells.append(f"{'-':>24}")
                    continue
                g = agg(list(d.values()))
                seeds |= set(d)
                sd = "" if g["sd"] != g["sd"] else f" ± {g['sd']:.3f}"
                cells.append(f"{g['mean']:>9.3f}{sd:<9}[{g['min']:.2f}..{g['max']:.2f}]".rjust(24))
                out_rows.append({"setting": s, "condition": cond, "metric": key,
                                 "seeds": " ".join(str(x) for x in sorted(d)),
                                 **{k: round(v, 4) for k, v in g.items()}})
            print(f"{s:<26} {','.join(str(x) for x in sorted(seeds)):<8}" + "".join(cells))

    out = a.summary.parent / "seeds_summary.csv"
    if out.exists():
        print(f"\n{out} already exists; not overwriting (table above is still valid).")
    else:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nsaved {out}")

    A, B = a.compare
    key = ("differentiated", "fat_pct_abs_error_mean")
    if all((s, *key) in values for s in (A, B)):
        ma, mb = (st.mean(values[(s, *key)].values()) for s in (A, B))
        if abs(ma - mb) < a.margin:
            choice, why = a.tie_goes_to, f"difference {abs(ma - mb):.2f} pp < {a.margin} pp -> tie"
        else:
            choice, why = (A, "lower typical miss") if ma < mb else (B, "lower typical miss")
        print(f"\nPRE-SET RULE: |fat% err| differentiated, seed mean: {A} {ma:.2f} pp vs {B} {mb:.2f} pp")
        print(f"              {why} -> CHOSEN: {choice}")
        paired_report(values, A, B)
    return 0


if __name__ == "__main__":
    sys.exit(main())
