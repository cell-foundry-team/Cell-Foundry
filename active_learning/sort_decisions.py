"""Sort the thumbnails into approve/ and reject/ folders from the expert's decisions.csv, and report the approve rate.

Use this when the expert returns decisions.csv (the "Download decisions (CSV)" button) instead of saving folders themselves.

    python sort_decisions.py --decisions decisions.csv --cycle-dir . --out sorted_cycle0

Output keeps the original full-quality PNG thumbnails, named  <candidate_id>__<source image>.png
"""
import argparse, csv, re, shutil
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--decisions", required=True)
ap.add_argument("--cycle-dir", default=".")
ap.add_argument("--thumb-dir", default="thumbnails_for_expert")
ap.add_argument("--out", required=True)
a = ap.parse_args()

safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s)
cyc, out = Path(a.cycle_dir), Path(a.out)
review = {r["candidate_id"]: r for r in csv.DictReader(open(cyc / "review.csv", encoding="utf-8"))}
(out / "approve").mkdir(parents=True, exist_ok=True)
(out / "reject").mkdir(parents=True, exist_ok=True)

counts = Counter()
rows_out = []
for r in csv.DictReader(open(a.decisions, encoding="utf-8")):
    cid, d = r["candidate_id"], r["decision"].strip().lower()
    if d not in ("approve", "reject") or cid not in review:
        counts["skipped"] += 1
        continue
    src = cyc / a.thumb_dir / review[cid]["thumbnail"]
    shutil.copy(src, out / d / f"{cid}__{safe(review[cid]['image'])}.png")
    counts[d] += 1
    rows_out.append({**review[cid], "decision": d})

with open(out / "decisions_merged.csv", "w", newline="", encoding="utf-8") as f:
    names = list(review[next(iter(review))].keys())
    w = csv.DictWriter(f, fieldnames=names if "decision" in names else names + ["decision"])
    w.writeheader(); w.writerows(rows_out)

n = counts["approve"] + counts["reject"]
print(f"approved {counts['approve']}, rejected {counts['reject']}, skipped/undecided {counts['skipped']}")
if n:
    print(f"Impact Precision (approved / decided) = {100 * counts['approve'] / n:.1f}%")
