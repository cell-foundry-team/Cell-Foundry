"""Make the fixed train / val / test split of the porcine brightfield images (18 / 4 / 6).

Images are split as WHOLE images (no crops), so no part of a test image can leak into
training. The split is stratified by image type, and within each type the images are
assigned at random with a fixed seed (not hand-picked: zero-shot results on all 28
images were already seen, so choosing images by hand could bias the test set).

    stratum       rule (on the file name)                 train  val  test
    control       "Negativ" or "_Ctrl"                       1     1     1
    pad11_diff    "PAD11" and "_Diff"                        1     0     1
    hard_21_26    "Positiv_21" ... "Positiv_26"              3     1     2
    positiv       every other "Positiv_<n>"                 13     2     2
                                                    total   18     4     6

"val" = the 4 check images used during development. "test" = used ONCE, at the end.
Positiv_21-26 are kept apart because every zero-shot model failed on them (large lipid
droplets segmented as cells), so each split gets some of them.

The script refuses to run if the folder does not contain exactly the expected number
of images per stratum (a changed folder must not silently give a different split).

By default it only PRINTS the split. Add --write to save it as JSON
(an existing file is never overwritten).

Usage (any Python 3.9+, no extra packages):
    python "<exp>\\scripts\\make_split.py" --group-dir "<data>\\Training_porcine_bf"
    python "<exp>\\scripts\\make_split.py" --group-dir "<data>\\Training_porcine_bf" --write --out "<repo>\\splits\\porcine_bf_v1.json"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from pathlib import Path

IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
SPLIT_NAME = "porcine_bf_v1"

# stratum -> (train, val, test); order matters: the first matching rule wins
PLAN = {
    "control": (1, 1, 1),
    "pad11_diff": (1, 0, 1),
    "hard_21_26": (3, 1, 2),
    "positiv": (13, 2, 2),
}


def stratum_of(base: str) -> str:
    """Assign an image (name without '_image') to one stratum. Raises if no rule fits."""
    low = base.lower()
    if "negativ" in low or re.search(r"(^|_)ctrl($|_)", low):
        return "control"
    if low.startswith("pad11") and re.search(r"(^|_)diff($|_)", low):
        return "pad11_diff"
    m = re.search(r"positiv_(\d+)$", low)
    if m:
        return "hard_21_26" if 21 <= int(m.group(1)) <= 26 else "positiv"
    raise SystemExit(f"No stratum rule matches '{base}'. Update PLAN / stratum_of first.")


def find_images(group_dir: Path) -> list[str]:
    """Names (without '_image' and extension) of all '<name>_image.<ext>' files, sorted."""
    names = sorted(p.stem[: -len("_image")] for p in group_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.endswith("_image"))
    if len(names) != len(set(names)):
        raise SystemExit("Two image files share one name (different extensions?). Fix the folder first.")
    return names


def make_split(names: list[str], seed: int) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """Return {split: [names]} and {stratum: {split: [names]}}."""
    by_stratum: dict[str, list[str]] = {s: [] for s in PLAN}
    for n in names:
        by_stratum[stratum_of(n)].append(n)

    for s, (n_tr, n_va, n_te) in PLAN.items():
        if len(by_stratum[s]) != n_tr + n_va + n_te:
            raise SystemExit(f"Stratum '{s}' has {len(by_stratum[s])} images, expected "
                             f"{n_tr + n_va + n_te}. The folder changed: check it before splitting.")

    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    detail: dict[str, dict[str, list[str]]] = {}
    for s, (n_tr, n_va, n_te) in PLAN.items():   # fixed order -> reproducible
        members = sorted(by_stratum[s])
        rng.shuffle(members)
        parts = {"train": members[:n_tr], "val": members[n_tr:n_tr + n_va],
                 "test": members[n_tr + n_va:]}
        detail[s] = {k: sorted(v) for k, v in parts.items()}
        for k, v in parts.items():
            split[k] += v
    return {k: sorted(v) for k, v in split.items()}, detail


def fingerprint(split: dict[str, list[str]]) -> str:
    """Short hash of the split, to check that two people use the same one."""
    text = json.dumps(split, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-dir", type=Path, required=True, help="folder with the *_image.* files")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--write", action="store_true", help="save the split as JSON (default: only print)")
    p.add_argument("--out", type=Path, default=None, help=f"JSON path (default: ./{SPLIT_NAME}.json)")
    a = p.parse_args(argv)

    if not a.group_dir.is_dir():
        raise SystemExit(f"Not a folder: {a.group_dir}")
    names = find_images(a.group_dir)
    split, detail = make_split(names, a.seed)

    print(f"split {SPLIT_NAME} | {len(names)} images | seed {a.seed} | fingerprint {fingerprint(split)}")
    for s, parts in detail.items():
        print(f"\n[{s}]")
        for k in ("train", "val", "test"):
            print(f"  {k:<5} ({len(parts[k])}): {', '.join(parts[k]) if parts[k] else '-'}")
    print("\ntotals: " + ", ".join(f"{k} {len(v)}" for k, v in split.items()))

    if not a.write:
        print("\n(not saved; add --write to save the JSON)")
        return 0
    out = a.out or Path.cwd() / f"{SPLIT_NAME}.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting.")
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": SPLIT_NAME, "seed": a.seed, "fingerprint": fingerprint(split),
        "group_dir_name": a.group_dir.name, "n_images": len(names),
        "plan": {s: dict(zip(("train", "val", "test"), c)) for s, c in PLAN.items()},
        "splits": split, "strata": detail,
        "note": "Whole-image split. val = development check set; test = use once at the end.",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
    }
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nsaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
