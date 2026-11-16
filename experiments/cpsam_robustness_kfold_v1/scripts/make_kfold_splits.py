"""Make k-fold split files (one JSON per fold) over the 28 porcine brightfield images.

Every fold file has the SAME format as splits/porcine_bf_v1.json (train / val / test, whole
images, fingerprint), so finetune_cpdino.py, predict_split.py and score_masks_v2.py read it
unchanged: just pass --split-file <fold file>.

Design (fixed here, before any result):
  * test  = fold k: every image is the test image of exactly ONE fold (7 images per fold for k=4);
  * val   = 3 images drawn at random (seed) from the other folds: only for watching the loss,
            nothing is selected with them (the fine-tune script saves the final epoch);
  * train = the remaining 18 images (same size as the hold-out split porcine_bf_v1).
  * Folds are stratified with the same strata as make_split.py (control, pad11_diff, hard_21_26,
    positiv): each stratum is shuffled with the seed and dealt round-robin to the folds.
    With 3 control images and 4 folds, one fold has NO control image (same limit as before).

Honest limits:
  * The 28 images include the 6 test images of porcine_bf_v1, which were already used once.
    A k-fold run is a NEW experiment, not a replacement for that test: report it separately,
    and do not use it to choose settings (settings are already fixed).
  * Folds tell how much the score depends on WHICH images are held out; they do not add images.

Usage (any Python 3.9+, no extra packages; does not read the images, only their names):
    python make_kfold_splits.py --group-dir <data>/Training_porcine_bf --out-dir <folder> [--k 4] [--seed 42]
An existing file is never overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpdino_vitb_finetune_v1" / "scripts"))
from make_split import IMAGE_EXTS, stratum_of  # noqa: E402

N_VAL = 3


def fingerprint(splits: dict) -> str:
    """Same definition as make_split.py / split_io.py."""
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def image_names(group_dir: Path) -> list[str]:
    names = sorted({p.name.rsplit("_image", 1)[0] for p in group_dir.iterdir()
                    if "_image" in p.name and p.suffix.lower() in IMAGE_EXTS})
    if not names:
        raise SystemExit(f"no *_image.* files in {group_dir}")
    return names


def make_folds(names: list[str], k: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    strata: dict[str, list[str]] = {}
    for n in names:
        strata.setdefault(stratum_of(n), []).append(n)
    folds: list[list[str]] = [[] for _ in range(k)]
    slot = 0
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        for n in members:
            folds[slot % k].append(n)
            slot += 1   # keeps dealing where the last stratum stopped, so fold sizes stay even
    return [sorted(f) for f in folds]


def fold_split(folds: list[list[str]], idx: int, seed: int) -> dict:
    test = folds[idx]
    rest = sorted(n for j, f in enumerate(folds) if j != idx for n in f)
    rng = random.Random(f"{seed}-val-{idx}")
    val = sorted(rng.sample(rest, N_VAL))
    train = sorted(n for n in rest if n not in val)
    return {"train": train, "val": val, "test": test}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    names = image_names(a.group_dir)
    if len(names) % a.k:
        print(f"note: {len(names)} images do not divide evenly into {a.k} folds")
    folds = make_folds(names, a.k, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    tested = [n for f in folds for n in f]
    assert sorted(tested) == names and len(set(tested)) == len(names), "every image must be tested exactly once"

    for idx in range(a.k):
        splits = fold_split(folds, idx, a.seed)
        assert not (set(splits["train"]) & set(splits["val"]) & set(splits["test"]))
        assert sum(len(v) for v in splits.values()) == len(names)
        name = f"porcine_bf_kfold{a.k}_fold{idx}"
        rec = {"name": name, "seed": a.seed, "fingerprint": fingerprint(splits),
               "group_dir_name": a.group_dir.name, "n_images": len(names),
               "splits": splits,
               "note": f"Whole-image k-fold, fold {idx} of {a.k}: test = this fold only, val = {N_VAL} images "
                       "(loss watching only), train = rest. A new experiment: not the porcine_bf_v1 test.",
               "created": dt.datetime.now().isoformat(timespec="seconds"), "script": "make_kfold_splits.py"}
        out = a.out_dir / f"{name}.json"
        if out.exists():
            raise SystemExit(f"{out} exists; not overwriting")
        out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        ctrl = sum(stratum_of(n) == "control" for n in splits["test"])
        print(f"{name}: train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])} "
              f"(controls in test: {ctrl}) fingerprint {rec['fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
