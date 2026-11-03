"""Inspect IMES annotation masks: numbers for every image, and an overlay picture.

Read-only: never edits the data. Needs numpy, scipy, scikit-image, tifffile, imageio (present in both
environments of this repo, imported at top level because the script is useless without them).

Usage (from the repo root):
  uv run python scripts/inspect_annotations.py                            # table for all images
  uv run python scripts/inspect_annotations.py --overlay Positiv_10 --out ~/pos10.png
  uv run python scripts/inspect_annotations.py --all-overlays ~/overlays   # one JPEG per image
Default data folder: data/Training_porcine_bf (override with --data).

CONFIDENTIALITY: overlays contain real IMES microscopy images. The script refuses to write them
inside this repository (outside data/, which is git-ignored); write to a local folder elsewhere.
"""
import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / "data" / "Training_porcine_bf"


def check_out(path):
    """Refuse to write derived image files inside the repo, except in the git-ignored data/ folder."""
    p = Path(path).expanduser().resolve()
    if REPO in p.parents or p == REPO:
        if (REPO / "data") not in p.parents and p != REPO / "data":
            raise SystemExit(f"Refusing to write IMES-derived images inside the repo: {p}\n"
                             "Choose a folder outside the repo (or under data/, which is git-ignored).")
    return p


def read(path):
    return tifffile.imread(path) if path.suffix.lower() in {".tif", ".tiff"} else iio.imread(path)


def pairs(root):
    imgs, masks = {}, {}
    for p in sorted(root.iterdir()):
        s = p.stem
        if s.endswith("_image"):
            imgs[s[:-6]] = p
        elif s.endswith("_instances"):
            masks[s[:-10]] = p
    return [(k, imgs[k], masks[k]) for k in sorted(set(imgs) & set(masks))]


def stats(m):
    areas = np.bincount(m.ravel())[1:]          # area of label 1..max (one pass)
    present = np.nonzero(areas)[0] + 1
    areas = areas[present - 1]
    objs = ndi.find_objects(m)                   # bounding box per label id
    multi = 0
    for i in present:
        sl = objs[i - 1]
        if ndi.label(m[sl] == i)[1] > 1:
            multi += 1
    return dict(n=len(present), max_id=int(m.max()), fg=100 * float((m > 0).mean()),
                med=int(np.median(areas)) if len(areas) else 0,
                amin=int(areas.min()) if len(areas) else 0, amax=int(areas.max()) if len(areas) else 0,
                multi=multi)


def overlay(img, m, out):
    ov = img[..., :3].copy()
    ov[find_boundaries(m, mode="inner")] = [255, 0, 0]
    iio.imwrite(out, np.concatenate([img[..., :3], ov], axis=1))   # left: image, right: with outlines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT)
    ap.add_argument("--overlay", help="substring of an image name, e.g. Positiv_10")
    ap.add_argument("--out", type=Path, help="output file for --overlay (required with --overlay)")
    ap.add_argument("--all-overlays", type=Path, metavar="DIR",
                    help="save an overlay (JPEG) for every image into DIR")
    a = ap.parse_args()
    if a.overlay and not a.out:
        ap.error("--overlay needs --out")
    if a.out:
        a.out = check_out(a.out)
    if a.all_overlays:
        a.all_overlays = check_out(a.all_overlays)
    ps = pairs(a.data)
    print(f"{len(ps)} image/mask pairs in {a.data}\n")
    print(f"{'name':26s}{'cells':>6s}{'max_id':>7s}{'fat%':>7s}{'median':>8s}{'min':>6s}{'max':>7s}{'multi-piece':>12s}")
    for k, ip, mp in ps:
        s = stats(read(mp))
        print(f"{k.replace('CLD-AIM-04 PIG 10x Training ', '')[-26:]:26s}{s['n']:6d}{s['max_id']:7d}{s['fg']:7.1f}{s['med']:8d}{s['amin']:6d}{s['amax']:7d}{s['multi']:12d}")
    if a.all_overlays:
        a.all_overlays.mkdir(parents=True, exist_ok=True)
        for k, ip, mp in ps:
            name = k.replace("CLD-AIM-04 PIG 10x Training ", "")
            overlay(read(ip), read(mp), a.all_overlays / f"{name}_overlay.jpg")
        print(f"\n{len(ps)} overlays saved to {a.all_overlays.resolve()} (left image, right with red outlines)")
    if a.overlay:
        k, ip, mp = next(p for p in ps if a.overlay in p[0])
        overlay(read(ip), read(mp), a.out)
        print(f"\noverlay of {k} saved to {a.out.resolve()} (left image, right with red outlines)")
