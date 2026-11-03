"""List and picture the multi-piece cells (one label ID made of disconnected islands).

Read-only. Usage (from the repo root):
  uv run python scripts/inspect_multipiece.py --image Positiv_29                      # list for one image
  uv run python scripts/inspect_multipiece.py --image Positiv_29 --crops ~/crops
Crop = raw image (left) | same crop, all cells outlined yellow, main piece red, other pieces cyan (right).
Crops contain real IMES images: they cannot be written inside the repo (see inspect_annotations.py).
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).parent))
import inspect_annotations as I  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="part of an image name, e.g. Positiv_29")
ap.add_argument("--data", type=Path, default=I.DEFAULT)
ap.add_argument("--crops", type=Path, help="folder to save one crop per multi-piece cell")
ap.add_argument("--pad", type=int, default=60)
a = ap.parse_args()
if a.crops:
    a.crops = I.check_out(a.crops)

k, ip, mp = next(p for p in I.pairs(a.data) if a.image in p[0])
img, m = I.read(ip)[..., :3], I.read(mp)
print(f"{k}: image {img.shape[:2]}")
rows = []
for i, sl in enumerate(ndi.find_objects(m), 1):
    if sl is None:
        continue
    lab, n = ndi.label(m[sl] == i)
    if n > 1:
        sz = np.bincount(lab.ravel())[1:]
        rows.append((i, n, sorted(map(int, sz), reverse=True), sl, lab, int(np.argmax(sz)) + 1))
print(f"{len(rows)} multi-piece cells (label id | pieces | piece sizes in px)")
for i, n, sz, *_ in rows:
    print(f"  {i:4d} | {n} | {sz}")
if a.crops:
    a.crops.mkdir(parents=True, exist_ok=True)
    H, W = m.shape
    for i, n, sz, sl, lab, main in rows:
        y0, y1 = max(0, sl[0].start - a.pad), min(H, sl[0].stop + a.pad)
        x0, x1 = max(0, sl[1].start - a.pad), min(W, sl[1].stop + a.pad)
        raw = img[y0:y1, x0:x1].copy()
        ov = raw.copy()
        ov[find_boundaries(m[y0:y1, x0:x1] > 0, mode="inner")] = [255, 255, 0]   # all annotated cells: yellow outline
        full = np.zeros(m.shape, int); full[sl][lab > 0] = lab[lab > 0]
        piece = full[y0:y1, x0:x1]
        for j in range(1, n + 1):
            mask = piece == j
            if j == main:
                ov[find_boundaries(mask, mode="inner")] = [255, 0, 0]
            else:                                        # dilate small pieces so a 1-px speck is visible
                big = ndi.binary_dilation(mask, iterations=3)
                ov[big] = [0, 255, 255]
        iio.imwrite(a.crops / f"{k.replace('CLD-AIM-04 PIG 10x Training ', '')}_label{i}_{'x'.join(map(str, sz[:3]))}px.png",
                    np.concatenate([raw, ov], axis=1))
    print(f"crops saved to {a.crops.resolve()}")
