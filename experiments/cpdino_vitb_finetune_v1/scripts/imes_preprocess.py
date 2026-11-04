"""IMES image preprocessing (per-channel CLAHE), re-written from the settings in IMES's
DINOCell Colab notebook, so that our models can be trained and evaluated on the same
inputs as IMES's model. Only the image-level step is reproduced here; the DINOCell-specific
steps (ImageNet normalisation, resize to 896, flow targets) belong to that model and are
NOT part of this preprocessing.

Recipe, applied to every image the same way (train, val and test):
  1. make the image 3-channel: grey -> repeated 3x; 1 channel -> repeated; 4 channels -> drop the 4th
  2. for EACH channel separately:
       a. min-max stretch to 0-255 using the single darkest and brightest pixel, cut to uint8
          (a flat channel becomes all zeros)
       b. CLAHE with clipLimit = 1.0 and an 8 x 8 tile grid
          (for a 1920 x 1080 image one tile is 240 x 135 px)
       c. min-max stretch to 0-255 again (uint8)
  3. stack the 3 channels -> H x W x 3 uint8

Known properties kept on purpose (faithful to IMES; possible later ablations):
  * CLAHE runs per colour channel; DINOCell's own inference converts to grey first (finding F4)
  * step 2a uses the absolute min/max, not percentiles, so one hot pixel can compress the range

Cellpose still applies its own 1-99 percentile normalisation afterwards (model-internal).

Quick visual check of one image (writes a new PNG, never overwrites):
    python "<exp>\\scripts\\imes_preprocess.py" --image "<data>\\<name>_image.png" --out "<folder>\\<name>_clahe.png"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID = (8, 8)


def to_three_channels(image: np.ndarray) -> np.ndarray:
    """H x W, H x W x 1 or H x W x 4 -> H x W x 3 (no other change)."""
    image = np.asarray(image)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.repeat(image, 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 4:
        return image[..., :3]
    if image.ndim == 3 and image.shape[-1] == 3:
        return image
    raise ValueError(f"Expected an H x W (x 1/3/4) image, got shape {image.shape}")


def stretch_to_uint8(channel: np.ndarray) -> np.ndarray:
    """Min-max stretch of one channel to 0-255 (uint8, truncated); a flat channel -> zeros."""
    channel = channel.astype(np.float32)
    lo, hi = float(channel.min()), float(channel.max())
    if hi <= lo:
        return np.zeros(channel.shape, dtype=np.uint8)
    return ((channel - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def imes_clahe(image: np.ndarray) -> np.ndarray:
    """Apply the IMES preprocessing (see module docstring). Returns H x W x 3 uint8."""
    import cv2

    rgb = to_three_channels(image)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    out = []
    for c in range(3):
        enhanced = clahe.apply(stretch_to_uint8(rgb[..., c]))
        out.append(cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    return np.stack(out, axis=-1)


def grey_clahe(image: np.ndarray) -> np.ndarray:
    """DINOCell's OWN preprocessing recipe (not IMES's): convert to grey FIRST, run CLAHE ONCE
    on the grey channel, then repeat back to 3 channels -- as opposed to imes_clahe() above,
    which runs CLAHE on R, G, B separately and never converts to grey. Same clip limit and tile
    grid, so the per-channel-vs-grey-first choice is the only thing that differs between the two.

    Untested against real training results as of writing (P1 in Test_Plan_Preprocessing.md,
    local-only/ -- not part of this repo). Added here so both the DINOCell and cpdino-vitb
    fine-tuning scripts can select it via --preprocess grey_clahe.
    """
    import cv2

    rgb = to_three_channels(image)
    grey = rgb.astype(np.float32).mean(axis=-1)                      # H x W, average of R/G/B
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(stretch_to_uint8(grey))
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.repeat(enhanced[..., None], 3, axis=-1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True, help="one *_image.* file")
    p.add_argument("--out", type=Path, required=True, help="where to write the preprocessed PNG")
    a = p.parse_args(argv)

    from cellpose import io as cpio   # same reader as our zero-shot scripts (RGB order, alpha dropped)
    import cv2

    if a.out.exists():
        raise SystemExit(f"{a.out} already exists; not overwriting.")
    raw = cpio.imread(str(a.image))
    pre = imes_clahe(raw)
    print(f"input : shape {raw.shape}, dtype {raw.dtype}, min {raw.min()}, max {raw.max()}")
    print(f"output: shape {pre.shape}, dtype {pre.dtype}, "
          f"per-channel mean {[round(float(pre[..., c].mean()), 1) for c in range(3)]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), pre[..., ::-1])   # cv2 writes BGR
    print(f"saved to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
