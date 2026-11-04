"""Shared helpers for the cpdino fine-tuning scripts (experiments/cpdino_finetune_v1/scripts).
(cpdino copy of experiments/cpdino_vitb_finetune_v1/scripts/split_io.py: same logic; only the defaults and examples now name cpdino, the ViT-L DINOv3 model.)

    load_split      read a split JSON made by make_split.py and re-check its fingerprint
    find_image      <group-dir>\\<name>_image.<ext>
    find_gt         <group-dir>\\<name>_instances.{tif,tiff,png}
    load_labels     GT label image -> 2D int32, labels 1..N (same rule as score_masks_v2.py)
    check_label     model labels become folder / file names -> only safe characters
    sha256_file     fingerprint of a weights file
    load_cellpose   build a CellposeModel safely (no silent fallback, strict key check)
    versions        software versions for the run config
    mask_path       where a predicted mask goes in the shared format
    save_mask       write a predicted mask in the shared format (usable by ANY model's script)
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
GT_EXTS = (".tif", ".tiff", ".png")
SUBSETS = ("train", "val", "test")
OPTIONAL_KEYS = ("diam_labels", "diam_mean")   # fixed diameter values, see BUGFIX 1 in load_cellpose


def _fingerprint(splits: dict) -> str:
    """Same definition as make_split.py."""
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def load_split(path: Path) -> dict:
    """Read the split JSON; stop if the image lists were edited (fingerprint no longer matches)."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = rec["splits"]
    if set(splits) != set(SUBSETS):
        raise SystemExit(f"{path}: expected subsets {SUBSETS}, found {sorted(splits)}")
    if _fingerprint(splits) != rec["fingerprint"]:
        raise SystemExit(f"{path}: fingerprint mismatch -> the image lists were changed by hand. "
                         "Re-create the split with make_split.py instead.")
    names = [n for s in SUBSETS for n in splits[s]]
    if len(names) != len(set(names)):
        raise SystemExit(f"{path}: an image appears in more than one subset.")
    return rec


def find_image(group_dir: Path, name: str) -> Path:
    hits = [group_dir / f"{name}_image{e}" for e in IMAGE_EXTS if (group_dir / f"{name}_image{e}").exists()]
    if len(hits) != 1:
        raise SystemExit(f"Expected exactly one '{name}_image.*' in {group_dir}, found {len(hits)}.")
    return hits[0]


def find_gt(group_dir: Path, name: str) -> Path:
    hits = [group_dir / f"{name}_instances{e}" for e in GT_EXTS if (group_dir / f"{name}_instances{e}").exists()]
    if len(hits) != 1:
        raise SystemExit(f"Expected exactly one '{name}_instances.*' in {group_dir}, found {len(hits)}.")
    return hits[0]


def load_labels(path: Path) -> np.ndarray:
    """Label image as 2D int32 with labels 1..N (0 = background). Same as score_masks_v2.py."""
    import tifffile
    from skimage import io as skio
    from skimage.segmentation import relabel_sequential

    arr = tifffile.imread(path) if path.suffix.lower() in (".tif", ".tiff") else skio.imread(path)
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D label image, got shape {arr.shape}: {path}")
    return relabel_sequential(arr.astype(np.int64))[0].astype(np.int32)


def check_label(label: str) -> str:
    """Labels become folder and file names (and a glob in score_masks_v2.py): keep them simple."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
        raise SystemExit(f"Label '{label}' may only contain letters, digits, '.', '_' and '-'.")
    return label


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cellpose(model: str, use_gpu: bool):
    """Build a CellposeModel from a built-in name or a weights file.

    Guards against two silent Cellpose 4.2.1.1 behaviours (checked in its source):
      * an unknown name or a missing file only gives a warning and loads cpsam_v2 instead;
      * weights are loaded with strict=False, so non-matching keys would be skipped quietly.
    """
    import torch
    from cellpose import models

    is_builtin = model in models.MODEL_NAMES
    if not is_builtin and not Path(model).is_file():
        raise SystemExit(f"'{model}' is neither a built-in model {models.MODEL_NAMES} nor an existing file.")
    cp = models.CellposeModel(gpu=use_gpu, pretrained_model=model)

    saved = torch.load(cp.pretrained_model, map_location="cpu", weights_only=True)
    saved = {k[7:] if k.startswith("module.") else k: v for k, v in saved.items()}
    loaded = cp.net.state_dict()
    missing, unexpected = sorted(set(loaded) - set(saved)), sorted(set(saved) - set(loaded))
    # BUGFIX 1: the built-in cpdino/cpdino-vitb files do not store the two fixed diameter values
    # 'diam_labels' and 'diam_mean' (default 30). Cellpose 4.2.1.1 never reads them at prediction
    # and uses diam_mean in training only with rescale=True (we use False), so they may be missing,
    # but only if they still hold the default 30. Any other missing key still stops the run.
    for k in [k for k in missing if k in OPTIONAL_KEYS]:
        if not torch.allclose(loaded[k].float().cpu(), torch.tensor([30.0])):
            raise SystemExit(f"'{k}' is missing from the file and is not the default 30: {loaded[k]}")
        missing.remove(k)
        print(f"    note: '{k}' not in the weights file; Cellpose default 30 kept (not used here)")
    if missing or unexpected:
        raise SystemExit(f"Weights do not match the network: {len(missing)} missing keys "
                         f"(e.g. {missing[:3]}), {len(unexpected)} unexpected keys (e.g. {unexpected[:3]}).")
    for k, v in saved.items():   # values must match up to the bf16 rounding of the loaded model
        if not torch.equal(loaded[k].cpu(), v.to(loaded[k].dtype)):
            raise SystemExit(f"Weight '{k}' differs from the file after loading.")
    return cp


def versions() -> dict:
    info = {"python": platform.python_version()}
    try:
        import cellpose, cv2, torch
        info.update(cellpose=cellpose.version, torch=torch.__version__, opencv=cv2.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:  # pragma: no cover
        info["version_error"] = repr(e)
    return info


def mask_path(run_dir: Path, label: str, name: str) -> Path:
    """Shared mask format read by score_masks_v2.py: <run_dir>/masks/<label>/<name>_<label>_instances.tif"""
    return Path(run_dir) / "masks" / check_label(label) / f"{name}_{label}_instances.tif"


def save_mask(run_dir: Path, label: str, name: str, mask: np.ndarray, image_shape: tuple) -> Path:
    """Write one predicted instance mask in the shared format.

    mask: 2D integer labels (0 = background, 1..N = objects) at the ORIGINAL image size.
    Refuses wrong shapes, more than 65535 objects and existing files (never overwrites).
    """
    import tifffile

    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2 or mask.shape != tuple(image_shape[:2]):
        raise ValueError(f"{name}: mask shape {mask.shape} != image shape {tuple(image_shape[:2])}")
    if mask.min() < 0 or mask.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"{name}: labels must be 0..65535, got {mask.min()}..{mask.max()}")
    out = mask_path(run_dir, label, name)
    if out.exists():
        raise FileExistsError(f"{out} exists; not overwriting")
    out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out, mask.astype(np.uint16))
    return out
