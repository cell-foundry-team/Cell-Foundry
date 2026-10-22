"""Zero-shot Cellpose models (cpsam_v2, cpsam, cpdino, cpdino-vitb) with a file-picker,
scored with the full metric set (fat %, fat % error, foreground Dice, count, F1/SA).

Same as scripts/run_cpsam.py, except the scoring: it uses ``score`` from
``score_masks_v2.py`` (must be in the SAME folder as this file), so the
single-image tool and the batch re-scoring use one identical metric definition.
The new score contains everything the old one had (n_true, n_pred, SA/F1/TP/FP/FN at
IoU 0.5 / 0.75 / 0.9, count error) plus fat %, fat % error, foreground Dice and IoU 0.7.

How to run (uv, cellpose-env; from cellpose-env\\):
    uv run python <path>\\run_cpsam_v2.py                      # file-explorer window to pick the image
    uv run python <path>\\run_cpsam_v2.py --image X.png        # skip the window, give the path directly
    uv run python <path>\\run_cpsam_v2.py --diameter 60        # cell diameter in px (image scaled by 30/60)
    uv run python <path>\\run_cpsam_v2.py --model cpdino       # other built-in Cellpose models

What happens:
    1. You pick an image (.png, .jpg, .jpeg, .tif, .tiff).
    2. The script looks for its ground-truth mask automatically: same name with
       "_image" replaced by "_instances" and extension .tif/.tiff/.png. If none is found,
       it asks whether you want to pick one by hand (Cancel = no scoring).
    3. Cellpose-SAM segments the image; metrics are printed if a mask was found.
    4. Results are shown in a window and saved to a results folder NEXT TO your dataset,
       never inside it (a NEW folder name, so earlier run_cpsam.py results are not overwritten), e.g.
           ...\\Project\\Cloud\\Training_porcine_bf\\X_image.png
        -> ...\\Project\\Cloud_results\\<model>_zero_shot_v2\\Training_porcine_bf\\
               X_<model>_instances.tif   predicted label image (0 = background, 1..N = cells)
               X_<model>_overlay.png     red = predicted outlines, green = ground-truth outlines
               X_<model>_metrics.txt     the printed metrics + model/version, for your notes
    5. You are asked whether to segment another image.

Metric definitions (see score_masks_v2.py):
    fat %        = (pixels with label > 0) / (H * W) * 100        (GT and prediction)
    fat % error  = fat %(pred) - fat %(GT), percentage points       (+ = over-estimate)
    foreground Dice = 2 |P n T| / (|P| + |T|)
    count error  = (n_pred - n_true) / n_true
    one-to-one matching of predicted and true cells by IoU:
    TP = matched pairs with IoU >= t, FP = unmatched predictions, FN = unmatched true cells
    SA(t) = TP / (TP + FP + FN)      # what Cellpose calls "average precision"; NOT COCO AP
    F1(t) = 2TP / (2TP + FP + FN)
Images with no annotated cells: F1/SA/count error are undefined; the false detections
and the area readouts are reported instead.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage.segmentation import find_boundaries

# One metric definition for every script: the score from score_masks_v2.py (same folder).
from score_masks_v2 import THRESHOLDS, load_labels, score

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MASK_EXTS = (".tif", ".tiff", ".png")  # lossless only; JPEG would corrupt label values


# --------------------------------------------------------------------------- file dialogs
def _tk_root():
    """Hidden Tk window so the dialogs appear on top of VS Code."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def pick_file(title: str, initial_dir: Path | None, patterns: tuple[str, ...]) -> Path | None:
    """Open a Windows file-explorer dialog; return the chosen path or None if cancelled."""
    from tkinter import filedialog

    root = _tk_root()
    pattern = " ".join(f"*{e}" for e in patterns)
    path = filedialog.askopenfilename(
        title=title,
        initialdir=str(initial_dir) if initial_dir else None,
        filetypes=[("Supported files", pattern), ("All files", "*.*")],
        parent=root,
    )
    root.destroy()
    return Path(path) if path else None


def ask_yes_no(title: str, question: str) -> bool:
    from tkinter import messagebox

    root = _tk_root()
    answer = messagebox.askyesno(title, question, parent=root)
    root.destroy()
    return bool(answer)


def cellpose_version() -> str:
    from importlib.metadata import version
    return version("cellpose")


# --------------------------------------------------------------------------- I/O
def load_image(path: Path) -> np.ndarray:
    """Read PNG/JPG/TIF as an array; convert palette images, drop alpha channels."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(path)
    else:
        pil = Image.open(path)
        if pil.mode == "P":
            pil = pil.convert("RGB")
        elif pil.mode == "LA":
            pil = pil.convert("L")
        img = np.array(pil)
    if img.ndim == 3 and img.shape[-1] == 4:  # RGBA -> RGB (some porcine phase images)
        img = img[..., :3]
    return img


def load_mask(path: Path) -> np.ndarray:
    """Read a single-channel label image (0 = background, 1..N = instances) as int32."""
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        raise ValueError("Masks must be lossless (.tif/.tiff/.png); JPEG changes label values.")
    m = tifffile.imread(path) if path.suffix.lower() in {".tif", ".tiff"} else np.array(Image.open(path))
    if m.ndim != 2:
        raise ValueError(f"Expected a 2D label mask, got shape {m.shape}")
    return m.astype(np.int32)


def find_matching_mask(image_path: Path) -> Path | None:
    """<name>_image.<ext>  ->  <name>_instances.<tif|tiff|png> in the same folder (README rule)."""
    stem = image_path.stem
    if not stem.endswith("_image"):
        return None
    base = stem[: -len("_image")] + "_instances"
    for ext in MASK_EXTS:
        cand = image_path.with_name(base + ext)
        if cand.exists():
            return cand
    return None


def default_out_dir(image_path: Path, model_name: str = "cpsam_v2") -> Path:
    """<root>/<dataset>/<group>/img  ->  <root>/<dataset>_results/<model>_zero_shot/<group>/"""
    group = image_path.parent
    dataset = group.parent
    return dataset.parent / f"{dataset.name}_results" / f"{model_name}_zero_shot_v2" / group.name


# --------------------------------------------------------------------------- metrics + overlay
def report_lines(res: dict) -> list[str]:
    """Readable report of score_masks_v2.score(), in the agreed order: area -> count -> instances."""
    lines = [
        "--- area (IMES primary readout)",
        f"    fat_pct_true: {res['fat_pct_true']:.2f} %",
        f"    fat_pct_pred: {res['fat_pct_pred']:.2f} %",
        f"   fat_pct_error: {res['fat_pct_error']:+.2f} pp  (+ = over-estimate)",
        f" foreground_dice: {res['fg_dice']:.3f}",
        "--- count",
        f"          n_true: {res['n_true']}",
        f"          n_pred: {res['n_pred']}",
    ]
    if res["negative_image"]:
        lines += ["   count_error_%: undefined (no annotated cells)",
                  f" false_detections: {res['n_pred']}",
                  "--- instances: undefined (no annotated cells)"]
        return lines
    lines += [f"   count_error_%: {100 * res['count_error']:+.1f}",
              "--- instances (one-to-one IoU matching)"]
    for t in THRESHOLDS:
        m = res["per_t"][t]
        lines.append(f"       IoU>={t:<4}: F1={m['F1']:.3f}  SA={m['SA']:.3f}  "
                     f"TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")
    return lines


def to_rgb8(img: np.ndarray) -> np.ndarray:
    """Any image (gray/RGB, 8/16-bit) -> RGB uint8 for display."""
    rgb = img if img.ndim == 3 else np.stack([img] * 3, -1)
    rgb = rgb.astype(np.float32)
    return (255 * (rgb - rgb.min()) / max(np.ptp(rgb), 1e-6)).astype(np.uint8)


def make_overlay(img: np.ndarray, pred: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    """Predicted outlines in red, ground-truth outlines in green."""
    rgb = to_rgb8(img)
    if gt is not None:
        rgb[find_boundaries(gt, mode="outer")] = (0, 255, 0)
    rgb[find_boundaries(pred, mode="outer")] = (255, 0, 0)
    return rgb


def show_result(img: np.ndarray, overlay: np.ndarray, title: str) -> None:
    """Side-by-side window: original | segmentation. Close the window to continue."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        Image.fromarray(overlay).show()  # fallback: default Windows image viewer
        return
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    ax[0].imshow(to_rgb8(img)); ax[0].set_title("Original")
    ax[1].imshow(overlay); ax[1].set_title("prediction (red) | ground truth (green)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- main
def segment_one(model, image_path: Path, args, use_gpu: bool) -> None:
    import torch

    img = load_image(image_path)
    print(f"\nimage {image_path.name}: shape {img.shape}, dtype {img.dtype}")

    mask_path = find_matching_mask(image_path)
    if mask_path is None and not args.no_dialog:
        if ask_yes_no("No mask found", f"No matching *_instances mask found for\n{image_path.name}.\n\n"
                                       "Pick a ground-truth mask by hand? (No = segment without scoring)"):
            mask_path = pick_file("Select the ground-truth mask", image_path.parent, MASK_EXTS)
    print(f"mask: {mask_path.name if mask_path else 'none (no scoring)'}")

    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pred, _, _ = model.eval(
        img,
        batch_size=args.batch_size,
        channel_axis=-1 if img.ndim == 3 else None,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
    )
    dt = time.perf_counter() - t0
    pred = pred.astype(np.int32)

    lines = [f"model: {args.model} (cellpose {cellpose_version()})", f"image: {image_path}", f"mask: {mask_path}",
             f"settings: diameter={args.diameter} flow_threshold={args.flow_threshold} "
             f"cellprob_threshold={args.cellprob_threshold} batch_size={args.batch_size}",
             f"inference: {dt:.1f} s, {pred.max()} cells predicted"]
    if use_gpu:
        lines.append(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    gt = None
    if mask_path is not None:
        gt = load_labels(mask_path)  # same loading as the batch re-scoring (labels 1..N)
        if gt.shape != pred.shape:
            raise ValueError(f"mask shape {gt.shape} != prediction shape {pred.shape}")
        lines += report_lines(score(gt, pred))
    print("\n".join(lines))

    out_dir = args.out if args.out else default_out_dir(image_path, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem.replace("_image", "")
    overlay = make_overlay(img, pred, gt)
    tag = args.model
    tifffile.imwrite(out_dir / f"{stem}_{tag}_instances.tif", pred.astype(np.uint16))
    Image.fromarray(overlay).save(out_dir / f"{stem}_{tag}_overlay.png")
    (out_dir / f"{stem}_{tag}_metrics.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved to {out_dir.resolve()}")

    if not args.no_show:
        show_result(img, overlay, f"{image_path.name}  —  {args.model}: {pred.max()} cells")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, default=None, help="skip the file dialog and use this image")
    p.add_argument("--model", default="cpsam_v2", choices=["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"],
                   help="built-in Cellpose model (cpdino* need: pip install git+https://github.com/facebookresearch/dinov3)")
    p.add_argument("--start-dir", type=Path, default=Path(r""),
                   help="folder the file dialog opens in")
    p.add_argument("--out", type=Path, default=None, help="override the results folder")
    p.add_argument("--diameter", type=float, default=None,
                   help="expected cell diameter in px; Cellpose resizes the image by 30/diameter (None = no resizing)")
    p.add_argument("--rescale", type=float, default=None,
                   help="DEPRECATED: Cellpose 4.2 ignores eval(rescale=...); converted to --diameter 30/rescale")
    p.add_argument("--batch-size", type=int, default=8, help="256x256 tiles per GPU batch; lower if out of memory")
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--no-show", action="store_true", help="do not open the result window")
    p.add_argument("--no-dialog", action="store_true", help="never open dialogs (for scripted runs)")
    args = p.parse_args()
    if args.rescale is not None:  # Cellpose 4.2.1 overwrites rescale with 1.0 inside eval(); only diameter works
        if args.diameter is None:
            args.diameter = 30.0 / args.rescale
        print(f"note: --rescale {args.rescale} -> --diameter {args.diameter:g} (Cellpose 4 ignores rescale)")

    import torch
    from cellpose import models

    use_gpu = torch.cuda.is_available() and not args.cpu
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | using GPU: {use_gpu}")
    if use_gpu:
        print("GPU:", torch.cuda.get_device_name(0))
    print("model:", args.model)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=args.model)  # loaded once, reused

    image_path = args.image
    start_dir = args.start_dir if args.start_dir.exists() else None
    while True:
        if image_path is None:
            image_path = pick_file("Select an image to segment", start_dir, IMAGE_EXTS)
            if image_path is None:
                print("No image selected - exiting.")
                return
        if image_path.suffix.lower() not in IMAGE_EXTS:
            print(f"Unsupported file type: {image_path.suffix}")
        else:
            segment_one(model, image_path, args, use_gpu)
        if args.no_dialog or not ask_yes_no("Continue?", "Segment another image?"):
            return
        start_dir, image_path = image_path.parent, None


if __name__ == "__main__":
    main()
