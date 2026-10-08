"""Zero-shot Cellpose models (cpsam_v2, cpsam, cpdino, cpdino-vitb) with a file-picker.

How to run (from the activated virtual environment):
    python scripts/run_cpsam.py                      # opens a file-explorer window to pick the image
    python scripts/run_cpsam.py --image X.png        # skip the window, give the path directly
    python scripts/run_cpsam.py --diameter 60        # tell Cellpose the cell diameter in px (image is scaled by 30/60)
    python scripts/run_cpsam.py --model cpdino       # other built-in Cellpose models (cpdino needs dinov3)

What happens:
    1. You pick an image (.png, .jpg, .jpeg, .tif, .tiff).
    2. The script looks for its ground-truth mask automatically: same name with
       "_image" replaced by "_instances" and extension .tif/.tiff/.png. If none is found,
       it asks whether you want to pick one by hand (Cancel = no scoring).
    3. Cellpose-SAM segments the image; metrics are printed if a mask was found.
    4. Results are shown in a window and saved to a results folder NEXT TO your dataset,
       never inside it, e.g.
           ...\\Project\\Cloud\\Training_porcine_bf\\X_image.png
        -> ...\\Project\\Cloud_results\\<model>_zero_shot\\Training_porcine_bf\\
               X_<model>_instances.tif   predicted label image (0 = background, 1..N = cells)
               X_<model>_overlay.png     red = predicted outlines, green = ground-truth outlines
               X_<model>_metrics.txt     the printed metrics + model/version, for your notes
    5. You are asked whether to segment another image.

Metric definitions (per image, one-to-one matching of predicted and true cells by IoU):
    TP = matched pairs with IoU >= t, FP = unmatched predictions, FN = unmatched true cells
    SA(t) = TP / (TP + FP + FN)      # what Cellpose calls "average precision"; NOT COCO AP
    F1(t) = 2TP / (2TP + FP + FN)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage.segmentation import find_boundaries

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
    return dataset.parent / f"{dataset.name}_results" / f"{model_name}_zero_shot" / group.name


# --------------------------------------------------------------------------- metrics + overlay
def score(gt: np.ndarray, pred: np.ndarray, thresholds=(0.5, 0.75, 0.9)) -> dict:
    """Instance metrics via Cellpose's one-to-one IoU matching."""
    from cellpose import metrics

    sa, tp, fp, fn = metrics.average_precision([gt], [pred], threshold=list(thresholds))
    out = {"n_true": int(len(np.unique(gt)) - (0 in gt)),
           "n_pred": int(len(np.unique(pred)) - (0 in pred))}
    for i, t in enumerate(thresholds):
        TP, FP, FN = tp[0, i], fp[0, i], fn[0, i]
        out[f"SA@{t}"] = float(sa[0, i])
        out[f"F1@{t}"] = float(2 * TP / max(2 * TP + FP + FN, 1))
        out[f"TP/FP/FN@{t}"] = (int(TP), int(FP), int(FN))
    out["count_error_%"] = 100.0 * (out["n_pred"] - out["n_true"]) / max(out["n_true"], 1)
    return out


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
        gt = load_mask(mask_path)
        if gt.shape != pred.shape:
            raise ValueError(f"mask shape {gt.shape} != prediction shape {pred.shape}")
        for k, v in score(gt, pred).items():
            lines.append(f"{k:>16}: {v:.3f}" if isinstance(v, float) else f"{k:>16}: {v}")
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
    p.add_argument("--start-dir", type=Path, default=Path(r"C:\Users\Priyam\Desktop\Project\Cloud"),
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
