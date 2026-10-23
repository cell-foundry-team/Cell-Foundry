"""Batch zero-shot evaluation of DINOCell on a whole group folder.

Counterpart to ``run_batch.py`` (Cellpose-family models), using the exact same
metric code (``score_masks.score``: Cellpose ``metrics.average_precision``,
one-to-one IoU matching, a match counts at IoU >= t) so DINOCell's numbers are
directly comparable in the shared summary/CSV/W&B format.

DINOCell is a separate package from the Cellpose-family models benchmarked in
``run_batch.py`` and must run in the MAIN project's ``.venv`` (dinocell==0.74),
not ``cellpose-env/`` (cellpose==4.2.1.1) -- the two can't be installed
together. See ``cellpose-env/README.md`` for the full explanation.

Inference uses ``dinocell``'s own sliding-window pipeline (``get_pipeline
("flows", ...)``), which tiles the image, runs the model per tile, and stitches
the tiles back together by averaging their overlapping predictions before
extracting instances -- this already handles cells that straddle a tile
boundary, so no custom tiling logic is needed here. ``--overlap-size`` controls
how much neighbouring tiles overlap (see the script's own help text for the
tradeoff); the default (128px, for the default 512px crop) is a reasonable
starting point, not a value IMES validated for this data -- check the
overlay images for boundary artefacts before trusting a number blindly.

Outputs go to a NEW folder per run (nothing is overwritten), same layout as
``run_batch.py``:

    <out-root>\\batch\\<YYYYmmdd-HHMMSS>_<group>_dinocell\\
        per_image.csv     one row per image
        summary.csv       one row per condition: mean +- std over images, pooled F1, counts, timing
        run_config.json   all settings + versions (for the decision log / reproducibility)
        masks\\<name>_dinocell_instances.tif
        overlays\\<name>_dinocell_overlay.png   (green = GT, red = prediction, yellow = both)

Usage (from the repo folder, in the MAIN .venv -- not cellpose-env):
    python scripts\\run_dinocell_batch.py --group-dir "<data>\\Training_porcine_bf" --weights "<path>\\DINOCell_demo_model.pt" --limit 2
    python scripts\\run_dinocell_batch.py --group-dir "<data>\\Training_porcine_bf" --overlap-size 64 --wandb
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_batch import PER_IMAGE_FIELDS, find_pairs, parse_condition, parse_series, save_overlay, summarise, write_csv  # noqa: E402
from score_masks import load_labels, score  # noqa: E402  (same metric code)

DEFAULT_GROUP_DIR = Path(r"")
DEFAULT_OUT_ROOT = Path(r"")
DEFAULT_WEIGHTS = Path.home() / "Documents" / "Cell Foundry" / "Cell-Foundry" / ".dinocell_models" / "DINOCell_demo_model.pt"


# ---------------------------------------------------------------------------- model
def build_model(weights_path: Path, device):
    """Load DINOCell with the fixed architecture hyperparameters tied to this
    checkpoint (confirmed against both the installed dinocell CLI and IMES's
    own commented training notebook -- crop_size/patch_size/feat_size are not
    safely changeable, they must match what the checkpoint was trained with)."""
    import torch
    from dinocell.model import DINOCell

    model = DINOCell(
        dino_model=None, decoder_type="upsample", objective_type="flows", use_dino_weights=False,
        patch_size=8, feat_size=64, crop_size=512, drop_rate=0.05, dropout_in_encoder=True,
        finetune_vision=False, finetune_decoder=True, finetune_prediction_head=True,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_pipeline(model, device, crop_size: int, overlap_size: int):
    from dinocell.pipeline import get_pipeline

    return get_pipeline("flows", model, device, crop_size=crop_size, overlap_size=overlap_size)


# -------------------------------------------------------------------------- helpers
def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (round(x, nd) if isinstance(x, float) else x)


def versions(device) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device)}
    try:
        import dinocell
        import torch

        info.update(dinocell=dinocell.__version__ if hasattr(dinocell, "__version__") else "unknown",
                    torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                    mps_available=torch.backends.mps.is_available())
    except Exception as e:  # pragma: no cover
        info["version_error"] = repr(e)
    try:
        import subprocess

        info["repo_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, timeout=10).stdout.strip() or "unknown"
    except Exception:
        info["repo_commit"] = "unknown"
    return info


# ----------------------------------------------------------------------------- main
def evaluate(pipeline, pairs, mask_dir: Path, overlay_dir: Path | None, model_dtype: str) -> list[dict]:
    name = "dinocell"

    first = next((img for img, gt in pairs if gt is not None), None)
    if first is not None:  # warm-up: not timed, not recorded
        t0 = time.time()
        pipeline.run(load_labels_image(first))
        print(f"    warm-up on {first.name}: {time.time() - t0:.1f} s (not recorded)")

    rows = []
    mask_dir.mkdir(parents=True, exist_ok=True)
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    for i, (img_path, gt_path) in enumerate(pairs, 1):
        base = img_path.stem[: -len("_image")]
        if gt_path is None:
            print(f"    [{i}/{len(pairs)}] {img_path.name}: no GT mask found -> skipped")
            continue
        img = load_labels_image(img_path)
        t0 = time.time()
        pred, _preds, _resized_img = pipeline.run(img, return_raw_preds=True)
        dt_s = time.time() - t0

        tifffile.imwrite(mask_dir / f"{base}_{name}_instances.tif", pred.astype(np.uint16))
        gt = load_labels(gt_path)
        pred_l = load_labels_array(pred)
        if gt.shape != pred_l.shape:
            # the pipeline may internally resize; re-score on the GT's own
            # resolution by resizing the prediction back, nearest-neighbour
            # to keep labels intact (no interpolation of integer IDs)
            from skimage.transform import resize

            pred_l = resize(pred_l, gt.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
        res = score(gt, pred_l)
        areas = np.bincount(gt.ravel())[1:]
        areas = areas[areas > 0]
        gt_diam = float(np.median(2 * np.sqrt(areas / np.pi))) if areas.size else float("nan")
        if overlay_dir:
            save_overlay(overlay_dir / f"{base}_{name}_overlay.png", img, gt, pred_l)

        row = {k: "" for k in PER_IMAGE_FIELDS}
        row.update(model=name, image=base, series=parse_series(base),
                   condition=parse_condition(base), n_true=res["n_true"],
                   n_pred=res["n_pred"], count_diff=res["n_pred"] - res["n_true"],
                   gt_median_diam_px=_fmt(gt_diam, 1), time_s=round(dt_s, 2),
                   _peak_gpu_gb=float("nan"),  # not tracked for DINOCell yet
                   _model_dtype=model_dtype)  # internal only (run_config), not written per image
        if res["negative_image"]:
            msg = f"no annotated cells, false detections={res['n_pred']}"
        else:
            m5, m75 = res["per_t"][0.5], res["per_t"][0.75]
            n_t, n_p = res["n_true"], res["n_pred"]
            row.update({
                "count_error": _fmt(res["count_error"]),
                "TP@0.5": m5["TP"], "FP@0.5": m5["FP"], "FN@0.5": m5["FN"],
                "precision@0.5": _fmt(m5["TP"] / n_p if n_p else float("nan")),
                "recall@0.5": _fmt(m5["TP"] / n_t), "F1@0.5": _fmt(m5["F1"]),
                "F1@0.75": _fmt(m75["F1"]),
            })
            msg = (f"F1@0.5={m5['F1']:.3f} F1@0.75={m75['F1']:.3f} "
                   f"n_pred/n_true={n_p}/{n_t} count={res['count_error']:+.1%}")
        print(f"    [{i}/{len(pairs)}] {base}: {msg}  ({dt_s:.1f} s)")
        rows.append(row)

    return rows


def load_labels_image(path: Path) -> np.ndarray:
    """Load a raw microscopy image (not a label mask) for feeding to the pipeline."""
    from cellpose import io as cpio

    return cpio.imread(str(path))


def load_labels_array(arr: np.ndarray) -> np.ndarray:
    """Same normalisation as score_masks.load_labels, for an in-memory prediction."""
    from skimage.segmentation import relabel_sequential

    arr = np.squeeze(np.asarray(arr))
    return relabel_sequential(arr.astype(np.int64))[0].astype(np.int32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help="path to DINOCell_demo_model.pt (or a fine-tuned checkpoint)")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    p.add_argument("--pattern", default="*_image.*",
                   help='glob for a subset of images, e.g. "*Positiv_2[1-6]_image.*"')
    p.add_argument("--crop-size", type=int, default=512,
                   help="sliding-window tile size in px. Must match the checkpoint's trained "
                        "crop_size (512 for the DINOCell demo model) -- this is architecturally "
                        "fixed, not a free tuning knob.")
    p.add_argument("--overlap-size", type=int, default=128,
                   help="how many px neighbouring tiles overlap by, on each side. Larger values "
                        "reduce the chance a cell gets cut by a hard tile boundary (more tiles "
                        "fully contain it, so stitching can average toward the better prediction) "
                        "but cost more compute (more pixels processed multiple times) and blend "
                        "predictions over a wider area. Not validated by IMES for this data -- "
                        "check overlay images for boundary artefacts before trusting a value.")
    p.add_argument("--no-overlays", action="store_true", help="don't save overlay PNGs")
    p.add_argument("--note", default="", help="free text stored in run_config.json")
    p.add_argument("--wandb", action="store_true", help="also log this run to Weights & Biases")
    p.add_argument("--wandb-project", default="cellfoundry", help="W&B project name")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B team/org name, so runs are visible to teammates, not just you "
                        "(or set the WANDB_ENTITY environment variable)")
    p.add_argument("--wandb-overlays", action="store_true",
                   help="ALSO upload overlay images to W&B (off by default: overlays are crops of "
                        "real IMES microscopy images, which must not leave our control without "
                        "IMES's explicit confirmation). Metrics-only logging (the default with "
                        "--wandb) uploads no image data.")
    a = p.parse_args(argv)

    pairs = find_pairs(a.group_dir, a.limit, a.pattern)
    if not pairs:
        raise SystemExit(f"No '*_image.*' files in {a.group_dir}")
    n_gt = sum(gt is not None for _, gt in pairs)
    print(f"group: {a.group_dir.name} | images: {len(pairs)} | with GT: {n_gt} | model: dinocell "
          f"| crop_size={a.crop_size} overlap_size={a.overlap_size}")

    device = _device()
    print(f"loading DINOCell from {a.weights} on {device}")
    model = build_model(a.weights, device)
    # Record the precision the checkpoint actually loaded in, the same way
    # run_batch.py does for the Cellpose-family models -- build_model() never
    # calls .half()/.bfloat16() or sets up autocast, so this model runs in
    # whatever dtype a.weights was saved in (most likely fp32, since the
    # notebook's own training run merged LoRA back into fp32 base weights
    # before saving -- but this is probed, not assumed, exactly because
    # three different precisions (IMES app BF16, Colab training FP16, this
    # checkpoint's actual saved dtype) are already known to disagree).
    model_dtype = str(next(model.parameters()).dtype)
    print(f"    dinocell loaded as {model_dtype}")
    pipeline = build_pipeline(model, device, a.crop_size, a.overlap_size)

    run_dir = a.out_root / "batch" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.group_dir.name}_dinocell"
    run_dir.mkdir(parents=True)
    config = {"args": {k: str(v) for k, v in vars(a).items()}, "versions": versions(device),
              "images": [img.name for img, _ in pairs], "started": dt.datetime.now().isoformat()}
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    run = None
    if a.wandb:
        import wandb  # optional dependency, only needed with --wandb

        wandb_args = {k: v for k, v in config["args"].items() if k not in ("group_dir", "out_root", "weights")}
        run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=run_dir.name,
                          config={**wandb_args, **config["versions"],
                                  "group": a.group_dir.name,
                                  "n_images": len(pairs), "n_with_gt": n_gt},
                          settings=wandb.Settings(x_disable_meta=True, x_disable_machine_info=True))

    rows = evaluate(pipeline, pairs, run_dir / "masks",
                     None if a.no_overlays else run_dir / "overlays", model_dtype)
    write_csv(run_dir / "per_image.csv", rows, PER_IMAGE_FIELDS)
    config["model_dtype"] = model_dtype
    if run is not None:
        wandb.config.update({"dinocell/model_dtype": model_dtype})
        for row in rows:
            numeric = {f"dinocell/{k}": v for k, v in row.items()
                       if isinstance(v, (int, float)) and v != ""}
            wandb.log(numeric)

    summary = summarise(rows)
    write_csv(run_dir / "summary.csv", summary)
    config["finished"] = dt.datetime.now().isoformat()
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if run is not None:
        try:
            if summary:
                wandb.log({"summary": wandb.Table(columns=list(summary[0].keys()),
                                                  data=[list(s.values()) for s in summary])})
            if a.wandb_overlays and not a.no_overlays:
                overlay_dir = run_dir / "overlays"
                if overlay_dir.is_dir():
                    for img_path in sorted(overlay_dir.glob("*.png"))[:10]:  # cap to avoid huge uploads
                        wandb.log({"overlay/dinocell": wandb.Image(str(img_path), caption=img_path.stem)})
        finally:
            run.finish()

    print("\n=== summary per condition (mean +- std over images; pooled = all cells together)")
    print(f"{'condition':<14} {'n':>3} {'F1@0.5':>13} {'pooled':>7} {'F1@0.75':>8} "
          f"{'cells pred/GT':>14} {'med |count err|':>15} {'s/img':>6}")
    for s in summary:
        print(f"{s['condition']:<14} {s['n_images']:>3} "
              f"{s['F1@0.5_mean']:>6} ±{s['F1@0.5_std']:<5} {s['pooled_F1@0.5']:>7} {s['F1@0.75_mean']:>8} "
              f"{str(s['n_pred_total']) + '/' + str(s['n_true_total']):>14} {s['abs_count_error_median']:>15} "
              f"{s['time_s_mean']:>6}")
    print(f"\nsaved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
