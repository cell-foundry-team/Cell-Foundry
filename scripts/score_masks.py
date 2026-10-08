"""Score any model's predicted label image against an IMES ground-truth mask.

Uses the same metric code as ``run_cpsam.py`` (Cellpose ``metrics.average_precision``:
one-to-one IoU matching, a match counts when IoU >= t), so every model is scored
identically.

    F1 = 2TP / (2TP + FP + FN)
    SA = TP / (TP + FP + FN)   (Cellpose calls this "AP"; it is NOT COCO AP)
    count error = (n_pred - n_true) / n_true

Negative images (no cells in the ground truth) have no F1/SA (0/0). For them the
script reports the number of false detections and their mean area instead.

Usage (in the ``cellfoundry`` env, from the repo folder):
    python scripts\\score_masks.py --pred "<..._microsam_instances.tif>" --model microsam_vit_b_lm_ais
    python scripts\\score_masks.py --pred "<pred.tif>" --gt "<..._instances.tif>" --model sam3_text

If ``--gt`` is omitted, the GT is looked up in the dataset folder (``--data-root``,
searched recursively) from the prediction's name: ``<name>_<model>_instances.tif``
-> ``<name>_instances.{tif,tiff,png}``.

Writes ``<pred name minus _instances>_metrics.txt`` next to the prediction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage import io as skio
from skimage.segmentation import relabel_sequential

DEFAULT_DATA_ROOT = Path(r"C:\Users\Priyam\Desktop\Project\Cloud")
THRESHOLDS = [0.5, 0.75, 0.9]
GT_EXTS = (".tif", ".tiff", ".png")


def load_labels(path: Path) -> np.ndarray:
    """Load a label image as a 2D int array with labels 1..N (0 = background)."""
    if path.suffix.lower() in (".jpg", ".jpeg"):
        raise ValueError(f"Refusing lossy JPEG mask: {path}")
    arr = tifffile.imread(path) if path.suffix.lower() in (".tif", ".tiff") else skio.imread(path)
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D label image, got shape {arr.shape}: {path}")
    labels, _, _ = relabel_sequential(arr.astype(np.int64))
    return labels.astype(np.int32)


def find_gt(pred: Path, model: str, data_root: Path) -> Path:
    """Derive the GT filename from the prediction filename and search the dataset for it."""
    stem = pred.stem
    suffix = f"_{model}_instances"
    if not stem.endswith(suffix):
        raise ValueError(f"Prediction name must end with '{suffix}' to auto-find GT; pass --gt instead.")
    base = stem[: -len(suffix)]
    for ext in GT_EXTS:
        hits = list(data_root.rglob(f"{base}_instances{ext}"))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"Several GT files match '{base}': {hits}. Pass --gt.")
    raise FileNotFoundError(f"No GT '{base}_instances.(tif|tiff|png)' under {data_root}. Pass --gt.")


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Compute matching metrics; handles negative images (n_true == 0)."""
    from cellpose import metrics  # imported here so --help works without cellpose

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs prediction {pred.shape}")
    n_true, n_pred = int(gt.max()), int(pred.max())
    out: dict = {"n_true": n_true, "n_pred": n_pred, "shape": gt.shape}

    if n_true == 0:  # negative image: F1/SA/count error are undefined
        areas = np.bincount(pred.ravel())[1:] if n_pred else np.array([])
        out["negative_image"] = True
        out["false_detections"] = n_pred
        out["mean_false_area_px"] = float(areas.mean()) if n_pred else 0.0
        return out

    sa, tp, fp, fn = metrics.average_precision([gt], [pred], threshold=THRESHOLDS)
    out["negative_image"] = False
    out["count_error"] = (n_pred - n_true) / n_true
    out["per_t"] = {}
    for i, t in enumerate(THRESHOLDS):
        TP, FP, FN = int(tp[0, i]), int(fp[0, i]), int(fn[0, i])
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else float("nan")
        out["per_t"][t] = {"F1": f1, "SA": float(sa[0, i]), "TP": TP, "FP": FP, "FN": FN}
    return out


def format_report(res: dict, pred: Path, gt: Path, model: str, note: str) -> str:
    lines = [
        f"date: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"model: {model}",
        f"prediction: {pred}",
        f"ground_truth: {gt}",
        f"image_shape: {res['shape']}",
        f"n_true: {res['n_true']}",
        f"n_pred: {res['n_pred']}",
    ]
    if note:
        lines.append(f"note: {note}")
    if res["negative_image"]:
        lines += [
            "negative_image: yes (no GT cells; F1/SA undefined)",
            f"false_detections: {res['false_detections']}",
            f"mean_false_area_px: {res['mean_false_area_px']:.1f}",
        ]
    else:
        lines.append(f"count_error: {res['count_error']:+.1%}")
        for t, m in res["per_t"].items():
            lines.append(
                f"IoU>={t}: F1={m['F1']:.3f}  SA={m['SA']:.3f}  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}"
            )
    try:
        import cellpose

        lines.append(f"metric_code: cellpose {cellpose.version} metrics.average_precision")
    except Exception:  # pragma: no cover
        pass
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, type=Path, help="predicted label image (.tif/.png)")
    p.add_argument("--model", required=True, help="model label used in the filename, e.g. microsam_vit_b_lm_ais")
    p.add_argument("--gt", type=Path, help="ground-truth label image; auto-found if omitted")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="dataset root for GT lookup")
    p.add_argument("--note", default="", help="free text saved in the report (settings, weights version...)")
    a = p.parse_args(argv)

    gt_path = a.gt or find_gt(a.pred, a.model, a.data_root)
    res = score(load_labels(gt_path), load_labels(a.pred))
    report = format_report(res, a.pred, gt_path, a.model, a.note)
    print(report)

    stem = a.pred.stem[: -len("_instances")] if a.pred.stem.endswith("_instances") else a.pred.stem
    out = a.pred.with_name(f"{stem}_metrics.txt")
    out.write_text(report, encoding="utf-8")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
