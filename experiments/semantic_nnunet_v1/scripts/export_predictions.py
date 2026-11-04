"""Copy nnU-Net predictions (case_XXX.png, 0 = background, 1 = fat) into the shared mask format
of PROTOCOL_porcine_bf_v1.md, so scripts/score_masks_v2.py scores them unchanged:
    <out-root>/predict/<YYYYmmdd-HHMMSS>_<subset>_<label>/masks/<label>/<name>_<label>_instances.tif

The whole fat area is written as ONE label (value 1). Area metrics (fat %, fat % error,
foreground Dice) are exact, because score_masks_v2.area_metrics() only looks at mask > 0.
Count error and F1 are NOT meaningful for this model (one "object" per image): report only the
area columns for nnU-Net.

Usage (environment: experiments/semantic_nnunet_v1):
    uv run --project <exp> python <exp>/scripts/export_predictions.py \\
        --pred-dir <nnUNetv2_predict output folder> \\
        --case-map <nnUNet_raw>/Dataset501_PorcineBFFat/case_map.json \\
        --group-dir <data>/Training_porcine_bf --subset test \\
        --out-root <results, outside repo>/nnunet --label nnunet-2d-raw-v1-s0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpdino_vitb_finetune_v1" / "scripts"))
from split_io import check_label, find_image, save_mask  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--case-map", type=Path, required=True)
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--subset", choices=("val", "test"), required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--label", required=True)
    a = p.parse_args(argv)

    import imageio.v3 as iio

    label = check_label(a.label)
    record = json.loads(a.case_map.read_text(encoding="utf-8"))
    cases = {c: v["name"] for c, v in record["cases"].items() if v["subset"] == a.subset}
    run_dir = a.out_root / "predict" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.subset}_{label}"

    missing = [c for c in cases if not (a.pred_dir / f"{c}.png").exists()]
    if missing:
        raise SystemExit(f"No prediction for {missing} in {a.pred_dir}")
    for case, name in sorted(cases.items()):
        pred = np.asarray(iio.imread(a.pred_dir / f"{case}.png"))
        if not set(np.unique(pred)) <= {0, 1}:
            raise SystemExit(f"{case}: expected labels 0/1, found {np.unique(pred)}")
        shape = iio.improps(find_image(a.group_dir, name)).shape
        out = save_mask(run_dir, label, name, pred.astype(np.uint16), shape)
        print(f"    {case} -> {out.name} (fat {100 * pred.mean():.1f} %)")

    (run_dir / "export_config.json").write_text(json.dumps({
        "pred_dir": str(a.pred_dir), "subset": a.subset, "label": label,
        "split": record["split"], "preprocess": record["preprocess"],
        "images": sorted(cases.values()), "note": "semantic model: only area metrics are meaningful",
        "exported": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"\n{len(cases)} masks -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
