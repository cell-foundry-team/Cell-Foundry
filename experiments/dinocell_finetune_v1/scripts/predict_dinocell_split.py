"""Segment ONE subset (train / val / test) of a split with a DINOCell checkpoint and save the masks
in the shared format, ready for scripts/score_masks_v2.py (same job as predict_split.py does for
the Cellpose models).

WHY THIS SCRIPT EXISTS (found by reading dinocell/pipeline.py, dinocell 0.74):
DINOCell's own pipeline (DINOFlowsSlidingWindowPipeline.run) ALWAYS (a) turns the image grey
(_resize) and (b) applies CLAHE (clip 1, 8 x 8 tiles) separately on EACH 512 x 512 CROP
(_preprocess, called per crop). There is no switch to turn this off. IMES's training notebook does
something different: CLAHE per colour channel on the WHOLE image, then crops. So a model trained
with finetune_dinocell.py --preprocess none / imes_clahe / grey_clahe would be scored on input it
never saw if it went through pipeline.run. This script therefore has two modes:

  default ("external"):  preprocess the WHOLE image with --preprocess (the same function as in
      training), then run the sliding window on the colour image with NO internal grey/CLAHE:
      per crop = ToTensor, ImageNet normalise, bicubic resize to 896 (exactly what training did,
      see patch_tensors in finetune_dinocell.py). Use this for every model trained with
      finetune_dinocell.py, with the SAME --preprocess as in training.
  --native-pipeline:  run DINOCell's pipeline exactly as shipped (grey + per-crop CLAHE). Use this
      for the ZERO-SHOT base checkpoint (DINOCell_demo_model.pt), and only with --preprocess none.
      This is what run_dinocell_batch.py does.

The sliding window itself (crop positions, overlap averaging, flows -> cells) is dinocell's own
code (SlidingWindowHelper, cells_from_flows); only the per-crop preprocessing is replaced.

Thresholds for turning flows into cells default to DINOCell's own (flow 0.0, cellprob 0.5; the
same values IMES's notebook uses for its validation F1).

Runs in the MAIN environment (dinocell==0.74), not cellpose-env. Output (new folder per run):
    <out-root>/predict/<YYYYmmdd-HHMMSS>_<subset>_<label>/masks/<label>/<name>_<label>_instances.tif
    <run>/predict_config.json
Then score (cellpose-env has the scorer's dependencies):
    uv run --project <repo>/cellpose-env python <repo>/scripts/score_masks_v2.py \\
        --batch-dirs <run folder> --group-dir <data>/Training_porcine_bf

NOT YET RUN on a real model. --subset test needs --confirm-test (test is used once, at the end).

Zero-shot base model, val:
    python predict_dinocell_split.py --split-file <repo>/splits/porcine_bf_v1.json --subset val \\
        --group-dir <data>/Training_porcine_bf --out-root <outside repo> \\
        --weights <repo>/.dinocell_models/DINOCell_demo_model.pt --preprocess none \\
        --native-pipeline --label dinocell-zeroshot
Fine-tuned model (weights file written by finetune_dinocell.py: <train run>/models/<label>):
    python predict_dinocell_split.py ... --weights <train run>/models/dinocell-ft-raw-v1-s0 \\
        --preprocess none --label dinocell-ft-raw-v1-s0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpdino_vitb_finetune_v1" / "scripts"))
from imes_preprocess import grey_clahe, imes_clahe  # noqa: E402
from split_io import SUBSETS, check_label, find_image, load_split, save_mask, sha256_file  # noqa: E402

PREPROCESS = {"none": lambda img: img, "imes_clahe": imes_clahe, "grey_clahe": grey_clahe}
CROP, OVERLAP = 512, 128   # same as run_dinocell_batch.py defaults; the model was built for crop 512


def build_model(weights: Path, device):
    """Same architecture arguments as the checkpoint (see run_dinocell_batch.build_model)."""
    import torch
    from dinocell.model import DINOCell

    model = DINOCell(dino_model=None, decoder_type="upsample", objective_type="flows",
                     use_dino_weights=False, patch_size=8, feat_size=64, crop_size=CROP, drop_rate=0.05,
                     dropout_in_encoder=True, finetune_vision=False, finetune_decoder=True,
                     finetune_prediction_head=True)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)   # a merged fine-tuned file has the plain layer names
    return model.to(device).eval()


def external_flows(pipeline, image: np.ndarray, device):
    """Sliding window over a COLOUR image with no internal grey/CLAHE. Returns [dx, dy, prob],
    each H x W, using dinocell's own SlidingWindowHelper for crop positions and overlap averaging."""
    import cv2
    import torch
    from torchvision import transforms

    h, w = image.shape[:2]
    if min(h, w) < pipeline.crop_size:
        raise SystemExit(f"image {h}x{w} is smaller than the crop size {pipeline.crop_size}; "
                         "this script does not resize small images")
    helper = pipeline.sliding_window_helper
    # the helper only accepts 2-D arrays, and the crop positions depend only on the image size:
    # ask it for the regions on a dummy 2-D array, then cut the colour image there
    _, regions = helper.seperate_into_crops_v2(np.zeros((h, w), dtype=np.uint8))
    to_input = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                   transforms.Resize((896, 896), interpolation=transforms.InterpolationMode.BICUBIC)])
    preds = []
    for x0, y0, x1, y1 in regions:
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        with torch.no_grad():
            out = pipeline.model(x=to_input(crop).unsqueeze(0).to(device))
        preds.append([cv2.resize(o[0].cpu().numpy(), (pipeline.crop_size, pipeline.crop_size)) for o in out])
    preds = np.array(preds)
    return [helper.combine_crops_v2((h, w), preds[:, i], regions) for i in range(preds.shape[1])]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--subset", choices=SUBSETS, required=True)
    p.add_argument("--confirm-test", action="store_true", help="required for --subset test")
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--preprocess", choices=sorted(PREPROCESS), default="none")
    p.add_argument("--native-pipeline", action="store_true",
                   help="DINOCell's own grey + per-crop CLAHE (zero-shot base checkpoint only)")
    p.add_argument("--flow-threshold", type=float, default=0.0)
    p.add_argument("--cellprob-threshold", type=float, default=0.5)
    p.add_argument("--note", default="")
    a = p.parse_args(argv)

    if a.subset == "test" and not a.confirm_test:
        p.error("--subset test needs --confirm-test (the test set is used ONCE, at the end)")
    if a.native_pipeline and a.preprocess != "none":
        p.error("--native-pipeline applies DINOCell's own preprocessing; combine it only with --preprocess none")
    label = check_label(a.label)
    split = load_split(a.split_file)
    if split["group_dir_name"] != a.group_dir.name:
        raise SystemExit(f"Split was made for '{split['group_dir_name']}', not '{a.group_dir.name}'.")
    names = split["splits"][a.subset]
    images = {n: find_image(a.group_dir, n) for n in names}   # stops early if one is missing
    if not a.weights.is_file():
        raise SystemExit(f"weights not found: {a.weights}")

    import torch
    from cellpose import io as cpio
    from dinocell.pipeline import get_pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    mode = "native pipeline (grey + per-crop CLAHE)" if a.native_pipeline else f"external, preprocess {a.preprocess}"
    print(f"split {split['name']} ({split['fingerprint']}) | subset {a.subset} | {len(names)} images | "
          f"device {device} | {mode} | thresholds flow {a.flow_threshold} cellprob {a.cellprob_threshold}")
    model = build_model(a.weights, device)
    dtype = str(next(model.parameters()).dtype)
    pipeline = get_pipeline("flows", model, device, crop_size=CROP, overlap_size=OVERLAP)

    run_dir = a.out_root / "predict" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.subset}_{label}"
    run_dir.mkdir(parents=True)   # fails if it exists: never overwrite
    config = {"args": {k: str(v) for k, v in vars(a).items()},
              "split": {"name": split["name"], "fingerprint": split["fingerprint"], "subset": a.subset},
              "images": names, "weights_sha256": sha256_file(a.weights), "model_dtype": dtype,
              "mode": mode, "started": dt.datetime.now().isoformat(timespec="seconds")}
    (run_dir / "predict_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    times = []
    for i, name in enumerate(names, 1):
        raw = cpio.imread(str(images[name]))
        t0 = time.time()
        if a.native_pipeline:
            labels, preds, _ = pipeline.run(raw, return_raw_preds=True)
            if (a.flow_threshold, a.cellprob_threshold) != (0.0, 0.5):
                # non-default thresholds: rebuild the cells from the same raw flows (no 2nd model pass)
                labels = pipeline.cells_from_flows(preds[0], preds[1], preds[2],
                                                   flow_threshold=a.flow_threshold,
                                                   cellprob_threshold=a.cellprob_threshold)
        else:
            flows = external_flows(pipeline, PREPROCESS[a.preprocess](raw), device)
            labels = pipeline.cells_from_flows(flows[0], flows[1], flows[2],
                                               flow_threshold=a.flow_threshold,
                                               cellprob_threshold=a.cellprob_threshold)
        times.append(time.time() - t0)
        labels = np.asarray(labels)
        if labels.shape != raw.shape[:2]:
            raise SystemExit(f"{name}: mask {labels.shape} != image {raw.shape[:2]}")
        save_mask(run_dir, label, name, labels, raw.shape)
        print(f"    [{i}/{len(names)}] {name}: {int(labels.max())} objects ({times[-1]:.1f} s)")

    config.update(finished=dt.datetime.now().isoformat(timespec="seconds"),
                  seconds_per_image=round(float(np.mean(times)), 2))
    (run_dir / "predict_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nall {len(names)} masks written -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
