"""Fine-tune DINOCell (LoRA) on the TRAIN images of a split, ported from IMES's own Colab
notebook (colab_dinocell_adipocyte_porcine.ipynb) to a script that uses the shared
porcine_bf_v1-style split and swappable preprocessing, the same way
experiments/cpdino_vitb_finetune_v1/scripts/finetune_cpdino.py does for cpdino-vitb.

Ported as-is from the notebook (not re-derived, not redesigned): LoRA insertion (LoRALinear /
install_lora / merge_lora), the flow-target generation (Cellpose dynamics.labels_to_flows), the
cell-centred random-crop sampler, the segmentation loss (BCE + weighted Dice + flow MSE), AMP
precision handling (bf16 falls back to fp16 if unsupported), checkpoint selection by validation
Dice. See local-only/DINOCell_Finetune_Scoping.md (Pavin's personal notes, not part of this repo)
for the full cell-by-cell mapping back to the notebook.

Changed from the notebook, on purpose:
  * Split comes from a shared split JSON (split_io.load_split), not the notebook's own internal
    80/20 shuffle. Train only on splits.train; splits.val is used only to print progress; the
    notebook's own 12-fixed-patches validation is kept (see "Validation" below) rather than
    switched to whole-image scoring, which run_dinocell_batch.py already does separately.
  * Preprocessing is a swappable --preprocess argument (none / imes_clahe / grey_clahe), the same
    PREPROCESS dict pattern as finetune_cpdino.py, instead of CLAHE being hard-coded inside
    prepare_pair() the way the notebook has it. imes_clahe and grey_clahe are IMPORTED from
    experiments/cpdino_vitb_finetune_v1/scripts/imes_preprocess.py (not reimplemented here), so
    "imes_clahe" means the identical pixel operation for both models.
  * argparse instead of the notebook's TrainingConfig dataclass with hardcoded paths.
  * Output in a new timestamped folder per run (never overwritten), same convention as
    finetune_cpdino.py and run_dinocell_batch.py.

MUST run in the repo's MAIN environment (dinocell==0.74), NOT cellpose-env (cellpose==4.2.1.1
breaks dinocell's flow-target generation). See cellpose-env/README.md for why the two can't be
installed together. To score a trained checkpoint afterwards, use run_dinocell_batch.py's
--weights flag with the merged .pt this script writes (scoring runs fine in either environment
once the mask is a plain .tif file).

NOT YET RUN by anyone. Written from a full read of the notebook (2026-11-05); not executed, not
tested end to end. The notebook's own result (best validation Dice 0.767 at step 750, instance
F1 0.3905 -- see project_cellfoundry_ground_truth.md) is the only real reference result that
exists for DINOCell fine-tuning; nothing from this script has been compared against it yet.

Preprocessing CLI examples (once this has actually been run once successfully):
    uv run python finetune_dinocell.py --split-file ../../../splits/porcine_bf_v1.json \\
        --group-dir "<data>/Training_porcine_bf" --out-root "<results, OUTSIDE the repo>" \\
        --preprocess none --label dinocell-ft-raw-v1-s0 --seed 0

    uv run python finetune_dinocell.py ... --preprocess imes_clahe --label dinocell-ft-clahe-v1-s0 --seed 0
    uv run python finetune_dinocell.py ... --preprocess grey_clahe --label dinocell-ft-greyclahe-v1-s0 --seed 0

Run the same --preprocess value 3 times (--seed 0, 1, 2) per setting, same rule as
PROTOCOL_porcine_bf_v1.md -- mean +/- SD over seeds, decide the winner on val only, test once.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpdino_vitb_finetune_v1" / "scripts"))
from imes_preprocess import grey_clahe, imes_clahe  # noqa: E402
from split_io import check_label, find_gt, find_image, load_split, sha256_file  # noqa: E402

PREPROCESS = {"none": lambda img: img, "imes_clahe": imes_clahe, "grey_clahe": grey_clahe}


def autocast_context(device, dtype):
    """torch.autocast("cuda", ...) on the real target (CUDA); a no-op on MPS/CPU, since MPS
    autocast support differs from CUDA's and this script's MPS path (see main()) is a local
    sanity check only, not the real precision behaviour -- it always runs fp32 there anyway."""
    import torch

    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype, enabled=dtype is not None)
    import contextlib
    return contextlib.nullcontext()


# --------------------------------------------------------------------------- config / LoRA
@dataclass
class Config:
    """Only the fields prepare_pair/load_patch/segmentation_loss/run_training actually use.
    Defaults are the notebook's own CONFIG values (colab_dinocell_adipocyte_porcine.ipynb
    cell 7), except steps=1000 kept, preprocess split out, paths replaced by CLI args."""
    crop_size: int = 512
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    amp_precision: str = "fp16"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_last_blocks: int = 4
    gamma_probability: float = 0.15
    gamma_max: float = 1.10
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    validate_every: int = 50
    validation_patches: int = 12
    instance_metrics: bool = True
    instance_iou_threshold: float = 0.5
    save_best_checkpoint: bool = True
    positive_weight_cap: float = 10.0
    dice_weight: float = 0.5
    seed: int = 0


class LoRALinear:  # placeholder name kept for the docstring cross-reference below; real class follows
    pass


def install_lora(model, config: Config):
    """Ported unchanged from the notebook (cell 13): freeze everything except the decoder, the
    prediction head, and LoRA adapters on the last `lora_last_blocks` attention blocks."""
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
            super().__init__()
            self.base = base
            self.scale = float(alpha) / rank
            self.dropout = nn.Dropout(dropout)
            self.lora_down = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
            self.lora_up = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
            self.base.requires_grad_(False)

        def forward(self, inputs):
            return self.base(inputs) + self.lora_up(self.lora_down(self.dropout(inputs))) * self.scale

        def merged(self):
            out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
            out = out.to(self.base.weight)
            with torch.no_grad():
                out.weight.copy_(self.base.weight + (self.lora_up.weight @ self.lora_down.weight) * self.scale)
                if out.bias is not None:
                    out.bias.copy_(self.base.bias)
            return out

    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.prediction_head.requires_grad_(True)
    blocks = model.encoder.dino_encoder.blocks
    selected = range(max(0, len(blocks) - config.lora_last_blocks), len(blocks))
    names = []
    for block_index in selected:
        attention = blocks[block_index].attn
        for target in ("qkv", "proj"):
            base = getattr(attention, target)
            setattr(attention, target, LoRALinear(base, config.lora_rank, config.lora_alpha, config.lora_dropout))
            names.append(f"encoder.dino_encoder.blocks.{block_index}.attn.{target}")
    model._lora_cls = LoRALinear  # stash the class so merge_lora can isinstance-check it below
    return names


def merge_lora(model):
    """Ported unchanged from the notebook: fold every LoRA update back into its base layer."""
    cls = model._lora_cls
    replacements = [(name, module) for name, module in model.named_modules() if isinstance(module, cls)]
    merged = []
    for name, module in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, module.merged())
        merged.append(name)
    return merged


def build_model(weights_path: Path, config: Config, device):
    """Ported from the notebook cell 23. Loads the base checkpoint strict, then installs LoRA."""
    import torch
    from dinocell.model import DINOCell

    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOCell weights not found: {weights_path}")
    model = DINOCell(dino_model=None, decoder_type="upsample", objective_type="flows", use_dino_weights=False,
                     patch_size=8, feat_size=64, crop_size=512, drop_rate=0.05, dropout_in_encoder=True,
                     finetune_vision=False, finetune_decoder=True, finetune_prediction_head=True)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    names = install_lora(model, config)
    return model.to(device), names


# --------------------------------------------------------------------------- data (ported from the notebook)
def prepare_pair(image: np.ndarray, labels: np.ndarray, preprocess: str) -> dict:
    """Ported from the notebook cell 17, with the CLAHE step pulled out into --preprocess
    (PREPROCESS[preprocess]) instead of being hard-coded. Everything else (3-channel conversion,
    label remapping, Cellpose flow-target generation) is a DINOCell requirement, not a
    preprocessing CHOICE, so it stays fixed regardless of --preprocess."""
    import torch
    from cellpose import dynamics

    image, labels = np.asarray(image), np.asarray(labels)
    if image.shape[:2] != labels.shape[:2]:
        raise ValueError("Mask and image have different sizes.")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    identifiers = np.unique(np.concatenate(([0], labels.ravel())))
    labels = np.searchsorted(identifiers, labels).astype(np.int32)

    image = PREPROCESS[preprocess](image.astype(np.uint8) if image.dtype != np.uint8 else image)

    if labels.max() == 0:
        flows = np.zeros((3, *labels.shape), dtype=np.float32)
    else:
        generated = np.asarray(
            dynamics.labels_to_flows([labels], files=None, device=torch.device("cpu"))[0],
            dtype=np.float32,
        )
        flows = generated[[3, 2, 1]]  # reorder to dx, dy, cell-probability (notebook's own comment)

    return {"image": image, "labels": labels, "flows": flows}


def prepare_dataset(pairs, directory: Path, prefix: str, preprocess: str):
    """Ported from the notebook cell 17: cache each image's preprocessed+flow result to .npz
    once, so load_patch (called many times per image across training) doesn't redo CLAHE/flows
    every step."""
    directory.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, (name, image_path, mask_path) in enumerate(pairs):
        from split_io import load_labels

        import tifffile
        from cellpose import io as cpio

        raw = cpio.imread(str(image_path))
        gt = load_labels(mask_path)
        sample = prepare_pair(raw, gt, preprocess)
        path = directory / f"{prefix}_{index:04d}.npz"
        np.savez(path, **sample)
        prepared.append(path)
        print(f"    prepared {prefix} {index + 1}/{len(pairs)}: {name} ({int(sample['labels'].max())} cells)")
    return prepared


def load_patch(path: Path, crop_size: int, rng: np.random.Generator) -> dict:
    """Ported unchanged from the notebook cell 19: cell-centred random crop (same logic
    tools/inspect_annotations.py simulates -- see Study_Log_Topic4_Patches.md, local-only/)."""
    with np.load(path, allow_pickle=False) as data:
        image, labels, flows = data["image"], data["labels"], data["flows"]
    height, width = labels.shape
    bottom, right = max(0, crop_size - height), max(0, crop_size - width)
    image = np.pad(image, ((0, bottom), (0, right), (0, 0)), mode="edge")
    labels = np.pad(labels, ((0, bottom), (0, right)))
    flows = np.pad(flows, ((0, 0), (0, bottom), (0, right)))
    foreground = np.argwhere(labels > 0)
    if len(foreground):
        center_y, center_x = foreground[int(rng.integers(len(foreground)))]
        offset_y, offset_x = rng.integers(crop_size // 4, 3 * crop_size // 4 + 1, size=2)
        top = int(np.clip(center_y - offset_y, 0, labels.shape[0] - crop_size))
        left = int(np.clip(center_x - offset_x, 0, labels.shape[1] - crop_size))
    else:
        top = int(rng.integers(labels.shape[0] - crop_size + 1))
        left = int(rng.integers(labels.shape[1] - crop_size + 1))
    return {"image": image[top:top + crop_size, left:left + crop_size].copy(),
            "labels": labels[top:top + crop_size, left:left + crop_size].copy(),
            "flows": flows[:, top:top + crop_size, left:left + crop_size].copy()}


def build_augmentation(config: Config, device):
    """Ported unchanged from the notebook cell 19."""
    if config.gamma_probability <= 0 or config.gamma_max <= 1:
        return None
    from kornia.augmentation import RandomGamma
    return RandomGamma((1.0, config.gamma_max), p=config.gamma_probability).to(device)


def patch_tensors(sample, device, config: Config, *, augment=False, gamma=None, rng=None):
    """Ported unchanged from the notebook cell 19: flip augmentation (flips flow signs to
    match), optional gamma jitter, ImageNet normalisation, resize to DINOv2's 896x896 input."""
    import torch
    from torchvision import transforms

    image = torch.from_numpy(np.ascontiguousarray(sample["image"])).permute(2, 0, 1)[None].to(
        device=device, dtype=torch.float32) / 255.0
    targets = torch.from_numpy(np.ascontiguousarray(sample["flows"]))[None].to(device=device, dtype=torch.float32)
    applied = []
    if augment:
        for probability, axis, channel, name in (
            (config.horizontal_flip_probability, -1, 0, "horizontal_flip"),
            (config.vertical_flip_probability, -2, 1, "vertical_flip"),
        ):
            if rng.random() < probability:
                image, targets = image.flip(axis), targets.flip(axis)
                targets[:, channel].neg_()
                applied.append(name)
        if gamma is not None:
            image = gamma(image).clamp(0, 1)
            if bool(gamma._params["batch_prob"].any().item()):
                applied.append("gamma")
    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
    image = transforms.Resize((896, 896), interpolation=transforms.InterpolationMode.BICUBIC)(image)
    return image, targets, applied


# --------------------------------------------------------------------------- loss / metrics (ported unchanged)
def segmentation_loss(predictions, targets, config: Config):
    flow_dx, flow_dy, probability = [p.float() for p in predictions]
    truth = targets[:, 2]
    probability = probability.clamp(1e-6, 1 - 1e-6)
    foreground = truth.sum()
    positive_weight = ((truth.numel() - foreground) / foreground.clamp_min(1)).clamp(1, config.positive_weight_cap)
    loss_bce = -(positive_weight * truth * probability.log() + (1 - truth) * (1 - probability).log1p().neg()).mean()
    loss_dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
    weights = targets.new_full(truth.shape, 0.05)
    weights[truth > 0.5] = 1.0
    loss_dx = ((flow_dx - targets[:, 0]).square() * weights).mean()
    loss_dy = ((flow_dy - targets[:, 1]).square() * weights).mean()
    total = loss_dx + loss_dy + loss_bce + config.dice_weight * loss_dice
    return total, {"loss": total, "loss_flow_dx": loss_dx, "loss_flow_dy": loss_dy,
                   "loss_cell_prob": loss_bce, "loss_dice": loss_dice}


def binary_metrics(predictions, targets):
    predicted, truth = predictions[2].detach() > 0.5, targets[:, 2] > 0.5
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    return {"dice": 2 * tp / max(1, 2 * tp + fp + fn) if tp + fp + fn else 1.0,
            "iou": tp / max(1, tp + fp + fn) if tp + fp + fn else 1.0,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


def instance_counts(predictions, labels, config: Config):
    import torch
    from cellpose import dynamics, metrics

    flow_dx, flow_dy, probability = [p.detach().float().cpu().numpy()[0] for p in predictions]
    predicted = dynamics.compute_masks(np.stack([flow_dy, flow_dx]), probability, cellprob_threshold=0.5,
                                       flow_threshold=0.0, min_size=15, max_size_fraction=0.95,
                                       device=torch.device("cpu"))
    if isinstance(predicted, tuple):
        predicted = predicted[0]
    identifiers = np.unique(np.concatenate(([0], labels.ravel())))
    truth = np.searchsorted(identifiers, labels).astype(np.int32)
    if truth.max() == 0 or np.max(predicted) == 0:
        return 0, int(np.max(predicted)), int(truth.max())
    _, tp, fp, fn = metrics.average_precision(truth, predicted, threshold=[config.instance_iou_threshold])
    return int(np.ravel(tp)[0]), int(np.ravel(fp)[0]), int(np.ravel(fn)[0])


def evaluate(model, prepared, config: Config, device, amp_dtype):
    """Ported unchanged from the notebook cell 21: the SAME 12 fixed patches every time (seeded
    by config.seed + 10000 + index), not whole images. This reproduces the known validation blind
    spot discussed in Study_Log_Topic4_Patches.md (local-only/): at most ~25% of each val image is
    ever scored here, and only near cells. Kept as-is for a faithful port; whole-image validation
    is a separate, larger change (see the scoping doc's open question), not done here."""
    import torch

    was_training = model.training
    model.eval()
    rows, counts = [], np.zeros(3, dtype=np.int64)
    try:
        with torch.no_grad():
            for index in range(config.validation_patches):
                sample = load_patch(prepared[index % len(prepared)], config.crop_size,
                                    np.random.default_rng(config.seed + 10000 + index))
                image, targets, _ = patch_tensors(sample, device, config)
                with autocast_context(device, amp_dtype):
                    predictions = model(image)
                _, components = segmentation_loss(predictions, targets, config)
                rows.append({**{k: float(v.detach()) for k, v in components.items()}, **binary_metrics(predictions, targets)})
                if config.instance_metrics:
                    counts += np.array(instance_counts(predictions, sample["labels"], config))
    finally:
        model.train(was_training)
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    if config.instance_metrics:
        tp, fp, fn = [int(c) for c in counts]
        result.update(instance_precision=tp / max(1, tp + fp), instance_recall=tp / max(1, tp + fn),
                      instance_f1=2 * tp / max(1, 2 * tp + fp + fn) if counts.sum() else 1.0,
                      instance_tp=tp, instance_fp=fp, instance_fn=fn)
    return result


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-file", type=Path, required=True, help="JSON made by make_split.py (porcine_bf_v1)")
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--label", required=True, help="name of the fine-tuned model, e.g. dinocell-ft-raw-v1-s0")
    p.add_argument("--weights", type=Path, default=Path.home() / "Documents" / "Cell Foundry" / "Cell-Foundry"
                   / ".dinocell_models" / "DINOCell_demo_model.pt", help="base DINOCell checkpoint")
    p.add_argument("--preprocess", choices=sorted(PREPROCESS), default="none")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--amp-precision", choices=("bf16", "fp16", "fp32"), default="fp16")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-last-blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validate-every", type=int, default=50)
    a = p.parse_args(argv)

    label = check_label(a.label)
    split = load_split(a.split_file)
    train_names, val_names = split["splits"]["train"], split["splits"]["val"]

    import tempfile

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # TEMPORARY, for a local sanity check only (no CUDA on this machine): MPS doesn't
        # support autocast/GradScaler the same way CUDA does, so AMP is forced off (fp32) below
        # and this path is meaningfully slower/different from the real CAIRO run. Remove this
        # branch, or gate it behind an explicit --allow-mps flag, before this is used for anything
        # beyond "does the code run at all".
        device = torch.device("mps")
        print("WARNING: no CUDA GPU found; falling back to MPS for a local sanity check only. "
              "This is NOT representative of the real (CAIRO/CUDA) run -- AMP is disabled.")
    else:
        raise SystemExit("No CUDA or MPS device found; DINOCell training on plain CPU is impractical.")

    config = Config(amp_precision=a.amp_precision, lora_rank=a.lora_rank,
                    lora_last_blocks=a.lora_last_blocks, seed=a.seed, validate_every=a.validate_every)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)

    precision = config.amp_precision
    if device.type != "cuda":
        precision = "fp32"  # autocast/GradScaler below are CUDA-specific; MPS sanity-check path only
    elif precision == "bf16" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
        print("This GPU does not support BF16; using FP16 with gradient scaling.")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]

    run_dir = a.out_root / "train" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{label}"
    run_dir.mkdir(parents=True)
    print(f"split {split['name']} ({split['fingerprint']}) | train {len(train_names)} | val {len(val_names)} "
          f"| preprocess {a.preprocess} | steps {a.steps} | precision {precision}")

    train_pairs = [(n, find_image(a.group_dir, n), find_gt(a.group_dir, n)) for n in train_names]
    val_pairs = [(n, find_image(a.group_dir, n), find_gt(a.group_dir, n)) for n in val_names]

    config_record = {"args": {k: str(v) for k, v in vars(a).items()},
                     "split": {"name": split["name"], "fingerprint": split["fingerprint"]},
                     "train_images": train_names, "val_images": val_names,
                     "amp_precision": precision, "started": dt.datetime.now().isoformat(timespec="seconds")}
    (run_dir / "train_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="dinocell_prepared_") as working:
        print("preparing training images (CLAHE/flows cached once per image)...")
        training = prepare_dataset(train_pairs, Path(working), "train", a.preprocess)
        validation = prepare_dataset(val_pairs, Path(working), "val", a.preprocess)

        model, lora_modules = build_model(a.weights, config, device)
        base_weights_sha256 = sha256_file(a.weights)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    base weights sha256 {base_weights_sha256[:12]} | {trainable_params:,} / {total_params:,} "
              f"trainable ({100 * trainable_params / total_params:.1f}%) | LoRA on {len(lora_modules)} layers")

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.999))
        # GradScaler is CUDA-only; the MPS sanity-check path runs fp32 (see precision= above), so
        # scaling is always disabled there regardless of the --amp-precision flag's value.
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
        gamma = build_augmentation(config, device)

        metrics_path = run_dir / "metrics.jsonl"
        selection = {"best_validation_dice": None, "best_step": None}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        for step in range(1, a.steps + 1):
            sample = load_patch(training[int(rng.integers(len(training)))], config.crop_size, rng)
            image, targets, applied = patch_tensors(sample, device, config, augment=True, gamma=gamma, rng=rng)
            with autocast_context(device, amp_dtype):
                predictions = model(image)
            loss, components = segmentation_loss(predictions, targets, config)
            scaler.scale(loss / config.grad_accum_steps).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            record = {"step": step, "train": {k: float(v.detach()) for k, v in components.items()},
                      "applied_augmentations": applied}
            if config.validate_every and step % config.validate_every == 0:
                record["valid"] = evaluate(model, validation, config, device, amp_dtype)
                score = float(record["valid"]["dice"])
                if selection["best_validation_dice"] is None or score > selection["best_validation_dice"]:
                    selection.update(best_validation_dice=score, best_step=step)
                    torch.save(model.state_dict(), run_dir / "best_raw.pt")  # pre-merge; merged at export time
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            if step == 1 or step % 25 == 0 or step == a.steps:
                msg = f"step {step}/{a.steps} loss {record['train']['loss']:.4f}"
                if "valid" in record:
                    msg += f" | valid dice {record['valid']['dice']:.4f} instance_f1 {record['valid'].get('instance_f1', float('nan')):.4f}"
                print(msg, flush=True)

        seconds = time.time() - t0
        # MPS doesn't track peak allocation the same way CUDA does; float("nan") here is a
        # known gap of the MPS sanity-check path, not a real measurement of anything.
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")

        if config.save_best_checkpoint and selection["best_step"] is not None:
            model.load_state_dict(torch.load(run_dir / "best_raw.pt", map_location=device, weights_only=True))
            export_step = selection["best_step"]
        else:
            export_step = a.steps
        model.to("cpu")
        merged_modules = merge_lora(model)
        destination = run_dir / "models" / label
        destination.parent.mkdir(exist_ok=True)
        tmp = destination.with_suffix(".tmp")
        torch.save(model.state_dict(), tmp)
        tmp.replace(destination)

        config_record.update(train_seconds=round(seconds, 1), peak_gpu_gb=round(peak_gb, 2),
                             export_step=export_step, best_validation_dice=selection["best_validation_dice"],
                             base_weights_sha256=base_weights_sha256, lora_modules=lora_modules,
                             merged_lora_modules=merged_modules, total_parameters=total_params,
                             trainable_parameters=trainable_params, weights_file=str(destination),
                             weights_sha256=sha256_file(destination),
                             finished=dt.datetime.now().isoformat(timespec="seconds"))
        (run_dir / "train_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

        print(f"\ntrained {a.steps} steps in {seconds / 60:.1f} min, peak GPU memory {peak_gb:.2f} GB")
        print(f"best validation Dice {selection['best_validation_dice']} at step {selection['best_step']} "
              f"(exported step {export_step})")
        print(f"weights: {destination}")
        print(f"run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
