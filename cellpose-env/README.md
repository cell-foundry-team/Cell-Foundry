# cellpose-env

Separate Python environment for running `scripts/run_batch.py`,
`run_cpsam.py`, and `score_masks.py` (Cellpose-family zero-shot
benchmarking: `cpdino`, `cpdino-vitb`).

## Why this is its own environment

The main project depends on `dinocell==0.74`, which requires exactly
`cellpose==4.0.9`. The DINOv3-backed Cellpose models (`cpdino`,
`cpdino-vitb`) need `cellpose==4.2.1.1` instead. The two versions can't
be installed together, so this folder is a standalone `uv` project
(`uv init --no-workspace`), independent of the parent project's
dependencies.

**Note: DINOCell (the main project's dependency) and `cpdino`/`cpdino-vitb`
(Cellpose's DINOv3-backed models, benchmarked here) are two different
models, not the same thing under different names.** DINOCell is a
separate package (DINOv2-based, per its own repo); `cpdino`/`cpdino-vitb`
are built into Cellpose itself using Meta's newer DINOv3 backbone. Both
get compared against each other as candidates, but they're architecturally
distinct — worth keeping straight when discussing results with the team
or IMES (e.g. the baseline Nicole refers to as "DINO-SAM" may be one or
the other; confirm which before assuming).

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
cd cellpose-env
uv sync
```

Without uv (e.g. inside an existing conda environment):

```bash
cd cellpose-env
pip install -r requirements.txt
```

`requirements.txt` is exported from `uv.lock` so both routes install the
exact same package versions. If you change dependencies via `uv add`,
regenerate it with:

```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

**After `uv sync` (or `pip install`), check the GPU is actually active:**

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available() or torch.backends.mps.is_available())"
```

This should print `True` at the end. On Windows specifically, PyPI's
plain `torch` wheel is CPU-only — `pyproject.toml` pins a CUDA build from
a separate index for `sys_platform == 'win32'` so `uv sync` gets this
right automatically. If you install via plain `pip` on Windows instead,
make sure the `--extra-index-url` line at the top of `requirements.txt`
is honored (pip reads it directly from the file, no extra flag needed),
or cpdino/cpdino-vitb will silently run on CPU instead of GPU.

## Running a benchmark

```bash
uv run python ../scripts/run_batch.py --group-dir "../data/Training_porcine_bf"
```

Add `--wandb` to also log the run to the shared `cellfoundry` Weights &
Biases project (set `--wandb-entity` or the `WANDB_ENTITY` environment
variable to the team org so teammates can see it, not just you).

**Overlay images are not uploaded to W&B by default** — they're crops of
real IMES microscopy data, which must stay off third-party services
unless IMES explicitly confirms it's fine. Pass `--wandb-overlays` only
if that's been confirmed.

## Where real IMES data lives

Real microscopy images, masks, and anything derived from them (overlays,
crops, visualizations) stay on:
- **This machine's local disk** (the gitignored `data/` folder), or
- **THWS Cloud** (how IMES shares data with us in the first place), or
- **The CAIRO NAS** (`/MAI/CellFoundry`, via the shared GPU server account).

These are the only approved locations. Third-party cloud services (W&B,
or anything else) may hold **metrics and config only** — numbers, not
image data — unless IMES explicitly approves otherwise.
