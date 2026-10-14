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
