# Cell-Foundry

Measuring the **fat proportion** in microscopy images of cultured adipocytes (fat cells) with open-source segmentation models. A project module at THWS (Technische Hochschule Würzburg-Schweinfurt), done for **IMES** (Institut für Medizintechnik Schweinfurt, THWS) under the mentorship of Prof. Dr. Magda Gregorová.

**Team:** Pavin Sumathi Palanichamy, Prethebha Muthukumaran, Priyam Jha.

## Goal

IMES grows mixed cultures of fat cells and stem cells (porcine and human) and wants to measure automatically how much of an image the fat cells cover. Their main readout is the **fat proportion**; the cell count is second.

> **Fat proportion = (number of image pixels covered by fat cells) / (total number of pixels of the image).**
> It is an area share, not a share of counts or diameters. A fat cell is taken as a whole, with its lipid droplets.

We compare open-source segmentation models (zero-shot and fine-tuned) against IMES's current baseline, DINOCell fine-tuned by IMES. A model is only useful if IMES can integrate it into its own application (image in, result out).

## Status

Early stage, porcine brightfield images first. What exists today:

- A reproducible zero-shot benchmark for four Cellpose-family models (`cpsam_v2`, `cpsam`, `cpdino`, `cpdino-vitb`) and a batch script for DINOCell.
- A second scoring pipeline (`*_v2.py`) with fat-area metrics: fat %, fat % error (percentage points), foreground Dice, count error, F1 at IoU 0.5 / 0.7 / 0.75 / 0.9.
- Experiment tracking with Weights & Biases (numbers only, never images).
- No fine-tuned model and no held-out test set yet. IMES has no test set either; they used 22 of their 28 annotated images for training and 6 for validation.

Zero-shot results on the 25 differentiated porcine brightfield images (default settings, whole images; details in the minutes of 22 Oct):

| Model | Fat % error | Typical miss | Fat-area Dice | Cell F1@0.5 | Count error |
|---|---|---|---|---|---|
| cpsam_v2 | -6.6 pp | 6.6 pp | 0.66 | 0.43 | +141 % |
| cpsam | -4.6 pp | 5.6 pp | 0.65 | 0.47 | +135 % |
| cpdino | -1.4 pp | 3.7 pp | 0.76 | 0.62 | +58 % |
| cpdino-vitb | -0.1 pp | 4.5 pp | 0.73 | 0.65 | -9 % |

pp = percentage points of the whole image; negative = too little fat; "typical miss" = mean absolute fat % error over the images. These are zero-shot numbers, not comparable with IMES's own validation score.

## Repository layout

| Path | What it holds |
|---|---|
| `scripts/` | Benchmark and scoring scripts (see below) |
| `cellpose-env/` | Stand-alone `uv` environment for the Cellpose-family models (`cellpose==4.2.1.1`) |
| `Environments/` | Windows setup guide for the `cf-core` conda environment |
| `Results/` | Saved benchmark results (CSV, PDF, run configuration) |
| `Mentor_MOM/`, `Client_MOM/` | Minutes of the mentor and client meetings |
| `Project Brief/` | The project brief we received |
| `Slides/` | Slides of the bi-weekly meetings |
| `colab_dinocell_adipocyte_porcine.ipynb` | Adapted DINOCell training notebook (IMES's notebook, adapted) |
| `data/` | **Not in the repository** (git-ignored, see below) |

## Scripts

| Script | Purpose | Environment |
|---|---|---|
| `run_batch.py`, `run_batch_v2.py` | Run chosen Cellpose-family models on a whole folder and score every image | `cellpose-env` |
| `run_cpsam.py`, `run_cpsam_v2.py` | Run one model on one image (file picker) and score it | `cellpose-env` |
| `score_masks.py`, `score_masks_v2.py` | Score saved masks against the ground truth; v2 adds the fat-area metrics | `cellpose-env` |
| `test_score_masks_v2.py` | Test of the v2 metrics on small synthetic masks with hand-computed answers (no real data) | `cellpose-env` |
| `run_dinocell_batch.py` | Zero-shot DINOCell on a folder, in the same output format | main environment (`dinocell==0.74`) |

Every batch run writes to a **new timestamped folder**; nothing is overwritten. Each run saves its settings, package versions and the git commit in `run_config.json`.

## Environments

The repository needs two Python environments that cannot be installed together: `dinocell==0.74` requires `cellpose==4.0.9`, while the DINOv3-based Cellpose models need `cellpose==4.2.1.1`. The reasons and setup commands are in [cellpose-env/README.md](cellpose-env/README.md). For Windows with conda see [Environments/cf-core-env-setup.md](Environments/cf-core-env-setup.md). A setup file for the main (DINOCell) environment is not yet part of the repository.

Quick start for the Cellpose models:

```bash
cd cellpose-env
uv sync
uv run python ../scripts/run_batch_v2.py --models cpdino-vitb --group-dir "../data/Training_porcine_bf" --out-root "../../Cloud_results"
```

`--out-root` is required. Choose a folder **outside the repository**: the results contain overlay images made from real IMES data.

Check that the GPU is active after installing (see `cellpose-env/README.md`); on Windows a CPU-only PyTorch is a known pitfall.

## Data and confidentiality

The microscopy images and annotation masks belong to IMES and are **not** part of this repository (`data/` and `*.tif`/`*.tiff` are git-ignored). Real images, masks and anything derived from them (overlays, crops, figures) may only be stored on a local disk, on THWS Cloud or on the CAIRO NAS. Third-party services such as Weights & Biases may hold **metrics and configuration only**, never image data. Do not commit overlays or crops, and do not paste images into issues or pull requests.

Expected data layout (provided by IMES): one folder per group, for example `data/Training_porcine_bf/`, with pairs `<name>_image.png` and `<name>_instances.tif` (a label image: 0 = background, every other number = one cell).

## Working agreements

- Everyone works on their own branch; changes reach `main` through a pull request reviewed by a teammate.
- Minutes of every meeting go into `Mentor_MOM/` or `Client_MOM/`.
- Open work is tracked in the GitHub issues.
