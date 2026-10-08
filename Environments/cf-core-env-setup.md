# `cf-core` conda environment: all setup commands (Windows PowerShell)

Replaces `cellfoundry-env-setup.md` (same env, new name, plus DINOv3 for cpdino). Verified on the RTX 5060 Laptop GPU, 2026-10-06.

**What runs in `cf-core`:** Cellpose-SAM (`cpsam_v2`, `cpsam`), Cellpose + DINOv3 (`cpdino`, `cpdino-vitb`), micro-SAM, the Cellpose app, napari, and our scripts (`run_cpsam.py`, `run_batch.py`, `score_masks.py`).

Two ways to build it:

- **Route A: exact copy from the recipe file** (recommended once `cf-core.yml` exists). Gives the identical versions.
- **Route B: from scratch** (first time, or on a new machine). Versions are pinned to the ones verified here.

---

## One-time PowerShell setup (only if conda doesn't work in VS Code's terminal yet)


```powershell


conda init powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# close and reopen VS Code / the terminal


```


---

## Route A: exact copy from `cf-core.yml` if you have already created cf-core



```powershell


conda deactivate
conda env create -f path\cf-core.yml
conda activate cf-core


```


Then run the checks (section 3 below).

The yml must contain, under `- pip:`, first the line `- --extra-index-url https://download.pytorch.org/whl/cu132`, and `dinov3` as `- git+https://github.com/facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751` (see section 5 for how to refresh the yml).

---

## Route B: from scratch

### 1. Create the env (Python 3.12)


```powershell


conda create -n cf-core python=3.12 -y
conda activate cf-core


```



### 2. Install packages, in this order


The torch versions might be different for Mac and Window users, Choose accordingly


```powershell


# 2a. PyTorch with CUDA 13.2 FIRST (RTX 50-series needs a CUDA >= 12.8 build; the cu126 build in the Cellpose docs does NOT work)
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132

# 2b. Segmentation tools: Cellpose 4 (with the app) and micro-SAM (brings torch-em and napari)
python -m pip install "cellpose[gui]==4.2.1.1" micro-sam==1.8.14 napari==0.9.2

# 2c. DINOv3 backbone for cpdino / cpdino-vitb (pinned to the commit verified on 2026-10-06)
python -m pip install "git+https://github.com/facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751"

# 2d. Analysis and notebook tools
python -m pip install tifffile==2026.9.20 scikit-image==0.26.0 scikit-learn==1.9.1 pandas==3.0.6 matplotlib==3.11.2 seaborn==0.13.2 tqdm==4.70.1 pyyaml==6.0.3 jupyter==1.1.1 ipykernel==6.31.0


```

### Do NOT install into `cf-core`

- `opencv-python` (Cellpose already brings `opencv-python-headless`; both together break `import cv2`)
- TensorFlow, `torchaudio` from the default index (can swap the GPU torch for a CPU build)
- CellSAM, DINOCell, SAM3, Medical SAM3: they have their own envs (`environments.md`)

---

## 3. Checks (after Route A or B; after every later install too)



```powershell


# GPU build of torch: expect "2.12.0+cu132 13.2 True NVIDIA GeForce RTX 5060 Laptop GPU"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Packages: expect "cellpose 4.2.1.1 | micro-sam 1.8.14 | napari 0.9.2" (cv2 must import without error)
python -c "import cellpose, micro_sam, napari, cv2; from importlib.metadata import version as v; print('cellpose', v('cellpose'), '| micro-sam', v('micro-sam'), '| napari', v('napari'))"

# DINOv3: expect "dinov3 ok"
python -c "from dinov3.hub.backbones import dinov3_vitl16; print('dinov3 ok')"

# Launchers belong to this env: expect ...\envs\cf-core\Scripts\micro_sam.annotator_2d.exe
(Get-Command micro_sam.annotator_2d).Source


```

**If the GPU check shows `+cpu` or `False`** (some install replaced torch):


```powershell

python -m pip install --force-reinstall torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132


```

---

## 4. Download the model weights once (~1.1–1.2 GB each, to `%USERPROFILE%\.cellpose\models`)



```powershell


python -c "from cellpose import models; [models.CellposeModel(gpu=True, pretrained_model=m) for m in ['cpsam_v2', 'cpsam', 'cpdino', 'cpdino-vitb']]"
dir $env:USERPROFILE\.cellpose\models


```


- Interrupted downloads leave only `tmp*` files; delete them when nothing is downloading: `del $env:USERPROFILE\.cellpose\models\tmp*`
- micro-SAM downloads its weights (`vit_b_lm` etc.) the first time a model is used.

---

## 5. Save the recipe after any change to the env



```powershell


conda env export -n cf-core --no-builds -f path\cf-core.yml
code path\cf-core.yml


```


Edit three things, then save:
1. Delete the last line (`prefix: ...`).
2. Directly under `- pip:`, add as the first entry: `- --extra-index-url https://download.pytorch.org/whl/cu132`
3. Replace the line `- dinov3==0.0.1` with `- git+https://github.com/facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751` (dinov3 isn't on PyPI, so the exported line would fail).

Use `-f`, never `> file.yml` (PowerShell would write UTF-16, which conda can't read).


---

## 6. Reference test (proves the env reproduces our numbers)

```powershell


cd path
python scripts\run_cpsam.py --image "img location" --out "path\env_check" --no-show


```
Expect: n_true 304, n_pred 315, F1@0.5 0.708 (TP 219 / FP 96 / FN 85), count error +3.6%.

---

## 7. Daily use

```powershell
conda activate cf-core
cd path
python -m cellpose                          # Cellpose app (File -> "Disable autosave _seg.npy file" first)
python scripts\run_cpsam.py                 # one image, file picker, scored
python scripts\run_batch.py                 # all 4 Cellpose models on the porcine BF folder
micro_sam.annotator_2d                      # micro-SAM napari app
```
VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> `cf-core`.

## Notes
- Harmless warning: `UserWarning: Sparse invariant checks are implicitly disabled` (Cellpose internals).
- The old env `cellfoundry` is the fallback until `cf-core` has been used for a few days; then `conda env remove -n cellfoundry`.
- For the repo, `cf-core.yml` needs a cross-platform variant first (it contains Windows-only packages like `pywin32`, and cu132 torch excludes older NVIDIA GPUs).
