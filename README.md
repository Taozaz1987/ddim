# Seismic Trace Reconstruction with DDIM

This repository provides a seismic-only workflow for reconstructing missing traces with diffusion models. The image-focused configs and datasets have been removed to reduce confusion—the remaining pipeline slices seismic sections into patches, trains a diffusion model on them, and reconstructs traces with DDIM-based samplers.

## Environment
Create the conda environment and install dependencies:

```bash
conda env create -f environment.yaml
conda activate copaint
```

## Data preparation
Use the helper script to slice seismic sections into normalized patches:

```bash
python prepare_data.py \
  --mat_path matdata.mat \
  --save_dir qiepian \
  --patch_size 128 \
  --stride 64 \
  --mat_key input
```

This produces `qiepian/marmousi_patches.npy`, which is the default training and sampling input. You can also run reconstruction directly from a `.mat` file without pre-saving to `.npy` (see inference below).

## Training
Train the diffusion model on the generated patches (defaults assume single-channel seismic data):

```bash
python scripts/image_train.py \
  --data_path qiepian/marmousi_patches.npy \
  --image_size 128 \
  --batch_size 8 \
  --max_steps 0 \
  --progress_bar
```

The training script wraps the seismic patches in a PyTorch `Dataset` and uses the guided-diffusion training loop. Checkpoints are written under `checkpoints/`.

## Inference
Run seismic reconstruction with DDIM samplers using the seismic config as a default:

```bash
python main.py \
  --config_file configs/seismic.yaml \
  --outdir images/seismic_run \
  --n_samples 4 \
  --n_iter 1
```

Key CLI flags (all populated from `configs/seismic.yaml` by default):

- `--data_path`: Path to the `.npy` patches (used when `--mat_path` is empty).
- `--mat_path`: Optional `.mat`/`.npy` section to slice on the fly instead of pre-generated patches.
- `--mat_key`: Variable name inside the `.mat` file (default: `input`).
- `--mat_stride`: Stride for sliding-window patching when using `--mat_path`.
- `--mask_type`: Mask generator for traces (`trace_dropout` by default).
- `--mask_drop_rate` / `--mask_drop_indices`: Control which traces are removed before reconstruction.

Outputs (ground truth, masks, and reconstructed grids) are saved under `--outdir`.

## Tests
Run the unit tests and smoke checks:

```bash
pytest
```
