# AngioFILM: Angiography Flow Interpolation Latent Matching

This repository contains the official implementation of our paper "AngioFILM: Angiography Flow Interpolation Latent Matching" (to be published).

## Abstract

We present AngioFILM, a novel deep learning framework for coronary angiogram video interpolation using flow matching. Our approach leverages a Vision Transformer (ViT) architecture, specifically based on the Latte model, with optional ECG-guided conditioning to generate high-quality intermediate frames in coronary angiography videos. The model aims to achieve temporal consistency and anatomical accuracy, enhancing temporal resolution in clinical settings.

## Key Features

- **Flow Matching-based Video Interpolation**: Utilizes flow matching with various path types including Linear, GVP (Generalized VP), VP (Variance Preserving), and Bridge paths. Supports velocity, noise, and score prediction models with configurable loss weighting strategies.
- **Optional ECG-guided Generation**: Integration of ECG signals for potentially improved cardiac cycle-aware frame generation (disabled by default in `config.yaml`).
- **Vision Transformer Architecture**: Employs a Latte-based ViT model (default: `Latte-L/4`) for spatial-temporal feature extraction.
- **Classifier-free Guidance**: Used during training and sampling for enhanced control over generation quality, controlled by `mask_cond_prob`.
- **Multiple Dataset Support**: Compatible with datasets formatted as `.npz` files, demonstrated with SNUBH data and potentially applicable to others like the public Coronary Dominance dataset after conversion.

## Model Architecture

Our model implementation is based on the Latte architecture with the following configuration by default (see `config.yaml`):

- **Backbone**: Vision Transformer (`Latte-L/4`) with adaptive layer normalization.
- **Autoencoder**: Training and sampling operate in the latent space using Cosmos-Tokenizer-CI.
- **Flow Matching Process**:
  - Multiple path types: Linear, GVP, VP, and Bridge paths (configurable via `transport/`)
  - Velocity prediction by default (`prediction: "velocity"`)
  - Configurable loss weighting: None, Velocity, or Likelihood weighting
  - Numerical stability controls with configurable epsilon values
- **Conditioning**:
  - Optional ECG signal embedding (`use_ecg: False` by default)
  - Parameter-free frame conditioning (using VAE latents of context frames)
  - Classifier-free guidance (`mask_cond_prob: 0.1` by default)

## Installation

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/SNUBH-CVC/AngioFILM
cd AngioFILM
git submodule update --init --recursive  # for Cosmos-Tokenizer

# Using conda (environment name: angio_film)
conda env create -f environment.yml
conda activate angio_film
cd Cosmos-Tokenizer && python setup.py install
```

### Using Docker

The repository includes a Dockerfile based on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` for containerized development and inference:

```bash
# Build the Docker image
docker build -t angio_film .

# Run a container (mounting current directory to /workspaces)
docker run --gpus all -it --rm -v $(pwd):/workspaces angio_film /bin/bash
```

Alternatively, you can use the provided devcontainer configuration(`.devcontainer/`) for VSCode development.

## Datasets

The model expects data in `.npz` format.

### Supported Datasets

- **SNUBH Dataset**: The default configuration (`configs/config.yaml`) points to `./data/SNUBH/train`. This is a private, in-house dataset.
- **Coronary Dominance Dataset**: This is an open dataset of coronary angiography videos.
  - **Paper**: [https://www.nature.com/articles/s41597-025-04676-8](https://www.nature.com/articles/s41597-025-04676-8)
  - **Huggingface**: [https://huggingface.co/datasets/BearSubj13/CoronaryDominance](https://huggingface.co/datasets/BearSubj13/CoronaryDominance)
  *Note: This dataset needs to be converted to the required `.npz` format first.*

### Dataset Structure (`.npz`)

Each `.npz` file should contain:
- `pixel_array`: A NumPy array of video frames with shape `(frames, height, width)` (loaded as `(F, 3, H, W)` grayscale in `dataset.py`).
- (Optional) `ecg_signal`: A NumPy array representing the ECG signal corresponding to the video.

## Training

### Basic Training

Training uses DistributedDataParallel (DDP).

```bash
# Single-GPU Training
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/config.yaml

# Multi-GPU Training (e.g., 2 GPUs)
torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/config.yaml
```

Training logs, checkpoints, and a copy of the config will be saved under the directory specified by `training.results_dir` in `config.yaml` (default: `./work_dirs`).

### Configuration

The training process is controlled by `config.yaml`. Key parameters include:

- **`dataset`**: Paths to train/test data (`.npz` format).
- **`transport`**: Flow matching configuration including path type (Linear/GVP/VP/Bridge), prediction type (velocity/noise/score), loss weighting, and stability parameters.
- **`model`**: ViT architecture (`name`), image size, number of frames, conditioning probabilities, ECG usage.
- **`training`**: Batch size, learning rate, optimizer, scheduler, checkpointing frequency, tokenizer name  (`cosmos_tokenizer_image_8x8` by default).
- **`system`**: Number of workers, mixed precision usage, etc.

## Inference

Generate interpolated frames for a test dataset using a trained checkpoint.

```bash
python sample.py \
    /path/to/your/checkpoint.pt \
    /path/to/test_data_npz_directory \
    --num_sampling_steps 50 \
    --fps 15
    --save_mp4 True \
```

**Arguments:**

- `ckpt_path`: Path to the trained model checkpoint (`.pt` file). The script expects the corresponding `config.yaml` to be in the parent directory of the checkpoint's parent directory (e.g., `work_dirs/experiment_name/config.yaml` if checkpoint is `work_dirs/experiment_name/checkpoints/model.pt`).
- `data_path`: Path to the directory containing test `.npz` files.
- `--output_root_dir`: Root directory where results will be saved (a subdirectory named after the experiment will be created). Default: `./outputs`.
- `--num_sampling_steps`: Number of steps for the sampler. Default: `50` for ODE
- `--fps`: Frames per second for the saved MP4 video. Default: `15`.
- `--save_mp4`: Whether to save the output as an MP4 video in addition to individual frames. Default: `True`.


## Evaluation

Evaluation metrics (PSNR, SSIM, LPIPS, FID) should be calculated based on the generated frames from the `sample.py` script compared against ground truth intermediate frames (if available).
```bash
# For PSNR, SSIM, LPIPS
python tools/evaluate_interpolation_lpips_psnr_ssim.py /path/to/output_directory
# For FID-DINOv2
python tools/evaluate_interpolation_fid.py /path/to/output_directory
```


## Acknowledgements
This repository uses components or pretrained weights from the following sources:

- Cosmos-Tokenizer (NVIDIA) — https://github.com/NVIDIA/Cosmos-Tokenizer — Licensed under Apache License 2.0
- Latte (Vchitect) — https://github.com/Vchitect/Latte — Licensed under Apache License 2.0
- SiT (willisma) - https://github.com/willisma/SiT — Licensed under MIT License

We thank the Seoul National University Bundang Hospital for providing the SNUBH dataset used in this research.

## References

This project builds upon the following research:
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Scalable Interpolant Transformers (SiT)](https://arxiv.org/abs/2401.08740)
- [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03046)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Vision Transformers](https://arxiv.org/abs/2010.11929)
- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)
