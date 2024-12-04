# DDPM Implementation

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) with support for:
- Basic DDPM
- DDIM sampling
- Latent DDPM with VAE
- Classifier-Free Guidance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the data:
```bash
# Download ImageNet-100 dataset
wget https://drive.google.com/drive/u/0/folders/1Hr8LU7HHPEad8ALmMo5cvisazsm6zE8Z/imagenet100_128x128.tar.gz
tar -xvf imagenet100_128x128.tar.gz
```

4. Download pretrained VAE (for Latent DDPM):
```bash
mkdir pretrained
# Download VAE checkpoint to pretrained/model.ckpt
```

## Training

### Basic DDPM:
```bash
python train.py --config configs/ddpm.yaml
```

### With Classifier-Free Guidance:
```bash
python train.py --config configs/ddpm.yaml --use_cfg True
```

### Latent DDPM:
```bash
python train.py --config configs/ddpm.yaml --latent_ddpm True
```

### DDIM:
```bash
python train.py --config configs/ddpm.yaml --use_ddim True
```

## Inference

```bash
python inference.py --config configs/ddpm.yaml --ckpt path/to/checkpoint.pth
```

## Project Structure
```
.
├── configs/
│   └── ddpm.yaml          # Configuration file
├── models/                # Model implementations
├── schedulers/            # DDPM and DDIM schedulers
├── pipelines/             # Generation pipeline
├── utils/                 # Utility functions
├── train.py              # Training script
└── inference.py          # Inference script
```

## Citation

If you use this code, please cite the original DDPM paper:

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2006.11239},
  year={2020}
}
```
