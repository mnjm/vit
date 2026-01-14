# Vision Transformers (ViT)

A minimal PyTorch implementation of [Vision Transformers(ViT)](https://arxiv.org/pdf/2010.11929), its varients [Data efficient Image Transformers (DeiT)](https://arxiv.org/pdf/2012.12877) and [Swin Transformers](https://arxiv.org/pdf/2103.14030). Experimented with CIFAR-100 (ViT-T/8 vs DeiT-T/8) and Tiny-Imagenet dataset with a (ViT-T/8 vs Swin-T-TinyImageNet), but supports other varients as well.

Architectural correctness is tested via parameter counts and output parity, matched against torchvision implementations (with exceptions for Swin due to differing internal choices).

Configuration is managed using Hydra, with optional experiment tracking via Weights & Biases (wandb).

## ViT-T/8 vs DeiT-T/8 on CIFAR-100

![CIFAR-100 Plots](https://raw.githubusercontent.com/mnjm/vision-transformers/refs/heads/assets/train-plots.png)

## ViT-B/8 vs Swin-T-TinyImageNet on TinyImageNet

![Tiny Image Net Plots](https://raw.githubusercontent.com/mnjm/vision-transformers/refs/heads/assets/train-plots-tiny-imagenet.png)

## Setup

- Install [uv](https://docs.astral.sh/uv/) and run
```bash
uv sync
```

## Training Runs

### Train ViT-T/8 on CIFAR-100

```bash
uv run train.py +run=vit-cifar100
```

### Train DeiT-T/8 on CIFAR-100

```bash
uv run train.py +run=deit-cifar100
```
Uses frozen `resnet18_cifar100` (via timm) as Teacher and is used for hard distillation (as it is showen to work well in DeiT paper)

### Train ViT-T/8 on Tiny-Imagenet

```bash
uv run train.py +run=vit-tiny-imagenet
```

### Train Swin-T on Tiny-Imagenet

```bash
uv run train.py +run=swin-tiny-imagenet
```

## Structure

```
.
├── config/
│   ├── dataset/        # Dataset configs
│   ├── model/          # Model configs (ViT / DeiT / Swin)
│   ├── run/            # Experiment presets
│   └── default.yaml    # Global defaults
├── model/              # Model implementations
├── data.py             # Dataset & dataloaders
├── train.py            # Training entry point
├── utils.py            # Training utilities
└── tests/              # Architecture & parity tests

````

## Configuration

* Explicit configs over implicit defaults
* Modular overrides:
  * `dataset`
  * `model`
  * `optimizer`
  * `lr_scheduler`
* Experiment outputs are auto-versioned and logged.

Example override:

```bash
python train.py model=ViT-B-16 dataset=cifar100
```