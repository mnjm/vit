# Vision Transformers (ViT)

A minimal PyTorch implementation of [Vision Transformers(ViT)](https://arxiv.org/pdf/2010.11929), its varients [Data efficient Image Transformers (DeiT)](https://arxiv.org/pdf/2012.12877), [Swin Transformers](https://arxiv.org/pdf/2103.14030) and [MobileViT](https://arxiv.org/pdf/2110.02178).  and Tiny-Imagenet dataset with a (ViT-16L-384D/8 vs Swin-T vs MobileViT-XXS), Experimented with CIFAR-100 (ViT-T/8 vs DeiT-T/8) but supports other varients as well.

Architectural correctness is tested via parameter counts and output parity, matched against torchvision implementations (with exceptions for Swin due to differing internal choices).

Configuration is managed using Hydra, with optional experiment tracking via Weights & Biases (wandb).


## ViT-16L-384D/8 vs Swin-T vs MobileViT-XXS on TinyImageNet

![Tiny Image Net Plots](https://raw.githubusercontent.com/mnjm/vision-transformers/refs/heads/assets/train-plots-tiny-imagenet.png)

Findings
- ViT-16L-384D/8 and Swin-T overfit early, as expected, since the dataset is small.
- There is a slight improvement of Swin-T over ViT-16L-384D/8, as it borrows some inductive biases from CNNs.
- MobileViT-XXS is underfitting / learning slowly because of:
  1. its low parameter count
  2. having a token count collapsed to 1 in most of the later ViT layers. MobileViT-XXS is not a good choice for 64×64 image classification tasks.

## ViT-T/8 vs DeiT-T/8 on CIFAR-100

![CIFAR-100 Plots](https://raw.githubusercontent.com/mnjm/vision-transformers/refs/heads/assets/train-plots-cifar100.png)

Findings
- DeiT-T/8 performs better than ViT-T/8 on CIFAR-100 as expected, as DeiT uses hard distillation from the frozen pretrained ResNet18 teacher to provide a richer context with some CNN inductive biases.

## Setup

- Install [uv](https://docs.astral.sh/uv/) and run
```bash
uv sync
```

## Training Runs


### Train ViT-16L-384D/8 on Tiny-Imagenet

```bash
uv run train.py +run=vit-tiny-imagenet
```

### Train Swin-T on Tiny-Imagenet

```bash
uv run train.py +run=swin-tiny-imagenet
```

### Train MobileViT-XXS on Tiny-Imagenet

```bash
uv run train.py +run=mobilevit-tiny-imagenet
```

### Train ViT-T/8 on CIFAR-100

```bash
uv run train.py +run=vit-cifar100
```

### Train DeiT-T/8 on CIFAR-100

```bash
uv run train.py +run=deit-cifar100
```
Uses frozen `resnet18_cifar100` (via timm) as Teacher and is used for hard distillation (as it is showen to work well in DeiT paper)

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

## Citations

```bibtex
@misc{2010.11929,
Author = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
Title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
Year = {2020},
Eprint = {arXiv:2010.11929},
}
```

```bibtex
@misc{2012.12877,
Author = {Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Hervé Jégou},
Title = {Training data-efficient image transformers & distillation through attention},
Year = {2020},
Eprint = {arXiv:2012.12877},
}
```

```bibtex
@misc{2103.14030,
Author = {Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
Title = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
Year = {2021},
Eprint = {arXiv:2103.14030},
}
```

```bibtex
@misc{2110.02178,
Author = {Sachin Mehta and Mohammad Rastegari},
Title = {MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
Year = {2021},
Eprint = {arXiv:2110.02178},
}
```
