# Vision Transformers (ViT)

A minimal PyTorch implementation of [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929), its variants [Data-efficient Image Transformers (DeiT)](https://arxiv.org/pdf/2012.12877), [Swin Transformer](https://arxiv.org/pdf/2103.14030), and [MobileViT](https://arxiv.org/pdf/2110.02178). Pretraining experiments were conducted on the Food101 dataset.

Architectural correctness is tested via parameter counts and output parity, matched against torchvision implementations (with exceptions for Swin due to differing internal choices).

Configuration is managed using Hydra, with optional experiment tracking via Weights & Biases (wandb).

## Findings

The objective was to observe the training behavior when using the same pretraining recipe across different popular ViT variants. The dataset used was [Food101](https://huggingface.co/datasets/ethz/food101). The training config is [here](./config/default.yaml).

![Food101 plot](https://raw.githubusercontent.com/mnjm/vision-transformers/refs/heads/assets/train-plots-food101.png)

- DeiT showed slightly better generalization than ViT, suggesting that distillation from a CNN is valuable.
- Swin performed the best because of its CNN-like ideas: local computation with hierarchical connections.
- MobileViT, while it underfit, had val metrics closer to its training metrics. This can likely be improved with slightly higher model capacity (XS).

## Setup

- Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync
```

## Train

```bash
uv run train.py +run=<run_name>
```

| Run | Dataset | Model |
| --- | --- | --- |
| `vit-food101` | Food-101 | `ViT-B-16` |
| `deit-food101` | Food-101 | `DeiT-B-16` |
| `swin-food101` | Food-101 | `Swin-T` |
| `mobilevit-food101` | Food-101 | `MobileViT-XXS` |
| `swin-tiny-imagenet` | Tiny-ImageNet | `Swin-T-TinyImageNet` |
| `mobilevit-tiny-imagenet` | Tiny-ImageNet | `MobileViT-XXS` |
| `vit-tiny-imagenet` | Tiny-ImageNet | `MobileVit-XXS` |
| `vit-cifar100` | CIFAR-100 | `ViT-T-8` |
| `deit-cifar100` | CIFAR-100 | `DeiT-T-8` |

Run-specific configurations can be found [here](./config/run).

## Hugging Face Checkpoints

The following Food101 checkpoints are available on Hugging Face:

- [`mnjm/vit-b16-food101`](https://huggingface.co/mnjm/vit-b16-food101)
- [`mnjm/deit-b16-food101`](https://huggingface.co/mnjm/deit-b16-food101)
- [`mnjm/swin-t-food101`](https://huggingface.co/mnjm/swin-t-food101)
- [`mnjm/mobilevit-xxs-food101`](https://huggingface.co/mnjm/mobilevit-xxs-food101)

## Compute Accuracy

To compute train and validation loss and accuracy for a saved checkpoint, run:

```bash
uv run compute_acc.py <checkpoint_file> [--device <device>]
```

## Structure

```
.
├── config/
│   ├── dataset/        # Dataset configs
│   ├── model/          # Model configs (ViT / DeiT / Swin)
│   ├── run/            # Run Experiment configs
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
