from collections.abc import Sequence

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

type Size2D = tuple[int, int]


def to_2tuple(x: int | Sequence[int]) -> Size2D:
    """Normalize an integer or length-2 sequence to a 2D tuple.

    Returns:
        Size2D: Two-element size tuple.
    """
    if isinstance(x, Sequence):
        assert len(x) == 2, f"Expected a pair, got {x}"
        return int(x[0]), int(x[1])
    return x, x


def _configure_optimizer(model: nn.Module, optim_cfg: DictConfig, device: torch.device):
    """Build optimizer parameter groups with weight-decay filtering.

    Args:
        model: Model whose parameters will be optimized.
        optim_cfg: Hydra optimizer config.
        device: Runtime device used to gate fused optimizers.

    Returns:
        torch.optim.Optimizer: Instantiated optimizer for the model parameters.
    """
    optim_cfg["fused"] = bool(optim_cfg.get("fused", False)) and device.type == "cuda"
    params_dict = {pn: p for pn, p in model.named_parameters()}
    params_dict = {
        pn: p for pn, p in params_dict.items() if p.requires_grad
    }  # filter params that requires grad
    # create optim groups of any params that is 2D or more. This group will be weight decayed ie
    # weight tensors in Linear and embeddings
    decay_params = [p for p in params_dict.values() if p.dim() >= 2]
    # create optim groups of any params that is 1D. All biases and layernorm params
    no_decay_params = [p for p in params_dict.values() if p.dim() < 2]
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    optim_cfg["weight_decay"] = 0.0
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = instantiate(optim_cfg, params=optim_groups, _convert_="all")
    return optimizer
