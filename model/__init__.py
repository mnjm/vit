"""Model factory helpers for ViT, DeiT, and Swin variants."""

import logging

import detectors  # noqa: F401
import timm
import torch
from omegaconf import DictConfig

from .deit import DeiT, DeitConfig
from .mobilevit import MobileViT, MobileViTConfig
from .swin import SwinTransformer, SwinTransformerConfig
from .vit import ViT, ViTConfig


def init_model(cfg: DictConfig, device: torch.device) -> ViT | DeiT | SwinTransformer | MobileViT:
    """Instantiate the configured model family for the target device.

    Returns:
        ViT | DeiT | SwinTransformer: Initialized model instance.
    """
    name = cfg.model.name
    if name.startswith("ViT"):
        model_cfg = ViTConfig(**cfg.model)
        model = ViT(model_cfg)
    elif name.startswith("DeiT"):
        assert getattr(cfg.model, "use_dist_token", False), (
            "Enable use_dist_token in DeiT model config"
        )
        model_cfg = DeitConfig(**cfg.model)
        model = DeiT(model_cfg)
    elif name.startswith("Swin"):
        model_cfg = SwinTransformerConfig(**cfg.model)
        model = SwinTransformer(model_cfg)
    elif name.startswith("MobileViT"):
        model_cfg = MobileViTConfig(**cfg.model)
        model = MobileViT(model_cfg)
    else:
        raise ValueError(f"Unsupported model name: {name}")
    return model


def init_deit(
    model: torch.nn.Module, cfg: DictConfig, device: torch.device, logger: logging.Logger
):
    """Load and attach the DeiT teacher model used for distillation.

    Returns:
        torch.nn.Module: Frozen teacher model attached to the student.
    """
    assert isinstance(model, DeiT), "Model should be DeiT"
    teacher_name = cfg.deit.teacher_name
    # load teacher model
    teacher = timm.create_model(teacher_name, pretrained=True)
    teacher.to(device)
    rand_img = torch.rand(
        (1, cfg.dataset.img_chls, cfg.dataset.img_size, cfg.dataset.img_size), device=device
    )
    out = teacher(rand_img)
    assert out.shape == (1, cfg.dataset.n_class), f"Invalid teacher model, model's {out.shape=}"
    # freeze teacher
    teacher.requires_grad_(False)
    teacher.eval()
    model.set_teacher(teacher)
    logger.info(f"Loaded teacher model {teacher_name=}")
    return teacher


__all__ = ["init_model", "init_deit"]
