"""Model factory helpers for ViT, DeiT, and Swin variants."""

import logging
import torch
import detectors  # noqa: F401
import timm
from omegaconf import DictConfig
from .vit import ViTConfig, ViT
from .deit import DeitConfig, DeiT
from .swin import SwinTransformer, SwinTransformerConfig


def init_model(cfg: DictConfig, device: torch.device) -> ViT | DeiT | SwinTransformer:
    """Instantiate the configured model family for the target device.

    Returns:
        ViT | DeiT | SwinTransformer: Initialized model instance.
    """
    name = cfg.model.name
    use_sdpa_attn = device.type != "mps"
    if name.startswith("DeiT"):
        assert getattr(cfg.model, "use_dist_token", False), (
            "Enable use_dist_token in DeiT model config"
        )
        model_cfg = DeitConfig(**cfg.model)
        model = DeiT(model_cfg, use_sdpa_attn=use_sdpa_attn)
    elif name.startswith("Swin"):
        model_cfg = SwinTransformerConfig(**cfg.model)
        model = SwinTransformer(model_cfg, use_sdpa_attn=use_sdpa_attn)
    else:
        model_cfg = ViTConfig(**cfg.model)
        model = ViT(model_cfg, use_sdpa_attn=use_sdpa_attn)
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
