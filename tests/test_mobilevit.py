"""Regression tests for the local MobileViT implementation."""

from pathlib import Path

from model.mobilevit import MobileViT, MobileViTConfig
from omegaconf import OmegaConf
import pytest
import timm
import torch

model_map = {
    "MobileViT-XXS": "mobilevit_xxs",
    "MobileViT-XS": "mobilevit_xs",
    "MobileViT-S": "mobilevit_s",
}
yml_files = [Path("./config/model") / f"{name}.yaml" for name in model_map.keys()]


@pytest.mark.parametrize("yml_file", yml_files)
def test_mobilevit_param_match(yml_file):
    """Check that parameter counts match timm reference models.

    Returns:
        None: Asserts parity for each configured model variant.
    """
    cfg = OmegaConf.load(yml_file)
    cfg.img_size = [224, 224]
    cfg.patch_size = (
        [cfg.patch_size, cfg.patch_size] if isinstance(cfg.patch_size, int) else cfg.patch_size
    )
    cfg.n_classes = 1000
    name = cfg.name
    kwargs = dict(cfg)
    if name not in model_map:
        pytest.skip(f"No timm mapping found for {name}")
    timm_model = timm.create_model(model_map[name], pretrained=False)
    timm_params = sum(p.numel() for p in timm_model.parameters())
    my_model = MobileViT(MobileViTConfig(**kwargs))
    my_params = sum(p.numel() for p in my_model.parameters())
    assert timm_params == my_params, f"{name} params mismatch {timm_params=} {my_params=}"


def _map_timm_key_to_local(key: str) -> str:
    stage_prefix_map = {
        "stages.0.0.": "stem.0.",
        "stages.1.0.": "stem.1.",
        "stages.1.1.": "stem.2.",
        "stages.1.2.": "stem.3.",
        "stages.2.0.": "trunk.0.",
        "stages.2.1.": "trunk.1.",
        "stages.3.0.": "trunk.2.",
        "stages.3.1.": "trunk.3.",
        "stages.4.0.": "trunk.4.",
        "stages.4.1.": "trunk.5.",
    }
    mapping_prefixes = [
        ("stem.conv.", "conv.0."),
        ("stem.bn.", "conv.1."),
        ("head.fc.", "head.3."),
    ]
    for old, new in mapping_prefixes:
        if key.startswith(old):
            return key.replace(old, new, 1)

    for old, new in stage_prefix_map.items():
        if key.startswith(old):
            key = key.replace(old, new, 1)
            break

    block_prefixes = [
        ("final_conv.conv.", "head.0.0."),
        ("final_conv.bn.", "head.0.1."),
        ("conv1_1x1.conv.", "block.0.0."),
        ("conv1_1x1.bn.", "block.0.1."),
        ("conv2_kxk.conv.", "block.1.0."),
        ("conv2_kxk.bn.", "block.1.1."),
        ("conv3_1x1.conv.", "block.2.0."),
        ("conv3_1x1.bn.", "block.2.1."),
        ("conv_kxk.conv.", "proj_in.0."),
        ("conv_kxk.bn.", "proj_in.1."),
        ("conv_1x1.", "conv1x1_1."),
        ("transformer.", "transformers."),
        ("conv_proj.conv.", "conv1x1_2.0."),
        ("conv_proj.bn.", "conv1x1_2.1."),
        ("conv_fusion.conv.", "proj_out.0."),
        ("conv_fusion.bn.", "proj_out.1."),
    ]
    for old, new in block_prefixes:
        if old in key:
            return key.replace(old, new, 1)
    return key


def load_weights_from_timm(my_model: MobileViT, timm_model):
    """Copy timm MobileViT weights into the local implementation.

    Returns:
        None: Updates ``my_model`` parameters and buffers in place.
    """
    mapped_state = {
        _map_timm_key_to_local(key): value for key, value in timm_model.state_dict().items()
    }
    my_model.load_state_dict(mapped_state, strict=True)


@torch.no_grad()
@pytest.mark.parametrize("yml_file", yml_files)
def test_mobilevit_output_match_timm(yml_file):
    """Verify output parity after copying timm MobileViT weights.

    Returns:
        None: Asserts the outputs match within tolerance.
    """
    cfg = OmegaConf.load(yml_file)
    cfg.img_size = [224, 224]
    cfg.patch_size = (
        [cfg.patch_size, cfg.patch_size] if isinstance(cfg.patch_size, int) else cfg.patch_size
    )
    cfg.n_classes = 1000
    name = cfg.name
    kwargs = dict(cfg)
    if name not in model_map:
        pytest.skip(f"No timm mapping found for {name}")
    timm_model = timm.create_model(model_map[name], pretrained=False)
    my_model = MobileViT(MobileViTConfig(**kwargs))
    timm_model.eval()
    my_model.eval()
    load_weights_from_timm(my_model, timm_model)

    x = torch.randn(2, 3, 256, 256)
    y_timm = timm_model(x)
    y_my = my_model(x)
    assert torch.allclose(y_timm, y_my, atol=1e-5, rtol=1e-5), "NOT bit-matching!"
