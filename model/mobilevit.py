"""MobileViT Model implementation"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .utils import Size2D, _configure_optimizer, to_2tuple
from .vit import ViTBlock


@dataclass
class MobileViTConfig:
    """Configuration values for MobileViT variants.

    Args:
        name: Variant name.
        img_size: Input image size as ``(height, width)``.
        dims: Transformer channel dimensions for MobileViT stages.
        chls: Channel widths across convolutional/trunk stages.
        patch_size: Patch size used inside MobileViT blocks.
        n_classes: Number of output classes.
        depths: Number of transformer blocks per MobileViT stage.
        expand_f: Expansion factor for MobileNetV2 blocks.
        mlp_ratio: Expansion ratio for transformer MLP hidden size.
    """

    name: str = "mobilevit-xxs"
    img_size: int | Size2D = (224, 224)
    dims: list[int] = field(default_factory=lambda: [64, 80, 96])
    chls: list[int] = field(default_factory=lambda: [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320])
    patch_size: int | Size2D = (2, 2)
    n_classes: int = 1000
    depths: list[int] = field(default_factory=lambda: [2, 4, 3])
    expand_f: float = 2.0
    mlp_ratio: float = 2.0


def conv_nxn_bn(
    in_chls: int,
    out_chls: int,
    kernel_size: int = 3,
    act_func: type[nn.Module] = nn.SiLU,
    **kwargs,
) -> nn.Sequential:
    """Build a Conv2d -> BatchNorm2d -> activation block.

    Args:
        in_chls: Number of input channels.
        out_chls: Number of output channels.
        kernel_size: Convolution kernel size.
        act_func: Activation module class to instantiate.
        **kwargs: Additional ``nn.Conv2d`` keyword arguments.

    Returns:
        nn.Sequential: Convolutional block with normalization and activation.
    """
    return nn.Sequential(
        nn.Conv2d(in_chls, out_chls, kernel_size=kernel_size, **kwargs),
        nn.BatchNorm2d(out_chls),
        act_func(),
    )


class MobileNetV2Block(nn.Module):
    """Inverted residual block used in the MobileViT convolutional stages."""

    def __init__(
        self,
        in_chls: int,
        out_chls: int,
        expand_f: float = 4.0,
        stride: int = 1,
        act_func: type[nn.Module] = nn.SiLU,
    ) -> None:
        """Initialize an inverted residual block.

        Args:
            in_chls: Number of input channels.
            out_chls: Number of output channels.
            expand_f: Hidden channel expansion factor.
            stride: Depthwise convolution stride.
            act_func: Activation module class to instantiate.

        Returns:
            None: This initializer does not return a value.
        """
        super(MobileNetV2Block, self).__init__()
        h_dim = int(in_chls * expand_f)
        self.res_path = in_chls == out_chls and stride == 1
        projection = (
            conv_nxn_bn(
                in_chls=in_chls,
                out_chls=h_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_func=act_func,
            )
            if expand_f != 1.0
            else nn.Identity()
        )
        self.block = nn.Sequential(
            projection,
            conv_nxn_bn(
                in_chls=h_dim,
                out_chls=h_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=h_dim,
                bias=False,
                act_func=act_func,
            ),
            conv_nxn_bn(
                in_chls=h_dim,
                out_chls=out_chls,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_func=nn.Identity,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block with an optional residual connection.

        Args:
            x: Input feature map of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Output feature map after block transformation.
        """
        if self.res_path:
            return x + self.block(x)
        return self.block(x)


class MobileViTBlock(nn.Module):
    """MobileViT block that fuses local convolutions with global token mixing."""

    def __init__(
        self,
        dim: int,
        depth: int,
        in_chls: int,
        patch_size: Size2D | int,
        mlp_dim: int,
        droprate_p: float = 0.0,
    ) -> None:
        """Initialize a MobileViT block.

        Args:
            dim: Transformer embedding dimension.
            depth: Number of transformer blocks.
            in_chls: Number of input/output feature channels.
            patch_size: Patch size as ``(height, width)`` or scalar.
            mlp_dim: Transformer MLP hidden dimension.
            droprate_p: Dropout rate for attention and projections.

        Returns:
            None: This initializer does not return a value.
        """
        super(MobileViTBlock, self).__init__()
        self.ph, self.pw = to_2tuple(patch_size)

        self.proj_in = conv_nxn_bn(in_chls, in_chls, kernel_size=3, padding=1, bias=False)
        self.conv1x1_1 = nn.Conv2d(in_chls, dim, kernel_size=1, padding=0, bias=False)
        self.conv1x1_2 = conv_nxn_bn(
            in_chls=dim, out_chls=in_chls, kernel_size=1, padding=0, bias=False
        )
        self.proj_out = conv_nxn_bn(2 * in_chls, in_chls, kernel_size=3, padding=1, bias=False)

        self.transformers = nn.Sequential(
            *(
                ViTBlock(
                    dim=dim,
                    n_heads=4,
                    mlp_dim=mlp_dim,
                    attn_drop_rate=droprate_p,
                    proj_drop_rate=droprate_p,
                    drop_path_prob=0.0,
                    act_fn=nn.SiLU,
                )
                for _ in range(depth)
            )
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process feature maps using local projection, token mixing, and fusion.

        Args:
            x: Input feature map of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Output feature map with fused local/global features.
        """
        res = x
        # Local features
        x = self.proj_in(x)
        x = self.conv1x1_1(x)
        b, d, h, w = x.shape
        ph, pw = self.ph, self.pw
        new_h = math.ceil(h / ph) * ph
        new_w = math.ceil(w / pw) * pw
        h_blk = new_h // ph
        w_blk = new_w // pw
        n_patches = h_blk * w_blk
        patch_area = ph * pw
        interpolate = False
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True
        x = x.reshape(b * d * h_blk, ph, w_blk, pw).transpose(1, 2)
        x = (
            x.reshape(b, d, n_patches, patch_area)
            .transpose(1, 3)
            .reshape(b * patch_area, n_patches, -1)
        )
        # Global features
        x = self.transformers(x)
        x = self.norm(x)
        # Fusion
        x = x.contiguous().view(b, patch_area, n_patches, -1)
        x = x.transpose(1, 3).reshape(b * d * h_blk, w_blk, ph, pw)
        x = x.transpose(1, 2).reshape(b, d, h_blk * ph, w_blk * pw)
        if interpolate:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.conv1x1_2(x)
        x = torch.cat([res, x], dim=1)
        x = self.proj_out(x)
        return x


class MobileViT(nn.Module):
    """MobileViT image classifier."""

    def __init__(self, cfg: MobileViTConfig) -> None:
        """Initialize the MobileViT backbone and classification head.

        Args:
            cfg: MobileViT model configuration.

        Returns:
            None: This initializer does not return a value.
        """
        super(MobileViT, self).__init__()
        self.cfg = cfg
        ih, iw = to_2tuple(cfg.img_size)
        ph, pw = to_2tuple(cfg.patch_size)
        assert ih % ph == 0 and iw % pw == 0
        init_dim, *_, last_dim = cfg.chls
        expand_f = cfg.expand_f
        dims = cfg.dims
        depths = cfg.depths
        patch_size = cfg.patch_size
        chls = cfg.chls
        n_classes = cfg.n_classes
        mlp_ratio = cfg.mlp_ratio

        self.conv = conv_nxn_bn(3, init_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem = nn.Sequential(
            MobileNetV2Block(in_chls=chls[0], out_chls=chls[1], expand_f=expand_f, stride=1),
            MobileNetV2Block(in_chls=chls[1], out_chls=chls[2], expand_f=expand_f, stride=2),
            MobileNetV2Block(in_chls=chls[2], out_chls=chls[3], expand_f=expand_f, stride=1),
            MobileNetV2Block(in_chls=chls[3], out_chls=chls[3], expand_f=expand_f, stride=1),
        )
        self.trunk = nn.Sequential(
            MobileNetV2Block(in_chls=chls[3], out_chls=chls[4], expand_f=expand_f, stride=2),
            MobileViTBlock(
                dim=dims[0],
                depth=depths[0],
                in_chls=chls[5],
                patch_size=patch_size,
                mlp_dim=int(dims[0] * mlp_ratio),
            ),
            MobileNetV2Block(in_chls=chls[5], out_chls=chls[6], expand_f=expand_f, stride=2),
            MobileViTBlock(
                dim=dims[1],
                depth=depths[1],
                in_chls=chls[7],
                patch_size=patch_size,
                mlp_dim=int(dims[1] * mlp_ratio),
            ),
            MobileNetV2Block(in_chls=chls[7], out_chls=chls[8], expand_f=expand_f, stride=2),
            MobileViTBlock(
                dim=dims[2],
                depth=depths[2],
                in_chls=chls[9],
                patch_size=patch_size,
                mlp_dim=int(dims[2] * mlp_ratio),
            ),
        )
        self.head = nn.Sequential(
            conv_nxn_bn(in_chls=chls[-2], out_chls=last_dim, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_dim, n_classes, bias=True),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize supported module weights in place.

        Args:
            m: Module instance to initialize.

        Returns:
            None: Mutates supported module weights in place.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out / m.groups
            nn.init.normal_(m.weight, mean=0, std=(2 / fan_out) ** 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run inference from input image tensor to class logits.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.
            target: Optional class indices with shape ``(B,)``.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Class logits, or logits with loss when labels are provided.
        """
        x = self.conv(x)
        x = self.stem(x)
        x = self.trunk(x)
        logits = self.head(x)
        if target is not None:
            loss = self.loss_fn(logits, target)
            return logits, loss
        return logits

    def configure_optimizer(
        self, optim_cfg: DictConfig, device: torch.device
    ) -> torch.optim.Optimizer:
        """Create the optimizer configured for this model.

        Args:
            optim_cfg: Optimizer configuration.
            device: Target device for optimizer state placement.

        Returns:
            torch.optim.Optimizer: Optimizer instance with decay groups applied.
        """
        return _configure_optimizer(self, optim_cfg, device)


if __name__ == "__main__":
    model = MobileViT(MobileViTConfig())
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(output.shape)
    print(f"{sum(p.numel() for p in model.parameters()):,}")
