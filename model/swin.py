"""Swin Transformer building blocks and classifier implementation."""

import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from torch import nn

from .utils import Size2D, _configure_optimizer, to_2tuple
from .vit import MLP, PatchEmbed, StochDepthDrop


@dataclass
class SwinTransformerConfig:
    """Configuration for Swin Transformer.

    Args:
        name: Model name.
        img_size: Input image size (square).
        patch_size: Patch size.
        patch_norm: Wheather to apply norm after patch embedding.
        img_chls: Number of image channels.
        n_class: Number of output classes.
        n_embed: Base embedding dimension.
        depths: Number of blocks per stage.
        n_heads: Number of attention heads per stage.
        window_size: Window size for attention.
        mlp_ratio: MLP expansion ratio.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        stoch_depth_drop_rate: Stochastic depth rate.
    """

    name: str = "Swin-T"
    img_size: int = 224
    patch_size: int = 4
    patch_norm: bool = True
    img_chls: int = 3
    n_class: int = 1000
    n_embed: int = 96
    depths: list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    n_heads: list[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    stoch_depth_drop_rate: float = 0.0


def window_partition(x: torch.Tensor, window_size: int | Size2D) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: Feature map (B, H, W, C).
        window_size: Window size.

    Returns:
        Windows (B * num_windows, Wh, Ww, C).
    """
    _, H, W, _ = x.shape
    wsh, wsw = to_2tuple(window_size)
    nh, nw = H // wsh, W // wsw
    windows = rearrange(
        x, "b (nh ws1) (nw ws2) c -> (b nh nw) ws1 ws2 c", ws1=wsh, ws2=wsw, nh=nh, nw=nw
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int | Size2D, H: int, W: int
) -> torch.Tensor:
    """Reconstruct feature map from windows.

    Args:
        windows: Windowed features.
        window_size: Window size.
        H: Output height.
        W: Output width.

    Returns:
        Feature map (B, H, W, C).
    """
    wsh, wsw = to_2tuple(window_size)
    nh, nw = H // wsh, W // wsw
    B = windows.shape[0] // (nh * nw)
    x = rearrange(
        windows, "(b nh nw) ws1 ws2 c -> b (nh ws1) (nw ws2) c", ws1=wsh, ws2=wsw, nh=nh, nw=nw, b=B
    )
    return x


class ShiftedWindowMHSA(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int | Size2D,
        n_heads: int,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        use_sdpa_attn: bool = True,
    ):
        """
        Args:
            dim: Embedding dimension.
            window_size: Attention window size.
            n_heads: Number of heads.
            attn_drop_rate: Attention dropout.
            proj_drop_rate: Projection dropout.
            use_sdpa_attn: Use SDPA attention.

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.n_heads = n_heads
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), n_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 2d coords to 1d
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.relative_position_index: torch.Tensor
        self.register_buffer("relative_position_index", relative_position_index)

        self.split_qkv = Rearrange("b n (t h d) -> t b h n d", t=3, h=n_heads, d=self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.use_sdpa_attn = use_sdpa_attn and hasattr(F, "scaled_dot_product_attention")
        if use_sdpa_attn and not self.use_sdpa_attn:
            warnings.warn("SDPA attn is enabled but not available.")
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        self.attn_drop_rate = attn_drop_rate
        self.scale = self.head_dim**-0.5
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Window tokens (Bw, N, C), where Bw = batch_size * windows_per_image, N = window_size^2
            mask: Optional attention mask (windows_per_image, N, N)

        Returns:
            Attention output (Bw, N, C).
        """
        Bw, N, C = x.shape

        q, k, v = self.split_qkv(self.qkv(x))

        rel_pos_bias = torch.index_select(
            self.relative_position_bias_table,
            0,
            torch.reshape(self.relative_position_index, (-1,)),
        )
        rel_pos_bias = (
            rel_pos_bias.view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        windows_per_img = mask.shape[0] if mask is not None else 0
        full_mask = rel_pos_bias
        if mask is not None:
            B = Bw // windows_per_img
            assert Bw % mask.shape[0] == 0
            # materializing the attention mask for the whole batch
            shift_mask = repeat(mask, "nw h w -> (b nw) 1 h w", b=B)
            full_mask = full_mask + shift_mask

        if self.use_sdpa_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=full_mask,
                dropout_p=self.attn_drop_rate if self.training else 0.0,
                scale=self.scale,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + full_mask
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, self.attn_drop_rate, training=self.training)
            x = attn @ v

        x = self.merge_heads(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with optional window shift."""

    def __init__(
        self,
        dim: int,
        input_res: int | Size2D,
        n_heads: int,
        window_size: int | Size2D,
        shift_size: int,
        mlp_ratio: float,
        attn_drop_rate: float,
        proj_drop_rate: float,
        path_drop_rate: float,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_sdpa_attn: bool = True,
    ):
        """
        Args:
            dim: Embedding dimension.
            input_res: Input resolution.
            n_heads: Number of heads.
            window_size: Window size.
            shift_size: if > 0, shifts the windows (SWMSHA) else (WMSHA)
            mlp_ratio: MLP ratio.
            attn_drop_rate: Attention dropout.
            proj_drop_rate: Projection dropout.
            path_drop_rate: Drop path rate.

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__()
        self.dim = dim
        self.input_res = to_2tuple(input_res)
        self.n_heads = n_heads
        self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size

        # if window size is larger than input resolution in either, do not partition windows
        if self.input_res[0] <= self.window_size[0] or self.input_res[1] <= self.window_size[1]:
            self.shift_size = 0
            self.window_size = self.input_res

        assert 0 <= self.shift_size <= min(self.window_size), (
            "shift size should be within 0 - window_size"
        )

        self.attn_norm = norm_layer(dim)
        self.attn = ShiftedWindowMHSA(
            dim=self.dim,
            window_size=self.window_size,
            n_heads=self.n_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            use_sdpa_attn=use_sdpa_attn,
        )

        self.drop_path = StochDepthDrop(drop_prob=path_drop_rate)

        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(
            in_features=self.dim,
            hidden_features=int(self.dim * mlp_ratio),
            out_features=self.dim,
            act_fn=act_layer,
            drop_rate=proj_drop_rate,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_res
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.attn_mask: torch.Tensor | None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens (B, H*W, C).

        Returns:
            Output tokens (B, H*W, C).
        """
        H, W = self.input_res
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.attn_norm(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windowed = window_partition(shifted_x, self.window_size)  # nW * B, wsh, wsw, C

        x_windowed = x_windowed.view(-1, self.window_size[0] * self.window_size[1], C)

        attn = self.attn(x_windowed, mask=self.attn_mask)  # nW * B, wsh * wsw, C

        attn = attn.view(-1, self.window_size[0], self.window_size[1], C)

        shifted_x = window_reverse(
            attn, self.window_size, H, W
        )  # B, n_h_win * wsh, n_w_win * wsw, C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))

        return x


class PatchMerge(nn.Module):
    """Patch merging layer (downsampling)."""

    def __init__(
        self, dim: int, input_res: int | Size2D, norm_lyr: type[nn.Module] = nn.LayerNorm
    ) -> None:
        """
        Args:
            dim: Input embedding dim.
            input_res: Input resolution.
            norm_lyr: Normalization layer after projection

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__()
        self.dim = dim
        self.input_res = to_2tuple(input_res)
        H, W = self.input_res
        assert H % 2 == 0 and W % 2 == 0, f"input size is not even {H}x{W}"
        self.re = Rearrange("b (h ph) (w pw) c -> b (h w) (pw ph c)", ph=2, pw=2)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_lyr(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tokens (B, H*W, C).

        Returns:
            Downsampled tokens (B, H/2 * W/2, 2 * C)
        """
        H, W = self.input_res
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = self.re(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinLayer(nn.Module):
    """One stage of Swin Transformer."""

    def __init__(
        self,
        dim: int,
        input_res: int | Size2D,
        depth: int,
        n_heads: int,
        window_size: int | Size2D,
        mlp_ratio: float,
        proj_drop_rate: float,
        attn_drop_rate: float,
        path_drop_rate: float | list[float],
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: bool = False,
        use_sdpa_attn: bool = True,
    ):
        """
        Args:
            dim: Embedding dim.
            input_res: Input resolution.
            depth: Number of blocks.
            n_heads: Number of heads.
            window_size: Window size.
            downsample: Apply patch merging.

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__()
        self.dim = dim
        self.input_res = to_2tuple(input_res)
        self.depth = depth
        window_size_2d = to_2tuple(window_size)
        self.blks = nn.ModuleList(
            SwinTransformerBlock(
                dim=dim,
                input_res=self.input_res,
                n_heads=n_heads,
                window_size=window_size_2d,
                shift_size=0 if (idx % 2 == 0) else window_size_2d[0] // 2,
                mlp_ratio=mlp_ratio,
                proj_drop_rate=proj_drop_rate,
                attn_drop_rate=attn_drop_rate,
                path_drop_rate=path_drop_rate[idx]
                if isinstance(path_drop_rate, list)
                else path_drop_rate,
                norm_layer=norm_layer,
                use_sdpa_attn=use_sdpa_attn,
            )
            for idx in range(depth)
        )

        if downsample:
            self.downsample = PatchMerge(dim=dim, input_res=self.input_res, norm_lyr=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all blocks in the stage and optional downsampling.

        Returns:
            torch.Tensor: Stage output tokens.
        """
        for blk in self.blks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SwinTransformer(nn.Module):
    """Swin Transformer"""

    def __init__(self, cfg: SwinTransformerConfig, use_sdpa_attn: bool = True) -> None:
        """Initialize the Swin Transformer backbone and classifier head.

        Args:
            cfg: Swin Transformer model configuration.
            use_sdpa_attn: Whether to use SDPA attention when available.

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__()

        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_dim=cfg.img_chls,
            out_dim=cfg.n_embed,
            norm_lyr=nn.LayerNorm if cfg.patch_norm else None,
        )
        self.n_patches = (cfg.img_size // cfg.patch_size) ** 2
        patches_res = cfg.img_size // cfg.patch_size, cfg.img_size // cfg.patch_size

        stoch_depth_drop_rates = [
            x.item() for x in torch.linspace(0, cfg.stoch_depth_drop_rate, sum(cfg.depths))
        ]

        self.n_layers = len(cfg.depths)

        self.layers = nn.ModuleList(
            SwinLayer(
                dim=int(cfg.n_embed * 2**idx),
                input_res=(patches_res[0] // (2**idx), patches_res[1] // (2**idx)),
                depth=cfg.depths[idx],
                n_heads=cfg.n_heads[idx],
                window_size=cfg.window_size,
                mlp_ratio=cfg.mlp_ratio,
                proj_drop_rate=cfg.drop_rate,
                attn_drop_rate=cfg.attn_drop_rate,
                downsample=(idx < self.n_layers - 1),
                path_drop_rate=stoch_depth_drop_rates[
                    sum(cfg.depths[:idx]) : sum(cfg.depths[: idx + 1])
                ],
                use_sdpa_attn=use_sdpa_attn,
            )
            for idx in range(self.n_layers)
        )

        self.n_features = int(cfg.n_embed * 2 ** (self.n_layers - 1))
        self.norm = nn.LayerNorm(self.n_features)
        self.head = nn.Linear(self.n_features, cfg.n_class) if cfg.n_class > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize supported module weights in place.

        Returns:
            None: Mutates the provided module parameters.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def loss_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Compute the classification loss for a batch.

        Returns:
            torch.Tensor: Scalar cross-entropy loss.
        """
        return F.cross_entropy(x, y, weight=weight, label_smoothing=label_smoothing)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the Swin model and optionally compute loss.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)``.
            y: Optional class indices with shape ``(B,)``.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Logits, or logits with loss when labels are provided.
        """
        x = self.patch_embed(x)

        for lyr in self.layers:
            x = lyr(x)

        x = self.norm(x)  # B, L, C
        x = x.mean(dim=1)
        x = self.head(x)  # B, n_class
        if y is not None:
            loss = self.loss_fn(x, y)
            return x, loss
        return x

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
    use_sdpa_attn = True
    model = SwinTransformer(SwinTransformerConfig(), use_sdpa_attn=use_sdpa_attn)
    x = torch.randn((2, 3, 224, 224))
    y = model(x)
    print(y.shape)
