import torch.nn as nn
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.layers import (
    Mlp,
    DropPath,
    use_fused_attn,
)


class Attention(nn.Module):
    """
    Multi-head self-attention layer with optional fused attention (scaled dot-product).

    Args:
        dim (int): Input embedding dimension.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to add bias in QKV projections. Defaults to False.
        qk_norm (bool, optional): Whether to apply LayerNorm to Q and K. Defaults to False.
        attn_drop (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout probability after output projection. Defaults to 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    """
    Applies a learnable scaling parameter (gamma) to the input.

    Args:
        dim (int): Input embedding dimension.
        init_values (float, optional): Initial value of gamma. Defaults to 1e-5.
        inplace (bool, optional): Whether to modify the tensor in-place. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LayerScale.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Scaled tensor of shape (batch_size, seq_len, dim).
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerBlock(nn.Module):
    """
    Single transformer block consisting of multi-head attention and MLP.

    Includes optional LayerScale, residual connections, and stochastic depth.

    Args:
        dim (int): Input embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Expansion ratio for MLP hidden dimension. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to add bias in QKV projections. Defaults to False.
        qk_norm (bool, optional): Whether to apply LayerNorm to Q and K. Defaults to False.
        proj_drop (float, optional): Dropout probability after projections. Defaults to 0.0.
        attn_drop (float, optional): Dropout probability in attention. Defaults to 0.0.
        init_values (float, optional): Initial value for LayerScale gamma. Defaults to None.
        drop_path (float, optional): Drop path (stochastic depth) probability. Defaults to 0.0.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        mlp_layer (nn.Module, optional): MLP implementation. Defaults to timm Mlp.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Transformer(nn.Module):
    """
    Transformer encoder consisting of stacked transformer blocks.

    Adapted from the timm library implementation.

    Args:
        embed_dim (int): Input embedding dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of stacked transformer blocks.
        mlp_ratio (float, optional): Expansion ratio for MLP hidden dimension. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to add bias in QKV projections. Defaults to False.
        qk_norm (bool, optional): Whether to apply LayerNorm to Q and K. Defaults to False.
        proj_drop (float, optional): Dropout probability after projections. Defaults to 0.0.
        attn_drop (float, optional): Dropout probability in attention. Defaults to 0.0.
        drop_path (float, optional): Drop path (stochastic depth) probability. Defaults to 0.0.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Forward pass of transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        for block in self.blocks:
            x = block(x)
        return x