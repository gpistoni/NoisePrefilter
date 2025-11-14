import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from pix2pix.nn_utils import *


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        in_channels,
        attn_heads=1,
        group_norm_num=32,
        use_checkpoint=True,
        use_xformer=True,
    ):
        super().__init__()
        self.channels = in_channels
        self.num_heads = attn_heads

        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(group_norm_num, in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads, use_xformer)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout=0.0,
        use_checkpoint=True,
        norm_type="batchnorm2d",
        group_norm_num=32,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ops = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            normalization("batchnorm2d", out_channels, group_norm_num),
            nn.Dropout(p=dropout),
            nn.SiLU(),
        )

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        return self.ops(x)


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        use_checkpoint=True,
        norm_type="batchnorm2d",
        group_norm_num=32,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ops = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            norm_type=norm_type,
            group_norm_num=group_norm_num,
        )

    def forward(self, x):
        return self.ops(x)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        use_checkpoint=True,
        norm_type="batchnorm2d",
        group_norm_num=32,
        mode="bilinear",
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ops = nn.Sequential(
            Interpolate(scale_factor=2, mode=mode),
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                norm_type=norm_type,
                group_norm_num=group_norm_num,
            ),
        )

    def forward(self, x):
        return self.ops(x)
