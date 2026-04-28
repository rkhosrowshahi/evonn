"""Depthwise separable 2D convolution (local replacement for separableconv-torch)."""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import torch.nn as nn
from torch.nn.common_types import _size_2_t


class SeparableConv2d(nn.Module):
    """Depthwise conv followed by pointwise 1x1 conv.

    Matches the layout of ``separableconv.nn.SeparableConv2d``: depthwise → norm →
    activation → pointwise → norm → activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        padding_mode: str = "zeros",
        dilation: _size_2_t = 1,
        depth_multiplier: int = 1,
        normalization_dw: Optional[str] = "bn",
        normalization_pw: Optional[str] = "bn",
        activation_dw: Callable[..., nn.Module] = nn.ReLU,
        activation_pw: Callable[..., nn.Module] = nn.ReLU,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        expansion_channels = max(in_channels * int(depth_multiplier), in_channels)
        if in_channels * depth_multiplier != expansion_channels:
            raise ValueError("depth_multiplier must be integer>=1")

        self.dwconv = nn.Conv2d(
            in_channels,
            expansion_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

        self.dwconv_normalization = self._norm2d(
            expansion_channels, normalization_dw, "normalization_dw"
        )
        self.dwconv_activation = activation_dw()

        self.pwconv = nn.Conv2d(
            expansion_channels,
            out_channels,
            1,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

        self.pwconv_normalization = self._norm2d(
            out_channels, normalization_pw, "normalization_pw"
        )
        self.pwconv_activation = activation_pw()

    @staticmethod
    def _norm2d(
        num_features: int, kind: Optional[str], param_name: str
    ) -> Optional[nn.Module]:
        if kind == "bn":
            return nn.BatchNorm2d(num_features)
        if kind == "in":
            return nn.InstanceNorm2d(num_features)
        if kind is None:
            return None
        warnings.warn(
            f"{param_name} is invalid. Default to ``None``. "
            "Use 'bn' for BatchNorm2d or 'in' for InstanceNorm2d.",
            stacklevel=3,
        )
        return None

    def forward(self, x):
        x = self.dwconv(x)
        if self.dwconv_normalization is not None:
            x = self.dwconv_normalization(x)
        x = self.dwconv_activation(x)
        x = self.pwconv(x)
        if self.pwconv_normalization is not None:
            x = self.pwconv_normalization(x)
        x = self.pwconv_activation(x)
        return x
