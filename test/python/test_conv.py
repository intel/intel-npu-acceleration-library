#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig
from sklearn.metrics import r2_score
import pytest
import torch


class DummyConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernels,
        bias,
        groups,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        if groups == -1:
            groups = out_channels
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernels,
            bias=bias,
            groups=groups,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        return self.conv(x)


@pytest.mark.parametrize("in_channels", [32, 128, 256])
@pytest.mark.parametrize("out_channels", [32, 128, 256])
@pytest.mark.parametrize("kernels", [1, 3])
@pytest.mark.parametrize("dim", [16, 32])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("groups", [1, -1])
def test_conv(
    in_channels, out_channels, kernels, dim, bias, dtype, stride, padding, groups
):
    torch.manual_seed(42)

    if groups != 1 and in_channels != out_channels:
        pytest.skip("DW convolutions require in_channels == out_channels")

    with torch.no_grad():
        X = torch.rand((1, in_channels, dim, dim), dtype=torch.float16)
        conv = DummyConv(
            in_channels,
            out_channels,
            kernels,
            bias=bias,
            groups=groups,
            stride=stride,
            padding=padding,
        ).half()
        conv.conv.weight.data *= 128
        y_ref = conv(X)

        compiler_conf = CompilerConfig(dtype=dtype)
        npu_conv = intel_npu_acceleration_library.compile(conv, compiler_conf)
        y = npu_conv(X)

        assert y.dtype == y_ref.dtype
        assert y.shape == y_ref.shape
        if dtype == torch.int8:
            assert 1 - r2_score(y_ref.flatten().numpy(), y.flatten().numpy()) < 0.05
        else:
            assert 1 - r2_score(y_ref.flatten().numpy(), y.flatten().numpy()) < 0.001
