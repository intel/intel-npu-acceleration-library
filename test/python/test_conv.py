#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


import intel_npu_acceleration_library
from sklearn.metrics import r2_score
import pytest
import torch


class DummyConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernels, bias):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernels, bias=bias)

    def forward(self, x):
        return self.conv(x)


@pytest.mark.parametrize("in_channels", [128, 256, 512])
@pytest.mark.parametrize("out_channels", [128, 256, 512])
@pytest.mark.parametrize("kernels", [1, 3])
@pytest.mark.parametrize("dim", [16, 128])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_conv(in_channels, out_channels, kernels, dim, bias, dtype):
    if dtype == torch.int8 and kernels > 1:
        pytest.skip("int8 only supports kernel size 1")

    with torch.no_grad():
        X = torch.rand((1, in_channels, dim, dim), dtype=torch.float32)

        conv = DummyConv(in_channels, out_channels, kernels, bias=bias)
        conv.conv.weight.data *= 128
        y_ref = conv(X)

        npu_conv = intel_npu_acceleration_library.compile(conv, dtype)
        y = npu_conv(X)

        assert y.dtype == y_ref.dtype
        assert y.shape == y_ref.shape
        if dtype == torch.int8:
            assert 1 - r2_score(y_ref.flatten().numpy(), y.flatten().numpy()) < 0.05
        else:
            assert 1 - r2_score(y_ref.flatten().numpy(), y.flatten().numpy()) < 0.001
