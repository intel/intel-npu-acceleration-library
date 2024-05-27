#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from sklearn.metrics import r2_score
from intel_npu_acceleration_library.backend import MatMul, QMatMul
from intel_npu_acceleration_library.quantization import quantize_tensor
import numpy as np
import itertools
import pytest
import torch
import time


channels = [512, 768, 1024, 2048]
batches = [16, 128, 512, 1024]


@pytest.mark.parametrize(
    "batch,inC,outC", itertools.product(batches, channels, channels)
)
def test_matmul(batch, inC, outC):
    X = torch.rand((batch, inC), requires_grad=False).to(torch.float16)
    W = torch.rand((outC, inC), requires_grad=False).to(torch.float16)

    cpu_mm = X @ W.T

    mm = MatMul(inC, outC, batch)

    assert mm

    npu_mm = mm.run(X.numpy(), W.numpy())

    assert np.isfinite(npu_mm).all()

    assert 1 - r2_score(cpu_mm.numpy(), npu_mm) < 0.001


@pytest.mark.parametrize(
    "batch,inC,outC", itertools.product(batches, channels, channels)
)
def test_qmatmul_per_channel_scales(batch, inC, outC):

    X = torch.rand((batch, inC), requires_grad=False).to(torch.float16) - 0.5
    W = torch.rand((outC, inC), requires_grad=False).to(torch.float16)

    # Compute reference matmul
    cpu_mm = X @ W.T

    assert W.shape == (outC, inC) and W.dtype == torch.float16

    # Quantize the weights
    weights_quant, scale = quantize_tensor(W)

    assert scale.shape == (outC, 1) and scale.dtype == torch.float16
    assert weights_quant.shape == (outC, inC) and weights_quant.dtype == torch.int8
    assert weights_quant.shape == W.shape

    # Conversion done properly
    expected_W = weights_quant.to(torch.float16) * scale
    assert 1 - r2_score(expected_W.numpy(), W.numpy()) < 0.001

    mm = QMatMul(inC, outC, batch)

    assert mm

    npu_mm = mm.run(X.numpy(), weights_quant.numpy(), scale.numpy())

    assert np.isfinite(npu_mm).all()

    assert 1 - r2_score(cpu_mm.numpy(), npu_mm) < 0.001
