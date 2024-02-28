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
    X = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
    W = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

    mm = MatMul(inC, outC, batch)

    assert mm

    npu_mm = mm.run(X, W)

    assert np.isfinite(npu_mm).all()

    cpu_mm = np.matmul(X, W.T)

    assert 1 - r2_score(cpu_mm, npu_mm) < 0.001


@pytest.mark.parametrize(
    "batch,inC,outC", itertools.product(batches, channels, channels)
)
def test_qmatmul_per_channel_scales(batch, inC, outC):

    X = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
    W = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

    # Compute reference matmul
    cpu_mm = np.matmul(X, W.T)

    assert W.shape == (outC, inC) and W.dtype == np.float16

    # Quantize the weights
    weights_quant, scale = quantize_tensor(torch.from_numpy(W))

    assert scale.shape == (outC, 1) and scale.dtype == torch.float16
    assert weights_quant.shape == (outC, inC) and weights_quant.dtype == torch.int8
    assert weights_quant.shape == W.shape

    # Conversion done properly
    assert 1 - r2_score(weights_quant.to(torch.float16) * scale, W) < 0.001

    mm = QMatMul(inC, outC, batch)

    assert mm

    npu_mm = mm.run(X, weights_quant.numpy(), scale.numpy())

    assert np.isfinite(npu_mm).all()

    assert 1 - r2_score(cpu_mm, npu_mm) < 0.001
