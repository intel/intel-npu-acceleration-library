#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.quantization import quantize_tensor
from intel_npu_acceleration_library.backend import MatMul, QMatMul, Linear, QLinear
import numpy as np
import intel_npu_acceleration_library
import pytest
import torch
import os


def test_profiling_matmul():

    if not intel_npu_acceleration_library.backend.npu_available():
        pytest.xfail("NPU not available")

    X = np.random.uniform(-1, 1, (512, 2048)).astype(np.float16)
    W = np.random.uniform(-1, 1, (512, 2048)).astype(np.float16)
    W_q, scale = quantize_tensor(torch.tensor(W))
    if os.path.exists("profiling.json"):
        os.remove("profiling.json")

    MatMul(W.shape[1], W.shape[0], X.shape[0], profile=True).run(X, W)
    assert os.path.isfile("profiling.json")
    os.remove("profiling.json")

    QMatMul(W.shape[1], W.shape[0], X.shape[0], profile=True).run(
        X, W_q.numpy(), scale.numpy()
    )
    assert os.path.isfile("profiling.json")
    os.remove("profiling.json")

    Linear(W.shape[1], W.shape[0], X.shape[0], profile=True).run(X, W, op_id=0)
    assert os.path.isfile("profiling.json")
    os.remove("profiling.json")

    QLinear(W.shape[1], W.shape[0], X.shape[0], profile=True).run(
        X, W_q.numpy(), scale.numpy(), op_id=0
    )
    assert os.path.isfile("profiling.json")
    os.remove("profiling.json")
