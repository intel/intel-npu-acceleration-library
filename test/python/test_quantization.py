#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from sklearn.metrics import r2_score
import numpy as np
import intel_npu_acceleration_library
import pytest
import torch

import intel_npu_acceleration_library.quantization


class NN(torch.nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        self.l1 = torch.nn.Linear(inC, outC, bias=False)

    def forward(self, x):
        return self.l1(x)


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("inC", [256, 512])
@pytest.mark.parametrize("outC", [256, 512])
def test_explicit_quantization(batch, inC, outC):
    module = intel_npu_acceleration_library.backend.NNFactory()
    assert module

    input = module.parameter((batch, inC))
    assert input

    output = module.linear(input, outC, inC)
    assert output

    module.compile(output)

    X = np.random.random((batch, inC)).astype(np.float16)
    W = np.random.randint(-127, 127, (outC, inC)).astype(np.int8)
    S = np.random.random((outC, 1)).astype(np.float32)

    w_float = W.astype(np.float16) * S
    y_ref = np.matmul(X, w_float.T)

    y = module.run(X, (W, S), op_id="0000")

    assert 1 - r2_score(y_ref, y) < 0.01


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("inC", [256, 512])
@pytest.mark.parametrize("outC", [256, 512])
def test_i8_quantization(batch, inC, outC):
    module = intel_npu_acceleration_library.backend.NNFactory()
    assert module

    input = module.parameter((batch, inC))
    assert input

    output = module.linear(input, outC, inC, False, wt_dtype=np.int8)
    assert output

    module.compile(output)

    X = np.random.random((batch, inC)).astype(np.float16)
    W = np.random.randint(-127, 127, (outC, inC)).astype(np.int8)
    S = np.random.random((outC, 1)).astype(np.float16)

    w_float = W.astype(np.float16) * S
    y_ref = np.matmul(X, w_float.T)

    y = module.run(X, (W, S), op_id="0000")

    assert 1 - r2_score(y_ref, y) < 0.01


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("inC", [256, 512])
@pytest.mark.parametrize("outC", [256, 512])
def test_compiled_quantized(batch, inC, outC):

    intel_npu_acceleration_library.backend.clear_cache()

    torch.manual_seed(0)
    X = torch.rand((batch, inC), dtype=torch.float16) - 0.5
    # X = np.random.random((batch, inC)).astype(np.float16)

    model = NN(inC, outC)
    y_ref = model(X.to(torch.float32)).detach()
    compiled_model = intel_npu_acceleration_library.compile(model, torch.int8)
    assert compiled_model

    y1 = compiled_model(X).detach()

    assert 1 - r2_score(y_ref.numpy(), y1.numpy()) < 0.01


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("inC", [256, 512])
@pytest.mark.parametrize("outC", [256, 512])
def test_i4_quantization(batch, inC, outC):

    module = intel_npu_acceleration_library.backend.NNFactory()
    assert module

    input = module.parameter((batch, inC))
    assert input
    # u8 represents packed i4 dtypes
    output = module.linear(input, outC, inC, False, wt_dtype=np.uint8)
    assert output

    module.compile(output)

    X = np.random.random((batch, inC)).astype(np.float16)
    S = np.random.random((outC, 1)).astype(np.float16)
    W = np.random.randint(-8, 7, (outC, inC)).astype(np.int8)

    w_float = W.astype(np.float16) * S
    y_ref = np.matmul(X, w_float.T)

    # Compress the weights for int4
    W_npu = intel_npu_acceleration_library.quantization.compress_to_i4(
        torch.from_numpy(W)
    ).numpy()

    y = module.run(X, (W_npu, S), op_id="0000")

    # assert y has no NaN
    assert not np.isnan(y).any()

    # assert y has no Inf
    assert not np.isinf(y).any()

    # Check for correctness vs reference
    assert 1 - r2_score(y_ref.flatten(), y.flatten()) < 0.01
