#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.tensor import Tensor
from intel_npu_acceleration_library.backend import NNFactory
from intel_npu_acceleration_library.dtypes import (
    float16,
    float32,
    float64,
    int4,
    int8,
    int16,
    int32,
    int64,
)
import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score


@pytest.mark.parametrize("shape", [[1, 128, 13, 13], [12, 231]])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        float16,
        float32,
        float64,
        int4,
        int8,
        int16,
        int32,
        int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bfloat16,
    ],
)
def test_tensor_creation(shape, dtype):
    model = NNFactory()
    tensor = model.parameter(shape, dtype)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == shape
    assert tensor.dtype == dtype


def test_matmul():
    model = NNFactory()
    tensor = model.parameter([16, 256], np.float16)
    weights = model.constant(np.ones([128, 256], dtype=np.float16))

    out = tensor @ weights
    model.compile()

    assert isinstance(out, Tensor)
    assert out.shape == [16, 128]
    assert out.dtype == np.float16


def test_model_creation():
    model = NNFactory()
    t1 = model.parameter([1, 128, 32, 64], float16)
    assert t1.shape == [1, 128, 32, 64]
    assert t1.dtype == float16
    t2 = model.parameter([128 * 32 * 64], int8)
    assert t2.shape == [128 * 32 * 64]
    assert t2.dtype == int8

    t2 = t2.to(float16)

    assert t2.dtype == float16

    t2 = t2.reshape(128, 64, 32)

    assert t2.shape == [128, 64, 32]

    t2 = t2.unsqueeze(0)

    assert t2.shape == [1, 128, 64, 32]

    t2 = t2.T

    assert t2.shape == [1, 128, 32, 64]

    sum = t1 + t2

    dd = sum.transpose(1, 2)

    assert dd.shape == [1, 32, 128, 64]

    ff = dd.reshape([32, 128, 64])

    assert ff.shape == [32, 128, 64]

    assert ff.dim() == 3

    gg = ff.view(1, -1, 1, 1)

    assert gg.shape == [1, 32 * 128 * 64, 1, 1]

    model.compile()


def test_slice():

    model = NNFactory()
    tensor = model.parameter([1, 128, 32, 64], float16)

    assert tensor[:, 0:64, :, :].shape == [1, 64, 32, 64]

    assert tensor[0, 0:64:2, :, :].shape == [1, 32, 32, 64]

    assert tensor[:, :-2, :, :].shape == [1, 126, 32, 64]

    assert tensor[..., :-2, :].shape == [1, 128, 30, 64]

    assert tensor[:, 10:20, ...].shape == [1, 10, 32, 64]


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize(
    "activation",
    [
        "acos",
        "asin",
        "atan",
        "acosh",
        "asinh",
        "atanh",
        "cosh",
        "sinh",
        "tanh",
        "cos",
        "sin",
        "tan",
        "ceiling",
        "clamp",
        "erf",
        "exp",
        "floor",
        "log",
        "round",
        "sign",
        "sigmoid",
        "softmax",
        "sqrt",
    ],
)
def test_operations_1(batch, hidden_dim, activation):

    # X in the range [-0.5, 0.5]
    X = torch.rand((batch, hidden_dim)).to(torch.float16) - 0.5

    if activation == "acosh":
        # acosh is only defined for x >= 1
        X += 1.5
    elif activation in ["sqrt", "tanh"]:
        # sqrt and tanh are only defined for x >= 0
        X += 0.5
    elif activation == "log":
        # log needs a bigger input to avoid negative overflow in fp16
        # log in range [0.5, 1.5]
        X += 1

    if activation == "softmax":
        reference = eval(f"X.{activation}(dim=-1).numpy()")
    elif activation == "clamp":
        reference = eval(f"X.{activation}(min=-0.5, max=0.5).numpy()")
    elif activation == "ceiling":
        reference = eval(f"X.ceil().numpy()")
    else:
        reference = eval(f"X.{activation}().numpy()")

    model = NNFactory()
    t1 = model.parameter(X.shape)

    if activation == "softmax":
        _ = eval(f"t1.{activation}(dim=-1)")
    elif activation == "clamp":
        _ = eval(f"t1.{activation}(min=-0.5, max=0.5)")
    else:
        _ = eval(f"t1.{activation}()")
    model.compile()

    result = model(X).numpy()

    assert result.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(result).all(), "NPU output contains NaN or Inf"

    assert 1 - r2_score(reference, result) < 0.001


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize(
    "activation",
    [
        "elu",
        "grn",
        "hsigmoid",
        "hswish",
        "mish",
        "relu",
        "softplus",
    ],
)
def test_operations_2(batch, hidden_dim, activation):

    # X in the range [-0.5, 0.5]
    X = torch.rand((batch, hidden_dim)).to(torch.float16) - 0.5

    if activation == "grn":
        reference = torch.nn.functional.normalize(X, p=2.0, dim=-1, eps=1e-12).numpy()
    elif activation == "hswish":
        reference = torch.nn.functional.hardswish(X).numpy()
    elif activation == "hsigmoid":
        reference = torch.nn.functional.hardsigmoid(X).numpy()
    else:
        reference = eval(f"torch.nn.functional.{activation}(X)").numpy()

    model = NNFactory()
    t1 = model.parameter(X.shape)

    if activation == "grn":
        _ = eval(f"t1.{activation}(bias=1e-12)")
    else:
        _ = eval(f"t1.{activation}()")

    model.compile()

    result = model(X).numpy()

    assert result.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(result).all(), "NPU output contains NaN or Inf"

    assert 1 - r2_score(reference, result) < 0.001


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [128, 256])
@pytest.mark.parametrize("axis", [0, 1, -1, -2, None])
@pytest.mark.parametrize("op", ["max", "mean", "min", "prod", "sum"])
def test_reduce_operations(batch, hidden_dim, axis, op):

    X = torch.rand((batch, hidden_dim)).to(torch.float16)

    if axis is None:
        reference = eval(f"X.{op}()")
    else:
        if op in ["max", "min"]:
            reference, _ = eval(f"X.{op}(dim=axis)")
        else:
            reference = eval(f"X.{op}(dim=axis)")
    reference = reference.numpy()

    print(X.sum())
    model = NNFactory()
    t1 = model.parameter(X.shape)
    _ = eval(f"t1.{op}()") if axis is None else eval(f"t1.{op}(dim=axis)")
    model.compile()

    result = model(X).numpy()

    assert result.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(result).all(), "NPU output contains NaN or Inf"

    if not result.shape:
        assert 1 - r2_score([reference, 1], [result, 1]) < 0.01
    else:
        assert 1 - r2_score(reference, result) < 0.01
