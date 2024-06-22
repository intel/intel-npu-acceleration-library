#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import NNFactory
from sklearn.metrics import r2_score
import numpy as np
import torch
import pytest


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize(
    "activation",
    [
        "log",
        "tanh",
        "sqrt",
        "abs",
        "acos",
        "asin",
        "atan",
        "cos",
        "cosh",
        "erf",
        "exp",
        "floor",
        "sin",
        "sinh",
        "tan",
        "acosh",
        "asinh",
        "atanh",
        "round",
        "sign",
        "ceil",
        "neg",
    ],
)
def test_activation(batch, hidden_dim, activation):

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

    torch_function = eval(f"torch.{activation}")

    reference = torch_function(X).numpy()

    model = NNFactory()
    input = model.parameter(X.shape)
    output = torch_function(input)
    model.compile(output)

    out = model.run(X.numpy())

    assert out.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(out).all(), "NPU output contains NaN or Inf"

    assert 1 - r2_score(reference, out) < 0.001


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize("min_val", [-1, -0.5, None])
@pytest.mark.parametrize("max_val", [1, 0.5, None])
def test_clamp(batch, hidden_dim, min_val, max_val):

    if min_val is None and max_val is None:
        pytest.skip("min_val and max_val cannot be both None")

    # X in the range [-0.5, 0.5]
    X = torch.rand((batch, hidden_dim)).to(torch.float16) - 0.5

    reference = torch.clamp(X, min_val, max_val).numpy()

    model = NNFactory()
    input = model.parameter(X.shape)
    output = torch.clamp(input, min_val, max_val)
    model.compile(output)

    out = model.run(X.numpy())

    assert out.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(out).all(), "NPU output contains NaN or Inf"

    assert 1 - r2_score(reference, out) < 0.001


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize(
    "activation",
    [
        "elu",
        "gelu",
        "relu",
        "sigmoid",
        "hardswish",
        "mish",
        "softplus",
        "hardsigmoid",
        "silu",
    ],
)
def test_functional_activation(batch, hidden_dim, activation):

    if activation == "gelu":
        torch_function = lambda x: torch.nn.functional.gelu(x, approximate="none")
        run_activation_test(torch_function, batch, hidden_dim)

        torch_function = lambda x: torch.nn.functional.gelu(x, approximate="tanh")
        run_activation_test(torch_function, batch, hidden_dim)
    else:
        torch_function = eval(f"torch.nn.functional.{activation}")
        run_activation_test(torch_function, batch, hidden_dim)


def run_activation_test(torch_function, batch, hidden_dim):

    # X in the range [-0.5, 0.5]
    X = torch.rand((batch, hidden_dim)).to(torch.float16) - 0.5

    reference = torch_function(X).numpy()

    model = NNFactory()
    input = model.parameter(X.shape)
    output = torch_function(input)
    model.compile(output)

    out = model.run(X.numpy())

    assert out.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(out).all(), "NPU output contains NaN or Inf"

    assert 1 - r2_score(reference, out) < 0.001


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize("start_dim", [0, 1])
@pytest.mark.parametrize("end_dim", [-1])
def test_flatten(batch, hidden_dim, start_dim, end_dim):

    x = torch.rand((batch, hidden_dim)).to(torch.float16)
    reference = torch.flatten(x, start_dim, end_dim).numpy()

    model = NNFactory()
    par = model.parameter(x.shape, np.float16)
    out = torch.flatten(par, start_dim, end_dim)
    model.compile(out)

    assert out.shape == list(reference.shape)

    result = model.run(x.numpy())

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


@pytest.mark.parametrize("channel", [16, 128])
@pytest.mark.parametrize("xydim", [4, 16])
@pytest.mark.parametrize(
    "fn",
    [torch.nn.functional.adaptive_avg_pool2d, torch.nn.functional.adaptive_max_pool2d],
)
@pytest.mark.parametrize("target_shape", [(1, 1), (2, 2), (4, 4)])
def test_adaptive_pooling(channel, xydim, fn, target_shape):

    x = torch.rand(1, channel, xydim, xydim).to(torch.float16)
    reference = fn(x, target_shape).detach().numpy()

    model = NNFactory()
    par = model.parameter([1, channel, xydim, xydim], np.float16)
    out = fn(par, target_shape)
    model.compile(out)

    assert out.shape == list(reference.shape)

    result = model.run(x.numpy())

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


@pytest.mark.parametrize("channel", [16, 128])
@pytest.mark.parametrize("xydim", [16, 32])
@pytest.mark.parametrize(
    "fn", [torch.nn.functional.avg_pool2d]  # , torch.nn.functional.max_pool2d],
)
@pytest.mark.parametrize("kernels", [1, 2, (2, 2), (4, 4)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
def test_avg_pooling(
    channel, xydim, fn, kernels, stride, padding, ceil_mode, count_include_pad
):

    if kernels == 1 and stride > 1:
        pytest.skip("Stride > 1 not supported for kernel size 1")

    pool = lambda x: fn(x, kernels, stride, padding, ceil_mode, count_include_pad)

    x = torch.rand(1, channel, xydim, xydim).to(torch.float16)
    reference = pool(x).detach().numpy()

    model = NNFactory()
    par = model.parameter([1, channel, xydim, xydim], np.float16)
    out = pool(par)
    model.compile(out)

    assert out.shape == list(reference.shape)

    result = model.run(x.numpy())

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01
