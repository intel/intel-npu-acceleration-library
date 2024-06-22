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
