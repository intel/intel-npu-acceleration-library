#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import NNFactory
import intel_npu_acceleration_library
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
    _ = torch_function(input)
    model.compile()

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
    _ = torch.clamp(input, min_val, max_val)
    model.compile()

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
    _ = torch_function(input)
    model.compile()

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
    model.compile()

    assert out.shape == list(reference.shape)

    result = model(x).detach().numpy()

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize("tensors", [2, 3, 4])
@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_concatenation(batch, hidden_dim, tensors, axis):
    x = [torch.rand((batch, hidden_dim)).to(torch.float16) for _ in range(tensors)]

    reference = torch.cat(x, dim=axis).numpy()

    model = NNFactory()
    par = [model.parameter(x[i].shape, np.float16) for i in range(tensors)]
    out = torch.cat(par, dim=axis)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model.run(*[x[i].numpy() for i in range(len(x))])

    assert 1 - r2_score(reference, result) < 0.01


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [128, 256])
@pytest.mark.parametrize("axis", [0, 1, -1, -2, None])
@pytest.mark.parametrize(
    "op", [torch.max, torch.mean, torch.min, torch.prod, torch.sum]
)
def test_reduce_operations(batch, hidden_dim, axis, op):

    x = torch.rand((batch, hidden_dim)).to(torch.float16)
    if axis is None:
        reference = op(x)
    else:
        if op in [torch.max, torch.min]:
            reference, _ = op(x, dim=axis)
        else:
            reference = op(x, dim=axis)
    reference = reference.numpy()

    model = NNFactory()
    par = model.parameter(x.shape, np.float16)
    out = op(par) if axis is None else op(par, dim=axis)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model.run(x.numpy())
    if not out.shape:
        assert 1 - r2_score([reference, 1], [result, 1]) < 0.01
    else:
        assert 1 - r2_score(reference, result) < 0.01


@pytest.mark.parametrize("channel", [16, 128])
@pytest.mark.parametrize("xydim", [4, 16])
@pytest.mark.parametrize(
    "fn",
    [torch.nn.functional.adaptive_avg_pool2d, torch.nn.functional.adaptive_max_pool2d],
)
@pytest.mark.parametrize("target_shape", [(1, 1), (2, 2), (4, 4)])
def test_adaptive_pooling(channel, xydim, fn, target_shape):

    if not intel_npu_acceleration_library.backend.npu_available() and any(
        shape > 1 for shape in target_shape
    ):
        pytest.xfail("Configuration unsupported on CPU")

    x = torch.rand(1, channel, xydim, xydim).to(torch.float16)
    reference = fn(x, target_shape).detach().numpy()

    model = NNFactory()
    par = model.parameter([1, channel, xydim, xydim], np.float16)
    out = fn(par, target_shape)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model(x).detach().numpy()

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


@pytest.mark.parametrize("channel", [16, 128])
@pytest.mark.parametrize("xydim", [16, 32])
@pytest.mark.parametrize("kernels", [1, 2, (2, 2), (4, 4)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
def test_avg_pooling(
    channel, xydim, kernels, stride, padding, ceil_mode, count_include_pad
):

    if kernels == 1 and stride > 1:
        pytest.skip("Stride > 1 not supported for kernel size 1")

    pool = lambda x: torch.nn.functional.avg_pool2d(
        x, kernels, stride, padding, ceil_mode, count_include_pad
    )

    x = torch.rand(1, channel, xydim, xydim).to(torch.float16)
    reference = pool(x).detach().numpy()

    model = NNFactory()
    par = model.parameter([1, channel, xydim, xydim], np.float16)
    out = pool(par)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model(x).detach().numpy()

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


@pytest.mark.parametrize("channel", [16, 128])
@pytest.mark.parametrize("xydim", [16, 32])
@pytest.mark.parametrize("kernels", [1, 2, (2, 2), (4, 4)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_max_pooling(channel, xydim, kernels, stride, padding, ceil_mode):

    if kernels == 1 and stride > 1:
        pytest.skip("Stride > 1 not supported for kernel size 1")

    pool = lambda x: torch.nn.functional.max_pool2d(
        x, kernels, stride, padding, dilation=1, ceil_mode=ceil_mode
    )

    x = torch.rand(1, channel, xydim, xydim).to(torch.float16)
    reference = pool(x).detach().numpy()

    model = NNFactory()
    par = model.parameter([1, channel, xydim, xydim], np.float16)
    out = pool(par)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model(x).detach().numpy()

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


@pytest.mark.parametrize(
    "shape", [(1, 3, 16, 16), (1, 16, 32, 32), (1, 64, 16, 16), (1, 128, 256)]
)
@pytest.mark.parametrize("op", [torch.add, torch.sub, torch.mul, torch.div])
@pytest.mark.parametrize("broadcast", [False, True])
def test_operations(shape, op, broadcast):
    torch.manual_seed(42)

    x = torch.rand(shape).to(torch.float16)
    if broadcast:
        y = torch.rand(shape[0]).to(torch.float16)
    else:
        y = torch.rand(shape).to(torch.float16)

    if torch.div == op:
        # Avoid NaNs and overflows
        y[y == 0] = 1
        y *= 10

    reference = op(x, y).detach().numpy()

    model = NNFactory()
    par = model.parameter(shape, np.float16)
    out = op(par, y)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model(x).detach().numpy()

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


@pytest.mark.parametrize("shape", [(1, 16, 16, 16), (1, 16, 32, 32), (1, 64, 16, 16)])
@pytest.mark.parametrize("mean", [0, 10, 120])
@pytest.mark.parametrize("variance", [1, 1.3])
@pytest.mark.parametrize("weight", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_batch_norm(shape, mean, variance, weight, bias):
    x = torch.rand(shape).to(torch.float16) * 10 + 3

    weight = torch.rand(shape[1]).to(torch.float16) if weight else None

    bias = torch.rand(shape[1]).to(torch.float16) if bias else None

    mean = torch.Tensor(shape[1] * [mean]).to(torch.float16)
    variance = torch.Tensor(shape[1] * [variance]).to(torch.float16)

    reference = (
        torch.nn.functional.batch_norm(x, mean, variance, weight=weight, bias=bias)
        .detach()
        .numpy()
    )

    model = NNFactory()
    par = model.parameter(shape, np.float16)
    out = torch.nn.functional.batch_norm(par, mean, variance, weight=weight, bias=bias)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model(x).detach().numpy()

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


@pytest.mark.parametrize("in_channels", [32, 128, 256])
@pytest.mark.parametrize("out_channels", [32, 128, 256])
@pytest.mark.parametrize("kernels", [1, 3])
@pytest.mark.parametrize("dim", [16, 32])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1, "same", "valid"])
@pytest.mark.parametrize("groups", [1, -1])
def test_conv(
    in_channels, out_channels, kernels, dim, bias, dtype, stride, padding, groups
):
    torch.manual_seed(42)

    if groups != 1 and in_channels != out_channels:
        pytest.skip("DW convolutions require in_channels == out_channels")

    if padding == "same" and stride > 1:
        pytest.skip("padding='same' is not supported for strided convolutions")

    if groups == -1:
        groups = in_channels

    x = torch.rand((1, in_channels, dim, dim)).to(torch.float16)

    weight = torch.rand((out_channels, in_channels // groups, kernels, kernels)).to(
        torch.float16
    )
    bias = torch.rand((out_channels,)).to(torch.float16) if bias else None

    reference = (
        torch.nn.functional.conv2d(x, weight, bias, stride, padding, groups=groups)
        .detach()
        .numpy()
    )

    model = NNFactory()
    par = model.parameter(x.shape, np.float16)

    out = torch.nn.functional.conv2d(par, weight, bias, stride, padding, groups=groups)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model(x).detach().numpy()

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01


def test_multiple_outputs():
    x = torch.rand((1, 16, 16, 16)).to(torch.float16)

    ref1 = torch.nn.functional.relu(x).detach().numpy()
    ref2 = torch.nn.functional.sigmoid(x).detach().numpy()

    model = NNFactory()
    par = model.parameter(x.shape, np.float16)
    _ = torch.nn.functional.relu(par)
    _ = torch.nn.functional.sigmoid(par)
    model.compile()

    result0, result1 = model(x)

    assert result0.shape == ref1.shape
    assert result1.shape == ref2.shape

    assert 1 - r2_score(result0.detach().numpy().flatten(), ref1.flatten()) < 0.01
    assert 1 - r2_score(result1.detach().numpy().flatten(), ref2.flatten()) < 0.01


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [128, 256])
@pytest.mark.parametrize("exponent", ["tensor", "float"])
@pytest.mark.parametrize("exponent_type", ["parameter", "constant"])
def test_power(batch, hidden_dim, exponent, exponent_type):

    x = torch.rand((batch, hidden_dim)).to(torch.float16)
    if exponent == "tensor":
        exponent = torch.rand((batch, hidden_dim)).to(torch.float16)
    else:
        exponent = torch.rand(1).to(torch.float16).item()

    reference = torch.pow(x, exponent=exponent).numpy()

    model = NNFactory()
    par = model.parameter(x.shape, np.float16)
    if exponent == "tensor" and exponent_type == "parameter":
        exponent_par = model.parameter(exponent.shape, np.float16)
        _ = torch.pow(par, exponent_par)
        model.compile()
        out = model(x, exponent).numpy()
    else:
        _ = torch.pow(par, exponent=exponent)
        model.compile()
        out = model(x).numpy()

    assert out.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(out).all(), "NPU output contains NaN or Inf"

    assert 1 - r2_score(reference, out) < 0.01


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [128, 256])
@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_logsoftmax(batch, hidden_dim, axis):
    x = torch.rand((batch, hidden_dim)).to(torch.float16)

    reference = torch.nn.functional.log_softmax(x, dim=axis).numpy()

    model = NNFactory()
    par = model.parameter(x.shape, np.float16)
    _ = torch.nn.functional.log_softmax(par, dim=axis)
    model.compile()

    out = model(x).numpy()

    assert out.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(out).all(), "NPU output contains NaN or Inf"

    assert 1 - r2_score(reference, out) < 0.01


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [128, 256])
@pytest.mark.parametrize("channels", [128, 256])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
# @pytest.mark.parametrize("beta", [0, 0.5, 1.0])
def test_addmm(batch, hidden_dim, channels, alpha, beta=1):
    torch.manual_seed(42)
    m1 = torch.rand((1, channels)).to(torch.float16)
    m2 = torch.rand((batch, hidden_dim)).to(torch.float16)
    m3 = torch.rand((hidden_dim, channels)).to(torch.float16)

    reference = torch.addmm(m1, m2, m3, alpha=alpha, beta=beta).numpy()

    model = NNFactory()
    par1 = model.parameter(m1.shape, np.float16)
    par2 = model.parameter(m2.shape, np.float16)
    par3 = model.parameter(m3.shape, np.float16)
    out = torch.addmm(par1, par2, par3, alpha=alpha, beta=beta)
    model.compile()

    assert out.shape == list(reference.shape)

    result = model(m1, m2, m3, alpha=alpha, beta=beta).detach().numpy()

    assert 1 - r2_score(reference.flatten(), result.flatten()) < 0.01
