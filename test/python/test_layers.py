#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import MLP
from sklearn.metrics import r2_score
import pytest
import torch


class MLP_PT(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        activation,
        min,
        max,
        grn_bias,
        bias=False,
        alpha=1.0,
    ):
        super().__init__()
        self.l1 = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        if activation == "swiglu":
            self.swiglu = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
            # pytorch call swish silu
            self.fn = lambda x: torch.nn.functional.silu(self.l1(x)) * self.swiglu(x)
        elif activation == "abs_act":
            self.fn = lambda x: torch.abs(self.l1(x))
        elif activation == "acos_act":
            self.fn = lambda x: torch.acos(self.l1(x))
        elif activation == "asin_act":
            self.fn = lambda x: torch.asin(self.l1(x))
        elif activation == "atan_act":
            self.fn = lambda x: torch.atan(self.l1(x))
        elif activation == "ceiling":
            self.fn = lambda x: torch.ceil(self.l1(x))
        elif activation == "clamp":
            self.fn = lambda x: torch.clamp(self.l1(x), min, max)
        elif activation == "cos_act":
            self.fn = lambda x: torch.cos(self.l1(x))
        elif activation == "cosh_act":
            self.fn = lambda x: torch.cosh(self.l1(x))
        elif activation == "erf_act":
            self.fn = lambda x: torch.erf(self.l1(x))
        elif activation == "elu":
            self.fn = lambda x: torch.nn.functional.elu(
                self.l1(x), alpha=alpha, inplace=False
            )
        elif activation == "exp_act":
            self.fn = lambda x: torch.exp(self.l1(x))
        elif activation == "floor_act":
            self.fn = lambda x: torch.floor(self.l1(x))
        elif activation == "grn":
            self.fn = lambda x: torch.nn.functional.normalize(
                self.l1(x), p=2.0, dim=-1, eps=grn_bias
            )
        elif activation == "gelu":
            self.fn = lambda x: torch.nn.functional.gelu(self.l1(x), approximate="tanh")
        elif activation == "log_act":
            self.fn = lambda x: torch.log(self.l1(x))
        elif activation == "negative":
            self.fn = lambda x: torch.neg(self.l1(x))
        elif activation == "relu":
            self.fn = lambda x: torch.nn.functional.relu(self.l1(x))
        elif activation == "sigmoid":
            self.fn = lambda x: torch.nn.functional.sigmoid(self.l1(x))
        elif activation == "sign":
            self.fn = lambda x: torch.sign(self.l1(x))
        elif activation == "sin_act":
            self.fn = lambda x: torch.sin(self.l1(x))
        elif activation == "sinh_act":
            self.fn = lambda x: torch.sinh(self.l1(x))
        elif activation == "sqrt_act":
            self.fn = lambda x: torch.sqrt(self.l1(x))
        elif activation == "tan_act":
            self.fn = lambda x: torch.tan(self.l1(x))
        elif activation == "tanh_act":
            self.fn = lambda x: torch.tanh(self.l1(x))
        elif activation == "acosh_act":
            self.fn = lambda x: torch.acosh(self.l1(x))
        elif activation == "asinh_act":
            self.fn = lambda x: torch.asinh(self.l1(x))
        elif activation == "atanh_act":
            self.fn = lambda x: torch.atanh(self.l1(x))
        elif activation == "hswish":
            self.fn = lambda x: torch.nn.functional.hardswish(self.l1(x))
        elif activation == "mish":
            self.fn = lambda x: torch.nn.functional.mish(self.l1(x))
        elif activation == "softplus":
            self.fn = lambda x: torch.nn.functional.softplus(self.l1(x))
        elif activation == "hsigmoid":
            self.fn = lambda x: torch.nn.functional.hardsigmoid(
                self.l1(x), inplace=False
            )
        elif activation == "round_act":
            self.fn = lambda x: torch.round(self.l1(x), decimals=0)
        elif activation == "softsign":
            self.fn = lambda x: torch.nn.functional.softsign(self.l1(x))
        else:
            raise RuntimeError(f"Unsupported activation: {activation}")
        self.l2 = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        y = self.fn(x)
        return self.l2(y)


@torch.no_grad
@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize("intermediate_dim", [512, 256])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("min", [-1.0])
@pytest.mark.parametrize("max", [1.0])
@pytest.mark.parametrize("grn_bias", [1e-12])
@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize(
    "activation",
    [
        "abs_act",
        "acos_act",
        "asin_act",
        "atan_act",
        "ceiling",
        "clamp",
        "cos_act",
        "cosh_act",
        "erf_act",
        "elu",
        "exp_act",
        "floor_act",
        "grn",
        "gelu",
        "log_act",
        "negative",
        "relu",
        "sigmoid",
        "sign",
        "sin_act",
        "sinh_act",
        "sqrt_act",
        "tan_act",
        "tanh_act",
        "acosh_act",
        "asinh_act",
        "atanh_act",
        "hswish",
        "mish",
        "softplus",
        "hsigmoid",
        "round_act",
        "softsign",
        "swiglu",
    ],
)
def test_mlp(
    batch, hidden_dim, intermediate_dim, bias, activation, min, max, grn_bias, alpha
):

    module = MLP_PT(
        hidden_dim, intermediate_dim, activation, min, max, grn_bias, bias, alpha
    )

    X = torch.rand((batch, hidden_dim)).to(torch.float16) - 0.5

    module.to(torch.float16)
    reference = module(X).numpy()

    model = MLP(
        (batch, hidden_dim),
        intermediate_dim,
        activation,
        bias,
        min=min,
        max=max,
        grn_bias=grn_bias,
        alpha=alpha,
    )
    weights = list(module.parameters())
    if activation == "swiglu":
        if bias:
            assert len(weights) == 6
        else:
            assert len(weights) == 3
    else:
        if bias:
            assert len(weights) == 4
        else:
            assert len(weights) == 2

    out = model.run(
        X.numpy(),
        *[w.to(torch.float16).numpy() for w in weights],
        op_id="000",
    )

    assert out.shape == reference.shape

    assert 1 - r2_score(reference, out) < 0.001
