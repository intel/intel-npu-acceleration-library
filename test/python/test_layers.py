#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import MLP
from sklearn.metrics import r2_score
import pytest
import torch


class MLP_PT(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation, bias=False):
        super().__init__()
        self.l1 = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        if activation == "swiglu":
            self.swiglu = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
            # pytorch call swish silu
            self.fn = lambda x: torch.nn.functional.silu(self.l1(x)) * self.swiglu(x)
        elif activation == "gelu":
            self.fn = lambda x: torch.nn.functional.gelu(self.l1(x), approximate="tanh")
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
@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
def test_mlp(batch, hidden_dim, intermediate_dim, bias, activation):

    module = MLP_PT(hidden_dim, intermediate_dim, activation, bias)

    X = torch.rand((batch, hidden_dim)).to(torch.float16) - 0.5

    reference = module(X.to(torch.float32)).to(torch.float16).numpy()

    model = MLP((batch, hidden_dim), intermediate_dim, activation, bias)
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
