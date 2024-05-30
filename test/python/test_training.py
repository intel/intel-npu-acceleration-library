#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


from sklearn.metrics import r2_score
from intel_npu_acceleration_library import compile
import torch
import pytest
import copy

# Three different values to pick the errors
in_c = 128
out_c = 512
batch = 256


class NN(torch.nn.Module):
    def __init__(self, inc, outc, bias) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(inc, outc, bias=bias)
        self.linear2 = torch.nn.Linear(outc, inc, bias=bias)

    def forward(self, x):
        return self.linear2(torch.nn.functional.relu(self.linear(x)))


@pytest.fixture
def model_no_bias():
    return compile(NN(inc=in_c, outc=out_c, bias=False))


@pytest.fixture
def model():
    return compile(NN(inc=in_c, outc=out_c, bias=True))


def test_parameters(model, model_no_bias):
    assert len(list(model.parameters())) == 4
    assert len(list(model_no_bias.parameters())) == 2


def test_gradient():

    npu_model = NN(inc=in_c, outc=out_c, bias=True).half()
    cpu_model = NN(inc=in_c, outc=out_c, bias=True).half()
    cpu_model.load_state_dict(copy.deepcopy(npu_model.state_dict()))

    # Compile one of the model on npu
    compile(npu_model, training=True)

    x = torch.rand([batch, in_c]).half()
    yref = torch.rand([batch, in_c]).half()

    opt1 = torch.optim.SGD(npu_model.parameters(), lr=0.5)
    opt2 = torch.optim.SGD(cpu_model.parameters(), lr=0.5)

    for idx in range(100):

        # Check the parameters are the same
        for p1, p2 in zip(npu_model.parameters(), cpu_model.parameters()):
            assert p1.dtype == p2.dtype
            assert 1 - r2_score(p1.detach().numpy(), p2.detach().numpy()) < 0.001, idx

        opt1.zero_grad()
        opt2.zero_grad()

        y1 = npu_model(x)
        y2 = cpu_model(x)

        model_loss = torch.mean(((yref - y1) ** 2))
        model_loss.backward()

        model_loss = torch.mean(((yref - y2) ** 2))
        model_loss.backward()

        assert (torch.abs(model_loss - model_loss) / model_loss).item() < 0.001

        opt1.step()
        opt2.step()
