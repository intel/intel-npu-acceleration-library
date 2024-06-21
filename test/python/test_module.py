#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.nn.module import NPUModule
from sklearn.metrics import r2_score
import pytest
import torch


class DummyModule(NPUModule):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(128, 128).to(torch.float16))
        self.l1 = torch.nn.Linear(256, 128).half()

    def forward(self, x, y, sum=False):
        if sum:
            z = y + x
        else:
            z = x - y
        return self.l1(z) * self.w


class DummyModule2(NPUModule):
    def __init__(self):
        super().__init__()
        self.l1 = DummyModule()
        self.register_buffer("xx", torch.rand(128, 128).to(torch.float16))

    def forward(self, x, y, sum=False):
        return (x * y)[:, :128] + self.l1(x, y, sum=sum) * self.xx


@pytest.mark.parametrize("sum", [True, False])
def test_torch_nested_module(sum):

    model = DummyModule2()
    x = torch.rand(128, 256).to(torch.float16)
    y = torch.rand(128, 256).to(torch.float16)

    reference = model(x, y=y, sum=sum)

    # Run on NPU
    model.to("NPU")

    result = model(x, y=y, sum=sum)

    assert 1 - r2_score(reference.detach().numpy(), result.detach().numpy()) < 0.001
