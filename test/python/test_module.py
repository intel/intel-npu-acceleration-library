#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.nn.module import (
    convert_to_npu_module,
    NPUModuleWrapper,
)
from sklearn.metrics import r2_score
import pytest
import torch


class DummyModule(torch.nn.Module):
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


class DummyModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = DummyModule()
        self.register_buffer("xx", torch.rand(128, 128).to(torch.float16))

    def forward(self, x, y, sum=False):
        return (x * y)[:, :128] + self.l1(x, y, sum=sum) * self.xx


@pytest.mark.parametrize("sum", [True, False])
def test_torch_nested_module(sum):

    model = DummyModule2()

    assert isinstance(model, torch.nn.Module)

    model = convert_to_npu_module(model)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, NPUModuleWrapper)

    x = torch.rand(128, 256).to(torch.float16)
    y = torch.rand(128, 256).to(torch.float16)

    reference = model(x, y=y, sum=sum)

    # Run on NPU
    model.to("NPU")

    result = model(x, y=y, sum=sum)

    assert 1 - r2_score(reference.detach().numpy(), result.detach().numpy()) < 0.001


@pytest.mark.parametrize("channels", [16, 128, 256])
@pytest.mark.parametrize("dim", [16, 32, 64])
def test_batch_norm(channels, dim):

    x = torch.rand(1, channels, dim, dim).to(torch.float16)

    model = torch.nn.BatchNorm2d(channels).half().eval()

    assert isinstance(model, torch.nn.Module)

    model = convert_to_npu_module(model)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, NPUModuleWrapper)

    reference = model(x)

    # Run on NPU
    model.to("NPU")

    result = model(x)

    assert (
        1
        - r2_score(
            reference.flatten().detach().numpy(), result.flatten().detach().numpy()
        )
        < 0.001
    )


def test_resnet():

    model = (
        torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)
        .half()
        .eval()
    )
    x = torch.randint(0, 256, (1, 3, 224, 224)).to(torch.float16)

    reference = model(x)

    model = model.to("npu")

    result = model(x.to("npu"))

    r2 = r2_score(
        reference.flatten().detach().numpy(), result.flatten().detach().numpy()
    )

    assert 1 - r2 < 0.01
