#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.nn.module import (
    convert_to_npu_module,
    NPUModuleWrapper,
    NPUContextManager,
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


@pytest.mark.parametrize("shape", [(1, 32, 16, 16), (256, 1024)])
def test_context_manager(shape):
    with NPUContextManager() as model:
        assert model is not None

        t0 = model.Tensor(shape, torch.float16)
        t1 = model.Tensor(shape, torch.float16)

        t2 = t0 * t1

        t3 = torch.nn.functional.relu(t2)
        t4 = torch.nn.functional.softmax(t2, dim=-1)

        assert t3 is not None
        assert t4 is not None

    t0_t = torch.rand(shape).to(torch.float16)
    t1_t = torch.rand(shape).to(torch.float16)
    r1, r2 = model(t0_t, t1_t)

    ref1 = torch.nn.functional.relu(t0_t * t1_t)
    ref2 = torch.nn.functional.softmax(t0_t * t1_t, dim=-1)

    assert (
        1 - r2_score(ref1.flatten().detach().numpy(), r1.flatten().detach().numpy())
        < 0.001
    )
    assert (
        1 - r2_score(ref2.flatten().detach().numpy(), r2.flatten().detach().numpy())
        < 0.001
    )
