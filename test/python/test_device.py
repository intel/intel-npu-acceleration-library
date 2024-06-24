#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.nn.module import NPUModule
import torch


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(128, 128)

    def forward(self, x):
        return self.l1(x)


def test_device():

    x = torch.rand((128, 128)).to(torch.float16).to("npu")

    model = NN().half().to("npu")

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, NPUModule)

    y = model(x)

    assert y.dtype == torch.float16
    assert y.device == torch.device("npu")
