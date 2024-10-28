#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.compiler import compile
from intel_npu_acceleration_library.compiler import CompilerConfig
from intel_npu_acceleration_library.dtypes import int4
from sklearn.metrics import r2_score
import intel_npu_acceleration_library
from packaging.version import Version
import pytest
import torch
import time
import sys


class NN(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(32, 128, bias=False)
        self.l2 = torch.nn.Linear(128, 32, bias=False)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        return self.relu(self.l2(self.relu(self.l1(x))))


torch.manual_seed(0)
x = 128 * (torch.rand((16, 32), dtype=torch.float16) - 0.5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8, int4])
def test_compilation(dtype):

    torch.manual_seed(42)
    model = NN().half()

    y_ref = model(x).detach()

    compiler_conf = CompilerConfig(dtype=dtype)
    compiled_model = compile(model, compiler_conf)

    assert compiled_model

    for name, layer in compiled_model.named_children():
        expected_cls = (
            intel_npu_acceleration_library.nn.Linear
            if dtype.is_floating_point
            else intel_npu_acceleration_library.nn.QuantizedLinear
        )
        assert isinstance(layer, expected_cls)
        if dtype == int4:
            assert layer.weight.dtype == torch.uint8
        else:
            assert layer.weight.dtype == dtype
        if layer.bias is not None:
            if dtype.is_floating_point:
                assert layer.bias.dtype == dtype
            else:
                layer.bias.dtype == torch.float32

    t0 = time.perf_counter()
    y1 = compiled_model(x).detach()
    t1 = time.perf_counter()

    y2 = compiled_model(x).detach()
    t2 = time.perf_counter()

    if dtype == int4:
        assert 1 - r2_score(y_ref.numpy(), y1.numpy()) < 0.05
    else:
        assert 1 - r2_score(y_ref.numpy(), y1.numpy()) < 0.01

    assert torch.allclose(y1, y2)

    # Check that for next iteration weights are prefetched
    # latency2 = t2 - t1
    # latency1 = t1 - t0
    # assert latency2 < latency1

    intel_npu_acceleration_library.backend.clear_cache()


def test_torch_compile():

    model = NN()
    y_ref = model(x.to(torch.float32)).detach()

    if sys.platform == "win32" and Version(torch.__version__) < Version("2.2.2"):
        with pytest.raises(RuntimeError) as e:
            compiled_model = torch.compile(model, backend="npu")
        assert str(e.value) == "Windows not yet supported for torch.compile"
    else:
        compiled_model = torch.compile(model, backend="npu")
        y = compiled_model(x.to(torch.float32)).detach()
        assert 1 - r2_score(y_ref.numpy(), y.numpy()) < 0.01


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int8])
def test_compile_training(dtype):

    model = NN()

    compiler_conf = CompilerConfig(dtype=dtype, training=True)
    compiled_model = compile(model, compiler_conf)

    for name, layer in compiled_model.named_children():
        if dtype == torch.int8:
            assert layer.training == False
        else:
            assert layer.training == True


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int8, int4])
def test_compile_inference(dtype):

    model = NN()

    compiler_conf = CompilerConfig(dtype=dtype)
    compiled_model = compile(model, compiler_conf)

    for name, layer in compiled_model.named_children():
        assert layer.training == False
