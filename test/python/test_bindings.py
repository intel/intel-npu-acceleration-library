#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
import numpy as np
import pytest
import ctypes


@pytest.mark.parametrize("device", ["CPU", "NPU"])
def test_bindings(device):

    device = ctypes.c_char_p(device.encode())
    matmul = backend_lib.createNNFactory(device, 32, 64, 16, False)

    assert isinstance(matmul, ctypes.POINTER(ctypes.c_char))

    backend_lib.destroyNNFactory(matmul)


@pytest.mark.parametrize("inC", [16, 32, 64, 128])
@pytest.mark.parametrize("outC", [16, 32, 64, 128])
@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("run_op", [True, False])
def test_factory_bindings(inC, outC, batch, run_op):

    ## Weights
    weights = np.zeros((outC, inC)).astype(np.float16)
    x = np.zeros((batch, inC)).astype(np.float16)
    out = np.empty((batch, outC), dtype=np.float16)

    # Create nn factory
    device = ctypes.c_char_p("NPU".encode())
    factory = backend_lib.createNNFactory(device, inC, outC, batch, False)

    # Create linear layer
    p0 = backend_lib.fp16parameter(factory, batch, inC)
    linear = backend_lib.linear(factory, p0, outC, inC, False, False, 8)
    backend_lib.compile(factory, linear)

    # Set parameters
    param = backend_lib.createParameters()
    backend_lib.addFloatParameter(param, weights, *weights.shape)
    backend_lib.setNNFactoryWeights(factory, param)

    # run
    if run_op:
        backend_lib.run(factory, x, out)

    # Call destuctors for parameters and weights
    backend_lib.destroyNNFactory(factory)
    backend_lib.destroyParameters(param)
