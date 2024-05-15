#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from sklearn.metrics import r2_score
from intel_npu_acceleration_library.backend import MatMul
import numpy as np
import intel_npu_acceleration_library
import pytest
import time
import sys
import os


def profile(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        ret = func(*args, **kwargs)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        return ret, elapsed

    return wrapper


def test_basic_functionality():

    X = np.random.uniform(-1, 1, (512, 2048)).astype(np.float16)
    W = np.random.uniform(-1, 1, (512, 2048)).astype(np.float16)

    mm = MatMul(2048, 512, X.shape[0])

    @profile
    def npu_run():
        return mm.run(X, W)

    @profile
    def cpu_run():
        return np.matmul(X, W.T)

    npu_val, npu_latency = npu_run()
    cpu_val, cpu_latency = cpu_run()

    assert 1 - r2_score(cpu_val, npu_val) < 0.001
    assert npu_latency < cpu_latency


def test_save_model():

    mm = MatMul(2048, 512, 512)
    mm.save("model.xml")
    assert os.path.isfile("model.xml")
    assert os.path.isfile("model.bin")
    os.remove("model.xml")
    os.remove("model.bin")


@pytest.mark.skipif(
    not intel_npu_acceleration_library.backend.npu_available(),
    reason="Cannot save model if NPU is not available",
)
def test_save_compiled_model():

    mm = MatMul(2048, 512, 512)
    mm.saveCompiledModel("model.blob")
    assert os.path.isfile("model.blob")
    os.remove("model.blob")


@pytest.mark.skipif(
    not intel_npu_acceleration_library.backend.npu_available(),
    reason="Skip test if NPU is not available",
)
@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Skip test if not on windows platform",
)
def test_driver_version():

    version = intel_npu_acceleration_library.backend.get_driver_version()
    assert version is not None
