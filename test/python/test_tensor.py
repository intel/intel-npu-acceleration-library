#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.tensor import create_tensor
from intel_npu_acceleration_library.backend import NNFactory
from intel_npu_acceleration_library.dtypes import (
    float16,
    float32,
    float64,
    int4,
    int8,
    int16,
    int32,
    int64,
)
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("shape", [[1, 128, 13, 13], [12, 231]])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        float16,
        float32,
        float64,
        int4,
        int8,
        int16,
        int32,
        int64,
    ],
)
def test_tensor_creation(shape, dtype):
    model = NNFactory()
    tensor = create_tensor(model, shape, dtype)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
