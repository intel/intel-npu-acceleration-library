#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.tensor import Tensor
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
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bfloat16,
    ],
)
def test_tensor_creation(shape, dtype):
    model = NNFactory()
    tensor = model.parameter(shape, dtype)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == shape
    assert tensor.dtype == dtype


def test_matmul():
    model = NNFactory()
    tensor = model.parameter([16, 256], np.float16)
    weights = model.constant(np.ones([128, 256], dtype=np.float16))

    out = tensor @ weights
    model.compile(out)

    assert isinstance(out, Tensor)
    assert out.shape == [16, 128]
    assert out.dtype == np.float16


def test_model_creation():
    model = NNFactory()
    t1 = model.parameter([1, 128, 32, 64], float16)
    assert t1.shape == [1, 128, 32, 64]
    assert t1.dtype == float16
    t2 = model.parameter([128 * 32 * 64], int8)
    assert t2.shape == [128 * 32 * 64]
    assert t2.dtype == int8

    t2 = t2.to(float16)

    assert t2.dtype == float16

    t2 = t2.reshape(128, 64, 32)

    assert t2.shape == [128, 64, 32]

    t2 = t2.unsqueeze(0)

    assert t2.shape == [1, 128, 64, 32]

    t2 = t2.T

    assert t2.shape == [1, 128, 32, 64]

    sum = t1 + t2

    dd = sum.transpose(1, 2)

    assert dd.shape == [1, 32, 128, 64]

    ff = dd.reshape([32, 128, 64])

    assert ff.shape == [32, 128, 64]

    assert ff.dim() == 3

    gg = ff.view(1, -1, 1, 1)

    assert gg.shape == [1, 32 * 128 * 64, 1, 1]

    model.compile(gg)


def test_slice():

    model = NNFactory()
    tensor = model.parameter([1, 128, 32, 64], float16)

    assert tensor[:, 0:64, :, :].shape == [1, 64, 32, 64]

    assert tensor[0, 0:64:2, :, :].shape == [1, 32, 32, 64]

    assert tensor[:, :-2, :, :].shape == [1, 126, 32, 64]

    assert tensor[..., :-2, :].shape == [1, 128, 30, 64]

    assert tensor[:, 10:20, ...].shape == [1, 10, 32, 64]
