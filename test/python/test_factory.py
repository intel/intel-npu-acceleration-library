#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import numpy as np
import intel_npu_acceleration_library
import pytest
import os


@pytest.mark.parametrize("batch", [16, 128])
@pytest.mark.parametrize("inC", [256, 512])
@pytest.mark.parametrize("outC", [256, 512])
@pytest.mark.parametrize("dtype", [np.float16, np.int8])
@pytest.mark.parametrize("activation", ["gelu", "swish", "softmax"])
def test_factory(batch, inC, outC, dtype, activation):
    module = intel_npu_acceleration_library.backend.NNFactory()
    assert module

    input = module.parameter((batch, inC))
    assert input

    weights = module.parameter((outC, inC), dtype)
    assert weights

    if dtype == np.int8:
        weights = module.convert_to_fp16(weights)

    mm = module.matmul(input, weights)
    assert mm

    act_fn = getattr(module, activation)
    output = act_fn(mm)
    assert output

    module.compile(output)

    output_shape = module.get_output_tensor_shape()
    assert output_shape == (batch, outC)

    filename = f"test_factory_mm_{batch}_{inC}_{outC}_{dtype.__name__}_{activation}"
    module.save(f"{filename}.xml")

    assert os.path.isfile(f"{filename}.xml")

    os.remove(f"{filename}.xml")
    os.remove(f"{filename}.bin")
