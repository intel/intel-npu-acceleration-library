#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import pytest
from intel_npu_acceleration_library.dtypes import float16, bfloat16, int4, int8


@pytest.fixture
def npu_dtypes():
    return [float16, bfloat16, int4, int8]


def test_NPUDtype_is_floating_point(npu_dtypes):
    for dtype in npu_dtypes:
        if dtype in (int4, int8):
            assert dtype.is_floating_point == False
        else:
            assert dtype.is_floating_point == True
