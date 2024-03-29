#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from dataclasses import dataclass
from functools import lru_cache
from typing import List


@dataclass(frozen=True)
class SupportedOp:
    """A class for supported runtime OPs in the NPU.

    Attrs:
        name (str): Operation name
        inputs (int): Number of inputs
    """

    name: str
    inputs: int


@lru_cache(maxsize=None)
def get_supported_ops() -> List[SupportedOp]:
    """Generate a list fo supported operations.

    Returns:
        List[SupportedOp]: list fo supported NPU operations
    """
    supported_ops = [
        SupportedOp(name="matmul", inputs=2),
        SupportedOp(name="eltwise_add", inputs=2),
        SupportedOp(name="eltwise_mul", inputs=2),
        SupportedOp(name="eltwise_div", inputs=2),
        SupportedOp(name="gelu", inputs=1),
        SupportedOp(name="softmax", inputs=1),
        SupportedOp(name="swish", inputs=1),
        SupportedOp(name="convert_to_fp16", inputs=1),
    ]
    return supported_ops
