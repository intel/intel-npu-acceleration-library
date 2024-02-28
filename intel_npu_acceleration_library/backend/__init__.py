#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from .bindings import lib
from .mlp import MLP
from .matmul import MatMul
from .linear import Linear
from .qmatmul import QMatMul
from .qlinear import QLinear
from .factory import NNFactory
from .runtime import run_matmul, run_factory
import warnings


def npu_available() -> bool:
    """Return if the NPU is available.

    Returns:
        bool: Return True if the NPU is available in the system
    """
    return lib.isNPUAvailable()


if not npu_available():
    warnings.warn(
        "NPU is not available in your system. Library will fallback to AUTO device selection mode",
        stacklevel=2,
    )


__all__ = [
    "NNFactory",
    "MLP",
    "MatMul",
    "Linear",
    "QMatMul",
    "QLinear",
    "run_matmul",
    "run_factory",
]
