#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from .bindings import lib
from .utils import npu_available, get_driver_version
from .mlp import MLP
from .matmul import MatMul
from .linear import Linear
from .qmatmul import QMatMul
from .qlinear import QLinear
from .factory import NNFactory
from .sdpa import SDPA
from .runtime import run_matmul, run_factory, clear_cache
import warnings


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
    "SDPA",
    "run_matmul",
    "run_factory",
    "clear_cache",
    "npu_available",
    "get_driver_version",
    "lib",
]
