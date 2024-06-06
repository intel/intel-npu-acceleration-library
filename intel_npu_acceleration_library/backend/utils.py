#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from .bindings import lib


def npu_available() -> bool:
    """Return if the NPU is available.

    Returns:
        bool: Return True if the NPU is available in the system
    """
    return lib.isNPUAvailable()


def get_driver_version() -> int:
    """Get the driver version for the Intel® NPU Acceleration Library.

    Raises:
        RuntimeError: an error is raised if the platform is not supported. Currently supported platforms are Windows and Linux

    Returns:
        int: NPU driver version
    """
    if not npu_available():
        raise RuntimeError("NPU is not available on this system")

    return lib.getNPUDriverVersion()
