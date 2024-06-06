#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from functools import lru_cache
from .bindings import lib
import warnings
import sys

__min_npu_driver_version__ = 2408


@lru_cache
def npu_available() -> bool:
    """Return if the NPU is available.

    Returns:
        bool: Return True if the NPU is available in the system
    """
    return lib.isNPUAvailable()


def get_driver_installation_url() -> str:
    """Get the driver installation URL.

    Returns:
        std: Return the driver installation url
    """
    if sys.platform == "win32":
        return "Driver Update URL: https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html"
    elif sys.platform == "linux":
        return "Driver Update URL: https://github.com/intel/linux-npu-driver"
    else:
        return ""


@lru_cache
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


def check_npu_and_driver_version():
    """Check NPU and driver version."""
    if not npu_available():
        warnings.warn(
            "NPU is not available in your system. Library will fallback to AUTO device selection mode",
            stacklevel=2,
        )
    elif get_driver_version() < __min_npu_driver_version__:

        warnings.warn(
            f"\nWarning: Outdated Driver Detected!!!\n"
            f"Current Driver Version: {get_driver_version()}, Minimum Required Version: {__min_npu_driver_version__}\n"
            f"Using an outdated driver may result in reduced performance and unexpected errors and crashes"
            f"To avoid these issues, please update your driver to the latest version.\n"
            f"{get_driver_installation_url()}\n",
            stacklevel=2,
        )
