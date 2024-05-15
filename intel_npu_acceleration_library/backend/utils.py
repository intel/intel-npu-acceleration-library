#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from .bindings import lib
import subprocess  # nosec B404
import sys
import re


def npu_available() -> bool:
    """Return if the NPU is available.

    Returns:
        bool: Return True if the NPU is available in the system
    """
    return lib.isNPUAvailable()


def get_driver_version() -> str:
    """Get the driver version for the Intel® NPU Acceleration Library.

    Raises:
        RuntimeError: an error is raised if the platform is not supported. Currently supported platforms are Windows and Linux

    Returns:
        str: _description_
    """
    if not npu_available():
        raise RuntimeError("NPU is not available on this system")

    if sys.platform == "win32":
        out = (
            subprocess.check_output(  # nosec
                'powershell -Command " Get-WmiObject Win32_PnPSignedDriver | select devicename, driverversion',
                shell=True,
            )
            .decode("utf-8")
            .strip()
        )

        npu_drivers = [
            driver for driver in out.split("\n") if "Intel(R) AI Boost" in driver
        ]
        if len(npu_drivers) == 0:
            raise RuntimeError("Cannot get driver version")
        elif len(npu_drivers) > 1:
            raise RuntimeError("Multiple drivers found")
        else:
            driver_info = re.search(r"(\d+\.\d+\.\d+\.\d+)", npu_drivers[0])
            if driver_info:
                driver_version = driver_info.group(1)
                return driver_version
            else:
                raise RuntimeError("Cannot get driver version")
    else:
        raise RuntimeError(f"Cannot get driver version for {sys.platform}")
