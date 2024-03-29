#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.ops import get_supported_ops
import numpy as np
import ctypes
import sys
import os

handler = ctypes.POINTER(ctypes.c_char)
c_fp16_array = np.ctypeslib.ndpointer(dtype=np.float16, ndim=2, flags="C_CONTIGUOUS")
c_fp32_array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
c_i8_array = np.ctypeslib.ndpointer(dtype=np.int8, ndim=2, flags="C_CONTIGUOUS")


def load_library() -> ctypes.CDLL:
    """Load the Intel® NPU Acceleration Library runtime library.

    Raises:
        RuntimeError: an error is raised if the platform is not supported. Currently supported platforms are WIndows and Linux

    Returns:
        ctypes.CDLL: The loaded dynamic library
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        dll_path = os.path.join(path, "..", "lib", "Release")
        os.add_dll_directory(os.path.abspath(dll_path))
        # Load DLL into memory.
        lib = ctypes.WinDLL(
            os.path.join(dll_path, "intel_npu_acceleration_library.dll")
        )  # , winmode=0)
    elif sys.platform == "linux":
        dll_path = os.path.join(path, "..", "lib")
        sys.path.append(dll_path)
        # In Linux it is required to explicitly load openvino lib
        _ = ctypes.CDLL(os.path.join(dll_path, "libopenvino.so"))
        lib = ctypes.CDLL(
            os.path.join(dll_path, "libintel_npu_acceleration_library.so")
        )
    else:
        raise RuntimeError(
            f"Platform {sys.platform} is not supported for intel-npu-acceleration-library library"
        )

    return lib


def init_common(lib: ctypes.CDLL):
    """Initialize common runtime bindings.

    Args:
        lib (ctypes.CDLL): Intel® NPU Acceleration Library runtime library
    """
    lib.saveModel.argtypes = [handler, ctypes.c_char_p]
    lib.saveCompiledModel.argtypes = [handler, ctypes.c_char_p]

    # Run a linar layer
    lib.run.argtypes = [handler, c_fp16_array, c_fp16_array]
    lib.run.restype = ctypes.c_float

    # Common destructor
    lib.destroyNNFactory.argtypes = [handler]

    lib.isNPUAvailable.restype = ctypes.c_bool


def init_network_factory(lib: ctypes.CDLL):
    """Initialize Netowrk factory bindings.

    Args:
        lib (ctypes.CDLL): Intel® NPU Acceleration Library runtime library
    """
    lib.createNNFactory.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
    ]
    lib.createNNFactory.restype = handler

    lib.setNNFactoryWeights.argtypes = [handler, handler]

    lib.fp16parameter.argtypes = [handler, ctypes.c_int, ctypes.c_int]
    lib.fp16parameter.restype = handler

    lib.i8parameter.argtypes = [handler, ctypes.c_int, ctypes.c_int]
    lib.i8parameter.restype = handler

    lib.compile.argtypes = [handler, handler]
    lib.compile.restype = handler

    lib.linear.argtypes = [
        handler,
        handler,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_bool,
    ]
    lib.linear.restype = handler

    for op in get_supported_ops():
        fn = getattr(lib, op.name)
        fn.argtypes = [handler] * (op.inputs + 1)
        fn.restype = handler


def init_parameters(lib: ctypes.CDLL):
    """Initialize Netowrk factory parameters.

    Args:
        lib (ctypes.CDLL): Intel® NPU Acceleration Library runtime library
    """
    lib.createParameters.argtypes = []
    lib.createParameters.restype = handler

    lib.destroyParameters.argtypes = [handler]

    lib.addFloatParameter.argtypes = [handler, c_fp16_array, ctypes.c_int, ctypes.c_int]
    lib.addIntParameter.argtypes = [
        handler,
        c_i8_array,
        c_fp16_array,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.addIntParameterConversion.argtypes = [
        handler,
        c_i8_array,
        c_fp32_array,
        ctypes.c_int,
        ctypes.c_int,
    ]


def initialize_bindings() -> ctypes.CDLL:
    """Load the Intel® NPU Acceleration Library runtime library, and initialize all c++ <-> python bindings.

    Returns:
        ctypes.CDLL: Initialize matmul bindings
    """
    lib = load_library()

    init_common(lib)
    init_network_factory(lib)
    init_parameters(lib)

    return lib


lib = initialize_bindings()
