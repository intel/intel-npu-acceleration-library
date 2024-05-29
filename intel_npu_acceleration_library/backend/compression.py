#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
import numpy as np


def compress_to_i4(weights: np.ndarray) -> np.ndarray:
    """Compress a int8 array to int4.

    Args:
        weights (np.ndarray): input array

    Returns:
        np.ndarray: compressed array
    """
    compressed_weights = np.zeros(
        (weights.shape[0], weights.shape[1] // 2), dtype=np.uint8
    )

    backend_lib.compressToI4(weights, compressed_weights, np.prod(weights.shape))
    return compressed_weights
