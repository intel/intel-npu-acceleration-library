#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import Tuple
import numpy as np


class SDPA(NNFactory):
    """Implementation of a ScaledDotProductAttention NPU operation."""

    def __init__(
        self,
        query_shapes: Tuple[int, int],
        key_shapes: Tuple[int, int],
        value_shapes: Tuple[int, int],
        mask_shapes: Tuple[int, int],
        is_causal: bool = False,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the SDPA.

        Args:
            query_shapes (Tuple[int, int]): shape of the query tensor
            key_shapes (Tuple[int, int]): shape of the key tensor
            value_shapes (Tuple[int, int]): shape of the value tensor
            mask_shapes (Tuple[int, int]): shape of the mask tensor
            is_causal (bool, optional): If the SDPA mask is is_causal or not. Defaults to False.
            profile (bool, optional): Enable/Disable profiling. Defaults to False.
            device (str, optional): Target device, default to "NPU".
        """
        super().__init__(profile, device)

        self.query = self.parameter(query_shapes)
        self.key = self.parameter(key_shapes)
        self.value = self.parameter(value_shapes)
        self.mask = self.parameter(mask_shapes)

        out = self.scaled_dot_product_attention(  # type: ignore[attr-defined]
            self.query, self.key, self.value, self.mask, is_causal
        )
        self.compile(out)

    def run(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Run the scaled dot product attention kernel.

        Args:
            query (np.ndarray): sdpa query tensor
            key (np.ndarray): sdpa key tensor
            value (np.ndarray): sdpa value tensor
            mask (np.ndarray): sdpa mask tensor

        Returns:
            np.ndarray: result
        """
        return super().run(query, key, value, mask)
