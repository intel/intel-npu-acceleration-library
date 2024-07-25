#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import Sequence, Union
import numpy as np


class Convolution(NNFactory):
    """Linear class, computing a matrix matrix multiplication with weights prefetching."""

    def __init__(
        self,
        input_shape: Sequence[int],
        weights_shape: Sequence[int],
        bias: bool = False,
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the Linear class.

        Args:
            input_shape (Sequence[int]): input shape
            weights_shape (Sequence[int]): weights shape
            bias (bool): Enable/Disable bias. Defaults to False.
            strides (Union[int, Sequence[int]], optional): Strides. Defaults to 1.
            padding (Union[int, Sequence[int]], optional): Padding. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): Dilation. Defaults to 1.
            groups (int, optional): Groups. Defaults to 1.
            profile (Optional[bool], optional): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
        """
        super().__init__(profile, device)
        input = self.parameter(input_shape)
        weights = self.parameter(weights_shape)
        if bias is not None:
            bias_node = self.parameter((1, weights_shape[0], 1, 1))
        else:
            bias_node = None

        _ = self.convolution(
            input,
            weights,
            bias=bias_node,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            act_dtype=np.float16,
        )

        self.compile()
