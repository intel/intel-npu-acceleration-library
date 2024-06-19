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

        # Get the number of spatial dimensions
        n_spatial_dims = len(input_shape) - 2

        if isinstance(strides, int):
            strides = [strides] * n_spatial_dims

        if isinstance(padding, int):
            padding_begins = [padding] * n_spatial_dims
            padding_ends = [padding] * n_spatial_dims
        else:
            padding_begins = list(padding)
            padding_ends = list(padding)

        if isinstance(dilation, int):
            dilation = [dilation] * n_spatial_dims

        conv = self.convolution(
            input,
            weights_shape,
            bias=bias,
            strides=strides,
            padding_begins=padding_begins,
            padding_ends=padding_ends,
            dilation=dilation,
            groups=groups,
            act_dtype=np.float16,
            wt_dtype=np.float16,
        )

        self.compile(conv)
