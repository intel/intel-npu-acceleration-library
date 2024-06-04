#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import Optional, Sequence


class MLP(NNFactory):
    """Linear class, computing a matrix matrix multiplication with weights prefetching."""

    def __init__(
        self,
        input_shape: Sequence[int],
        intermediate_size: int,
        activation: str = "swiglu",
        bias: Optional[bool] = False,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the Linear class.

        Args:
            input_shape (Sequence[int]): input shape channels
            intermediate_size (int): intermediate_size
            activation (str): activation function to use
            bias (Optional[bool], optional): Enable/Disable bias. Defaults to False.
            profile (bool): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
        """
        super().__init__(profile, device)
        self.intermediate_size = intermediate_size
        self.batch, self.hidden_size = input_shape
        input = self.parameter((self.batch, self.hidden_size))

        mm1 = self.linear(input, self.intermediate_size, self.hidden_size, bias=bias)

        if activation == "swiglu":
            mm2 = self.linear(input, self.intermediate_size, self.hidden_size, bias=bias)  # type: ignore[attr-defined]
            mm1 = self.eltwise_mul(self.swish(mm1), mm2)  # type: ignore[attr-defined]
        else:
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1)

        out = self.linear(mm1, self.hidden_size, self.intermediate_size, bias=bias)
        self.compile(out)
