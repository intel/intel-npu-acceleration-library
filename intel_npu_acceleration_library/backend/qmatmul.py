#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.factory import NNFactory
import numpy as np


class QMatMul(NNFactory):
    """Quantized Linear class, computing a matrix matrix multiplication."""

    def __init__(
        self,
        inC: int,
        outC: int,
        batch: int,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the QMatmul class.

        Args:
            inC (int): input channels
            outC (int): output channels
            batch (int): batch
            profile (bool): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
        """
        super().__init__(inC, outC, batch, profile, device)
        out = self.linear(self.input, outC, inC, bias=False, wt_dtype=np.int8)
        self.compile(out)

    def run(self, X: np.ndarray, W: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Run the layer:  X * (W * S)^T.

        Args:
            X (np.ndarray): activation
            W (np.ndarray): quantized weights
            scale (np.ndarray): quantization scale

        Raises:
            RuntimeError: Input, weights or scale shape mismatch

        Returns:
            np.ndarray: result
        """
        if not (X.shape[0] == self.batch and X.shape[1] == self.inC):
            raise RuntimeError(
                f"Input shape {X.shape} different from expected one {(self.batch, self.inC)}"
            )
        if not (X.shape[0] == self.batch and X.shape[1] == self.inC):
            raise RuntimeError(
                f"Weight shape {W.shape} different from expected one {(self.outC, self.inC)}"
            )
        if not (X.shape[0] == self.batch and X.shape[1] == self.inC):
            raise RuntimeError(
                f"Scale shape {W.shape} different from expected one {(self.outC, 1)}"
            )

        return super().run(X, (W, scale))
