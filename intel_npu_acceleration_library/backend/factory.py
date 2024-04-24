#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.base import BaseNPUBackendWithPrefetch
from intel_npu_acceleration_library.backend.ops import get_supported_ops
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
from typing import Optional, Tuple, Any, Union
from functools import partial
import numpy.typing as npt
import numpy as np
import ctypes


class NNFactory(BaseNPUBackendWithPrefetch):
    """Linear class, computing a matrix matrix multiplication with weights prefetching."""

    def __init__(
        self,
        inC: int,
        outC: int,
        batch: int,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the Linear class.

        Args:
            inC (int): input channels
            outC (int): output channels
            batch (int): batch
            profile (Optional[bool], optional): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
        """
        super().__init__(profile)
        self.batch = batch
        self.inC, self.outC = inC, outC
        self.device = device
        self._mm = backend_lib.createNNFactory(
            ctypes.c_char_p(self.device.encode()),
            self.inC,
            self.outC,
            self.batch,
            profile,
        )
        self.out = np.empty((self.batch, self.outC), dtype=np.float16)
        self.elapsed = None
        self.input = self.parameter((self.batch, self.inC))

        for op in get_supported_ops():
            setattr(self, op.name, partial(self._call_backend_op, op.name))

    def _call_backend_op(self, op_name: str, *parameters: Any) -> Any:
        """Dynamically call a backend operation.

        Args:
            op_name (str): operation name
            parameters (Any): variable list of operation parameters

        Returns:
            Any: Operation
        """
        fn = getattr(backend_lib, op_name)
        return fn(self._mm, *parameters)

    def get_backend_dtype(self, dtype) -> ctypes.c_char_p:
        """Get the string representation of the dtype.

        Args:
            dtype: numpy dtype

        Raises:
            RuntimeError: Unsupported datatype

        Returns:
            ctypes.c_char_p: string representation of the dtype
        """
        if dtype == np.int8:
            str_dtype = "int8"
        elif dtype == np.uint8:
            # u8 represents packed i4 dtypes
            str_dtype = "int4"
        elif dtype == np.float16:
            str_dtype = "float16"
        else:
            raise RuntimeError(f"DType is not supported {dtype}")
        return ctypes.c_char_p(str_dtype.encode())

    def parameter(
        self, shape: Tuple[int, int], dtype: npt.DTypeLike = np.float16
    ) -> ctypes._Pointer:
        """Generate a model input parameter.

        Args:
            shape (Tuple[int, int]): Parameter shape (only 2D tensors supported atm)
            dtype (np.dtype, optional): parameter type np.int8, np.uint8 and np.float16 supported. Defaults to np.float16. Unit8 represents packed i4 dtypes

        Returns:
            ctypes._Pointer: an instance to a parameter object

        """
        shape_ptr = np.array(shape, dtype=np.uint32)
        return backend_lib.parameter(
            self._mm, shape_ptr.size, shape_ptr, self.get_backend_dtype(dtype)
        )

    def linear(
        self,
        input_node: ctypes._Pointer,
        output_channels: int,
        input_channels: int,
        bias: Optional[bool] = False,
        act_dtype: npt.DTypeLike = np.float16,
        wt_dtype: npt.DTypeLike = np.float16,
    ) -> ctypes._Pointer:
        """Generate a linear layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            output_channels (int): number of output channels
            input_channels (int): number of input channels
            bias (bool, optional): enable/disable bias. Defaults to False.
            act_dtype (npt.DTypeLike, optional): activation dtype. Defaults to np.float16.
            wt_dtype (npt.DTypeLike, optional): weight dtype. Defaults to np.float16.

        Returns:
            ctypes._Pointer: _description_
        """
        return backend_lib.linear(
            self._mm,
            input_node,
            output_channels,
            input_channels,
            bias,
            self.get_backend_dtype(act_dtype),
            self.get_backend_dtype(wt_dtype),
        )

    def compile(self, output_node: ctypes._Pointer):
        """Finalize and compile a model.

        Args:
            output_node (ctypes._Pointer): Model output node
        """
        backend_lib.compile(self._mm, output_node)

    def run(
        self,
        X: np.ndarray,
        *weights: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> np.ndarray:
        """Run the layer: X * W^T.

        Args:
            X (np.ndarray): lhs operator
            weights (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): rhs operators
            kwargs (Any): additional arguments

        Raises:
            RuntimeError: Input tensor shape mismatch

        Returns:
            np.ndarray: result
        """
        if not (X.shape[0] == self.batch and X.shape[1] == self.inC):
            raise RuntimeError(
                f"Input shape {X.shape} different from expected one {(self.batch, self.inC)}"
            )

        prefetch = self.setWeights(kwargs.get("op_id", None), *weights)

        self.elapsed = backend_lib.run(self._mm, X, self.out)

        if prefetch:
            self.prefetchWeights()

        return self.out
