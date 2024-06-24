#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.base import BaseNPUBackendWithPrefetch
from intel_npu_acceleration_library.backend.ops import get_supported_ops
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
from intel_npu_acceleration_library.backend.tensor import Tensor
from intel_npu_acceleration_library.dtypes import int4, bfloat16
from typing import Optional, Tuple, Any, Union, Sequence, TypeVar, Callable, cast
from functools import partial
import numpy.typing as npt
import numpy as np
import ctypes
import torch


F = TypeVar("F", bound=Callable[..., Any])


class NNFactory(BaseNPUBackendWithPrefetch):
    """Linear class, computing a matrix matrix multiplication with weights prefetching."""

    def __init__(
        self,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the Linear class.

        Args:
            profile (Optional[bool], optional): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
        """
        super().__init__(profile)
        self.device = device
        self._mm = backend_lib.createNNFactory(
            ctypes.c_char_p(self.device.encode()),
            profile,
        )
        self.elapsed = None

        for op in get_supported_ops():
            if not hasattr(self, op.name.replace("_act", "")):
                setattr(
                    self,
                    op.name.replace("_act", ""),
                    partial(self._call_backend_op, op.name),
                )

    def return_tensor(fn: F) -> F:  # type: ignore
        """Wrap the output of a function in a Tensor object.

        Args:
            fn (function): Function

        Returns:
            function: A function that wraps the output in a Tensor object
        """

        def wrapper(self, *args: Any, **kwargs: Any) -> Tensor:
            """Wrap the output of a function in a Tensor object.

            Args:
                args (Any): Variable length argument list
                kwargs (Any): Arbitrary keyword arguments

            Returns:
                Tensor: Tensor object
            """
            # Convert Tensor objects to their underlying node
            args = tuple(arg.node if isinstance(arg, Tensor) else arg for arg in args)
            kwargs = {
                k: v.node if isinstance(v, Tensor) else v for k, v in kwargs.items()
            }
            # Call the function
            node = fn(self, *args, **kwargs)
            # Wrap the node in a Tensor object
            return Tensor(factory=self, node=node)

        return cast(F, wrapper)

    @return_tensor
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
        elif dtype == np.uint8 or dtype == int4:
            # u8 represents packed i4 dtypes
            str_dtype = "int4"
        elif dtype == np.int16:
            str_dtype = "int16"
        elif dtype == np.int32:
            str_dtype = "int32"
        elif dtype == np.int64:
            str_dtype = "int64"
        elif dtype == np.float16:
            str_dtype = "float16"
        elif dtype == np.float32:
            str_dtype = "float32"
        elif dtype == np.float64:
            str_dtype = "float64"
        elif dtype == bfloat16:
            str_dtype = "bfloat16"
        else:
            raise RuntimeError(f"DType is not supported {dtype}")
        return ctypes.c_char_p(str_dtype.encode())

    @return_tensor
    def parameter(
        self, shape: Sequence[int], dtype: npt.DTypeLike = np.float16
    ) -> ctypes._Pointer:
        """Generate a model input parameter.

        Args:
            shape (Sequence[int]): Parameter shape
            dtype (np.dtype, optional): parameter type np.int8, np.uint8 and np.float16 supported. Defaults to np.float16. Unit8 represents packed i4 dtypes

        Returns:
            ctypes._Pointer: an instance to a parameter object

        """
        shape_ptr = np.array(shape, dtype=np.uint32)
        return backend_lib.parameter(
            self._mm, shape_ptr.size, shape_ptr, self.get_backend_dtype(dtype)
        )

    @return_tensor
    def to(self, tensor: ctypes._Pointer, dtype: npt.DTypeLike) -> ctypes._Pointer:
        """Convert a tensor to a different dtype.

        Args:
            tensor (ctypes._Pointer): input tensor
            dtype (npt.DTypeLike): target dtype

        Returns:
            ctypes._Pointer: output tensor
        """
        dtype_ptr = self.get_backend_dtype(dtype)
        return backend_lib.to(self._mm, tensor, dtype_ptr)

    @return_tensor
    def constant(
        self,
        data: Union[np.array, Sequence[int], Sequence[float], int, float, torch.Tensor],
    ) -> ctypes._Pointer:
        """Generate a model input constant.

        Args:
            data (Union[np.array, Sequence[int], Sequence[float], int, float, torch.Tensor]): constant data

        Returns:
            ctypes._Pointer: an instance to a constant object

        """
        if isinstance(data, (list, tuple)):
            if all(isinstance(i, int) for i in data):
                data = np.array(data, dtype=np.int64)
            else:
                data = np.array(data, dtype=np.float32)
        elif isinstance(data, int):
            data = np.array([data], dtype=np.int64)
        elif isinstance(data, float):
            data = np.array([data], dtype=np.float32)
        elif isinstance(data, torch.Tensor):
            data = data.detach().numpy()

        dst = data.ctypes.data_as(ctypes.c_void_p)
        shape_ptr = np.array(data.shape, dtype=np.uint32)
        return backend_lib.constant(
            self._mm, shape_ptr.size, shape_ptr, self.get_backend_dtype(data.dtype), dst
        )

    @return_tensor
    def convolution(
        self,
        input_node: ctypes._Pointer,
        weights_shape: Sequence[int],
        bias: bool,
        strides: Sequence[int] = (1, 1),
        padding_begins: Sequence[int] = (0, 0),
        padding_ends: Sequence[int] = (0, 0),
        dilation: Sequence[int] = (1, 1),
        groups: int = 1,
        act_dtype: npt.DTypeLike = np.float16,
        wt_dtype: npt.DTypeLike = np.float16,
    ) -> ctypes._Pointer:
        """Generate a convolution layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            weights_shape (Sequence[int]): weights shape
            strides (Sequence[int]): strides
            padding_begins (Sequence[int]): padding
            padding_ends (Sequence[int]): padding
            dilation (Sequence[int]): dilation
            groups (int): groups
            bias (bool): enable/disable bias
            act_dtype (npt.DTypeLike, optional): activation dtype. Defaults to np.float16.
            wt_dtype (npt.DTypeLike, optional): weight dtype. Defaults to np.float16.

        Returns:
            ctypes._Pointer: output node
        """
        weights_shape_ptr = np.array(weights_shape, dtype=np.uint32)
        strides_ptr = np.array(strides, dtype=np.uint32)
        padding_begins_ptr = np.array(padding_begins, dtype=np.uint32)
        padding_ends_ptr = np.array(padding_ends, dtype=np.uint32)
        dilation_ptr = np.array(dilation, dtype=np.uint32)

        return backend_lib.convolution(
            self._mm,
            input_node,
            weights_shape_ptr.size,
            weights_shape_ptr,
            strides_ptr.size,
            strides_ptr,
            padding_begins_ptr.size,
            padding_begins_ptr,
            padding_ends_ptr.size,
            padding_ends_ptr,
            dilation_ptr.size,
            dilation_ptr,
            groups,
            bias,
            self.get_backend_dtype(act_dtype),
            self.get_backend_dtype(wt_dtype),
        )

    @return_tensor
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
            ctypes._Pointer: output node
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

    @return_tensor
    def reshape(
        self, input_node: ctypes._Pointer, shape: Sequence[int]
    ) -> ctypes._Pointer:
        """Generate a reshape layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            shape (Sequence[int]): shape

        Returns:
            ctypes._Pointer: output node
        """
        shape_node = self.constant(shape).node  # type: ignore
        return backend_lib.reshape(self._mm, input_node, shape_node)

    @return_tensor
    def transpose(
        self, input_node: ctypes._Pointer, input_order: Sequence[int]
    ) -> ctypes._Pointer:
        """Generate a transpose layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            input_order (Sequence[int]): input order

        Returns:
            ctypes._Pointer: output node
        """
        input_order_node = self.constant(input_order).node  # type: ignore
        return backend_lib.transpose(self._mm, input_node, input_order_node)

    @return_tensor
    def unsqueeze(
        self, input_node: ctypes._Pointer, axis: Sequence[int]
    ) -> ctypes._Pointer:
        """Generate an unsqueeze layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            axis (Sequence[int]): axis

        Returns:
            ctypes._Pointer: output node
        """
        axis_node = self.constant(axis).node  # type: ignore
        return backend_lib.unsqueeze(self._mm, input_node, axis_node)

    @return_tensor
    def slice(
        self,
        input_node: ctypes._Pointer,
        begin: Sequence[int],
        end: Sequence[int],
        stride: Optional[Sequence[int]] = None,
    ) -> ctypes._Pointer:
        """Generate an unsqueeze layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            begin (Sequence[int]): begin
            end (Sequence[int]): end
            stride (Optional[Sequence[int]]): stride

        Raises:
            ValueError: begin and end must have the same length

        Returns:
            ctypes._Pointer: output node
        """
        if len(begin) != len(end):
            raise ValueError("begin and end must have the same length")

        if stride is None:
            stride = [1] * len(begin)

        begin_mask_ptr = np.zeros([len(begin)], dtype=np.uint32)
        end_mask_ptr = np.zeros([len(begin)], dtype=np.uint32)

        begin = self.constant(begin).node  # type: ignore
        end = self.constant(end).node  # type: ignore
        stride = self.constant(stride).node  # type: ignore

        return backend_lib.slice(
            self._mm,
            input_node,
            begin,
            end,
            stride,
            begin_mask_ptr.size,
            begin_mask_ptr,
            end_mask_ptr.size,
            end_mask_ptr,
        )

    @return_tensor
    def concat(
        self, input_node_1: ctypes._Pointer, input_node_2: ctypes._Pointer, axis: int
    ) -> ctypes._Pointer:
        """Generate a concatenation layer.

        Args:
            input_node_1 (ctypes._Pointer): first layer input node
            input_node_2 (ctypes._Pointer): second layer input node
            axis (int): axis

        Returns:
            ctypes._Pointer: output node
        """
        if axis < 0:
            shape_size = backend_lib.op_shape_size(input_node_1)
            axis = (axis + shape_size) % shape_size
        axis = np.int64(axis)
        return backend_lib.concat(self._mm, input_node_1, input_node_2, axis)

    @return_tensor
    def normL2(
        self, input_node: ctypes._Pointer, axis: int, eps: Optional[float] = 1e-12
    ) -> ctypes._Pointer:
        """Generate an L2 normalization layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            axis (int): axis
            eps (float, optional): epsilon added to L2 norm. Defaults to 1e-12

        Returns:
            ctypes._Pointer: output node
        """
        if axis < 0:
            axis = abs(axis)
        axis_node = self.constant(axis).node  # type: ignore
        return backend_lib.normL2(self._mm, input_node, axis_node, eps)

    @return_tensor
    def avg_pooling(
        self,
        input: ctypes._Pointer,
        kernel_size: Union[int, Sequence[int]],
        strides: Optional[Union[int, Sequence[int]]] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        n_spatial_dims: int = 2,
    ) -> ctypes._Pointer:
        """Generate an average pooling layer.

        Args:
            input (ctypes._Pointer): layer input node
            kernel_size (Sequence[int]): kernel size
            strides (Sequence[int]): strides
            padding (int): padding
            ceil_mode (bool): ceil mode
            count_include_pad (bool): count include pad
            divisor_override (int): divisor override
            n_spatial_dims (int): number of spatial dimensions

        Raises:
            NotImplementedError: divisor_override is not supported

        Returns:
            ctypes._Pointer: output node
        """
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * n_spatial_dims

        if strides is None:
            strides = kernel_size
        elif isinstance(strides, int):
            strides = [strides] * n_spatial_dims

        if isinstance(padding, int):
            padding_begins = [padding] * n_spatial_dims
            padding_ends = [padding] * n_spatial_dims
        else:
            padding_begins = list(padding)
            padding_ends = list(padding)

        strides_ptr = np.array(strides, dtype=np.uint32)
        padding_begins_ptr = np.array(padding_begins, dtype=np.uint32)
        padding_ends_ptr = np.array(padding_ends, dtype=np.uint32)
        kernel_size_ptr = np.array(kernel_size, dtype=np.uint32)

        rounding_type = 1 if ceil_mode else 0
        auto_pad = 0  # Hardcoded to explicit padding

        if divisor_override:
            raise NotImplementedError("divisor_override is not supported")

        return backend_lib.avg_pooling(
            self._mm,
            input,
            strides_ptr.size,
            strides_ptr,
            padding_begins_ptr.size,
            padding_begins_ptr,
            padding_ends_ptr.size,
            padding_ends_ptr,
            kernel_size_ptr.size,
            kernel_size_ptr,
            not count_include_pad,  # exclude_pad
            rounding_type,  # rounding_type
            auto_pad,  # auto_pad
        )

    @return_tensor
    def max_pooling(
        self,
        input: ctypes._Pointer,
        kernel_size: Union[int, Sequence[int]],
        strides: Optional[Union[int, Sequence[int]]] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        n_spatial_dims: int = 2,
    ) -> ctypes._Pointer:
        """Generate an average pooling layer.

        Args:
            input (ctypes._Pointer): layer input node
            kernel_size (Sequence[int]): kernel size
            strides (Sequence[int]): strides
            padding (int): padding
            ceil_mode (bool): ceil mode
            n_spatial_dims (int): number of spatial dimensions

        Returns:
            ctypes._Pointer: output node
        """
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * n_spatial_dims

        if strides is None:
            strides = kernel_size
        elif isinstance(strides, int):
            strides = [strides] * n_spatial_dims

        if isinstance(padding, int):
            padding_begins = [padding] * n_spatial_dims
            padding_ends = [padding] * n_spatial_dims
        else:
            padding_begins = list(padding)
            padding_ends = list(padding)

        strides_ptr = np.array(strides, dtype=np.uint32)
        padding_begins_ptr = np.array(padding_begins, dtype=np.uint32)
        padding_ends_ptr = np.array(padding_ends, dtype=np.uint32)
        kernel_size_ptr = np.array(kernel_size, dtype=np.uint32)

        rounding_type = 1 if ceil_mode else 0
        auto_pad = 0  # Hardcoded to explicit padding

        return backend_lib.max_pooling(
            self._mm,
            input,
            strides_ptr.size,
            strides_ptr,
            padding_begins_ptr.size,
            padding_begins_ptr,
            padding_ends_ptr.size,
            padding_ends_ptr,
            kernel_size_ptr.size,
            kernel_size_ptr,
            rounding_type,  # rounding_type
            auto_pad,  # auto_pad
        )

    def get_output_tensor_shape(self):
        """Get output tensor shape.

        Returns:
            tuple[int]: output tensor shape
        """
        size = backend_lib.get_output_tensor_shape_size(self._mm, 0)
        return tuple(
            [
                backend_lib.get_output_tensor_shape(self._mm, 0, idx)
                for idx in range(size)
            ]
        )

    def compile(self, output_node: Union[ctypes._Pointer, Tensor]):
        """Finalize and compile a model.

        Args:
            output_node (ctypes._Pointer): Model output node
        """
        if isinstance(output_node, Tensor):
            output_node = output_node.node

        backend_lib.compile(self._mm, output_node)
        self.output_shape = self.get_output_tensor_shape()
        if len(self.output_shape) != 2:
            out_shape_1d = np.prod(self.output_shape)
            self.out = np.empty((1, out_shape_1d), dtype=np.float16)
        else:
            self.out = np.empty(self.output_shape, dtype=np.float16)
        backend_lib.set_output(self._mm, self.out, 0)

    def set_input_tensor(self, tensor: np.ndarray, idx: int):
        """Set input tensor.

        Args:
            tensor (np.ndarray): Input tensor
            idx (int): tensor index
        """
        if len(tensor.shape) != 2:
            backend_lib.set_activation(self._mm, tensor.reshape(1, -1), idx)
        else:
            backend_lib.set_activation(self._mm, tensor, idx)

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

        Returns:
            np.ndarray: result
        """
        op_id = kwargs.get("op_id", None)
        if op_id is None and all(isinstance(tensor, np.ndarray) for tensor in weights):
            for idx, weight in enumerate(weights):
                self.set_input_tensor(weight, idx + 1)
            prefetch = False
        else:
            prefetch = self.setWeights(kwargs.get("op_id", None), *weights)

        self.set_input_tensor(X, 0)
        self.elapsed = backend_lib.run(self._mm)

        if prefetch:
            self.prefetchWeights()

        return self.out.reshape(self.output_shape)
