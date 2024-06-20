#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import lib as backend_lib
from intel_npu_acceleration_library.dtypes import (
    float16,
    bfloat16,
    float32,
    float64,
    int4,
    int8,
    int16,
    int32,
    int64,
    NPUDtype,
)
from dataclasses import dataclass
from typing import Sequence, Any
import numpy as np
import ctypes


@dataclass
class Tensor:
    """
    Represents a tensor object.

    Attrs:
        factory (NNFactory): The factory object used to create the tensor.
        node (ctypes._Pointer): The pointer to the underlying tensor node.
        shape (Sequence[int]): The shape of the tensor.
        dtype (NPUDtype): The data type of the tensor.
        T (Tensor): The transpose of the tensor.

    Methods:
        __add__(self, other): Adds two tensors element-wise.
        __sub__(self, other): Subtracts two tensors element-wise.
        __mul__(self, other): Multiplies two tensors element-wise.
        __truediv__(self, other): Divides two tensors element-wise.
        __neg__(self): Negates the tensor.
        __repr__(self): Returns a string representation of the tensor.
        __str__(self): Returns a string representation of the tensor.
        __len__(self): Returns the total number of elements in the tensor.
        T: Returns the transpose of the tensor.
        squeeze(self): Removes dimensions of size 1 from the tensor.
        unsqueeze(self, axis): Adds a dimension of size 1 to the tensor.
        __matmul__(self, other): Performs matrix multiplication between two tensors.
    """

    factory: "NNFactory"  # type: ignore # noqa: F821
    node: ctypes._Pointer

    @property
    def shape(self) -> Sequence[int]:
        """
        Returns the shape of the tensor.

        Returns:
            Sequence[int]: The shape of the tensor.
        """
        shape_size = backend_lib.op_shape_size(self.node)
        return [backend_lib.op_shape(self.node, i) for i in range(shape_size)]

    @property
    def dtype(self) -> NPUDtype:
        """
        Returns the data type of the tensor.

        Returns:
            type: The data type of the tensor.
        """
        dtype_int = backend_lib.op_dtype(self.node)

        if dtype_int == 2:
            return np.bool
        elif dtype_int == 3:
            return bfloat16
        elif dtype_int == 4:
            return float16
        elif dtype_int == 5:
            return float32
        elif dtype_int == 6:
            return float64
        elif dtype_int == 7:
            return int4
        elif dtype_int == 8:
            return int8
        elif dtype_int == 9:
            return int16
        elif dtype_int == 10:
            return int32
        elif dtype_int == 11:
            return int64
        else:
            raise RuntimeError("Unsupported dtype")

    def __add__(self, other) -> "Tensor":
        """
        Add two tensors element-wise.

        Args:
            other (Tensor): The tensor to be added.

        Returns:
            Tensor: The result of the addition.
        """
        return generate_op([self, other], "eltwise_add")

    def __sub__(self, other) -> "Tensor":
        """
        Subtract two tensors element-wise.

        Args:
            other (Tensor): The tensor to be subtracted.

        Returns:
            Tensor: The result of the subtraction.
        """
        return generate_op([self, -other], "eltwise_add")

    def __mul__(self, other) -> "Tensor":
        """
        Multiply two tensors element-wise.

        Args:
            other (Tensor): The tensor to be multiplied.

        Returns:
            Tensor: The result of the multiplication.
        """
        return generate_op([self, other], "eltwise_mul")

    def __truediv__(self, other) -> "Tensor":
        """
        Divide two tensors element-wise.

        Args:
            other (Tensor): The tensor to be divided.

        Returns:
            Tensor: The result of the division.
        """
        return generate_op([self, other], "eltwise_div")

    def __neg__(self) -> "Tensor":
        """
        Negate the tensor.

        Returns:
            Tensor: The negated tensor.
        """
        return generate_op([self], "negative")

    def __repr__(self) -> str:
        """
        Return a string representation of the tensor.

        Returns:
            str: The string representation of the tensor.
        """
        return f"Tensor({self.shape}, {self.dtype})"

    def __str__(self) -> str:
        """
        Return a string representation of the tensor.

        Returns:
            str: The string representation of the tensor.
        """
        return f"Tensor({self.shape}, {self.dtype})"

    def __len__(self) -> int:
        """
        Return the total number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return np.product(self.shape)

    @property
    def T(self) -> "Tensor":
        """
        Return the transpose of the tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        input_order = list(range(len(self.shape)))
        input_order[-1], input_order[-2] = input_order[-2], input_order[-1]
        return generate_op([self], "transpose", input_order)

    def transpose(self, input_order: Sequence[int]) -> "Tensor":
        """
        Return the transpose of the tensor.

        Args:
            input_order (Sequence[int]): The order of the dimensions in the transposed tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        return generate_op([self], "transpose", input_order)

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        """
        Return the transpose of the tensor.

        Args:
            shape (Sequence[int]): The new shape of the tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        return generate_op([self], "reshape", shape)

    def squeeze(self) -> "Tensor":
        """
        Remove dimensions of size 1 from the tensor.

        Returns:
            Tensor: The squeezed tensor.
        """
        return generate_op([self], "squeeze")

    def unsqueeze(self, axis) -> "Tensor":
        """
        Add a dimension of size 1 to the tensor.

        Args:
            axis (int): The axis along which to add the dimension.

        Returns:
            Tensor: The unsqueezed tensor.
        """
        return generate_op([self], "unsqueeze", axis)

    def __matmul__(self, other) -> "Tensor":
        """
        Perform matrix multiplication between two tensors.

        Args:
            other (Tensor): The tensor to be multiplied.

        Returns:
            Tensor: The result of the matrix multiplication.
        """
        return generate_op([self, other], "matmul")

    def to(self, dtype: NPUDtype) -> "Tensor":
        """
        Convert the tensor to the specified data type.

        Args:
            dtype (NPUDtype): The data type to convert the tensor to.

        Returns:
            Tensor: The converted tensor.
        """
        return generate_op([self], "to", dtype)


def generate_op(
    tensors: Sequence[Tensor], op: str, *args: Any, **kwargs: Any
) -> "Tensor":
    """
    Generate a new tensor by applying the specified operation to a sequence of tensors.

    Args:
        tensors (Sequence[Tensor]): A sequence of tensors.
        op (str): The name of the operation to apply.
        args (Any): Variable length argument list.
        kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor generated by applying the operation to the input tensors.

    Raises:
        ValueError: If the tensors are not from the same factory.

    """
    # Check that all tensors are from the same factory
    if not len({tensor.factory for tensor in tensors}) == 1:
        raise ValueError("All tensors must be from the same factory")

    factory = tensors[0].factory
    # Create the operation
    return factory.__getattribute__(op)(*tensors, *args, **kwargs)
