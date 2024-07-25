#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from dataclasses import dataclass
from typing import Union
import numpy as np
import torch


@dataclass(frozen=True)
class NPUDtype:
    """Represents a custom data type for NPUs (Neural Processing Units).

    Attrs:
        name: str: The name of the data type.
        bits: int: The number of bits used to represent the data type.
        min: int: The minimum value that can be represented by the data type.
        max: int: The maximum value that can be represented by the data type.
        torch_dtype: torch.dtype: The corresponding torch data type.
        is_floating_point: bool: True if the data type is floating-point, False otherwise.
    """

    name: str
    bits: int
    min: int
    max: int
    torch_dtype: torch.dtype

    @property
    def is_floating_point(self) -> bool:
        """
        Check if the data type is a floating-point type.

        Returns:
            bool: True if the data type is floating-point, False otherwise.
        """
        return self.torch_dtype.is_floating_point

    def __eq__(self, value: Union["NPUDtype", torch.dtype]) -> bool:
        """
        Compare the NPUDtype object with another NPUDtype or torch.dtype object.

        Args:
            value (Union["NPUDtype", torch.dtype]): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if isinstance(value, torch.dtype):
            if value.is_floating_point:
                info = torch.finfo(value)
            else:
                info = torch.iinfo(value)
            return (
                self.bits == info.bits
                and self.max == info.max
                and self.min == info.min
                and self.torch_dtype == value
            )
        if isinstance(value, type):
            value = np.dtype(value)
            if value.kind == "f":
                info = np.finfo(value)
            else:
                info = np.iinfo(value)
            return (
                self.bits == info.bits and self.max == info.max and self.min == info.min
            )
        else:
            return super().__eq__(value)

    def __repr__(self) -> str:
        """
        Return a string representation of the NPUDtype object.

        Returns:
            str: The string representation of the NPUDtype object.
        """
        return self.name


float16 = NPUDtype(
    "fp16",
    16,
    torch.finfo(torch.float16).min,
    torch.finfo(torch.float16).max,
    torch.float16,
)
bfloat16 = NPUDtype(
    "bf16",
    16,
    torch.finfo(torch.bfloat16).min,
    torch.finfo(torch.bfloat16).max,
    torch.bfloat16,
)
float32 = NPUDtype(
    "fp32",
    32,
    torch.finfo(torch.float32).min,
    torch.finfo(torch.float32).max,
    torch.float32,
)
float64 = NPUDtype(
    "fp64",
    64,
    torch.finfo(torch.float64).min,
    torch.finfo(torch.float64).max,
    torch.float64,
)
int4 = NPUDtype("int4", 4, -8, 7, torch.int8)
int8 = NPUDtype("int8", 8, -128, 127, torch.int8)
int16 = NPUDtype(
    "int16", 16, torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max, torch.int16
)
int32 = NPUDtype(
    "int32", 32, torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, torch.int32
)
int64 = NPUDtype(
    "int64", 64, torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max, torch.int64
)
