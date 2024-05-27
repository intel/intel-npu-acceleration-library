#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from dataclasses import dataclass
from typing import Union
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
        else:
            return super().__eq__(value)


float16 = NPUDtype("fp16", 16, -65504, 65504, torch.float16)
bfloat16 = NPUDtype("bfloat16", 16, -65504, 65504, torch.float16)
int4 = NPUDtype("int4", 4, -8, 7, torch.int8)
int8 = NPUDtype("int8", 8, -128, 127, torch.int8)
