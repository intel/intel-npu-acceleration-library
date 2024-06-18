#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import run_factory, Convolution
from typing import Optional, Sequence, Union
from functools import partial
import torch
import uuid


class Conv2d(torch.nn.Module):
    """
    2D convolutional layer implementation.

    Attrs:
        weight (torch.Tensor): The weight tensor of the layer.
        bias (torch.Tensor): The bias tensor of the layer.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
    ) -> None:
        """Initialize a Convolutional layer.

        Args:
            weights (torch.Tensor): The weight tensor of the layer.
            bias (Optional[torch.Tensor], optional): The bias tensor of the layer. Defaults to None.
            strides (Union[int, Sequence[int]], optional): Strides. Defaults to 1.
            padding (Union[int, Sequence[int]], optional): Padding. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): Dilation. Defaults to 1.
            groups (int, optional): Groups. Defaults to 1.
        """
        super().__init__()

        self.op_id = str(uuid.uuid4())
        self.parameters = [weights]
        if bias is not None:
            self.parameters.append(bias)
        self.backend_cls = partial(
            Convolution,
            weights_shape=weights.shape,
            bias=bias is not None,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    @property
    def weight(self) -> torch.Tensor:
        """
        Get the weight tensor of the layer.

        Returns:
            torch.Tensor: The weight tensor.
        """
        return self.parameters[0]

    @property
    def bias(self) -> torch.Tensor:
        """
        Get the bias tensor of the layer.

        Returns:
            torch.Tensor: The bias tensor.
        """
        if len(self.parameters) > 1:
            return self.parameters[1]
        return None

    def compute_output_dim(self, dim, idx) -> int:
        """
        Compute the output dimension for a given input dimension.

        Args:
            dim (int): Input dimension.
            idx (int): Index of the dimension.

        Returns:
            int: Output dimension.
        """
        return (
            dim
            + 2 * self.padding[idx]
            - self.dilation[idx] * (self.kernel_size[idx] - 1)
            - 1
        ) // self.stride[idx] + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        return run_factory(x, self.parameters, self.backend_cls, self.op_id)

    @staticmethod
    def fromTorch(layer, dtype: torch.dtype = torch.float16) -> "Conv2d":
        """
        Create a Conv2d layer from a torch.nn.Conv2d layer.

        Args:
            layer (torch.nn.Conv2d): The torch Conv2d layer.
            dtype (torch.dtype, optional): Data type of the layer.

        Returns:
            Conv2d: The converted Conv2d layer.
        """
        new_layer = Conv2d(
            layer.weight,
            layer.bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
        )

        return new_layer
