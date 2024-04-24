#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import intel_npu_acceleration_library.nn as nn
import torch


class Conv2d(torch.nn.Module):
    """
    2D convolutional layer implementation.

    Attrs:
        weight (torch.Tensor): The weight tensor of the layer.
        bias (torch.Tensor): The bias tensor of the layer.

    Args:
        matmul (torch.nn.Module): The matrix multiplication module.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution. Defaults to (1, 1).
        padding (Union[int, Tuple[int, int]], optional): Padding added to the input. Defaults to (0, 0).
        dilation (Union[int, Tuple[int, int]], optional): Dilation rate of the convolution. Defaults to (1, 1).
    """

    def __init__(
        self,
        matmul,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    ) -> None:
        """Initialize a Convolutional layer.

        Args:
            matmul: The matrix multiplication function to be used.
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            kernel_size: The size of the convolutional kernel.
            stride: The stride of the convolution. Defaults to (1, 1).
            padding: The padding added to the input. Defaults to (0, 0).
            dilation: The dilation rate of the convolution. Defaults to (1, 1).
        """
        super().__init__()

        self.matmul = matmul
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride

    @property
    def weight(self) -> torch.Tensor:
        """
        Get the weight tensor of the layer.

        Returns:
            torch.Tensor: The weight tensor.
        """
        return self.matmul.weight

    @property
    def bias(self) -> torch.Tensor:
        """
        Get the bias tensor of the layer.

        Returns:
            torch.Tensor: The bias tensor.
        """
        return self.matmul.bias

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

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the convolutional layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Unfold the input
        inp_unf = torch.nn.functional.unfold(
            x, self.kernel_size, self.dilation, self.padding, self.stride
        ).transpose(1, 2)
        out_unf = self.matmul(inp_unf).transpose(1, 2)

        out_shape = [x.shape[0], self.out_channels] + [
            self.compute_output_dim(dim, idx) for idx, dim in enumerate(x.shape[2:])
        ]
        out = out_unf.view(out_shape)

        return out

    @staticmethod
    def fromTorch(layer, dtype) -> "Conv2d":
        """
        Create a Conv2d layer from a torch.nn.Conv2d layer.

        Args:
            layer (torch.nn.Conv2d): The torch Conv2d layer.
            dtype (torch.dtype): Data type of the layer.

        Returns:
            Conv2d: The converted Conv2d layer.
        """
        weight = layer.weight.view(layer.weight.shape[0], -1)
        matmul = nn.Linear.fromTensor(weight, getattr(layer, "bias", None), dtype)
        new_layer = Conv2d(
            matmul,
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
        )

        return new_layer
