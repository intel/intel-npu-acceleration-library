#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import run_matmul
from typing import Optional, Iterable, Union
import torch


class AutogradMatMul(torch.autograd.Function):
    """Autograd module for Linear operation."""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, w: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run a linear forward pass. Depending on the datatype of the weights it runs a float or quantized operation.

            Equivalent pytorch code:
            result = x @ w.T

        Args:
            ctx (Any): the autograd context
            x (torch.Tensor): Activation tensor. Its dtype must be torch.float16
            w (torch.Tensor): Weight tensor. Its dtype must be torch.float16
            scale (Optional[torch.Tensor], optional): Quantization scale. If weights.dtype == torch.int8 then it must be set. Defaults to None.

        Returns:
            torch.Tensor: result
        """
        result = run_matmul(x, w, scale, None)
        ctx.save_for_backward(w, x)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Iterable[Union[torch.Tensor, None]]:
        """Run a linear backward pass.

        grad_output shape: [batch, output_channels]
        x shape: [batch, input_channels]
        w shape: [output_channels, input_channels]

        Expected gradients
        dl_dx shape: [batch, input_channels]
        dl_dw shape: [output_channels, input_channels]

        Equivalent pytorch code:
        dl_dx = grad_output @ w.to(torch.float32)
        dl_dw =  (x.T @ grad_output).T

        Args:
            ctx (Any): the autograd context
            grad_output (torch.Tensor): output gradient

        Returns:
            Iterable[Union[torch.Tensor, None]]: Input and parameters gradients
        """
        (
            w,
            x,
        ) = ctx.saved_tensors

        dl_dx = run_matmul(grad_output, torch.transpose(w, -1, -2))
        dl_dw = run_matmul(
            torch.transpose(grad_output, -1, -2), torch.transpose(x, -1, -2)
        )
        return dl_dx, dl_dw, None
