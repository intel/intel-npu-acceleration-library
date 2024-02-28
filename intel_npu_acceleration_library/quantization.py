#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from typing import Tuple
import torch


def quantize_tensor(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a fp16 tensor symmetrically.

    Produces a quantize tensor (same shape, dtype == `torch.int8`) and a scale tensor (dtype == `torch.float16)
    The quantization equation is the following W = S * W_q

    Args:
        weight (torch.Tensor): The tensor to quantize

    Raises:
        RuntimeError: Error in the quantization step

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scale
    """
    scale = torch.max(torch.abs(weight), dim=-1).values

    # Compute scale and zero point
    scale = (scale / 127).to(torch.float16).view(-1, 1)

    weights_quant = torch.floor(weight / scale)

    if not (torch.max(weights_quant) <= 127 and torch.min(weights_quant) >= -128):
        raise RuntimeError(
            f"Quantization error: range of quantized weghts = {(torch.min(weights_quant), torch.max(weights_quant))} instead of (-128, 127)"
        )
    return weights_quant.to(torch.int8), scale
