#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
import intel_npu_acceleration_library.backend.compression as compression
from typing import Tuple
import torch


def quantize_tensor(
    weight: torch.Tensor, min_max_range: Tuple[int, int] = (-128, 127)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a fp16 tensor symmetrically.

    Produces a quantize tensor (same shape, dtype == `torch.int8`) and a scale tensor (dtype == `torch.float16)
    The quantization equation is the following W = S * W_q

    Args:
        weight (torch.Tensor): The tensor to quantize
        min_max_range (Tuple[int, int]): The min and max range for the quantized tensor. Defaults to (-128, 127).

    Raises:
        RuntimeError: Error in the quantization step

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scale
    """
    scale = torch.max(torch.abs(weight), dim=-1).values

    # if any of the elements are zeros set the scale to the max value
    if torch.any(scale == 0):
        scale = torch.ones_like(scale) * torch.max(torch.abs(weight))

    # Compute scale and zero point
    scale = (scale / max(min_max_range)).to(torch.float16).view(-1, 1)

    weights_quant = torch.floor(weight / scale)

    if not (
        torch.max(weights_quant) <= max(min_max_range)
        and torch.min(weights_quant) >= min(min_max_range)
    ):
        raise RuntimeError(
            f"Quantization error: range of quantized weghts = {(torch.min(weights_quant), torch.max(weights_quant))} instead of ({min_max_range})"
        )
    return weights_quant.to(torch.int8), scale


def compress_to_i4(weights: torch.Tensor) -> torch.Tensor:
    """
    Compresses a given tensor to 4-bit representation.

    Args:
        weights (torch.Tensor): The input tensor to be compressed.

    Returns:
        torch.Tensor: The compressed tensor with 4-bit representation.
    """
    return torch.tensor(compression.compress_to_i4(weights.numpy()))
