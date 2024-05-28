#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
import intel_npu_acceleration_library.backend.compression as compression
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
from intel_npu_acceleration_library.dtypes import NPUDtype
from neural_compressor.quantization import fit
from typing import Tuple
import logging
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


def quantize_fit(
    model: torch.nn.Module, weights_dtype: str, algorithm: str = "RTN"
) -> torch.nn.Module:
    """Quantize a model with a given configuration.

    Args:
        model (torch.nn.Module): The model to quantize
        weights_dtype (str): The datatype for the weights
        algorithm (str, optional): The quantization algorithm. Defaults to "RTN".

    Raises:
        RuntimeError: Quantization error: unsupported datatype

    Returns:
        torch.nn.Module: The quantized model
    """
    if weights_dtype == "int4":
        bits = 4
    elif weights_dtype == "int8":
        bits = 8
    else:
        raise RuntimeError(f"Quantization error: unsupported datatype {weights_dtype}")

    conf = PostTrainingQuantConfig(
        approach="weight_only",
        tuning_criterion=TuningCriterion(timeout=100000),
        op_type_dict={
            ".*": {  # match all ops
                "weight": {
                    "dtype": weights_dtype,
                    "bits": bits,
                    "group_size": -1,
                    "scheme": "sym",
                    "algorithm": algorithm,
                },
                "activation": {
                    "dtype": "fp16",
                },
            }
        },
    )

    return fit(model=model, conf=conf)


def quantize_i8_model(
    model: torch.nn.Module, algorithm: str = "RTN"
) -> torch.nn.Module:
    """Quantize a model to 8-bit representation.

    Args:
        model (torch.nn.Module): The model to quantize
        algorithm (str, optional): The quantization algorithm. Defaults to "RTN".

    Returns:
        torch.nn.Module: The quantized model
    """
    quantized_model = quantize_fit(model, "int8", algorithm)

    return quantized_model.export_compressed_model(
        scale_dtype=torch.float16, use_optimum_format=False
    )


def quantize_i4_model(
    model: torch.nn.Module, algorithm: str = "RTN"
) -> torch.nn.Module:
    """Quantize a model to 4-bit representation.

    Args:
        model (torch.nn.Module): The model to quantize
        algorithm (str, optional): The quantization algorithm. Defaults to "RTN".

    Returns:
        torch.nn.Module: The quantized model
    """
    quantized_model = quantize_fit(model, "int4", algorithm)

    return quantized_model.export_compressed_model(
        compression_dtype=torch.int8,
        scale_dtype=torch.float16,
        use_optimum_format=False,
    )


def quantize_model(model: torch.nn.Module, dtype: NPUDtype) -> torch.nn.Module:
    """Quantize a model.

    Args:
        model (torch.nn.Module): The model to quantize
        dtype (NPUDtype): The desired datatype

    Raises:
        RuntimeError: Quantization error: unsupported datatype

    Returns:
        torch.nn.Module: The quantized model
    """
    # Silence neural compressor logger
    logger = logging.getLogger("neural_compressor")
    logger.setLevel(logging.ERROR)

    if dtype.bits == 4:
        return quantize_i4_model(model)
    elif dtype == torch.int8:
        return quantize_i8_model(model)
    else:
        raise RuntimeError(f"Quantization error: unsupported datatype {dtype}")
