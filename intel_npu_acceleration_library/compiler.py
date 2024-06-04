#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.optimizations import horizontal_fusion_linear
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
from transformers.models.gemma.modeling_gemma import GemmaMLP, GemmaAttention
from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear
from intel_npu_acceleration_library.quantization import quantize_model
from intel_npu_acceleration_library.dtypes import int8, int4
import intel_npu_acceleration_library.nn as nn
from torch._dynamo import register_backend
from typing import Union, Callable, Any
from typing import List
import torch


def compile(
    model: torch.nn.Module, dtype: torch.dtype = torch.float16, training: bool = False
) -> torch.nn.Module:
    """Compile a model for the NPU.

    Args:
        model (torch.nn.Module): a pytorch nn.Module to compile and optimize for the npu
        dtype (torch.dtype): the model target datatype, default to torch.float16
        training (bool): enable training. Default disabled

    Raises:
        RuntimeError: invalid datatypes

    Returns:
        torch.nn.Module: compiled NPU nn.Module
    """
    if not (dtype.is_floating_point or dtype in (int8, int4)):
        raise RuntimeError(
            f"intel-npu-acceleration-library library do not support yet the requeste datatype: {dtype}"
        )

    # Prepare and optimize model for NPU
    with torch.no_grad():
        # General optimizations
        apply_horizontal_fusion(model)
        optimize_llama_attention(model, dtype)
        if dtype in (int8, int4):
            # Quantize model
            model = quantize_model(model, dtype)

        # Model lowering to NPU ops
        lower_linear(model, dtype)

    if dtype.is_floating_point and training:
        # Set model to evaluation only as quantized training is not supported yet
        return model

    return model.eval()


def module_optimization(func: Callable) -> torch.nn.Module:
    """Optimize recursively a torch.nn.Module with a specific function.

    The function `func` get called recursively to every module in the network.

    Args:
        func (Callable): optimization function

    Returns:
        torch.nn.Module: optimized module
    """

    def wrapper(model: torch.nn.Module, *args: Any, **kwargs: Any):
        """Recursively apply the optimization function.

        Args:
            model (torch.nn.Module): original module
            args (Any): positional arguments
            kwargs (Any): keyword arguments

        """
        for name, layer in model.named_children():
            new_layer = func(name, layer, *args, **kwargs)
            if new_layer:
                model.add_module(name, new_layer)
                wrapper(new_layer, *args, **kwargs)
            else:
                wrapper(layer, *args, **kwargs)

    return wrapper


@module_optimization
def lower_linear(
    name: str, layer: torch.nn.Module, dtype: torch.dtype
) -> Union[torch.nn.Module, None]:
    """Lower torch.nn.Linear layer to NPU equivalent operators.

    Args:
        name (str): Layer name
        layer (torch.nn.Module): Original torch.nn.Linear module
        dtype (torch.dtype): Target datatype

    Raises:
        RuntimeError: unsupported quantization bits

    Returns:
        Union[torch.nn.Module, None]: Return the new NPU operator or None
    """
    if isinstance(layer, torch.nn.Linear):
        return nn.Linear.fromTorch(layer, dtype)
    if isinstance(layer, torch.nn.Conv2d):
        return nn.Conv2d.fromTorch(layer, dtype)
    if isinstance(layer, WeightOnlyLinear):
        if layer.bits == 4:
            return nn.QuantizedLinear(
                layer.qweight.to(torch.uint8), layer.scales, layer.bias
            )
        elif layer.bits == 8:
            return nn.QuantizedLinear(
                layer.qweight.view(torch.int8), layer.scales, layer.bias
            )
        else:
            raise RuntimeError(f"Unsupported quantization bits: {layer.bits}")
    return None


@module_optimization
def apply_horizontal_fusion(
    name: str, layer: torch.nn.Module
) -> Union[torch.nn.Module, None]:
    """Apply horizontal fusion (merging two linear layers with same input) when necessary.

    Args:
        name (str): Layer name
        layer (torch.nn.Module): Original module

    Returns:
        Union[torch.nn.Module, None]: optimized module
    """
    if isinstance(layer, (LlamaMLP, GemmaMLP)):
        return horizontal_fusion_linear(layer)
    return None


@module_optimization
def optimize_llama_attention(
    name: str, layer: torch.nn.Module, dtype: torch.dtype
) -> Union[torch.nn.Module, None]:
    """Optimize LLAMA attention block.

    Args:
        name (str): Module name
        layer (torch.nn.Module): Original Module
        dtype (torch.dtype): Target datatype

    Returns:
        Union[torch.nn.Module, None]: optimized llama module
    """
    if isinstance(layer, (LlamaAttention, GemmaAttention)):
        return nn.LlamaAttention.fromTorch(layer, dtype)
    return None


@register_backend
def npu(
    gm: Union[torch.nn.Module, torch.fx.GraphModule], example_inputs: List[torch.Tensor]
) -> Union[torch.nn.Module, torch.fx.GraphModule]:
    """Implement the custom torch 2.0 compile backend for the NPU.

    Args:
        gm (Union[torch.nn.Module, torch.fx.GraphModule]): The torch fx Module
        example_inputs (List[torch.Tensor]): A list of example inputs

    Returns:
        Union[torch.nn.Module, torch.fx.GraphModule]: The compiled model
    """
    # Run some optimizations
    gm = horizontal_fusion_linear(gm)

    # For now compile in fp16
    return compile(gm)
