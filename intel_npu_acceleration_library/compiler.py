#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.optimizations import horizontal_fusion_linear
from torch._dynamo import register_backend
from typing import Union

try:
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
    from transformers.models.gemma.modeling_gemma import GemmaMLP, GemmaAttention

    is_transformers_available = True
except ModuleNotFoundError:
    # Transformer library is not installed
    is_transformers_available = False


import intel_npu_acceleration_library.nn as nn
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
    if not (dtype.is_floating_point or dtype == torch.int8):
        raise RuntimeError(
            f"intel-npu-acceleration-library library do not support yet the requeste datatype: {dtype}"
        )
    # Prepare and optimize model for NPU
    with torch.no_grad():
        model = prepare_model_for_npu(model, dtype=dtype)

    if dtype.is_floating_point and training:
        # Set model to evaluation only as quantized training is not supported yet
        return model

    return model.eval()


def prepare_model_for_npu(
    model: torch.nn.Module, dtype: torch.dtype = torch.float16
) -> torch.nn.Module:
    """Prepare a torch.nn.Module model to run on the NPU.

    Args:
        model (torch.nn.Module): The model to offload to the NPU
        dtype (torch.dtype): the model target datatype

    Returns:
        torch.nn.Module: The torch.nn.Module compiled and optimized for the NPU
    """
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.Linear):
            model.add_module(name, nn.Linear.fromTorch(layer, dtype))
        elif is_transformers_available:
            if isinstance(layer, (LlamaMLP, GemmaMLP)):
                new_layer = horizontal_fusion_linear(layer)
                model.add_module(name, new_layer)
                prepare_model_for_npu(new_layer, dtype)
            elif isinstance(layer, (LlamaAttention, GemmaAttention)):
                model.add_module(name, nn.LlamaAttention.fromTorch(layer, dtype))
            # elif layer.__class__.__name__ == "PhiMLP":
            #     model.add_module(name, nn.PhiMLP.fromTorch(layer, dtype))
            else:
                prepare_model_for_npu(layer, dtype)
        else:
            prepare_model_for_npu(layer, dtype)

    return model


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
