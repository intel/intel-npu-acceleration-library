#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.optimizations import horizontal_fusion_linear
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
from transformers.models.gemma.modeling_gemma import GemmaMLP, GemmaAttention
from transformers.models.phi3.modeling_phi3 import Phi3MLP
from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear
from intel_npu_acceleration_library.quantization import quantize_model
from intel_npu_acceleration_library.dtypes import int8, int4
from intel_npu_acceleration_library.nn.module import NPUModuleWrapper
import intel_npu_acceleration_library.nn as nn
from torch._dynamo import register_backend
from typing import Union, Callable, Any
from typing import List
import torch
from functools import partial


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

        if dtype in (int8, int4):
            # Quantize model
            model = quantize_model(model, dtype)
            weights_quantization(model)

        # Model lowering to NPU ops
        if isinstance(model, Phi3MLP):
            model = model.to("npu")
        else:
            # General optimizations
            apply_general_optimizations(model)
            create_npu_kernels(model)

    if dtype.is_floating_point and training:
        # Set model to evaluation only as quantized training is not supported yet
        return model

    return model.eval()


def apply_general_optimizations(model: torch.nn.Module):
    """Apply general optimizations to a torch.nn.Module.

    Args:
        model (torch.nn.Module): a pytorch nn.Module to compile and optimize for the npu
    """
    apply_horizontal_fusion(model)
    optimize_llama_attention(model)
    optimize_phi3_MLP(model)


def create_npu_kernels(model: torch.nn.Module):
    """Create NPU kernels.

    Args:
        model (torch.nn.Module): a pytorch nn.Module to compile and optimize for the npu
    """
    lower_linear(model)


def module_optimization(func: Callable) -> torch.nn.Module:
    """Optimize recursively a torch.nn.Module with a specific function.

    The function `func` get called recursively to every module in the network.

    Args:
        func (Callable): optimization function

    Returns:
        torch.nn.Module: optimized module
    """
    module_optimization.counter = 0  # type: ignore[attr-defined]

    def wrapper(model: torch.nn.Module, *args: Any, **kwargs: Any):
        """Recursively apply the optimization function.

        Args:
            model (torch.nn.Module): original module
            args (Any): positional arguments
            kwargs (Any): keyword arguments

        """
        if not isinstance(model, NPUModuleWrapper):
            for name, layer in model.named_children():
                new_layer = func(name, layer, *args, **kwargs)
                if (func.__name__ == "optimize_phi3_MLP") and (
                    module_optimization.counter >= 5  # type: ignore[attr-defined]
                ):
                    new_layer = None

                if new_layer:
                    module_optimization.counter += 1  # type: ignore[attr-defined]
                    model.add_module(name, new_layer)
                    if not isinstance(new_layer, NPUModuleWrapper):
                        wrapper(new_layer, *args, **kwargs)
                else:
                    if not isinstance(layer, NPUModuleWrapper):
                        wrapper(layer, *args, **kwargs)

    return wrapper


@module_optimization
def lower_linear(name: str, layer: torch.nn.Module) -> Union[torch.nn.Module, None]:
    """Lower torch.nn.Linear layer to NPU equivalent operators.

    Args:
        name (str): Layer name
        layer (torch.nn.Module): Original torch.nn.Linear module

    Raises:
        RuntimeError: unsupported quantization bits

    Returns:
        Union[torch.nn.Module, None]: Return the new NPU operator or None
    """
    if isinstance(layer, torch.nn.Linear):
        return nn.Linear.fromTorch(layer)
    if isinstance(layer, torch.nn.Conv2d):
        return nn.Conv2d.fromTorch(layer)
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
    name: str, layer: torch.nn.Module
) -> Union[torch.nn.Module, None]:
    """Optimize LLAMA attention block.

    Args:
        name (str): Module name
        layer (torch.nn.Module): Original Module

    Returns:
        Union[torch.nn.Module, None]: optimized llama module
    """
    if isinstance(layer, (LlamaAttention, GemmaAttention)):
        return nn.LlamaAttention.fromTorch(layer)
    return None


@module_optimization
def optimize_phi3_MLP(
    name: str, layer: torch.nn.Module
) -> Union[torch.nn.Module, None]:
    """Optimize Phi-3 MLP block.

    Args:
        name (str): Module name
        layer (torch.nn.Module): Original Module

    Returns:
        Union[torch.nn.Module, None]: optimized Phi-3 module
    """
    if layer.__class__.__name__ == "Phi3MLP":
        return layer.to("npu")
    return None


@module_optimization
def weights_quantization(
    name: str, layer: torch.nn.Module
) -> Union[torch.nn.Module, None]:
    """Apply weights quantization.

    Args:
        name (str): Layer name
        layer (torch.nn.Module): Original torch.nn.Linear module

    Raises:
        RuntimeError: unsupported quantization bits

    Returns:
        None: Returns None
    """
    if isinstance(layer, WeightOnlyLinear):
        if layer.bits == 4:
            print("This works - int4 !!")
            layer.forward = partial(forward, layer)
        elif layer.bits == 8:
            print("This works - int8 !!")
            layer.forward = partial(forward, layer)
        else:
            raise RuntimeError(f"Unsupported quantization bits: {layer.bits}")
    return None


def forward(self, input):
    """Override forward method for WeightOnlyLinear class.

    Args:
        input: Thr input tensor.

    Returns:
        torch.Tensor: The output tensor.
    """
    w = self.qweight.to(torch.float16)
    # output = torch.nn.functional.linear(input, w, None) * self.scales
    # if self.bias:
    #     return output + self.bias
    output = torch.nn.functional.linear(input.to(w.dtype), w, self.bias) * self.scales
    if self.bias:
        output = torch.nn.functional.linear(input, w, self.bias) * self.scales
    else:
        output = torch.nn.functional.linear(input, w, None) * self.scales
    return output


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
