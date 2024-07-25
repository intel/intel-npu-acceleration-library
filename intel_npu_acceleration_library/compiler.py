#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.optimizations import horizontal_fusion_linear
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
from transformers.models.gemma.modeling_gemma import GemmaMLP, GemmaAttention
from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear
from intel_npu_acceleration_library.quantization import quantize_model
from intel_npu_acceleration_library.dtypes import int8, int4, NPUDtype
from intel_npu_acceleration_library.nn.module import NPUModuleWrapper
import intel_npu_acceleration_library.nn as nn
from torch._dynamo import register_backend
from typing import Union, Callable, Any
from typing import List
import torch
from functools import partial


class CompilerConfig:
    """Configuration class to store the compilation configuration of a model for the NPU."""

    def __init__(
        self,
        use_to: bool = False,
        dtype: Union[torch.dtype, NPUDtype] = torch.float16,
        training: bool = False,
    ) -> None:
        """Initialize the configuration class.

        Args:
            use_to (bool): Enable model compiling using .to() . Defaults to disabled
            dtype (Union[torch.dtype, NPUDtype]): The dtype to compile the model with. Defaults to torch.float16
            training (bool): Enable training. Defaults to disabled
        """
        self.use_to = use_to
        self.dtype = dtype
        self.training = training


def compile(model: torch.nn.Module, config: CompilerConfig) -> torch.nn.Module:
    """Compile a model for the NPU.

    Args:
        model (torch.nn.Module): a pytorch nn.Module to compile and optimize for the npu
        config (CompilerConfig): the compiler configuration

    Raises:
        RuntimeError: invalid datatypes

    Returns:
        torch.nn.Module: compiled NPU nn.Module
    """
    if not (config.dtype.is_floating_point or config.dtype in (int8, int4)):
        raise RuntimeError(
            f"intel-npu-acceleration-library library do not support yet the requeste datatype: {config.dtype}"
        )

    # Prepare and optimize model for NPU
    with torch.no_grad():
        # Model lowering to NPU ops
        if config.use_to:
            model = model.to("npu")
        else:
            # General optimizations
            apply_general_optimizations(model)

        if config.dtype in (int8, int4):
            # Quantize model
            model = quantize_model(model, config.dtype)
            weights_quantization(model)

        if not config.use_to:
            create_npu_kernels(model)

    if config.dtype.is_floating_point and config.training:
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

    def wrapper(model: torch.nn.Module, *args: Any, **kwargs: Any):
        """Recursively apply the optimization function.

        Args:
            model (torch.nn.Module): original module
            args (Any): positional arguments
            kwargs (Any): keyword arguments

        """
        if not isinstance(model, NPUModuleWrapper) or kwargs.get(
            "ignore_isinstance", False
        ):
            for name, layer in model.named_children():
                new_layer = func(name, layer, *args, **kwargs)
                if new_layer:
                    model.add_module(name, new_layer)
                    if not isinstance(new_layer, NPUModuleWrapper) or kwargs.get(
                        "ignore_isinstance", False
                    ):
                        wrapper(new_layer, *args, **kwargs)
                else:
                    if not isinstance(layer, NPUModuleWrapper) or kwargs.get(
                        "ignore_isinstance", False
                    ):
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
def weights_quantization(
    name: str, layer: torch.nn.Module, ignore_isinstance: bool = True
) -> Union[torch.nn.Module, None]:
    """Apply weights quantization.

    Args:
        name (str): Layer name
        layer (torch.nn.Module): Original torch.nn.Linear module
        ignore_isinstance (bool): ignore isinstance check in module_optimization. Defaults to True.

    Raises:
        RuntimeError: unsupported quantization bits

    Returns:
        None: Returns None
    """
    if isinstance(layer, WeightOnlyLinear):
        if (layer.bits == 4) or (layer.bits == 8):
            layer.forward = partial(forward, layer)
        else:
            raise RuntimeError(f"Unsupported quantization bits: {layer.bits}")
    return None


def forward(self, input):
    """Override forward method for WeightOnlyLinear class.

    Args:
        input: The input tensor.

    Returns:
        torch.Tensor: The output tensor.
    """
    if self.bits == 4:
        # Unpack the int4 values
        lower_int4 = self.qweight & 0x0F
        lower_int4 = lower_int4 - (lower_int4 & 0x8) * 2
        upper_int4 = (self.qweight >> 4) & 0x0F
        upper_int4 = upper_int4 - (upper_int4 & 0x8) * 2

        w = torch.stack((lower_int4, upper_int4), dim=2)
        w = w.contiguous().view(self.qweight.shape[0], -1)

    elif self.bits == 8:
        w = self.qweight.view(torch.int8)

    output = (
        torch.nn.functional.linear(input.to(torch.float16), w.to(torch.float16), None)
        * self.scales.T
    )

    if self.bias:
        return output + self.bias

    return output


@register_backend
def npu(
    gm: Union[torch.nn.Module, torch.fx.GraphModule],
    example_inputs: List[torch.Tensor],
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
    config = CompilerConfig()
    return compile(gm, config)
