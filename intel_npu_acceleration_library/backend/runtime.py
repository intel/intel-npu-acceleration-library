#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import Linear, QLinear
from intel_npu_acceleration_library.backend import MatMul, QMatMul
from intel_npu_acceleration_library.backend import NNFactory
from torch.profiler import record_function
from typing import Optional, Any, List, Dict, Deque, Union
from functools import partial
from collections import deque
import numpy as np
import torch

_model_cache: Dict[str, Deque[NNFactory]] = {}


def clear_cache():
    """Clear the cache of models."""
    global _model_cache
    _model_cache = {}


@torch.no_grad()
def run_matmul(
    x: torch.Tensor,
    weights: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    op_id: Optional[str] = None,
) -> torch.Tensor:
    """Run a matmul operation. Depending on the datatype of the weights it runs a float or quantized operation.

    Args:
        x (torch.Tensor): Activation tensor. Its dtype must be torch.float16
        weights (torch.Tensor): Weights tensor.  Its dtype can be torch.float16 or torch.int8
        scale (Optional[torch.Tensor], optional): Quantization scale. If weights.dtype == torch.int8 then it must be set. Defaults to None.
        op_id (Optional[str], optional): Operation ID. Defaults to None.

    Raises:
        RuntimeError: Unsupported weights datatype. Supported types: [torch.float16, torch.int8]

    Returns:
        torch.Tensor: result
    """
    global _model_cache

    outC, inC = weights.shape[-2:]

    if weights.dtype == torch.uint8:
        # In case is Int4 we need to double the input channels because weights are compressed
        inC *= 2

    # Set tensors as contiguous in memory
    x = set_contiguous(x)
    weights = set_contiguous(weights)
    if len(weights.shape) > 2:
        weights = weights.view([-1, weights.shape[-1]])

    if weights.dtype.is_floating_point:
        op_class = Linear if op_id is not None else MatMul
        op_class_name = op_class.__name__
        create_op = partial(op_class)
        op_args = [weights.numpy()]
    elif weights.dtype in (torch.int8, torch.uint8):
        if scale is None:
            raise RuntimeError("Quantized weights require a not null scale")
        op_class = QLinear if op_id is not None else QMatMul
        op_class_name = op_class.__name__
        np_dtype = np.int8 if weights.dtype == torch.int8 else np.uint8
        create_op = partial(op_class, dtype=np_dtype)
        if scale is None:
            raise RuntimeError(
                f"Quantized matmul (weights dtype == {weights.dtype}) requires scale (scale = {scale})"
            )
        op_args = [weights.numpy(), scale.numpy()]
    else:
        raise RuntimeError(f"Unsupported dtype for weights {weights.dtype}")

    if not x.dtype.is_floating_point:
        raise RuntimeError(f"Unsupported dtype for activation {x.dtype}")

    # Use or not op_id depending on the class used
    op_kwargs = {"op_id": op_id} if op_id else {}

    original_input_shape = x.shape
    expected_output_shape = list(original_input_shape[:-1]) + [outC]

    if not (len(x.shape) >= 2):
        raise RuntimeError(f"Input shape {x.shape} must me >= 2")

    # Reshape input
    input_dtype = x.dtype
    x = x.to(torch.float16) if input_dtype != torch.float16 else x
    if len(x.shape) > 2 or x.shape[-1] != inC:
        x = x.view([-1, inC])
    x_np = x.numpy()

    batch = x_np.shape[0]

    key = f"{str(op_class_name)}_{batch}_{inC}_x_{outC}_{inC}_{x_np.dtype}"
    models = _model_cache.get(key, None)

    if models is None:
        _model_cache[key] = deque([create_op(inC, outC, batch)])
    elif len(models) < 1:
        _model_cache[key].append(create_op(inC, outC, batch))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    profiling_name = "matvec" if batch == 1 else "matmul"
    with record_function(f"npu_{profiling_name}_{key}"):
        ret = model.run(x_np, *op_args, **op_kwargs)

    return adapt_output_tensor(ret, expected_output_shape, input_dtype)


def adapt_output_tensor(
    output: np.ndarray, original_shape: torch.Size, input_dtype: torch.dtype
) -> torch.Tensor:
    """Adapt the output tensor to the original shape and dtype.

    Args:
        output (np.ndarray): output tensor
        original_shape (torch.Size): original shape
        input_dtype (torch.dtype): input dtype

    Returns:
        torch.Tensor: output tensor
    """
    output = torch.from_numpy(output)
    if output.shape != original_shape:
        output = output.view(original_shape)
    # needs to copy as the same buffer can be reutilized
    return output.to(input_dtype, copy=True)


def set_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Set tensor to be contiguous in memory.

    Args:
        tensor (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output, contiguous tensor
    """
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


@torch.no_grad()
def run_factory(
    x: Union[torch.Tensor, List[torch.Tensor]],
    weights: List[torch.Tensor],
    backend_cls: Any,
    op_id: Optional[str] = None,
) -> torch.Tensor:
    """Run a factory operation. Depending on the datatype of the weights it runs a float or quantized operation.

    Args:
        x (Union[torch.Tensor, List[torch.Tensor]]): Activation tensor(s). Its dtype must be torch.float16
        weights (torch.Tensor): Weights tensor.  Its dtype can be torch.float16 or torch.int8
        backend_cls (Any): Backend class to run
        op_id (Optional[str], optional): Operation ID. Defaults to None.

    Returns:
        torch.Tensor: result
    """
    global _model_cache

    # Use or not op_id depending on the class used
    op_kwargs = {"op_id": op_id} if op_id else {}

    if not isinstance(x, (list, tuple)):
        x = [x]

    # Reshape input
    input_dtype = x[0].dtype
    x_np = [set_contiguous(elem).to(torch.float16).numpy() for elem in x]
    op_args = [set_contiguous(w).to(torch.float16).numpy() for w in weights]

    shape_dtype_signature = "_".join(
        ["_".join(str(dim) for dim in t.shape) + f"_{t.dtype}" for t in x_np + op_args]
    )
    key = f"{backend_cls.func.__name__}_{shape_dtype_signature}"
    models = _model_cache.get(key, None)

    input_shapes = [elem.shape for elem in x_np]
    if models is None:
        _model_cache[key] = deque([backend_cls(*input_shapes)])
    elif len(models) < 1:
        _model_cache[key].append(backend_cls(*input_shapes))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    with record_function(f"npu_factory_mul_{key}"):
        ret = model.run(*x_np, *op_args, **op_kwargs)

    return adapt_output_tensor(ret, model.output_shape, input_dtype)
