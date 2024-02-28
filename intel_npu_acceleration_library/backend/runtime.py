#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import Linear, QLinear
from intel_npu_acceleration_library.backend import MatMul, QMatMul
from intel_npu_acceleration_library.backend import NNFactory
from torch.profiler import record_function
from typing import Optional, List, Any, Dict, Deque
from collections import deque
import numpy as np
import torch

_model_cache: Dict[str, Deque[NNFactory]] = {}


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

    # Set tensors as contiguous in memory
    x = set_contiguous(x)
    weights = set_contiguous(weights)
    weights = weights.view([-1, weights.shape[-1]])

    if weights.dtype.is_floating_point:
        op_class = Linear if op_id is not None else MatMul
        op_args = [weights.to(torch.float16).numpy()]
    elif weights.dtype == torch.int8:
        if scale is None:
            raise RuntimeError("Quantized weights require a not null scale")
        op_class = QLinear if op_id is not None else QMatMul
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
    x_np = x.to(torch.float16).view([-1, inC]).numpy()

    real_batch = x_np.shape[0]

    # If the real batch is 1, we need to use 16 and then slice it
    if real_batch == 1:
        batch = 16
    else:
        batch = real_batch

    key = f"{str(op_class.__name__)}_{batch}_{inC}_x_{outC}_{inC}_{x_np.dtype}"
    models = _model_cache.get(key, None)

    if models is None:
        _model_cache[key] = deque([op_class(inC, outC, batch)])
    elif len(models) < 1:
        _model_cache[key].append(op_class(inC, outC, batch))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    if real_batch == 1:
        # Expand and then slice
        with record_function(f"npu_matvec_{key}"):
            ret = model.run(np.vstack(16 * [x_np]), *op_args, **op_kwargs)[:1, ...]
    else:
        with record_function(f"npu_matmul_{key}"):
            ret = model.run(x_np, *op_args, **op_kwargs)
    return torch.from_numpy(ret).view(expected_output_shape).to(input_dtype)


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
    x: torch.Tensor,
    weights: List[torch.Tensor],
    backend_cls: Any,
    op_id: Optional[str] = None,
) -> torch.Tensor:
    """Run a factory operation. Depending on the datatype of the weights it runs a float or quantized operation.

    Args:
        x (torch.Tensor): Activation tensor. Its dtype must be torch.float16
        weights (torch.Tensor): Weights tensor.  Its dtype can be torch.float16 or torch.int8
        backend_cls (Any): Backend class to run
        op_id (Optional[str], optional): Operation ID. Defaults to None.

    Returns:
        torch.Tensor: result
    """
    global _model_cache

    inC = x.shape[-1]
    # TODO: fix this
    outC = inC

    # Use or not op_id depending on the class used
    op_kwargs = {"op_id": op_id} if op_id else {}

    original_input_shape = x.shape
    expected_output_shape = list(original_input_shape[:-1]) + [outC]

    # Reshape input
    input_dtype = x.dtype
    x_np = x.to(torch.float16).view((-1, inC)).numpy()

    op_args = [w.to(torch.float16).numpy() for w in weights]

    real_batch = x_np.shape[0]

    # If the real batch is 1, we need to use 16 and then slice it
    if real_batch == 1:
        batch = 16
    else:
        batch = real_batch

    key = f"{backend_cls.func.__name__}_{batch}_{inC}_{outC}_{x_np.dtype}"
    models = _model_cache.get(key, None)

    if models is None:
        _model_cache[key] = deque([backend_cls(batch=batch)])
    elif len(models) < 1:
        _model_cache[key].append(backend_cls(batch=batch))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    if real_batch == 1:
        # Expand and then slice
        with record_function(f"npu_factory_vect_{key}"):
            ret = model.run(np.vstack(16 * [x_np]), *op_args, **op_kwargs)[:1, ...]
    else:
        with record_function(f"npu_factory_mul_{key}"):
            ret = model.run(x_np, *op_args, **op_kwargs)

    return torch.from_numpy(ret).view(expected_output_shape).to(input_dtype)
