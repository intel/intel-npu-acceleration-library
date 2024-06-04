#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend import run_factory, SDPA
from typing import Optional
from functools import partial
import torch


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Execute SDPA kernel.

    Args:
        query (torch.Tensor): query tensor
        key (torch.Tensor): key tensor
        value (torch.Tensor): value tensor
        attn_mask (torch.Tensor, optional): attention mask tensor. Defaults to None.
        dropout_p (float, optional): optional dropout. Defaults to 0.0.
        is_causal (bool, optional): enable causal mask. Defaults to False.
        scale (Optional[float], optional): custom scale. Defaults to None.

    Raises:
        RuntimeError: _description_

    Returns:
        torch.Tensor: _description_
    """
    backend_cls = partial(SDPA, is_causal=is_causal)
    if dropout_p != 0:
        raise RuntimeError("dropout_p != 0 is not supported yet")
    if scale is not None:
        raise RuntimeError("scale != 0 is not supported yet")

    return run_factory([query, key, value, attn_mask], [], backend_cls)
