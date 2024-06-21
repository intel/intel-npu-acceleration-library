#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


from intel_npu_acceleration_library.backend.tensor import (
    implements,
    generate_op,
    Tensor,
)
from typing import Optional, Sequence
import torch


def __generate_activation(x: Tensor, op: str, out: Optional[Tensor] = None) -> Tensor:
    """Generate an activation function.

    Args:
        x (Tensor): The input tensor.
        op (str): The operation to perform.
        out (Optional[Tensor], optional): The output tensor. Defaults to None.

    Returns:
        Tensor: The output tensor
    """
    out = generate_op(x, op)
    return out


@implements(torch.log)
def log(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the natural logarithm of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "log", out)


@implements(torch.cos)
def cos(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the cosine of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "cos", out)


@implements(torch.tanh)
def tanh(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the hyperbolic tangent of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "tanh", out)


@implements(torch.sqrt)
def sqrt(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the square root of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "sqrt", out)


@implements(torch.abs)
def abs(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the absolute value of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "abs", out)


@implements(torch.acos)
def acos(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the arc-cosine value of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "acos", out)


@implements(torch.asin)
def asin(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the arc-sine value of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "asin", out)


@implements(torch.atan)
def atan(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the arc-tangent value of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "atan", out)


@implements(torch.cosh)
def cosh(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the hyperbolic-cosine value of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "cosh", out)


@implements(torch.erf)
def erf(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the erf function of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "erf", out)


@implements(torch.exp)
def exp(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the exp function of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "exp", out)


@implements(torch.floor)
def floor(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the floor function of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "floor", out)


@implements(torch.sin)
def sin(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the sine of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "sin", out)


@implements(torch.sinh)
def sinh(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the hyperbolic sine of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "sinh", out)


@implements(torch.tan)
def tan(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the tangent of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "tan", out)


@implements(torch.acosh)
def acosh(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the hyperbolic arc-cosine of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "acosh", out)


@implements(torch.asinh)
def asinh(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the hyperbolic arc-sine of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "asinh", out)


@implements(torch.atanh)
def atanh(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the hyperbolic arc-tangent of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "atanh", out)


@implements(torch.round)
def round(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the round value of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "round", out)


@implements(torch.sign)
def sign(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the sign of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "sign", out)


@implements(torch.nn.functional.linear)
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Return the sign of a tensor element-wise.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.
        bias (Optional[Tensor], optional): the bias tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    mm = generate_op([input, weight], "matmul")
    if bias:
        return generate_op([mm, bias], "eltwise_add")
    return mm


@implements(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute scaled dot product attention on query, key and value tensors, using an optional attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.

    Args:
        query (Tensor): query tensor
        key (Tensor): key tensor
        value (Tensor): value tensor
        attn_mask (Tensor, optional): attention mask tensor. Defaults to None.
        dropout_p (float, optional): optional dropout. Defaults to 0.0.
        is_causal (bool, optional): enable causal mask. Defaults to False.
        scale (Optional[float], optional): custom scale. Defaults to None.

    Raises:
        RuntimeError: dropout_p != 0 is not supported yet, scale != 0 is not supported yet

    Returns:
        Tensor: output tensor
    """
    if dropout_p != 0:
        raise RuntimeError("dropout_p != 0 is not supported yet")
    if scale is not None:
        raise RuntimeError("scale != 0 is not supported yet")

    if attn_mask is None:
        return generate_op(
            [query, key, value], "scaled_dot_product_attention_simple", is_causal
        )

    return generate_op(
        [query, key, value, attn_mask], "scaled_dot_product_attention", is_causal
    )


@implements(torch.nn.functional.dropout)
def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Return dropout operation.

    Args:
        input (Tensor): The input tensor.
        p (float): The probability of an element to be zeroed. Defaults to 0.5.
        training (bool): apply dropout if True. Defaults to True.
        inplace (bool): apply dropout in place. Defaults to False.

    Raises:
        NotImplementedError: Training mode is not supported yet, Inplace mode is not supported yet

    Returns:
        Tensor: Output tensor.
    """
    if training:
        raise NotImplementedError("Training mode is not supported yet")

    if inplace:
        raise NotImplementedError("Inplace mode is not supported yet")

    return input


@implements(torch.nn.functional.layer_norm)
def layer_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-05,
) -> Tensor:
    """Return layer normalization operation.

    Args:
        input (Tensor): The input tensor.
        normalized_shape (Sequence[int]): The shape of the normalized tensor.
        weight (Optional[Tensor], optional): The weight tensor. Defaults to None.
        bias (Optional[Tensor], optional): The bias tensor. Defaults to None.
        eps (float): The epsilon value. Defaults to 1e-05.

    Returns:
        Tensor: Output tensor.
    """
    axis = input.shape.index(normalized_shape[0])
    ln = generate_op([input], "normL2", axis, eps)
    if weight:
        ln = ln * weight

    if bias:
        ln = ln + bias

    return ln


@implements(torch.nn.functional.gelu)
def gelu(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the gelu of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "gelu", out)
