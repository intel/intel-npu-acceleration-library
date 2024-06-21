#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


from intel_npu_acceleration_library.backend.tensor import (
    implements,
    generate_op,
    Tensor,
)
from typing import Optional
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
