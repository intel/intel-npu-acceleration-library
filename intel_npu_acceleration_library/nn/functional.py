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


@implements(torch.ceil)
def ceil(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the ceil of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "ceiling", out)


@implements(torch.clamp)
def clamp(
    x: Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Return the clamp of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        min (Optional[float], optional): The minimum value. Defaults to None.
        max (Optional[float], optional): The maximum value. Defaults to None.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    if min is None:
        min = x.dtype.min
    if max is None:
        max = x.dtype.max
    out = generate_op(x, "clamp", min, max)
    return out


@implements(torch.neg)
def neg(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Return the negative of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    return __generate_activation(x, "negative", out)


@implements(torch.flatten)
def flatten(x, start_dim=0, end_dim=-1) -> "Tensor":
    """
    Flatten the tensor.

    Args:
        x (Tensor): The input tensor.
        start_dim (int): The first dim to flatten. Defaults to 0.
        end_dim (int): The last dim to flatten. Defaults to -1.

    Returns:
        Tensor: The flattened tensor.
    """
    return x.flatten(start_dim, end_dim)


# Functional activations


@implements(torch.nn.functional.elu)
def elu(x: Tensor, alpha: float = 1.0, inplace=False) -> Tensor:
    """Return the clamp of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        alpha (float): The alpha value. Defaults to 1.0.
        inplace (bool): apply elu in place. Defaults to False.

    Returns:
        Tensor: Output tensor.
    """
    out = generate_op(x, "elu", alpha)
    if inplace:
        x = out
    return out


@implements(torch.nn.functional.gelu)
def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """Return the gelu of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        approximate (str): The approximation method. Defaults to 'none'. When the approximate argument is 'tanh', Gelu is estimated with tanh approximation. When the approximate argument is 'erf', Gelu is estimated with erf approximation. When the approximate argument is 'none', Gelu is estimated with the original formula.

    Returns:
        Tensor: Output tensor.
    """
    if approximate == "tanh":
        return __generate_activation(x, "gelu")
    else:
        return __generate_activation(x, "gelu_erf")


@implements(torch.nn.functional.relu)
def relu(x: Tensor, inplace=False) -> Tensor:
    """Return the relu of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        inplace (bool): apply elu in place. Defaults to False.

    Returns:
        Tensor: Output tensor.
    """
    out = generate_op(x, "relu")
    if inplace:
        x = out
    return out


@implements(torch.nn.functional.sigmoid)
def sigmoid(x: Tensor) -> Tensor:
    """Return the sigmoid of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: Output tensor.
    """
    return generate_op(x, "sigmoid")


@implements(torch.nn.functional.hardswish)
def hardswish(x: Tensor, inplace=False) -> Tensor:
    """Return the hardswish of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        inplace (bool): apply elu in place. Defaults to False.

    Returns:
        Tensor: Output tensor.
    """
    out = generate_op(x, "hswish")
    if inplace:
        x = out
    return out


@implements(torch.nn.functional.mish)
def mish(x: Tensor, inplace=False) -> Tensor:
    """Return the mish of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        inplace (bool): apply elu in place. Defaults to False.

    Returns:
        Tensor: Output tensor.
    """
    out = generate_op(x, "mish")
    if inplace:
        x = out
    return out


@implements(torch.nn.functional.softplus)
def softplus(x: Tensor, beta: float = 1, threshold: float = 20) -> Tensor:
    """Return the softplus of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        beta (float): The beta value. Defaults to 1.
        threshold (float): The threshold value. Defaults to 20.

    Raises:
        NotImplementedError: Only default values are supported

    Returns:
        Tensor: Output tensor.
    """
    if beta == 1 and threshold == 20:
        return generate_op(x, "softplus")
    else:
        raise NotImplementedError("Only default values are supported")


@implements(torch.nn.functional.hardsigmoid)
def hardsigmoid(x: Tensor, inplace=False) -> Tensor:
    """Return the hardsigmoid of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        inplace (bool): apply elu in place. Defaults to False.

    Returns:
        Tensor: Output tensor.
    """
    out = generate_op(x, "hsigmoid")
    if inplace:
        x = out
    return out


@implements(torch.nn.functional.silu)
def silu(x: Tensor, inplace=False) -> Tensor:
    """Return the silu of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        inplace (bool): apply elu in place. Defaults to False.

    Returns:
        Tensor: Output tensor.
    """
    out = x * generate_op(x, "sigmoid")
    if inplace:
        x = out
    return out


@implements(torch.nn.functional.adaptive_avg_pool2d)
def adaptive_avg_pool2d(input: Tensor, output_size: Sequence[int]):
    """Return the adaptive average pool2d of a tensor given the desired output shape.

    Args:
        input (Tensor): The input tensor.
        output_size (Sequence[int]): The desired output shape.

    Returns:
        Tensor: Output tensor.
    """
    return generate_op([input, output_size], "adaptive_avg_pool")


@implements(torch.nn.functional.adaptive_max_pool2d)
def adaptive_max_pool2d(
    input: Tensor, output_size: Sequence[int], return_indices: bool = False
):
    """Return the adaptive_max_pool2d of a tensor given the desired output shape.

    Args:
        input (Tensor): The input tensor.
        output_size (Sequence[int]): The desired output shape.
        return_indices (bool): Not supported yet. Defaults to False.

    Raises:
        NotImplementedError: return_indices is not supported yet

    Returns:
        Tensor: Output tensor.
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported yet")
    return generate_op([input, output_size], "adaptive_max_pool")
