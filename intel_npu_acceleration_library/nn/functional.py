#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


from intel_npu_acceleration_library.backend.tensor import (
    implements,
    generate_op,
    Tensor,
)
from typing import Optional, Sequence, Union
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
    """Apply a linear transformation to the incoming data: y = x * A^T + b.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.
        bias (Optional[Tensor], optional): the bias tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    mm = generate_op([input, weight], "matmul", False, True)
    if bias is not None:
        return generate_op([mm, bias], "eltwise_add")
    return mm


@implements(torch.addmm)
def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    beta: float = 1,
    alpha: float = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Return the addmm of a tensor element-wise.

    Args:
        input (Tensor): The input tensor.
        mat1 (Tensor): The first matrix tensor.
        mat2 (Tensor): The second matrix tensor.
        beta (float): The beta value. Defaults to 1.
        alpha (float): The alpha value. Defaults to 1.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    out = beta * input + alpha * (mat1 @ mat2)
    return out


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
    if weight is not None:
        ln = ln * weight

    if bias is not None:
        ln = ln + bias

    return ln


@implements(torch.nn.functional.normalize)
def normalize(
    input: Tensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Return the normalized tensor.

    Args:
        input (Tensor): The input tensor.
        p (float): The power value. Defaults to 2.0.
        dim (int): The dim to normalize. Defaults to 1.
        eps (float): The epsilon value. Defaults to 1e-12.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Raises:
        NotImplementedError: p != 2 is not supported yet

    Returns:
        Tensor: Output tensor.
    """
    if p != 2:
        raise NotImplementedError("p != 2 is not supported yet")

    out = generate_op([input], "normL2", dim, eps)
    return out


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


@implements(torch.add)
def add(
    x: Tensor,
    other: Union[Tensor, torch.Tensor, int, float],
    alpha: float = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Return the sum of two tensors element-wise.

    Args:
        x (Tensor): The input tensor.
        other (Union[Tensor, torch.Tensor, int, float]): The other tensor.
        alpha (float): The alpha value. Defaults to 1.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    if alpha != 1:
        other = torch.mul(other, alpha)

    out = generate_op([x, other], "eltwise_add")
    return out


@implements(torch.sub)
def sub(
    x: Tensor,
    other: Union[Tensor, torch.Tensor, int, float],
    alpha: float = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Return the subtraction of two tensors element-wise.

    Args:
        x (Tensor): The input tensor.
        other (Union[Tensor, torch.Tensor, int, float]): The other tensor.
        alpha (float): The alpha value. Defaults to 1.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    if alpha != 1:
        other = torch.mul(other, alpha)

    return torch.add(x, torch.neg(other), out=out)


@implements(torch.mul)
def mul(
    x: Tensor,
    other: Union[Tensor, torch.Tensor, int, float],
    out: Optional[Tensor] = None,
) -> Tensor:
    """Return the elementwise multiplication of two tensors.

    Args:
        x (Tensor): The input tensor.
        other (Union[Tensor, torch.Tensor, int, float]): The other tensor.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    out = generate_op([x, other], "eltwise_mul")
    return out


@implements(torch.div)
def div(
    x: Tensor,
    other: Union[Tensor, torch.Tensor, int, float],
    rounding_mode: Optional[str] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Return the elementwise division of two tensors.

    Args:
        x (Tensor): The input tensor.
        other (Union[Tensor, torch.Tensor, int, float]): The other tensor.
        rounding_mode (Optional[str]): The rounding mode. Defaults to None. Options are 'trunc', 'floor'.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Raises:
        NotImplementedError: trunc is not supported yet

    Returns:
        Tensor: Output tensor.
    """
    out = generate_op([x, other], "eltwise_div")

    if rounding_mode == "trunc":
        raise NotImplementedError("trunc is not supported yet")
    elif rounding_mode == "floor":
        return torch.floor(out)
    else:
        return out


@implements(torch.unsqueeze)
def unsqueeze(x, dim: int) -> Tensor:
    """Return the unsqueezed tensor.

    Args:
        x (Tensor): The input tensor.
        dim (int): The dim to unsqueeze

    Returns:
        Tensor: The squeezed tensor.
    """
    return generate_op([x], "unsqueeze", dim)


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


@implements(torch.cat)
def cat(input: Sequence[Tensor], dim: int, out: Optional[Tensor] = None) -> Tensor:
    """Return the concatenation of tensors given the desired output shape.

    Args:
        input (Sequence[Tensor]): The input tensors.
        dim (int): The dimension to concatenation tensors along.
        out (Optional[Tensor], optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    if len(input) == 2:
        tensor = generate_op([input[0], input[1]], "concat", axis=dim)
    else:
        tensor = torch.cat([input[0], input[1]], dim=dim)
        for x in range(2, len(input)):
            tensor = torch.cat([tensor, input[x]], dim=dim)
    return tensor


@implements(torch.max)
def max(x, dim: Optional[int] = None, keep_dims: Optional[bool] = False) -> Tensor:
    """Return the reduced max tensor.

    Args:
        x (Tensor): The input tensor.
        dim (Optional[int], optional): The dim to reduce. Default is None, and all dimensions are reduced.
        keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.

    Returns:
        Tensor: The the reduced max tensor.
    """
    return generate_op(x, "reduce_max", reduction_axes=dim, keep_dims=keep_dims)


@implements(torch.mean)
def mean(
    x,
    dim: Optional[Union[int, Sequence[int]]] = None,
    keep_dims: Optional[bool] = False,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Return the reduced mean tensor.

    Args:
        x (Tensor): The input tensor.
        dim (Optional[Union[int, Sequence[int]]], optional): The dim(s) to reduce. Default is None, and all dimensions are reduced.
        keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.
        dtype (Optional[torch.dtype], optional): The data type. Defaults to None.

    Returns:
        Tensor: The the reduced mean tensor.
    """
    mean = generate_op(x, "reduce_mean", reduction_axes=dim, keep_dims=keep_dims)
    if dtype:
        mean = mean.to(dtype)
    return mean


@implements(torch.min)
def min(x, dim: Optional[int] = None, keep_dims: Optional[bool] = False) -> Tensor:
    """Return the reduced min tensor.

    Args:
        x (Tensor): The input tensor.
        dim (Optional[int], optional): The dim to reduce. Default is None, and all dimensions are reduced.
        keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.

    Returns:
        Tensor: The the reduced min tensor.
    """
    return generate_op(x, "reduce_min", reduction_axes=dim, keep_dims=keep_dims)


@implements(torch.prod)
def prod(
    x,
    dim: Optional[int] = None,
    keep_dims: Optional[bool] = False,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Return the reduced product tensor.

    Args:
        x (Tensor): The input tensor.
        dim (Optional[int], optional): The dim to reduce. Default is None, and all dimensions are reduced.
        keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.
        dtype (Optional[torch.dtype], optional): The data type. Defaults to None.

    Returns:
        Tensor: The the reduced product tensor.
    """
    prod = generate_op(x, "reduce_prod", reduction_axes=dim, keep_dims=keep_dims)
    if dtype:
        prod = prod.to(dtype)
    return prod


@implements(torch.sum)
def sum(
    x,
    dim: Optional[Union[int, Sequence[int]]] = None,
    keep_dims: Optional[bool] = False,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Return the reduced sum tensor.

    Args:
        x (Tensor): The input tensor.
        dim (Optional[Union[int, Sequence[int]]], optional): The dim(s) to reduce. Default is None, and all dimensions are reduced.
        keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.
        dtype (Optional[torch.dtype], optional): The data type. Defaults to None.

    Returns:
        Tensor: The the reduced sum tensor.
    """
    sum = generate_op(x, "reduce_sum", reduction_axes=dim, keep_dims=keep_dims)
    if dtype:
        sum = sum.to(dtype)
    return sum


# Functional activations


@implements(torch.nn.functional.elu)
def elu(x: Tensor, alpha: float = 1.0, inplace=False) -> Tensor:
    """Return the elu of a tensor element-wise.

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


@implements(torch.nn.functional.prelu)
def prelu(x: Tensor, weight: Tensor) -> Tensor:
    """Return the parametric relu of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        weight (Tensor): The weights tensor.

    Returns:
        Tensor: Output tensor.
    """
    return generate_op([x, weight], "prelu")


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


@implements(torch.nn.functional.softmax)
def softmax(
    x: Tensor,
    dim: Optional[int] = None,
    _stacklevel=3,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Return the softmax of a tensor element-wise.

    Args:
        x (Tensor): The input tensor.
        dim (int): The dim to apply softmax. Defaults to -1.
        _stacklevel (int): The stack level. Defaults to 3.
        dtype (torch.dtype): The data type. Defaults to None.


    Returns:
        Tensor: Output tensor.
    """
    if dim is None:
        dim = -1
    smax = generate_op(x, "softmax", dim)

    if dtype:
        smax = smax.to(dtype)
    return smax


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
    if output_size == 1:
        return generate_op(
            [input], "reduce_mean", reduction_axes=[-2, -1], keep_dims=True
        )
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
    if output_size == 1:
        return generate_op(
            [input], "reduce_max", reduction_axes=[-2, -1], keep_dims=True
        )
    return generate_op([input, output_size], "adaptive_max_pool")


@implements(torch.nn.functional.avg_pool2d)
def avg_pool2d(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Optional[Union[int, Sequence[int]]] = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
):
    """Generate an average pooling layer.

    Args:
        input (Tensor): layer input node
        kernel_size (Sequence[int]): kernel size
        stride (Sequence[int]): strides
        padding (int): padding
        ceil_mode (bool): ceil mode
        count_include_pad (bool): count include pad
        divisor_override (int): divisor override

    Returns:
        Tensor: output node
    """
    return generate_op(
        [input],
        "avg_pooling",
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        2,
    )


@implements(torch.nn.functional.max_pool2d)
def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Generate an average pooling layer.

    Args:
        input (ctypes._Pointer): layer input node
        kernel_size (Sequence[int]): kernel size
        stride (Sequence[int]): strides
        padding (int): padding
        dilation (int): dilation
        ceil_mode (bool): ceil mode
        return_indices (bool): return indices

    Raises:
        NotImplementedError: return_indices and dilation are not supported

    Returns:
        ctypes._Pointer: output node
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported yet")

    if dilation != 1:
        raise NotImplementedError("dilation is not supported yet")

    return generate_op(
        [input],
        "max_pooling",
        kernel_size,
        stride,
        padding,
        ceil_mode,
        2,
    )


@implements(torch.nn.functional.batch_norm)
def batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    """Generate a batch normalization layer.

    Args:
        input (Tensor): layer input node
        running_mean (Tensor): running mean
        running_var (Tensor): running variance
        weight (Tensor): weight
        bias (Tensor): bias
        training (bool): training
        momentum (float): momentum
        eps (float): epsilon

    Raises:
        NotImplementedError: Training mode is not supported yet

    Returns:
        Tensor: output node
    """
    if training:
        raise NotImplementedError("Training mode is not supported yet")

    dtype = input.dtype.torch_dtype
    running_mean = running_mean.view(1, -1, 1, 1).to(dtype)
    running_var = running_var.view(1, -1, 1, 1).to(dtype)

    result = (input - running_mean) / torch.sqrt(
        running_var + torch.tensor([eps]).to(dtype)
    ).to(dtype)

    if weight is not None:
        result = result * weight.view(1, -1, 1, 1)

    if bias is not None:
        result = result + bias.view(1, -1, 1, 1)

    return result


@implements(torch.nn.functional.conv2d)
def conv2d(
    input: Tensor,
    weight: Union[Tensor, torch.Tensor],
    bias: Optional[Union[Tensor, torch.Tensor]] = None,
    stride: int = 1,
    padding: Union[int, str] = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """Generate a convolution layer.

    Args:
        input (Tensor): layer input node
        weight (Union[Tensor, torch.Tensor]): weight
        bias (Union[Tensor, torch.Tensor]): bias
        stride (int): stride
        padding (Union[int, str]): padding
        dilation (int): dilation
        groups (int): groups

    Raises:
        ValueError: Padding mode not supported

    Returns:
        Tensor: output node
    """
    if isinstance(padding, str):
        if padding == "valid":
            padding = 0
        elif padding == "same":
            padding = weight.shape[2] // 2
        else:
            raise ValueError(f"Padding mode {padding} not supported")

    if bias is not None:
        bias = bias.view((1, weight.shape[0], 1, 1))

    if groups > 1:
        new_shape = [groups, weight.shape[0] // groups] + list(weight.shape[1:])
        weight = weight.view(new_shape)

    conv = generate_op(
        [input, weight, bias],
        "convolution",
        strides=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    return conv


@implements(torch.pow)
def pow(input: Tensor, exponent: Union[Tensor, torch.Tensor, float]):
    """Return the tensor raised to the power of the exponent.

    Args:
        input (Tensor): The input tensor.
        exponent (Union[Tensor, torch.Tensor, float]): The exponent value.

    Returns:
        Tensor: Output tensor.
    """
    if isinstance(exponent, float):
        exponent = torch.full(input.shape, exponent).to(torch.float16)
    return generate_op([input], "power", exponent=exponent)


@implements(torch.nn.functional.log_softmax)
def log_softmax(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel=3,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Return the log softmax of a tensor element-wise.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension along which log_softmax will be computed. Defaults to -1.
        _stacklevel (int): The stack level. Defaults to 3.
        dtype (torch.dtype): The data type. Defaults to None.

    Returns:
        Tensor: Output tensor.
    """
    if dim is None:
        dim = -1
    log_smax = generate_op([input], "log_softmax", dim)

    if dtype:
        log_smax = log_smax.to(dtype)
    return log_smax
