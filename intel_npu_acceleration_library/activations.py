#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import torch


def get_activation(act_function: str):
    """Return an activation function for the NPU.

    Args:
        act_function (str): an NPU supported activation function

    Returns:
        torch.nn: activation function
    """
    match act_function:
        case "cos":
            return torch.cos
        case "sin":
            return torch.sin
        case "tan":
            return torch.tan
        case "acos":
            return torch.acos
        case "asin":
            return torch.asin
        case "atan":
            return torch.atan
        case "cosh":
            return torch.cosh
        case "sinh":
            return torch.sinh
        case "tanh":
            return torch.tanh
        case "acosh":
            return torch.acosh
        case "asinh":
            return torch.asinh
        case "atanh":
            return torch.atanh
        case "abs":
            return torch.abs
        case "ceil":
            return torch.ceil
        case "clamp":
            return torch.clamp
        case "elu":
            return torch.nn.functional.elu
        case "erf":
            return torch.erf
        case "exp":
            return torch.exp
        case "floor":
            return torch.floor
        case "gelu":
            return torch.nn.functional.gelu
        case "hardsigmoid":
            return torch.nn.functional.hardsigmoid
        case "hardswish":
            return torch.nn.functional.hardswish
        case "log":
            return torch.log
        case "mish":
            return torch.nn.functional.mish
        case "neg":
            return torch.neg
        case "relu":
            return torch.nn.functional.relu
        case "round":
            return torch.round
        case "sigmoid":
            return torch.nn.functional.sigmoid
        case "sign":
            return torch.sign
        case "silu":
            return torch.nn.functional.silu
        case "softmax":
            return torch.nn.functional.softmax
        case "softplus":
            return torch.nn.functional.softplus
        case "sqrt":
            return torch.sqrt
        case _:
            return torch.nn.functional.silu
