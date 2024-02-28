#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from .linear import Linear, QuantizedLinear  # noqa

try:
    from .llm import LlamaAttention, PhiMLP  # noqa

    llm_modules = ["LlamaAttention", "PhiMLP"]
except ModuleNotFoundError:
    # Transformer library is not installed
    llm_modules = []


__all__ = ["Linear", "QuantizedLinear"] + llm_modules
