#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend.sdpa import SDPA
from intel_npu_acceleration_library.functional import scaled_dot_product_attention
from sklearn.metrics import r2_score
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("sequence", [16, 32, 128, 256])
@pytest.mark.parametrize("dim", [512, 1024])
@pytest.mark.parametrize("kv_cache", [True, False])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sdpa(heads, sequence, dim, kv_cache, is_causal):

    min_value = torch.finfo(torch.float16).min

    query = torch.rand(1, heads, 1 if kv_cache else sequence, dim // heads).to(
        torch.float16
    )
    key = torch.rand(1, heads, sequence, dim // heads).to(torch.float16)
    value = torch.rand(1, heads, sequence, dim // heads).to(torch.float16)
    mask = min_value * torch.ones(1, heads, 1 if kv_cache else sequence, sequence).to(
        torch.float16
    )
    mask = torch.triu(mask)

    npu_sdpa = SDPA(
        query.shape, key.shape, value.shape, mask.shape, is_causal=is_causal
    )

    npu_result = npu_sdpa.run(query.numpy(), key.numpy(), value.numpy(), mask.numpy())

    ref_result = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        None if is_causal else mask,
        dropout_p=0,
        is_causal=is_causal,
        scale=None,
    )

    assert npu_result.shape == (1, heads, 1 if kv_cache else sequence, dim // heads)

    assert np.isfinite(npu_result).all()

    r2 = r2_score(ref_result.numpy().flatten(), npu_result.flatten())

    assert 1 - r2 < 0.05


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("sequence", [16, 32, 128, 256])
@pytest.mark.parametrize("dim", [512, 1024])
@pytest.mark.parametrize("kv_cache", [True, False])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sdpa_runtime(heads, sequence, dim, kv_cache, is_causal):

    min_value = torch.finfo(torch.float16).min

    query = torch.rand(1, heads, 1 if kv_cache else sequence, dim // heads).to(
        torch.float16
    )
    key = torch.rand(1, heads, sequence, dim // heads).to(torch.float16)
    value = torch.rand(1, heads, sequence, dim // heads).to(torch.float16)
    mask = min_value * torch.ones(1, heads, 1 if kv_cache else sequence, sequence).to(
        torch.float16
    )
    mask = torch.triu(mask)

    npu_result = scaled_dot_product_attention(
        query, key, value, mask, is_causal=is_causal
    )

    ref_result = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        None if is_causal else mask,
        dropout_p=0,
        is_causal=is_causal,
        scale=None,
    )

    assert npu_result.shape == (1, heads, 1 if kv_cache else sequence, dim // heads)

    assert np.isfinite(npu_result).all()

    r2 = r2_score(ref_result.numpy().flatten(), npu_result.numpy().flatten())

    assert 1 - r2 < 0.05
