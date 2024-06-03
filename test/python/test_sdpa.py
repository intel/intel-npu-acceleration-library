#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend.sdpa import SDPA
from sklearn.metrics import r2_score
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("sequence", [16, 32, 128, 256])
@pytest.mark.parametrize("dim", [512, 1024])
@pytest.mark.parametrize("kv_cache", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
def test_sdpa(heads, sequence, dim, kv_cache, is_causal):
    pytest.skip("not working yet")

    min_value = torch.finfo(torch.float16).min

    key = torch.rand(1, heads, sequence, dim // heads).to(torch.float16)
    query = torch.rand(1, heads, 1 if kv_cache else sequence, dim // heads).to(
        torch.float16
    )
    value = torch.rand(1, heads, sequence, dim // heads).to(torch.float16)
    mask = min_value * torch.ones(1, heads, 1 if kv_cache else sequence, sequence).to(
        torch.float16
    )
    mask = torch.triu(mask)

    npu_sdpa = SDPA(
        query.shape, key.shape, value.shape, mask.shape, is_causal=is_causal
    )

    ref_result = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, mask, dropout_p=0, is_causal=is_causal, scale=None
    )
    npu_result = npu_sdpa.run(query.numpy(), key.numpy(), value.numpy(), mask.numpy())

    assert npu_result.shape == (1, heads, 1 if kv_cache else sequence, dim // heads)

    assert np.isfinite(npu_result).all()

    r2 = r2_score(ref_result.numpy().flatten(), npu_result.flatten())

    assert 1 - r2 < 0.05
    print("DONE")


# test_sdpa(32, 1024, 1024, kv_cache=True, is_causal=True)
