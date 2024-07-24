#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers.models.phi.modeling_phi import PhiConfig, PhiMLP
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3MLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from intel_npu_acceleration_library.dtypes import int8, int4
from intel_npu_acceleration_library.compiler import CompilerConfig
from sklearn.metrics import r2_score
from torch.profiler import profile, ProfilerActivity
import intel_npu_acceleration_library
import pytest
import torch
import numpy as np


@pytest.fixture
def config():
    return LlamaConfig(num_hidden_layers=1)


@pytest.fixture
def decoder_model(config):
    return LlamaForCausalLM(config)


@pytest.fixture
def model():
    return AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


@pytest.mark.parametrize("model_seq_length", [128, 256])
def test_warm_up(tokenizer, model, model_seq_length):
    compiler_conf = CompilerConfig()
    compiled_model = intel_npu_acceleration_library.compile(model, compiler_conf)
    intel_npu_acceleration_library.nn.llm.warm_up_decoder_model(
        tokenizer, compiled_model, model_seq_length
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_compilation(tokenizer, decoder_model, dtype):
    prefill = tokenizer("test sentence", return_tensors="pt")["input_ids"].to("cpu")
    y_ref = decoder_model(prefill).logits.detach()

    compiler_conf = CompilerConfig(dtype=dtype)
    compiled_model = intel_npu_acceleration_library.compile(
        decoder_model, compiler_conf
    )

    assert compiled_model

    y = compiled_model(prefill).logits.detach()

    assert 1 - r2_score(y_ref.flatten().numpy(), y.flatten().numpy()) < 0.01


@torch.no_grad
@pytest.mark.parametrize("seq_len", [16, 128, 256])
@pytest.mark.parametrize("hidden_size", [256, 512])
@pytest.mark.parametrize("intermediate_size", [512])
def test_phi2_mlp(seq_len, hidden_size, intermediate_size):
    conf = PhiConfig.from_pretrained("microsoft/phi-2")
    conf.num_hidden_layers = 1
    conf.hidden_size = hidden_size
    conf.intermediate_size = intermediate_size

    mlp = PhiMLP(conf)

    x = torch.rand((seq_len, conf.hidden_size), dtype=torch.float16)
    reference = mlp(x.to(torch.float32)).to(torch.float16)

    model = intel_npu_acceleration_library.nn.PhiMLP.fromTorch(mlp)

    assert model

    out = model(x)

    assert 1 - r2_score(reference.numpy(), out.numpy()) < 0.001


@torch.no_grad
@pytest.mark.parametrize("seq_len", [16, 128, 256])
@pytest.mark.parametrize("hidden_size", [256, 512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("dtype", ["float16", "int8", "int4"])
def test_phi3_mlp_compile(seq_len, hidden_size, intermediate_size, dtype):
    conf = Phi3Config.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    conf.num_hidden_layers = 1
    conf.hidden_size = hidden_size
    conf.intermediate_size = intermediate_size

    if dtype == "int8":
        dtype = int8
    elif dtype == "int4":
        dtype = int4
    else:
        dtype = torch.float16

    mlp = Phi3MLP(conf)

    hidden_states = torch.rand((seq_len, conf.hidden_size))

    reference = mlp(hidden_states.to(torch.float32)).to(torch.float16).detach().numpy()

    compiler_conf = CompilerConfig(use_to=True, dtype=dtype)
    model = intel_npu_acceleration_library.compile(mlp, compiler_conf)

    assert model

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        out = model(hidden_states)

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=20
        )
    )

    out = out.detach().numpy()

    assert out.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(out).all(), "NPU output contains NaN or Inf"

    if dtype == int4:
        assert 1 - r2_score(reference, out) < 0.05
    else:
        assert 1 - r2_score(reference, out) < 0.001
