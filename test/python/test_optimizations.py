#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers.models.phi.modeling_phi import PhiConfig, PhiMLP
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP, LlamaModel
from transformers.models.gemma.modeling_gemma import GemmaConfig, GemmaMLP, GemmaModel
from intel_npu_acceleration_library.optimizations import horizontal_fusion_linear
from intel_npu_acceleration_library.compiler import CompilerConfig
from sklearn.metrics import r2_score
import torch.nn as nn
import intel_npu_acceleration_library
import torch
import pytest


class SimpleModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        return torch.cat((gate, up), dim=-1)


class SimpleModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)
        self.down_proj = nn.Linear(input_dim, hidden_dim // 2, bias=bias)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        down = self.down_proj(x)

        return torch.cat((gate, up, down), dim=-1)


class SimpleModelList(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False, n_nodes=2):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim, bias=bias) for i in range(n_nodes)]
        )

    def forward(self, x):
        layers = [layer(x) for layer in self.linears]

        return torch.cat(layers, dim=-1)


def get_model(model_name, hidden_size, intermediate_size, bias):
    if model_name == "SimpleModel2":
        return SimpleModel2(hidden_size, intermediate_size, bias)
    elif model_name == "SimpleModel3":
        return SimpleModel3(hidden_size, intermediate_size, bias)
    elif model_name == "LlamaMLP":
        conf = LlamaConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        conf.num_hidden_layers = 1
        conf.hidden_size = hidden_size
        conf.intermediate_size = intermediate_size

        return LlamaMLP(conf)
    elif model_name == "GemmaMLP":
        conf = GemmaConfig()
        conf.num_hidden_layers = 1
        conf.hidden_size = hidden_size
        conf.head_dim = conf.hidden_size // conf.num_attention_heads
        conf.intermediate_size = intermediate_size

        return GemmaMLP(conf)
    elif model_name == "LlamaModel":
        conf = LlamaConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        conf.num_hidden_layers = 1
        conf.hidden_size = hidden_size
        conf.intermediate_size = intermediate_size
        conf.head_dim = conf.hidden_size // conf.num_attention_heads

        return LlamaModel(conf)
    elif model_name == "GemmaModel":
        conf = GemmaConfig()
        conf.num_hidden_layers = 1
        conf.hidden_size = hidden_size
        conf.head_dim = conf.hidden_size // conf.num_attention_heads
        conf.intermediate_size = intermediate_size

        return GemmaModel(conf)
    elif model_name == "PhiMLP":
        conf = PhiConfig.from_pretrained("microsoft/phi-2")
        conf.num_hidden_layers = 1
        conf.hidden_size = hidden_size
        conf.intermediate_size = intermediate_size

        return PhiMLP(conf)
    else:
        raise RuntimeError(f"Invalid model name: {model_name}")


@pytest.mark.parametrize(
    "model_name", ["SimpleModel2", "SimpleModel3", "LlamaMLP", "PhiMLP", "GemmaMLP"]
)
@pytest.mark.parametrize("hidden_size", [256, 512])
@pytest.mark.parametrize("intermediate_size", [512, 1024])
@pytest.mark.parametrize("batch", [1, 128])
@pytest.mark.parametrize("bias", [True, False])
def test_fusion(model_name, hidden_size, intermediate_size, batch, bias):

    model = get_model(model_name, hidden_size, intermediate_size, bias)
    example_input = torch.rand((batch, hidden_size)) - 0.5

    reference = model(example_input)

    optimized = horizontal_fusion_linear(model)

    output = optimized(example_input)

    assert torch.allclose(reference, output, rtol=1e-03, atol=1e-4)


@pytest.mark.parametrize("model_name", ["LlamaModel", "GemmaModel"])
@pytest.mark.parametrize("hidden_size", [256, 512])
@pytest.mark.parametrize("intermediate_size", [512, 1024])
@pytest.mark.parametrize("sequence_length", [1, 128])
@pytest.mark.parametrize("bias", [True, False])
def test_model(model_name, hidden_size, intermediate_size, sequence_length, bias):

    with torch.no_grad():
        model = get_model(model_name, hidden_size, intermediate_size, bias).eval()
        example_input = torch.randint(
            0,
            1024,
            (
                1,
                sequence_length,
            ),
        )

        reference = model(example_input)[0]

        compiler_conf = CompilerConfig(dtype=torch.float16)
        optimized = intel_npu_acceleration_library.compile(model, compiler_conf)

        output = optimized(example_input)[0]

    assert 1 - r2_score(reference.flatten().numpy(), output.flatten().numpy()) < 0.01


@pytest.mark.parametrize("layers", [2, 3, 10])
def test_fusion_module_list(layers):
    model = SimpleModelList(512, 128, True, layers)
    example_input = torch.rand((1, 512)) - 0.5

    reference = model(example_input)

    optimized = horizontal_fusion_linear(model)

    output = optimized(example_input)

    assert torch.allclose(reference, output, rtol=1e-03, atol=1e-4)

    assert len(list(optimized.parameters())) == 2

    assert sum([torch.numel(par) for par in optimized.parameters()]) == sum(
        [torch.numel(par) for par in model.parameters()]
    )
