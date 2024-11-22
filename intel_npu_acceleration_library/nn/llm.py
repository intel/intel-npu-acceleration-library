#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaConfig,
)
from transformers import AutoTokenizer
from intel_npu_acceleration_library.nn import Linear
from intel_npu_acceleration_library.backend import run_factory, MLP
from functools import partial
from typing import Optional, List, Generator, Tuple
from transformers.cache_utils import Cache
import torch
import uuid


class PhiMLP(torch.nn.Module):
    """Phi-2 MLP operation NPU backend."""

    def __init__(
        self,
        parameters: List[torch.Tensor],
    ):
        """Initialize LLAMA MLP operation.

        Args:
            parameters (List[torch.Tensor]): model weights
        """
        super().__init__()
        self.op_parameters = parameters
        self.op_id = str(uuid.uuid4())
        intermediate_size, _ = parameters[0].shape
        self.backend_cls = partial(
            MLP,
            intermediate_size=intermediate_size,
            activation="gelu",
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        return run_factory(x, self.op_parameters, self.backend_cls, self.op_id)

    @staticmethod
    def fromTorch(
        layer: torch.nn.Module, dtype: torch.dtype = torch.float16
    ) -> "PhiMLP":
        """Generate a NPU PhiMLP layer from a transformer one.

        Args:
            layer (torch.nn.Linear): the original PhiMLP model to run on the NPU
            dtype (torch.dtype): the desired datatype

        Returns:
            PhiMLP: A NPU PhiMLP layer
        """
        new_layer = PhiMLP(
            parameters=[weight.to(dtype) for weight in layer.parameters()],
        )

        return new_layer


class FusedLlamaMLP(torch.nn.Module):
    """LLAMA MLP operation NPU backend."""

    def __init__(
        self,
        parameters: List[torch.Tensor],
    ):
        """Initialize LLAMA MLP operation.

        Args:
            parameters (List[torch.Tensor]): model weights
        """
        super().__init__()
        self.op_parameters = parameters
        self.op_id = str(uuid.uuid4())
        intermediate_size, _ = parameters[0].shape
        self.backend_cls = partial(MLP, intermediate_size=intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        return run_factory(x, self.op_parameters, self.backend_cls, self.op_id)

    @staticmethod
    def fromTorch(
        layer: torch.nn.Module, dtype: torch.dtype = torch.float16
    ) -> "FusedLlamaMLP":
        """Generate a NPU LlamaMLP layer from a transformer LlamaMLP one.

        Args:
            layer (torch.nn.Linear): the original LlamaMLP model to run on the NPU
            dtype (torch.dtype): the desired datatype

        Returns:
            FusedLlamaMLP: A NPU LlamaMLP layer
        """
        new_layer = FusedLlamaMLP(
            parameters=[weight.to(dtype) for weight in layer.parameters()],
        )

        return new_layer


class LlamaAttention(torch.nn.Module):
    """LlamaAttention operation NPU backend."""

    def __init__(
        self,
        config: LlamaConfig,
        q_weights: torch.Tensor,
        kv_weights: torch.Tensor,
        o_proj: torch.Tensor,
        rotary_emb: torch.nn.Module,
        dtype: torch.dtype = torch.float16,
        layer_idx: Optional[int] = None,
    ):
        """Initialize the LlamaAttention class.

        Args:
            config (LlamaConfig): LlamaAttention configuration
            q_weights (torch.Tensor): Weights for the query Linear layer
            kv_weights (torch.Tensor): Concatentation of the weights for the Key and Value Linear layer
            o_proj (torch.Tensor): Weights for the output projection Linear layer
            rotary_emb (torch.nn.Module): Rotary embedding module
            dtype (torch.dtype): the desired datatype
            layer_idx (Optional[int], optional): Layer index. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.rotary_emb = rotary_emb
        self.kv_proj = Linear.fromTensor(kv_weights, None, dtype)
        self.q_proj = Linear.fromTensor(q_weights, None, dtype)
        self.o_proj = Linear.fromTensor(o_proj, None, dtype)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in Transformers v4.45
    ):
        """Torch module forward method.

        Args:
            hidden_states (torch.Tensor): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (Optional[torch.Tensor], optional): attention mask of shape `(batch_size, sequence_length)`. Defaults to None.
            position_ids (Optional[torch.Tensor], optional): position_ids of shape `(batch_size, sequence_length)`. Defaults to None.
            past_key_value (Optional[Cache], optional): Pre-computed hidden-states (key and values in the self-attention blocks). Defaults to None.
            output_attentions (Optional[bool], optional):  Whether or not to return the attentions tensors of all attention layers.. Defaults to False.
            use_cache (Optional[bool], optional): If set to `True`, `past_key_values` key value states are returned. Defaults to False.
            cache_position (Optional[torch.LongTensor], optional): Cache position useful for static cache applications . Defaults to None.
            position_embeddings (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): If set to a tuple, it means the `sin` and `cos` are uniformly calculated by the outer `LlamaModel` and passed in. Defaults to None.

        Returns:
            _type_: result
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        kv_states = self.kv_proj(hidden_states)

        key_states = kv_states[..., : self.num_key_value_heads * self.head_dim]
        value_states = kv_states[..., self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    @staticmethod
    def fromTorch(
        layer: torch.nn.Module, dtype: torch.dtype = torch.float16
    ) -> "LlamaAttention":
        """Generate a NPU LlamaAttention layer from a transformer LlamaAttention one.

        Args:
            layer (torch.nn.Linear): the original LlamaAttention model to run on the NPU
            dtype (torch.dtype): the desired datatype

        Returns:
            LlamaAttention: A NPU LlamaAttention layer
        """
        kv_weights = torch.cat((layer.k_proj.weight, layer.v_proj.weight), dim=0)

        new_layer = LlamaAttention(
            config=layer.config,
            q_weights=layer.q_proj.weight,
            kv_weights=kv_weights,
            o_proj=layer.o_proj.weight,
            rotary_emb=layer.rotary_emb,
            dtype=dtype,
            layer_idx=layer.layer_idx,
        )

        return new_layer


def lshift_insert(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """Compute shift left and insert a value into a tensor.

    Args:
        tensor (torch.Tensor): input tensor
        value (float): value to add

    Returns:
        torch.Tensor: output tensor
    """
    tensor = torch.roll(tensor, shifts=-1, dims=-1)
    tensor[0, -1] = value
    return tensor


# Generate function
@torch.no_grad()
def generate_with_static_shape(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    use_past: Optional[bool] = True,
    pad_token_id: Optional[int] = None,
    **kwargs,
) -> Generator[int, None, None]:
    """Run LLM generator routine wiht static shapes.

    Args:
        model (torch.nn.Module): LLM mode
        input_ids (torch.Tensor): model input_ids
        max_length (int): model max lenght.
        attention_mask (Optional[torch.Tensor], optional): input attention mask. Defaults to None.
        use_past (Optional[bool], optional): Enable/disable KV caching. Defaults to True.
        pad_token_id (Optional[int], optional): Padding token. Defaults to None.
        kwargs: Additional arguments

    Raises:
        RuntimeError: pad_token_id is not set and needed for static shape generation

    Yields:
        Generator[int, None, None]: Return a generator of new tokens
    """
    # Get sequence lenght
    batch, seq_lenght = input_ids.shape

    if pad_token_id is None:
        raise RuntimeError(
            "pad_token_id is not set and needed for static shape generation"
        )

    # Padding attention mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.int32).to(model.device)
    attention_mask_padding = torch.zeros(
        (batch, max_length - seq_lenght), dtype=input_ids.dtype, device=input_ids.device
    )
    attention_mask = torch.cat((attention_mask_padding, attention_mask), dim=-1)

    # Padding input_ids with left padding
    padding_input_ids = pad_token_id * torch.ones(
        (batch, max_length - seq_lenght), dtype=input_ids.dtype, device=input_ids.device
    )
    input_ids = torch.cat((padding_input_ids, input_ids), dim=-1).to(model.device)

    # Set the proper position ids
    position_ids = kwargs.get("position_ids", None)
    if position_ids is None:
        position_ids = torch.tensor(
            [[0] * (max_length - seq_lenght) + list(range(seq_lenght))],
            dtype=torch.int32,
        ).to(model.device)
    else:
        raise RuntimeError("Cannot set position_ids with in static shape generation")

    # past_key_values for KV-cache
    past_key_values = None

    for idx in range(seq_lenght, max_length):

        # Run the inference
        # position_ids=position_ids,
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        # Here I do greedy search as an example, but in general is where you want to select the next token with your fancy decoding algorithm
        logits = out.logits
        new_token = torch.argmax(logits[0, -1, :]).item()

        yield int(new_token)

        if not use_past:
            # Shift left input and position ids and set the new token and idx to the proper values
            input_ids = lshift_insert(input_ids, new_token)
            position_ids = lshift_insert(position_ids, idx)
        else:
            # Set input_ids and position_ids to their new value
            input_ids = torch.tensor([[new_token]], dtype=input_ids.dtype).to(
                model.device
            )
            position_ids = torch.tensor([[idx]], dtype=input_ids.dtype).to(model.device)

            # Select the proper KV cached keys for next inference
            past_key_values = [
                [item[:, :, 1:, :] for item in layer_past]
                for layer_past in out.past_key_values
            ]

        # Shift left attention mask and set the last value to one
        attention_mask = lshift_insert(attention_mask, 1)


def warm_up_decoder_model(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    model_seq_length: int,
    use_past: Optional[bool] = True,
):
    """Warm up the model on the NPU.

    This function JIT compile all the layers offloaded to the NPU and load and warm them into the NPU. This is particolarly useful for LLM decoders

    Args:
        tokenizer (AutoTokenizer): a tokenizer
        model (torch.nn.Module): a torch Module representing a language model decoder
        model_seq_length (int): Max sequence lenght for the tokenizer padding
        use_past (Optional[bool], optional): Enable or Disable KV-caching. Defaults to True.
    """
    input_ids = tokenizer(tokenizer.eos_token, return_tensors="pt")["input_ids"].to(
        "cpu"
    )

    results = generate_with_static_shape(
        model,
        input_ids=input_ids,
        max_length=model_seq_length,
        use_past=use_past,
        pad_token_id=tokenizer.pad_token_id,
    )
    idx = 0
    # Only two inferences
    for _ in results:
        if idx < 1:
            idx += 1
        else:
            break
