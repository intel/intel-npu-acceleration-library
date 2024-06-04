#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = NPUModelForCausalLM.from_pretrained(
    model_id, dtype=torch.int8, use_cache=True
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

print("Run inference with Llama3 on NPU\n")


query = input(">")


messages = [
    {
        "role": "system",
        "content": "You are an helpful chatbot that can provide information about the Intel NPU",
    },
    {"role": "user", "content": query},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    streamer=streamer,
)
