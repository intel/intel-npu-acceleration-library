#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
import intel_npu_acceleration_library
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)


print("Compile model for the NPU")
model = intel_npu_acceleration_library.compile(model, dtype=torch.int8)

query = input("Ask something: ")
prefix = tokenizer(query, return_tensors="pt")["input_ids"]


generation_kwargs = dict(
    input_ids=prefix,
    streamer=streamer,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    max_new_tokens=512,
)

print("Run inference")
_ = model.generate(**generation_kwargs)
