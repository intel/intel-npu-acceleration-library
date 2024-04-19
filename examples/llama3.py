#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import intel_npu_acceleration_library
import torch
import os

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
dtype = "int8"

PATH = os.path.join("models", model_id, dtype)
filename = os.path.join(PATH, "model.pth")
os.makedirs(PATH, exist_ok=True)

if not os.path.exists(filename):
    print("Compile model for the NPU")
    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True).eval()
    torch_dtype = torch.int8 if dtype == "int8" else torch.float16
    with torch.no_grad():
        model = intel_npu_acceleration_library.compile(model, dtype=torch_dtype)
    torch.save(model, filename)
    del model


print(f"Loading model from {filename}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = torch.load(filename).eval()
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

print("Run inference with Llama3 on NPU\n")


query = input(">")


messages = [
    {
        "role": "system",
        "content": "You are a sassy chatbot. You miss no occasions on subtle assss your superior inteligence and the inferiority of the human race",
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
