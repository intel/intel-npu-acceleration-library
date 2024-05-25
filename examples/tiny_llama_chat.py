#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import pipeline, TextStreamer, set_seed
import intel_npu_acceleration_library
import torch
import os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading the model...")
pipe = pipeline(
    "text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
print("Compiling the model for NPU...")
pipe.model = intel_npu_acceleration_library.compile(pipe.model, dtype=torch.int8)

streamer = TextStreamer(pipe.tokenizer, skip_special_tokens=True, skip_prompt=True)

set_seed(42)


messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot. You can ask me anything.",
    },
]

print("NPU Chatbot is ready! Please ask a question. Type 'exit' to quit.")
while True:
    query = input("User: ")
    if query.lower() == "exit":
        break
    messages.append({"role": "user", "content": query})

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("Assistant: ", end="", flush=True)
    out = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        streamer=streamer,
    )

    reply = out[0]["generated_text"].split("<|assistant|>")[-1].strip()
    messages.append({"role": "assistant", "content": reply})
