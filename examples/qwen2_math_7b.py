#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM, int8
from intel_npu_acceleration_library.compiler import CompilerConfig
import time

model_id = "Qwen/Qwen2-Math-7B-Instruct"

compiler_conf = CompilerConfig(dtype=int8)
model = NPUModelForCausalLM.from_pretrained(
    model_id, use_cache=True, config=compiler_conf
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

print("Run inference with Qwen2-Math-7B on NPU\n")

# sample query:  Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$.

query = input(">")

messages = [
    {
        "role": "system",
        "content": "You are an helpful chatbot",
    },
    {"role": "user", "content": query},
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Measure the start time
start_time = time.time()

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.01,
    streamer=streamer,
)

generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Calculate the total number of generated tokens
num_tokens_generated = sum(len(tokens) for tokens in generated_ids)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Measure the end time
end_time = time.time()

# Calculate the number of tokens generated
num_tokens_generated = sum(len(tokens) for tokens in generated_ids)

# Calculate the tokens per second
time_taken = end_time - start_time
print("Total generated tokens:", num_tokens_generated)
print("Total Time taken:", time_taken)

tokens_per_second = num_tokens_generated / time_taken

# Print the tokens per second
print(f"Tokens per second: {tokens_per_second:.2f}")
