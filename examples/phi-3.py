#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import intel_npu_acceleration_library
import warnings

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype="auto",
    trust_remote_code=True,
)
model = intel_npu_acceleration_library.compile(model, torch.float16)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
streamer = TextStreamer(tokenizer)

messages = [
    {
        "role": "system",
        "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
    },
    {
        "role": "user",
        "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
    },
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
    "streamer": streamer,
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipe(messages, **generation_args)
