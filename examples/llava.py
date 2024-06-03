#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import requests
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoTokenizer,
    CLIPImageProcessor,
    TextStreamer,
)
from transformers.feature_extraction_utils import BatchFeature
import intel_npu_acceleration_library
import torch


checkpoint = "Intel/llava-gemma-2b"

# Load model
model = LlavaForConditionalGeneration.from_pretrained(checkpoint)

model = intel_npu_acceleration_library.compile(model)

image_processor = CLIPImageProcessor.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

# Prepare inputs
# Use gemma chat template
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "<image>\nWhat's the content of the image?"}],
    tokenize=False,
    add_generation_prompt=True,
)
text_inputs = tokenizer(prompt, return_tensors="pt")

# clean the console
print("\033[H\033[J")
print("LLaVA Gemma Chatbot\n")
print("Please provide an image URL to generate a response.\n")
url = input("Image URL: ")

print("Description: ", end="", flush=True)
# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]

inputs = BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

# Generate
model.generate(**inputs, max_new_tokens=150, streamer=streamer)
