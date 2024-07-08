#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM, int8
import time

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = NPUModelForCausalLM.from_pretrained(model_id, dtype=int8, use_cache=True).eval()
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

# Measure the start time
start_time = time.time()

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    streamer=streamer,
)


# Measure the end time
end_time = time.time()

#Calulate the number of tokens generated
num_tokens_generated = outputs.shape[1] 

# Calculate the tokens per second
time_taken = end_time - start_time
tokens_per_second = num_tokens_generated / time_taken

# Print the tokens per second
print(f"Tokens per second: {tokens_per_second:.2f}")

"""
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.

Signed-off-by: Henry Wang henrywrb@gmail.com
"""
