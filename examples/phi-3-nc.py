#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit
import intel_npu_acceleration_library
import warnings

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype="auto",
    trust_remote_code=True,
)


woq_conf = PostTrainingQuantConfig(
    approach="weight_only",
    op_type_dict={
        ".*": {  # match all ops
            "weight": {
                "dtype": "int4",
                "bits": 4,
                "group_size": -1,
                "scheme": "sym",
                # "algorithm": "AUTOROUND",
                "algorithm": "minmax",
            },
            "activation": {
                "dtype": "fp16",
            },
        }
    },
)
quantized_model = fit(model=model, conf=woq_conf)  # , calib_dataloader=dataloader)

compressed_model = quantized_model.export_compressed_model(
    compression_dtype=torch.int8, scale_dtype=torch.float16, use_optimum_format=False
)

model = intel_npu_acceleration_library.compile(compressed_model)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
streamer = TextStreamer(tokenizer, skip_prompt=True)

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
