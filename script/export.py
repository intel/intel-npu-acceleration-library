#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from intel_npu_acceleration_library.compiler import compile
import argparse
import torch
import os


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Export a for NPU")
    parser.add_argument("--model", "-m", type=str, help="Name of the model to export")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "int8", "int4"],
        help="type of quantization to perform",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="models", help="Output path"
    )

    return parser.parse_args()


def export(model_id, dtype, output):

    print(f"Loading {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
    config = AutoConfig.from_pretrained(model_id)

    PATH = os.path.join(output, model_id, dtype)

    tokenizer.save_pretrained(PATH)
    config.save_pretrained(PATH)

    if dtype == "fp16":
        print(f"Compiling model {model_id}")
        torch_dtype = torch.float16
    elif dtype == "int8":
        print(f"Quantizing & Compiling model {model_id}")
        torch_dtype = torch.int8
    else:
        raise RuntimeError(f"Invalid dtype {dtype}")

    with torch.no_grad():
        compile(model, dtype=torch_dtype)

    filename = os.path.join(PATH, "model.pth")
    os.makedirs(PATH, exist_ok=True)

    print("Saving model...")
    torch.save(model, filename)

    print(f"Model saved in {filename}")


if __name__ == "__main__":
    args = define_and_parse_args()
    export(args.model, args.dtype, args.output)
