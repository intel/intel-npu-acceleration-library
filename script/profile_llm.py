#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoTokenizer, AutoModelForCausalLM
from intel_npu_acceleration_library.nn.llm import generate_with_static_shape
from intel_npu_acceleration_library.dtypes import int8, int4

from torch.profiler import profile, ProfilerActivity
import intel_npu_acceleration_library
import argparse
import torch
import time
import os


def main(
    prompt="List all numbers in the Fibonacci sequence: 1, 1, 2, 3, ",
    context_size=512,
    max_new_tokens=50,
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="NPU",
    dtype="float16",
    disable_intel_npu_acceleration_library=False,
):

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Load model
    if os.path.isdir(model_id) and os.path.isfile(f"{model_id}//model.pth"):
        compiled = True
        model = torch.load(f"{model_id}//model.pth")
        model.eval()
    else:
        compiled = False
        model = (
            AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            .to("cpu")
            .eval()
        )

    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "int8":
        dtype = int8
    elif dtype == "int4":
        dtype = int4
    else:
        raise RuntimeError(f"Invalid dtype: {dtype}")

    if not disable_intel_npu_acceleration_library:
        if not compiled:
            model = intel_npu_acceleration_library.compile(model, dtype)
        intel_npu_acceleration_library.nn.llm.warm_up_decoder_model(
            tokenizer, model, context_size
        )

    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cpu")

    results = generate_with_static_shape(
        model,
        input_ids=input_ids,
        max_length=context_size,
        use_past=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    times = [time.perf_counter()]
    idx = 0
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for new_token_id in results:
            times.append(time.perf_counter())
            if idx >= max_new_tokens:
                break
            idx += 1
            token = tokenizer.decode([new_token_id], skip_special_tokens=True)

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=20
        )
    )
    prof.export_chrome_trace("trace.json")

    elapsed = [y - x for x, y in zip(times, times[1:])]

    prefix_time = elapsed[0]
    tps = len(elapsed[1:]) / sum(elapsed[1:])

    print(
        f"model {model_id} (context: {context_size}): prefill-phase {prefix_time:.3f} s, tokens/s {tps:.3f}"
    )


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Profiling a LLM in the NPU")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=128,
        help="Context size (default: %(default)s)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Set the number of CPU threads to use (default: %(default))",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Set the max number of new tokens to generate (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        "-d",
        default="NPU",
        choices=["NPU", "CPU", "GPU"],
        help="Select the target device (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "int8", "int4"],
        help="Select the target dtype (default: %(default)s)",
    )

    parser.add_argument(
        "--disable-intel-npu-acceleration-library",
        action="store_true",
        help="Disable Intel® NPU Acceleration Library optimizations",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = define_and_parse_args()

    print(
        f"Profiling {args.model} with context size {args.context_size} and dtype {args.dtype}"
    )
    if args.n_threads:
        print(f"Setting number of pytorch thread to {args.n_threads}")
        torch.set_num_threads(args.n_threads)
        print(f"Pytorch thread: {torch.get_num_threads()}")

    main(
        context_size=args.context_size,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        disable_intel_npu_acceleration_library=args.disable_intel_npu_acceleration_library,
    )
