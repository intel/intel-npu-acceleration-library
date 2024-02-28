#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import subprocess
import itertools
import intel_npu_acceleration_library
import platform
import datetime
import socket
import tqdm
import json
import re


def profile_model(
    model_id,
    context_size,
    device="NPU",
    dtype="float16",
    use_intel_npu_acceleration_library=True,
):

    profiling_data = {
        "model": model_id,
        "context_size": context_size,
        "device": device,
        "dtype": dtype,
        "intel_npu_acceleration_library": use_intel_npu_acceleration_library,
        "prefill": None,
        "tps": None,
        "error": None,
    }
    try:
        disable_intel_npu_acceleration_library = (
            "--disable-intel-npu-acceleration-library"
            if not use_intel_npu_acceleration_library
            else ""
        )
        output = subprocess.check_output(
            f"python profile_llm.py -m {model_id} --context-size {context_size} --device {device} {disable_intel_npu_acceleration_library} ",
            shell=True,
        ).decode()

        profiling_line = output.strip().split("\n")[-1].strip()

        pattern = r"prefill-phase (\d+\.\d+) s, tokens/s (\d+\.\d+) s"

        match = re.search(pattern, profiling_line)

        # Check if a match is found
        if match:
            # Extract the prefill phase and tokens/s values
            profiling_data["prefill"] = float(match.group(1))
            profiling_data["tps"] = float(match.group(2))
        else:
            profiling_data["error"] = f"parsing error: profiling output {output}"
    except:
        profiling_data["error"] = "runtime error"

    return profiling_data


def save_data(data):
    date = data["config"]["time"].replace(" ", "_").replace(":", "_")
    with open(f"leaderboard_{date}.json", "w") as fp:
        json.dump(data, fp, indent=4)


def main():

    data = {
        "config": {
            "time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "arch": platform.machine(),
            "version": platform.version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "npu": "yes"
            if intel_npu_acceleration_library.backend.npu_available()
            else "no",
            "unit": "seconds",
        },
        "profiling": [],
    }
    save_data(data)

    models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "stabilityai/stablelm-3b-4e1t",
        # "qnguyen3/quan-1.8b-chat",
        "facebook/opt-1.3b",
        "gpt2-large",
        "openlm-research/open_llama_3b_v2",
        "EleutherAI/pythia-2.8b-v0",
        "tiiuae/falcon-rw-1b",
        "EleutherAI/gpt-neo-1.3B",
        "stabilityai/stable-code-3b",
        "google/gemma-2b-it",
    ]

    contexts = [64, 128, 256, 512]
    use_intel_npu_acceleration_library_lst = [True]
    devices = ["NPU"]
    dtypes = ["float16", "int8"]
    configurations = list(
        itertools.product(
            models, contexts, devices, dtypes, use_intel_npu_acceleration_library_lst
        )
    )

    for model, context, device, dtype, use_intel_npu_acceleration_library in tqdm.tqdm(
        configurations
    ):
        profiling_data = profile_model(
            model, context, device, dtype, use_intel_npu_acceleration_library
        )
        data["profiling"].append(profiling_data)
        save_data(data)


if __name__ == "__main__":

    main()
