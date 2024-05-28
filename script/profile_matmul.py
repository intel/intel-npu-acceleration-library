#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.quantization import quantize_tensor, compress_to_i4
from intel_npu_acceleration_library.dtypes import int4
from intel_npu_acceleration_library.backend import Linear, QLinear
from functools import partial
import numpy as np
import argparse
import torch
import time
import json


def print_profile_data(hwp_data, data):
    config_keys = ["batch", "inC", "outC", "dtype"]
    config = ", ".join([f"{key}: {hwp_data[key]}" for key in config_keys])

    e2e_runtimes = [elem["runtime"] for elem in data]
    print(
        f"MatMul ({config}) => HWP: {hwp_data['runtime']:.3f} ms, E2E: {np.mean(e2e_runtimes):.3f} ± {2 * np.std(e2e_runtimes):.3f} ms"
    )


def profile(inC, outC, batch, dtype, n_iters=500, skip_first=10):
    data = []
    mac = inC * outC * batch
    memcpy = (inC + outC) * batch

    X = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
    W = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

    if dtype == "float16":
        matmul_csl = Linear
        args = [W]
    elif dtype == "int8":
        weights, scale = quantize_tensor(torch.tensor(W))
        matmul_csl = partial(QLinear, dtype=np.int8)
        args = [weights.numpy(), scale.numpy()]
    elif dtype == "int4":
        weights, scale = quantize_tensor(torch.tensor(W), (int4.min, int4.max))
        weights = compress_to_i4(weights)
        matmul_csl = partial(QLinear, dtype=np.uint8)
        args = [weights.numpy(), scale.numpy()]
    else:
        raise RuntimeError(f"Invalid dtype: {dtype}")

    args.append("0000")

    mm_prof = matmul_csl(inC, outC, batch, profile=True)
    mm = matmul_csl(inC, outC, batch, profile=False)

    # Get the HWP data
    mm_prof.run(X, *args)
    with open("profiling.json") as fp:
        hwp_runtime = (
            json.load(fp)["taskStatistics"]["total duration"] / 1000.0
        )  # in us
    hwp_data = dict(
        batch=batch,
        inC=inC,
        outC=outC,
        memcpy=memcpy,
        mac=mac,
        runtime=hwp_runtime,
        dtype=dtype,
    )

    for idx in range(n_iters):
        t0 = time.perf_counter()
        mm.run(X, *args)
        t1 = time.perf_counter()
        if idx > (skip_first - 1):
            data.append(
                dict(
                    batch=batch,
                    inC=inC,
                    outC=outC,
                    memcpy=memcpy,
                    mac=mac,
                    runtime=(t1 - t0) * 1000,
                    dtype=W.dtype,
                )
            )

    print_profile_data(hwp_data, data)

    return hwp_data, data


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Profiling a MatMul model in the NPU")
    parser.add_argument("--batch", "-b", type=int, required=True, help="MatMul batch")
    parser.add_argument(
        "--input-channels", "-c", type=int, required=True, help="MatMul input channels"
    )
    parser.add_argument(
        "--output-channels",
        "-k",
        type=int,
        required=True,
        help="MatMul output channels",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "int8", "int4"],
        help="Select the target dtype (default: %(default)s)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = define_and_parse_args()
    profile(args.input_channels, args.output_channels, args.batch, dtype=args.dtype)
