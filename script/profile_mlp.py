#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3MLP
from intel_npu_acceleration_library.dtypes import int8, int4
from intel_npu_acceleration_library.compiler import CompilerConfig
from torch.profiler import profile, ProfilerActivity
from sklearn.metrics import r2_score
import intel_npu_acceleration_library
import argparse
import torch
import numpy as np


def main(
    seq_len=128,
    hidden_size=256,
    intermediate_size=512,
    dtype="float16",
    _profile=False,
    enable_graph_mode=False,
):

    conf = Phi3Config.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    conf.num_hidden_layers = 1
    conf.hidden_size = hidden_size
    conf.intermediate_size = intermediate_size

    # Define a single Phi-3 MLP layer
    mlp = Phi3MLP(conf)

    hidden_states = torch.rand((seq_len, conf.hidden_size))

    reference = mlp(hidden_states.to(torch.float32)).to(torch.float16)

    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "int8":
        dtype = int8
    elif dtype == "int4":
        dtype = int4
    else:
        raise RuntimeError(f"Invalid dtype: {dtype}")

    # Compile model
    compiler_conf = CompilerConfig(use_to=enable_graph_mode, dtype=dtype)
    model = intel_npu_acceleration_library.compile(mlp, compiler_conf)
    if _profile:
        model.profile = True

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(1000):
            results = model(hidden_states)

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=20
        )
    )

    prof.export_chrome_trace("trace.json")

    results = results.detach().numpy()
    reference = reference.detach().numpy()

    assert results.shape == reference.shape, "Output shape mismatch"
    assert np.isfinite(reference).all(), "Pytorch Reference contains NaN or Inf"
    assert np.isfinite(results).all(), "NPU output contains NaN or Inf"

    if dtype == int4:
        assert 1 - r2_score(reference, results) < 0.05
    else:
        assert 1 - r2_score(reference, results) < 0.001


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Profiling a MLP layer in the NPU")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Hidden size (default: %(default)s)",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=512,
        help="Intermediate size (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "int8", "int4"],
        help="Select the target dtype (default: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable the profiling (default: False)",
    )
    parser.add_argument(
        "--enable_graph_mode",
        action="store_true",
        default=False,
        help="Enable graph mode for MLP, otherwise use eager mode (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = define_and_parse_args()

    print(
        f"Profiling with sequence length {args.seq_len}, hidden size {args.hidden_size}, intermediate size {args.intermediate_size}, dtype {args.dtype}"
    )

    main(
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dtype=args.dtype,
        _profile=args.profile,
        enable_graph_mode=args.enable_graph_mode,
    )
