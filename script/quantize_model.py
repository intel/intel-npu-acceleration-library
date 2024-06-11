#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_npu_acceleration_library as npu_lib
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit
from neural_compressor.adaptor.torch_utils.auto_round import get_dataloader
import argparse
import torch
import os


def define_and_parse_arguments():
    parser = argparse.ArgumentParser(description="Export a model to NPU")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The name of the model to export",
    )
    parser.add_argument(
        "-b",
        "--bits",
        type=int,
        default=4,
        help="The number of bits to use for quantization",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="models",
        help="The directory where to save the exported model",
    )
    parser.add_argument(
        "-s",
        "--sequence-lenght",
        type=int,
        default=2048,
        help="The sequence lenght to use for the dataloader",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="RTN",
        help="The quantization algorithm to use",
    )
    return parser.parse_args()


def export_model(
    model_name: str,
    algorithm: str,
    bits: int = 4,
    sequence_lenght: int = 2048,
    output_dir: str = "models",
):
    """Quantize and export a model.

    Args:
        model_name (str): the name of the model to export
        algorithm (str, optional): the neural compressor quantization algorithm
        bits (int, optional): the number of bits. Defaults to 4.
        sequence_lenght (int, optional): the model sequence lenght. Defaults to 2048.
        output_dir (str, optional): the output directory. Defaults to "models".
    """
    print(f"Exporting model {model_name} with {bits} bits")
    output_folder = os.path.join(output_dir, model_name, algorithm, f"int{bits}")
    os.makedirs(output_folder, exist_ok=True)

    float_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    float_model.config.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)

    dataloader = get_dataloader(tokenizer, seqlen=sequence_lenght)

    woq_conf = PostTrainingQuantConfig(
        approach="weight_only",
        op_type_dict={
            ".*": {  # match all ops
                "weight": {
                    "dtype": "int",
                    "bits": bits,
                    "group_size": -1,
                    "scheme": "sym",
                    "algorithm": algorithm.upper(),
                },
                "activation": {
                    "dtype": "fp16",
                },
            }
        },
    )

    print("Apply generic model optimizations")
    npu_lib.compiler.apply_general_optimizations(float_model)
    print("Quantize model")
    quantized_model = fit(model=float_model, conf=woq_conf, calib_dataloader=dataloader)
    print("Quantize model")
    compressed_model = quantized_model.export_compressed_model(
        scale_dtype=torch.float16, use_optimum_format=False
    )

    print("Create NPU kernels")
    npu_model = npu_lib.compiler.create_npu_kernels(compressed_model)

    torch.save(npu_model, os.path.join(output_folder, "pytorch_npu_model.bin"))
    print(f"Model succesfully exported to {output_folder}")


if __name__ == "__main__":
    args = define_and_parse_arguments()
    export_model(
        args.model, args.algorithm, args.bits, args.sequence_lenght, args.output_dir
    )
