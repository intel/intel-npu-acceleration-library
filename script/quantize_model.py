#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_npu_acceleration_library as npu_lib
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit
from neural_compressor.adaptor.torch_utils.auto_round import get_dataloader
import torch
import os


def export_model(model_name, bits=4, output_dir="models"):
    output_folder = os.path.join(output_dir, model_name, f"int{bits}")
    os.makedirs(output_folder, exist_ok=True)

    float_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    float_model.config.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)

    dataloader = get_dataloader(tokenizer, seqlen=2048)

    woq_conf = PostTrainingQuantConfig(
        approach="weight_only",
        op_type_dict={
            ".*": {  # match all ops
                "weight": {
                    "dtype": "int",
                    "bits": 4,
                    "group_size": -1,
                    "scheme": "sym",
                    "algorithm": "AUTOROUND",
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
    export_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
