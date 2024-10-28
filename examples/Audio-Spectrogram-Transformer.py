#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


import sys
import subprocess
import pkg_resources

required = {"librosa", "soundfile", "datasets", "intel-npu-acceleration-library"}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch
import intel_npu_acceleration_library

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation",
    trust_remote_code=True,
)
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
print("Compile model for the NPU")
model = intel_npu_acceleration_library.compile(model)

# audio file is decoded on the fly
inputs = feature_extractor(
    dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_ids]
predicted_label

# compute loss - target_label is e.g. "down"
target_label = model.config.id2label[0]
inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
loss = model(**inputs).loss
print(round(loss.item(), 2))
