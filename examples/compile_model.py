#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


from intel_npu_acceleration_library import compile
from sklearn.metrics import r2_score
import intel_npu_acceleration_library
import pytest
import torch
import sys

# Define a
class NN(torch.nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(hidden_dim, intermediate_dim)
        self.l2 = torch.nn.Linear(intermediate_dim, hidden_dim)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        return self.relu(self.l2(self.relu(self.l1(x))))


if __name__ == "__main__":

    # Define a NN module
    model = NN(32, 128)
    # Generate the input
    x = torch.rand((16, 32), dtype=torch.float16) - 0.5

    # Get the reference output
    with torch.no_grad():
        y_ref = model(x.to(torch.float32))

    # Compile the model
    print("Compile the model for the NPU")
    if sys.platform == "win32":
        # Windows do not support torch.compile
        print(
            "Windows do not support torch.compile, fallback to intel_npu_acceleration_library.compile"
        )
        compiled_model = intel_npu_acceleration_library.compile(model)
    else:
        compiled_model = torch.compile(model, backend="npu")

    # Get the NPU output
    with torch.no_grad():
        y = compiled_model(x)

    print(f"Reference vs actual R2 score: {r2_score(y_ref, y):.2f}")
