#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import MatMul
import numpy as np


def run_matmul(inC, outC, batch):

    # Create both inputs
    X1 = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
    X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

    mm = MatMul(inC, outC, batch, profile=False)

    return mm.run(X1, X2)


if __name__ == "__main__":
    result = run_matmul(128, 128, 32)
    print(result)
