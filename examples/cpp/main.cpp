//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu_acceleration_library/nn_factory.h"

using namespace intel_npu_acceleration_library;
#include <iostream>

int main() {
    const size_t batch = 128, inC = 256, outC = 512, N = 100000;

    std::cout << "Create a ModelFactory" << std::endl;
    auto factory = std::make_shared<ModelFactory>("NPU");

    // create parameter
    auto input = factory->parameter({batch, inC}, ov::element::f16);
    auto weights = factory->parameter({outC, inC}, ov::element::f16);

    // create matmul
    auto matmul = factory->matmul(input, weights);

    // Compile the model
    factory->compile(matmul);

    // Save OV model
    std::cout << "Saving model to matmul.xml" << std::endl;
    factory->saveModel("matmul.xml");

    // Here you can create float16 buffers and run inference by using
    half_ptr input_buffer = new uint16_t[batch * inC];
    half_ptr weights_buffer = new uint16_t[outC * inC];
    half_ptr output_buffer = new uint16_t[batch * outC];

    memset(input_buffer, 0, 128 * 256 * sizeof(uint16_t));
    memset(weights_buffer, 0, 128 * 256 * sizeof(uint16_t));
    memset(output_buffer, 0, 128 * 512 * sizeof(uint16_t));

    factory->setInputTensor(input_buffer, 0);
    factory->setInputTensor(weights_buffer, 1);
    factory->setOutputTensor(output_buffer, 0);

    // Run inference
    std::cout << "Run inference on " << N << " workloads" << std::endl;
    for (auto idx = 0; idx < N; idx++)
        factory->run();
    std::cout << "Inference done" << std::endl;

    delete[] input_buffer;
    delete[] weights_buffer;
    delete[] output_buffer;
    return 0;
}