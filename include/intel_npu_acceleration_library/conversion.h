//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <immintrin.h>
#include <iostream>
#include <thread>
#include <vector>
#include "intel_npu_acceleration_library/common.h"

namespace intel_npu_acceleration_library {

/**
 * @brief Compress a int8 vector to I4 format.
 *
 * @param src pointer to the source int8 buffer
 * @param dst pointer to the destination uint8 buffer
 * @param size size of the src and dst buffers
 */
void compressToI4(const int8_t* src, uint8_t* dst, size_t size) {
    for (int i = 0; i < size / 2; i++) {
        dst[i] = (src[2 * i] & 0x0F) | ((src[2 * i + 1] & 0x0F) << 4);
    }
}

/**
 * @brief Convert a int8 vector to fp16 given a scalar scale.
 *
 * @param src pointer to the source int8 buffer
 * @param scale Float scale
 * @param dst pointer to the destination float16 buffer
 * @param size size of the src and dst buffers
 */
void vector_to_fp16(const int8_t* src, float scale, half_ptr dst, size_t size) {
    constexpr size_t VEC_SIZE = 8;             // Use AVX2: process 8 values per loop iteration for 32-bit floats
    __m256 scale_vec = _mm256_set1_ps(scale);  // Broadcast scale

    for (size_t idx = 0; idx < size; idx += VEC_SIZE) {
        // Load int8_t and extend to int32_t for conversion
        __m128i input_8 = _mm_loadl_epi64((__m128i const*)(src + idx));  // Load 8 int8_t values
        __m256i input_32 = _mm256_cvtepi8_epi32(input_8);                // Extend to 32-bit integers

        // Convert integers to float and apply scaling
        __m256 float_vec = _mm256_mul_ps(_mm256_cvtepi32_ps(input_32), scale_vec);

        // Convert float to fp16
        __m128i fp16_vec = _mm256_cvtps_ph(float_vec, _MM_FROUND_TO_NEAREST_INT);

        // Store the result
        _mm_store_si128((__m128i*)(dst + idx), fp16_vec);
    }
}

/**
 * @brief Convert a int8 array to fp16 given a per output channel scale vector.
 *
 * @param input pointer to the source int8 buffer of shape [output_channels, input_channels]
 * @param scale pointer of a float scale vector of shape [output_channels]
 * @param output dst pointer to the destination float16 buffer of shape [output_channels, input_channels]
 * @param input_channels number of input channels
 * @param output_channels number of output channels
 */
void array_to_fp16_worker(const int8_t* input, float* scale, half_ptr output, size_t input_channels,
                          size_t output_channels) {
    for (size_t idx = 0; idx < output_channels; idx++) {
        vector_to_fp16(input + idx * input_channels, scale[idx], output + idx * input_channels, input_channels);
    }
}

/**
 * @brief Convert a int8 array to fp16 given a per output channel scale vector.
 *
 * @param input pointer to the source int8 buffer of shape [output_channels, input_channels]
 * @param scale pointer of a float scale vector of shape [output_channels]
 * @param output dst pointer to the destination float16 buffer of shape [output_channels, input_channels]
 * @param input_channels number of input channels
 * @param output_channels number of output channels
 * @param num_threads number of parallel threads to use
 */
void to_fp16(const int8_t* input, float* scale, half_ptr output, size_t input_channels, size_t output_channels,
             unsigned int num_threads) {
    std::vector<std::thread> threads;

    // Calculate chunk size per thread
    size_t channels_per_thread = (output_channels + num_threads - 1) / num_threads;  // Ceiling division

    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t start_channel = i * channels_per_thread;
        size_t end_channel = std::min((i + 1) * channels_per_thread, output_channels);

        if (start_channel < output_channels) {
            threads.emplace_back(array_to_fp16_worker, input + start_channel * input_channels, scale + start_channel,
                                 output + start_channel * input_channels, input_channels, end_channel - start_channel);
        }
    }

    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}
}  // namespace intel_npu_acceleration_library
