//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
#define intel_npu_acceleration_library_DLL_API __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define intel_npu_acceleration_library_DLL_API __declspec(dllexport)
#endif

namespace intel_npu_acceleration_library {

static constexpr ov::Property<std::string> npu_compiler_type{"NPU_COMPILER_TYPE"};
static constexpr ov::Property<std::string> npu_parameters{"NPU_COMPILATION_MODE_PARAMS"};

/**
 * @brief Return true if the NPU is available on the system, otherwise return false
 *
 * @param core ov::Cor object
 * @return true NPU AI accelerator is available
 * @return false NPU AI accelerator is not available
 */
bool _isNPUAvailable(ov::Core& core) {
    std::vector<std::string> availableDevices = core.get_available_devices();
    return std::find(availableDevices.begin(), availableDevices.end(), "NPU") != availableDevices.end();
}

}  // namespace intel_npu_acceleration_library

// Define half pointer as uint16_t pointer datatype
#define half_ptr uint16_t*