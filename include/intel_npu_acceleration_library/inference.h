//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "intel_npu_acceleration_library/common.h"
#include "intel_npu_acceleration_library/parameters.h"

namespace intel_npu_acceleration_library {

/**
 * @brief The OVInferenceModel implements the basic of NN inference on NPU
 *
 */
class OVInferenceModel {
private:
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::mutex mutex_;

protected:
    std::shared_ptr<ov::Model> model;  ///< @brief OpenVINO model
    size_t inC;                        ///< @brief Model input channels
    size_t outC;                       ///< @brief Model output channels
    size_t batch;                      ///< @brief Model batch szie
    std::string device;                ///< @brief Target device
    bool profile;                      ///< @brief Enable/disable profiling

    /**
     * @brief Compile a generated OV model to a specific device
     *
     * @param device target compialtion device
     */
    void compile_model(std::string device) {
        if (!_isNPUAvailable(core)) {
            // Fallback to auto in case there is no NPU device. Handle this situation at python level
            device = "CPU";
        }
        // set letency hint
        core.set_property(ov::cache_dir("cache"));
        core.set_property(device, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        if (device == "NPU") {
            core.set_property(device, intel_npu_acceleration_library::npu_compiler_type("DRIVER"));
            if (profile) {
                core.set_property(device, ov::enable_profiling(true));
                core.set_property(device, intel_npu_acceleration_library::npu_parameters(
                                                  "dpu-profiling=true dma-profiling=true sw-profiling=true"));
            }
        }

        // Compile model
        compiled_model = core.compile_model(model, device);
        // Create inference request
        infer_request = compiled_model.create_infer_request();
        // First inference
        infer_request.infer();
    }

    /**
     * @brief Create a ov model object. This class needs to be override in child classes
     *
     */
    virtual void create_ov_model() {
        throw std::runtime_error("create_ov_model not implemented for OVInferenceModel class");
    };

public:
    ov::Tensor X;           ///< @brief Model input tensor
    ov::Tensor Out;         ///< @brief Model output tensor
    std::thread wt_thread;  ///< @brief Async weight prefetch thread

    /**
     * @brief Construct a new OVInferenceModel object
     *
     * @param device target device
     * @param inC number of input channels
     * @param outC number of output channels
     * @param batch batch size
     * @param profile enable/disable profiling
     */
    OVInferenceModel(std::string device, size_t inC, size_t outC, size_t batch, bool profile = false)
            : inC(inC), outC(outC), batch(batch), device(device), profile(profile) {
    }

    virtual ~OVInferenceModel() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (wt_thread.joinable())
            wt_thread.join();
    }

    /**
     * @brief Save the model to a local path
     *
     * @param path
     */
    void saveCompiledModel(const std::string& path) {
        std::ofstream model_path(path);
        compiled_model.export_model(model_path);
        model_path.flush();
        model_path.close();

        std::cout << "Model saved to " << path << std::endl;
    }

    /**
     * @brief Save the model to a local path
     *
     * @param path
     */
    void saveModel(const std::string& path) {
        ov::save_model(model, path, true);
        std::cout << "Model saved to " << path << std::endl;
    }

    /**
     * @brief Run an inference
     *
     * @return void
     */
    void run() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (wt_thread.joinable())
            wt_thread.join();

        // Start async request for the first time
        infer_request.infer();
        // auto end = std::chrono::system_clock::now();
        if (profile) {
            infer_request.get_profiling_info();
        }
    }

    /**
     * @brief Set the input and output activations
     *
     * @param _X pointer to the float16 input activation
     * @param _Out pointer to the float16 output activation
     */
    void setActivations(half_ptr _X, half_ptr _Out) {
        std::lock_guard<std::mutex> lock(mutex_);
        X = ov::Tensor(ov::element::f16, ov::Shape({batch, inC}), (void*)_X);
        Out = ov::Tensor(ov::element::f16, ov::Shape({batch, outC}), (void*)_Out);

        infer_request.set_input_tensor(0, X);
        infer_request.set_output_tensor(Out);
    }

    /**
     * @brief Set the network parameters
     *
     * @param _weights vector of network parameters
     * @param started atomic bool variable that is set to true once the conversion started. Useful for thread
     * syncronizations
     */
    void setWeights(std::vector<std::shared_ptr<Parameter>> _weights, std::atomic_bool& started) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::tuple<Parameter*, void*, size_t>> memcpy_vector;

        // Start from idx == 1
        size_t idx = 1;
        for (auto& _W : _weights) {
            auto W = infer_request.get_input_tensor(idx++);
            auto addr = static_cast<void*>(W.data());
            auto size = W.get_byte_size();
            memcpy_vector.push_back({_W.get(), addr, size});
        }

        started = true;
        for (auto& elem : memcpy_vector) {
            std::get<0>(elem)->set_data(std::get<1>(elem), std::get<2>(elem));
        }
    }
};

}  // namespace intel_npu_acceleration_library