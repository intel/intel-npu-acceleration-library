//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu_acceleration_library/nn_factory.h"

extern "C" {

intel_npu_acceleration_library_DLL_API bool isNPUAvailable() {
    ov::Core core;
    return intel_npu_acceleration_library::_isNPUAvailable(core);
}

// ######################## Compression ########################

intel_npu_acceleration_library_DLL_API void compressToI4(const int8_t* src, uint8_t* dst, size_t size) {
    intel_npu_acceleration_library::compressToI4(src, dst, size);
}

// ######################### Parameters #########################

intel_npu_acceleration_library_DLL_API intel_npu_acceleration_library::Parameters* createParameters() {
    return new intel_npu_acceleration_library::Parameters();
}

intel_npu_acceleration_library_DLL_API void destroyParameters(intel_npu_acceleration_library::Parameters* parameters) {
    delete parameters;
}

intel_npu_acceleration_library_DLL_API void addFloatParameter(intel_npu_acceleration_library::Parameters* parameters,
                                                              half_ptr data, size_t dim0, size_t dim1) {
    parameters->add_parameter(data, intel_npu_acceleration_library::Shape({dim0, dim1}));
}

intel_npu_acceleration_library_DLL_API void addIntParameter(intel_npu_acceleration_library::Parameters* parameters,
                                                            int8_t* data, half_ptr scale, size_t dim0, size_t dim1) {
    parameters->add_parameter(data, scale, intel_npu_acceleration_library::Shape({dim0, dim1}));
}

intel_npu_acceleration_library_DLL_API void addInt4Parameter(intel_npu_acceleration_library::Parameters* parameters,
                                                             uint8_t* data, half_ptr scale, size_t dim0, size_t dim1) {
    parameters->add_parameter(data, scale, intel_npu_acceleration_library::Shape({dim0, dim1}));
}

intel_npu_acceleration_library_DLL_API void addIntParameterConversion(
        intel_npu_acceleration_library::Parameters* parameters, int8_t* data, float* scale, size_t dim0, size_t dim1) {
    parameters->add_parameter(data, scale, intel_npu_acceleration_library::Shape({dim0, dim1}));
}

// ######################### NN Factory #########################

intel_npu_acceleration_library_DLL_API intel_npu_acceleration_library::ModelFactory* createNNFactory(
        char* device, bool profile = false) {
    return new intel_npu_acceleration_library::ModelFactory(std::string(device), profile);
}

intel_npu_acceleration_library_DLL_API void destroyNNFactory(intel_npu_acceleration_library::OVInferenceModel* matmul) {
    delete matmul;
}

intel_npu_acceleration_library_DLL_API void saveModel(intel_npu_acceleration_library::OVInferenceModel* matmul,
                                                      char* path) {
    matmul->saveModel(std::string(path));
}

intel_npu_acceleration_library_DLL_API void saveCompiledModel(intel_npu_acceleration_library::OVInferenceModel* matmul,
                                                              char* path) {
    matmul->saveCompiledModel(std::string(path));
}

intel_npu_acceleration_library_DLL_API void setNNFactoryWeights(
        intel_npu_acceleration_library::OVInferenceModel* mm, intel_npu_acceleration_library::Parameters* parameters) {
    mm->wt_thread = std::thread(&intel_npu_acceleration_library::OVInferenceModel::setWeights, mm,
                                parameters->get_parameters());
}

intel_npu_acceleration_library_DLL_API void compile(intel_npu_acceleration_library::ModelFactory* factory,
                                                    ov::op::Op* result) {
    factory->compile(result);
}

intel_npu_acceleration_library_DLL_API size_t
get_output_tensor_shape_size(intel_npu_acceleration_library::ModelFactory* factory, size_t tensor_idx) {
    ov::Tensor tensor = factory->getOutputTensors(tensor_idx);
    return tensor.get_shape().size();
}

intel_npu_acceleration_library_DLL_API size_t
get_output_tensor_shape(intel_npu_acceleration_library::ModelFactory* factory, size_t tensor_idx, size_t idx) {
    ov::Tensor tensor = factory->getOutputTensors(tensor_idx);
    return tensor.get_shape()[idx];
}

intel_npu_acceleration_library_DLL_API void set_activation(intel_npu_acceleration_library::OVInferenceModel* mm,
                                                           half_ptr X, size_t idx) {
    mm->setInputTensor(X, idx);
}

intel_npu_acceleration_library_DLL_API void set_output(intel_npu_acceleration_library::OVInferenceModel* mm,
                                                       half_ptr Out, size_t idx) {
    mm->setOutputTensor(Out, idx);
}

intel_npu_acceleration_library_DLL_API float run(intel_npu_acceleration_library::OVInferenceModel* mm) {
    auto start = std::chrono::system_clock::now();

    mm->run();

    auto stop = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return static_cast<float>(elapsed.count()) / static_cast<float>(1000.0);
}

// ######################### NN Factory layers #########################

intel_npu_acceleration_library_DLL_API ov::op::Op* parameter(intel_npu_acceleration_library::ModelFactory* factory,
                                                             size_t size, unsigned int* data, char* dtype) {
    ov::element::Type_t ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(dtype));
    std::vector<size_t> shape(data, data + size);
    return factory->parameter(shape, ov_dtype);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* matmul(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* in0, ov::op::Op* in1) {
    return factory->matmul(in0, in1);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* eltwise_add(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* in0, ov::op::Op* in1) {
    return factory->eltwise_add(in0, in1);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* eltwise_mul(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* in0, ov::op::Op* in1) {
    return factory->eltwise_mul(in0, in1);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* eltwise_div(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* in0, ov::op::Op* in1) {
    return factory->eltwise_div(in0, in1);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* gelu(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->gelu(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* swish(intel_npu_acceleration_library::ModelFactory* factory,
                                                         ov::op::Op* in0) {
    return factory->swish(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* softmax(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->softmax(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* convert_to_fp16(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* in0) {
    return factory->convert_to(in0, ov::element::Type_t::f16);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* linear(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* in0, size_t dim0, size_t dim1, bool bias,
                                                          char* act_dtype, char* wt_dtype) {
    ov::element::Type_t act_ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(act_dtype));
    ov::element::Type_t wt_ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(wt_dtype));

    bool quantized = wt_ov_dtype == ov::element::Type_t::i8 || wt_ov_dtype == ov::element::Type_t::i4;

    auto weights = factory->parameter({dim0, dim1}, wt_ov_dtype);
    if (quantized) {
        weights = factory->convert_to(weights, act_ov_dtype);
    }

    auto mm = factory->matmul(in0, weights);

    if (quantized) {
        auto scale = factory->parameter({1, dim0}, act_ov_dtype);
        mm = factory->eltwise_mul(mm, scale);
    }

    if (bias) {
        auto bias = factory->parameter({1, dim0}, act_ov_dtype);
        return factory->eltwise_add(mm, bias);
    }
    return mm;
}

intel_npu_acceleration_library_DLL_API ov::op::Op* scaled_dot_product_attention(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* query, ov::op::Op* key, ov::op::Op* value,
        ov::op::Op* attn_mask, bool is_causal) {
    return factory->scaled_dot_product_attention(query, key, value, attn_mask, is_causal);
}
};