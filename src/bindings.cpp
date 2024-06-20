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

intel_npu_acceleration_library_DLL_API uint32_t getNPUDriverVersion() {
    ov::Core core;
    return intel_npu_acceleration_library::driver_version(core);
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

intel_npu_acceleration_library_DLL_API ov::op::Op* constant(intel_npu_acceleration_library::ModelFactory* factory,
                                                            size_t size, unsigned int* data, char* dtype, void* dst) {
    ov::element::Type_t ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(dtype));
    std::vector<size_t> shape(data, data + size);
    return factory->constant(ov_dtype, shape, dst);
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

intel_npu_acceleration_library_DLL_API ov::op::Op* abs_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->abs_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* acos_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->acos_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* asin_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->asin_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* atan_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->atan_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* ceiling(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->ceiling(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* clamp(intel_npu_acceleration_library::ModelFactory* factory,
                                                         ov::op::Op* in0, float min, float max) {
    return factory->clamp(in0, min, max);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* cos_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->cos_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* cosh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->cosh_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* erf_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->erf_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* elu(intel_npu_acceleration_library::ModelFactory* factory,
                                                       ov::op::Op* in0, float alpha) {
    return factory->elu(in0, alpha);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* floor_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->floor_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* grn(intel_npu_acceleration_library::ModelFactory* factory,
                                                       ov::op::Op* in0, float bias) {
    return factory->grn(in0, bias);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* exp_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->exp_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* gelu(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->gelu(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* log_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->log_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* negative(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->negative(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* relu(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->relu(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sigmoid(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->sigmoid(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sign(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->sign(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sin_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->sin_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sinh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->sinh_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sqrt_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->sqrt_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* tan_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->tan_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* tanh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->tanh_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* acosh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->acosh_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* asinh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->asinh_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* atanh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->atanh_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* hswish(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* in0) {
    return factory->hswish(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* mish(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->mish(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* softplus(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->softplus(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* hsigmoid(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->hsigmoid(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* round_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->round_act(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* softsign(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->softsign(in0);
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
        auto scale = factory->constant<double>(act_ov_dtype, std::vector<size_t>({1, 1}), sqrt(1.0 / dim1));
        in0 = factory->eltwise_mul(in0, scale);
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

intel_npu_acceleration_library_DLL_API ov::op::Op* convolution(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* in0, size_t weight_shape_size,
        unsigned int* weight_shape_data, size_t strides_size, unsigned int* strides_data, size_t pad_begins_size,
        unsigned int* pad_begins_data, size_t pad_ends_size, unsigned int* pad_ends_data, size_t dilations_size,
        unsigned int* dilations_data, size_t groups, bool bias, char* act_dtype, char* wt_dtype) {
    ov::element::Type_t act_ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(act_dtype));
    ov::element::Type_t wt_ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(wt_dtype));

    // Create vectors from the input data
    std::vector<size_t> weight_shape(weight_shape_data, weight_shape_data + weight_shape_size);
    std::vector<size_t> strides(strides_data, strides_data + strides_size);
    std::vector<size_t> pad_begins(pad_begins_data, pad_begins_data + pad_begins_size);
    std::vector<size_t> pad_ends(pad_ends_data, pad_ends_data + pad_ends_size);
    std::vector<size_t> dilations(dilations_data, dilations_data + dilations_size);

    bool quantized = wt_ov_dtype == ov::element::Type_t::i8 || wt_ov_dtype == ov::element::Type_t::i4;

    auto weights = factory->parameter(weight_shape, wt_ov_dtype);

    if (quantized) {
        weights = factory->convert_to(weights, act_ov_dtype);
        auto scale =
                factory->constant<double>(act_ov_dtype, std::vector<size_t>({1, 1, 1, 1}), sqrt(1.0 / weight_shape[1]));
        in0 = factory->eltwise_mul(in0, scale);
    }

    auto mm = factory->convolution(in0, weights, strides, pad_begins, pad_ends, dilations, groups);

    if (quantized) {
        auto scale = factory->parameter({1, weight_shape[0], 1, 1}, act_ov_dtype);
        mm = factory->eltwise_mul(mm, scale);
    }

    if (bias) {
        auto bias = factory->parameter({1, weight_shape[0], 1, 1}, act_ov_dtype);
        return factory->eltwise_add(mm, bias);
    }
    return mm;
}

intel_npu_acceleration_library_DLL_API ov::op::Op* scaled_dot_product_attention(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* query, ov::op::Op* key, ov::op::Op* value,
        ov::op::Op* attn_mask, bool is_causal) {
    return factory->scaled_dot_product_attention(query, key, value, attn_mask, is_causal);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* normL2(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* data, ov::op::Op* axes, float eps) {
    return factory->normL2(data, axes, eps);
}
}