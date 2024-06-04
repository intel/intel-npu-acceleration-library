//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include "intel_npu_acceleration_library/inference.h"

typedef ov::Output<ov::Node> OVNode;

namespace intel_npu_acceleration_library {

/**
 * @brief The ModelFactory class implements a generic interface for NPU network generation and inference.
 * It supports only single input single output operations with input of shape [batch, input_channels] and output of
 * shape [batch, output_channels]
 *
 */
class ModelFactory : public intel_npu_acceleration_library::OVInferenceModel {
private:
    ov::ParameterVector parameters;
    std::vector<std::shared_ptr<ov::op::Op>> operations;

public:
    /**
     * @brief Construct a new Model Factory object
     *
     * @param device target device
     * @param profile enable/disable profiling
     */
    ModelFactory(std::string device, bool profile = false)
            : intel_npu_acceleration_library::OVInferenceModel(device, profile) {
    }

    /**
     * @brief Create a new N-Dimensional network parameter
     *
     * @param shape parameter shape
     * @param dtype parameter datatype
     * @return ov::op::Op*
     */
    ov::op::Op* parameter(std::vector<size_t> shape, ov::element::Type_t dtype) {
        auto param = std::make_shared<ov::opset8::Parameter>(dtype, ov::Shape(shape));
        parameters.push_back(param);
        return param.get();
    }

    /**
     * @brief Create a new matmul operation
     *
     * @param input matmul lhs input
     * @param weights matmul rhs input, a.k.a. weights
     * @param trA transpose the lhs input
     * @param trB transpose the rhs input
     * @return ov::op::Op*
     */
    ov::op::Op* matmul(ov::op::Op* input, ov::op::Op*& weights, bool trA = false, bool trB = true) {
        auto matmul = std::make_shared<ov::opset1::MatMul>(input->output(0), weights->output(0), trA, trB);
        operations.push_back(matmul);
        return matmul.get();
    }

    /**
     * @brief Create a new convolution operation
     *
     * @param input convolution input
     * @param weights convolution weights
     * @param strides convolution strides
     * @param pads_begin convolution padding begin
     * @param pads_ends convolution padding end
     * @param dilations convolution dilations
     * @return ov::op::Op*
     */
    ov::op::Op* convolution(ov::op::Op* input, ov::op::Op*& weights, std::vector<size_t> strides,
                            std::vector<size_t> pads_begin, std::vector<size_t> pads_ends,
                            std::vector<size_t> dilations) {
        auto conv = std::make_shared<ov::opset8::Convolution>(
                input->output(0), weights->output(0), ov::Strides(strides),
                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_ends.begin(), pads_ends.end())),
                ov::Strides(dilations));
        operations.push_back(conv);
        return conv.get();
    }

    /**
     * @brief Create a new gelu operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* gelu(ov::op::Op* input) {
        auto gelu = std::make_shared<ov::opset7::Gelu>(input->output(0), ov::op::GeluApproximationMode::TANH);
        operations.push_back(gelu);
        return gelu.get();
    }

    /**
     * @brief Create a new swish operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* swish(ov::op::Op* input) {
        auto swish = std::make_shared<ov::opset4::Swish>(input->output(0));
        operations.push_back(swish);
        return swish.get();
    }

    /**
     * @brief Create a new softmax operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* softmax(ov::op::Op* input) {
        auto smax = std::make_shared<ov::opset8::Softmax>(input->output(0), -1);
        operations.push_back(smax);
        return smax.get();
    }

    /**
     * @brief Create a new conversion to dtype operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* convert_to(ov::op::Op* input, ov::element::Type_t dtype) {
        auto convert = std::make_shared<ov::opset1::Convert>(input->output(0), dtype);
        operations.push_back(convert);
        return convert.get();
    }

    /**
     * @brief Create a new elementwise add operation
     *
     * @param x1 eltwise lhs input
     * @param x2 eltwise rhs input
     * @return ov::op::Op*
     */
    ov::op::Op* eltwise_add(ov::op::Op* x1, ov::op::Op*& x2) {
        auto eltwise = std::make_shared<ov::opset1::Add>(x1->output(0), x2->output(0));
        operations.push_back(eltwise);
        return eltwise.get();
    }

    /**
     * @brief Create a new elementwise multiply operation
     *
     * @param x1 eltwise lhs input
     * @param x2 eltwise rhs input
     * @return ov::op::Op*
     */
    ov::op::Op* eltwise_mul(ov::op::Op* x1, ov::op::Op*& x2) {
        auto eltwise = std::make_shared<ov::opset1::Multiply>(x1->output(0), x2->output(0));
        operations.push_back(eltwise);
        return eltwise.get();
    }

    /**
     * @brief Create a new elementwise division operation
     *
     * @param x1 eltwise lhs input
     * @param x2 eltwise rhs input
     * @return ov::op::Op*
     */
    ov::op::Op* eltwise_div(ov::op::Op* x1, ov::op::Op*& x2) {
        auto eltwise = std::make_shared<ov::opset1::Divide>(x1->output(0), x2->output(0));
        operations.push_back(eltwise);
        return eltwise.get();
    }

    /**
     * @brief Create a new ScaledDotProductAttention operation
     *
     * @param query sdpa query input
     * @param key sdpa key input
     * @param value sdpa value input
     * @param attn_mask sdpa attn_mask input
     * @param is_causal set the attention mask to causal. If it is set, attn_mask is ignored
     * @return ov::op::Op*
     */
    ov::op::Op* scaled_dot_product_attention(ov::op::Op* query, ov::op::Op* key, ov::op::Op* value,
                                             ov::op::Op* attn_mask, bool is_causal) {
        auto sdpa = std::make_shared<ov::opset13::ScaledDotProductAttention>(
                query->output(0), key->output(0), value->output(0), attn_mask->output(0), is_causal);

        operations.push_back(sdpa);
        return sdpa.get();
    }

    /**
     * @brief Compile the model
     *
     * @param result the last operation in the network. Must have a [batch, output_channel] shape
     */
    void compile(ov::op::Op* result) {
        model = std::make_shared<ov::Model>(std::make_shared<ov::opset8::Result>(result->output(0)), parameters,
                                            "NNFactory");

        compile_model(device);
    }
};

}  // namespace intel_npu_acceleration_library
