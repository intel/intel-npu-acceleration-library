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
     * @param inC number of input channels
     * @param outC number of output channels
     * @param batch batch size
     * @param profile enable/disable profiling
     */
    ModelFactory(std::string device, size_t inC, size_t outC, size_t batch, bool profile = false)
            : intel_npu_acceleration_library::OVInferenceModel(device, inC, outC, batch, profile) {
    }

    /**
     * @brief Create a new 2D [dim0, dim1] network parameter
     *
     * @param dim0 dimension 0
     * @param dim1 dimension 1
     * @param dtype parameter datatype
     * @return ov::op::Op*
     */
    ov::op::Op* parameter(size_t dim0, size_t dim1, ov::element::Type_t dtype) {
        auto param = std::make_shared<ov::opset8::Parameter>(dtype, ov::Shape({dim0, dim1}));
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
     * @brief Create a new conversion to fp16 operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* convert_to_fp16(ov::op::Op* input) {
        auto convert = std::make_shared<ov::opset1::Convert>(input->output(0), ov::element::Type_t::f16);
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
