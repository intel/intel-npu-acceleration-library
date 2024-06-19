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
     * @brief Create a new constant object
     *
     * @param dtype element type of the tensor constant
     * @param shape shape of the tensor constant
     * @param values vector of literals for initializing the tensor constant
     * @return ov::op::Op*
     */
    template <typename T>
    ov::op::Op* constant(ov::element::Type_t dtype, std::vector<size_t> shape, std::vector<T>& values) {
        auto constant = std::make_shared<ov::opset1::Constant>(dtype, ov::Shape(shape), values);
        operations.push_back(constant);
        return constant.get();
    }

    /**
     * @brief Create a new constant object
     *
     * @param dtype element type of the tensor constant
     * @param shape shape of the tensor constant
     * @param value value for initializing the tensor constant
     * @return ov::op::Op*
     */
    template <class T, class = typename std::enable_if<std::is_fundamental<T>::value>::type>
    ov::op::Op* constant(ov::element::Type_t dtype, std::vector<size_t> shape, T value) {
        auto constant = std::make_shared<ov::opset1::Constant>(dtype, ov::Shape(shape), value);
        operations.push_back(constant);
        return constant.get();
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
     * @param groups convolution groups
     * @return ov::op::Op*
     */
    ov::op::Op* convolution(ov::op::Op* input, ov::op::Op*& weights, std::vector<size_t> strides,
                            std::vector<size_t> pads_begin, std::vector<size_t> pads_ends,
                            std::vector<size_t> dilations, size_t groups = 1) {
        if (groups > 1) {
            auto conv = std::make_shared<ov::opset8::GroupConvolution>(
                    input->output(0), weights->output(0), ov::Strides(strides),
                    ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                    ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_ends.begin(), pads_ends.end())),
                    ov::Strides(dilations));
            operations.push_back(conv);
            return conv.get();
        }
        auto conv = std::make_shared<ov::opset8::Convolution>(
                input->output(0), weights->output(0), ov::Strides(strides),
                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_ends.begin(), pads_ends.end())),
                ov::Strides(dilations));
        operations.push_back(conv);
        return conv.get();
    }

    /**
     * @brief Create a new gather operation
     *
     * @param input tensor from which slices are gathered
     * @param indices tensor with indexes to gather
     * @param axis The tensor is a dimension index to gather data from
     * @param batch_dims The number of batch dimension in data and indices tensors.
     * @return ov::op::Op*
     */
    ov::op::Op* gather(ov::op::Op* input, ov::op::Op* indices, ov::op::Op* axis, const size_t batch_dims = 0) {
        auto gather =
                std::make_shared<ov::opset8::Gather>(input->output(0), indices->output(0), axis->output(0), batch_dims);
        operations.push_back(gather);
        return gather.get();
    }

    /**
     * @brief create a new reshape operation
     *
     * @param input tensor to be reshaped.
     * @param shape new shape tensor, -1 is allowed for one dimension, it will be calculated automatically.
     * @return ov::op::Op*
     */
    ov::op::Op* reshape(ov::op::Op* input, ov::op::Op* shape) {
        auto reshape = std::make_shared<ov::opset1::Reshape>(input->output(0), shape->output(0), true);
        operations.push_back(reshape);
        return reshape.get();
    }

    /**
     * @brief create a new transpose operation
     *
     * @param input tensor to be transposed.
     * @param shape permutation tensor, the new order of dimensions.
     * @return ov::op::Op*
     */
    ov::op::Op* transpose(ov::op::Op* input, ov::op::Op* input_order) {
        auto reshape = std::make_shared<ov::opset1::Transpose>(input->output(0), input_order->output(0));
        operations.push_back(reshape);
        return reshape.get();
    }

    /**
     * @brief Create a new absolute activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* abs_act(ov::op::Op* input) {
        auto abs_act = std::make_shared<ov::opset1::Abs>(input->output(0));
        operations.push_back(abs_act);
        return abs_act.get();
    }

    /**
     * @brief Create a new arccos activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* acos_act(ov::op::Op* input) {
        auto acos_act = std::make_shared<ov::opset1::Acos>(input->output(0));
        operations.push_back(acos_act);
        return acos_act.get();
    }

    /**
     * @brief Create a new arcsin activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* asin_act(ov::op::Op* input) {
        auto asin_act = std::make_shared<ov::opset1::Asin>(input->output(0));
        operations.push_back(asin_act);
        return asin_act.get();
    }

    /**
     * @brief Create a new arctan activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* atan_act(ov::op::Op* input) {
        auto atan_act = std::make_shared<ov::opset1::Atan>(input->output(0));
        operations.push_back(atan_act);
        return atan_act.get();
    }

    /**
     * @brief Create a new ceiling operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* ceiling(ov::op::Op* input) {
        auto ceiling = std::make_shared<ov::opset1::Ceiling>(input->output(0));
        operations.push_back(ceiling);
        return ceiling.get();
    }

    /**
     * @brief Create a new clamp operation
     *
     * @param input operation's input node
     * @param min lower bound of the <min;max> range
     * @param max the upper bound of the <min;max> range
     * @return ov::op::Op*
     */
    ov::op::Op* clamp(ov::op::Op* input, float min, float max) {
        auto clamp = std::make_shared<ov::opset1::Clamp>(input->output(0), min, max);
        operations.push_back(clamp);
        return clamp.get();
    }

    /**
     * @brief Create a new cosine activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* cos_act(ov::op::Op* input) {
        auto cos_act = std::make_shared<ov::opset1::Cos>(input->output(0));
        operations.push_back(cos_act);
        return cos_act.get();
    }

    /**
     * @brief Create a new cosh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* cosh_act(ov::op::Op* input) {
        auto cosh_act = std::make_shared<ov::opset1::Cosh>(input->output(0));
        operations.push_back(cosh_act);
        return cosh_act.get();
    }

    /**
     * @brief Create a new elu operation
     *
     * @param input operation's input node
     * @param alpha multiplier for negative values
     * @return ov::op::Op*
     */
    ov::op::Op* elu(ov::op::Op* input, float alpha) {
        auto elu = std::make_shared<ov::opset1::Elu>(input->output(0), alpha);
        operations.push_back(elu);
        return elu.get();
    }

    /**
     * @brief Create a new erf activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* erf_act(ov::op::Op* input) {
        auto erf_act = std::make_shared<ov::opset1::Erf>(input->output(0));
        operations.push_back(erf_act);
        return erf_act.get();
    }

    /**
     * @brief Create a new exp activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* exp_act(ov::op::Op* input) {
        auto exp_act = std::make_shared<ov::opset1::Exp>(input->output(0));
        operations.push_back(exp_act);
        return exp_act.get();
    }

    /**
     * @brief Create a new floor activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* floor_act(ov::op::Op* input) {
        auto floor_act = std::make_shared<ov::opset1::Floor>(input->output(0));
        operations.push_back(floor_act);
        return floor_act.get();
    }

    /**
     * @brief Create a new grn operation
     *
     * @param input operation's input node
     * @param bias bias added to the variance
     * @return ov::op::Op*
     */
    ov::op::Op* grn(ov::op::Op* input, float bias) {
        auto grn = std::make_shared<ov::opset1::GRN>(input->output(0), bias);
        operations.push_back(grn);
        return grn.get();
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
     * @brief Create a new natural log operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* log_act(ov::op::Op* input) {
        auto log_act = std::make_shared<ov::opset1::Log>(input->output(0));
        operations.push_back(log_act);
        return log_act.get();
    }

    /**
     * @brief Create a new negative operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* negative(ov::op::Op* input) {
        auto negative = std::make_shared<ov::opset1::Negative>(input->output(0));
        operations.push_back(negative);
        return negative.get();
    }

    /**
     * @brief Create a new relu operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* relu(ov::op::Op* input) {
        auto relu = std::make_shared<ov::opset1::Relu>(input->output(0));
        operations.push_back(relu);
        return relu.get();
    }

    /**
     * @brief Create a new sigmoid operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* sigmoid(ov::op::Op* input) {
        auto sigmoid = std::make_shared<ov::opset1::Sigmoid>(input->output(0));
        operations.push_back(sigmoid);
        return sigmoid.get();
    }

    /**
     * @brief Create a new sign operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* sign(ov::op::Op* input) {
        auto sign = std::make_shared<ov::opset1::Sign>(input->output(0));
        operations.push_back(sign);
        return sign.get();
    }

    /**
     * @brief Create a new sine activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* sin_act(ov::op::Op* input) {
        auto sin_act = std::make_shared<ov::opset1::Sin>(input->output(0));
        operations.push_back(sin_act);
        return sin_act.get();
    }

    /**
     * @brief Create a new sinh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* sinh_act(ov::op::Op* input) {
        auto sinh_act = std::make_shared<ov::opset1::Sinh>(input->output(0));
        operations.push_back(sinh_act);
        return sinh_act.get();
    }

    /**
     * @brief Create a new sqrt activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* sqrt_act(ov::op::Op* input) {
        auto sqrt_act = std::make_shared<ov::opset1::Sqrt>(input->output(0));
        operations.push_back(sqrt_act);
        return sqrt_act.get();
    }

    /**
     * @brief Create a new tan activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* tan_act(ov::op::Op* input) {
        auto tan_act = std::make_shared<ov::opset1::Tan>(input->output(0));
        operations.push_back(tan_act);
        return tan_act.get();
    }

    /**
     * @brief Create a new tanh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* tanh_act(ov::op::Op* input) {
        auto tanh_act = std::make_shared<ov::opset1::Tanh>(input->output(0));
        operations.push_back(tanh_act);
        return tanh_act.get();
    }

    /**
     * @brief Create a new arccosh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* acosh_act(ov::op::Op* input) {
        auto acosh_act = std::make_shared<ov::opset4::Acosh>(input->output(0));
        operations.push_back(acosh_act);
        return acosh_act.get();
    }

    /**
     * @brief Create a new arcsinh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* asinh_act(ov::op::Op* input) {
        auto asinh_act = std::make_shared<ov::opset4::Asinh>(input->output(0));
        operations.push_back(asinh_act);
        return asinh_act.get();
    }

    /**
     * @brief Create a new arctanh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* atanh_act(ov::op::Op* input) {
        auto atanh_act = std::make_shared<ov::opset4::Atanh>(input->output(0));
        operations.push_back(atanh_act);
        return atanh_act.get();
    }

    /**
     * @brief Create a new hswish operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* hswish(ov::op::Op* input) {
        auto hswish = std::make_shared<ov::opset4::HSwish>(input->output(0));
        operations.push_back(hswish);
        return hswish.get();
    }

    /**
     * @brief Create a new mish operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* mish(ov::op::Op* input) {
        auto mish = std::make_shared<ov::opset4::Mish>(input->output(0));
        operations.push_back(mish);
        return mish.get();
    }

    /**
     * @brief Create a new softplus operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* softplus(ov::op::Op* input) {
        auto softplus = std::make_shared<ov::opset4::SoftPlus>(input->output(0));
        operations.push_back(softplus);
        return softplus.get();
    }

    /**
     * @brief Create a new hsigmoid operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* hsigmoid(ov::op::Op* input) {
        auto hsigmoid = std::make_shared<ov::opset5::HSigmoid>(input->output(0));
        operations.push_back(hsigmoid);
        return hsigmoid.get();
    }

    /**
     * @brief Create a new round activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* round_act(ov::op::Op* input) {
        auto round_act =
                std::make_shared<ov::opset5::Round>(input->output(0), ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        operations.push_back(round_act);
        return round_act.get();
    }

    /**
     * @brief Create a new softsign operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* softsign(ov::op::Op* input) {
        auto softsign = std::make_shared<ov::opset9::SoftSign>(input->output(0));
        operations.push_back(softsign);
        return softsign.get();
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
