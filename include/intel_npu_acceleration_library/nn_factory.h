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
    ov::OutputVector results;

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
     * @param dst data pointer of the tensor constant
     * @return ov::op::Op*
     */
    ov::op::Op* constant(ov::element::Type_t dtype, std::vector<size_t> shape, const void* dst) {
        auto constant = std::make_shared<ov::opset1::Constant>(dtype, ov::Shape(shape), dst);
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
     * @brief Create a new linear operation
     *
     * @param input matmul lhs input
     * @param weights matmul rhs input, a.k.a. weights
     * @param bias matmul bias input
     * @return ov::op::Op*
     */
    ov::op::Op* linear(ov::op::Op* input, ov::op::Op* weights, ov::op::Op* bias) {
        auto mm_op = matmul(input, weights);
        if (bias != nullptr) {
            return eltwise_add(mm_op, bias);
        }
        return mm_op;
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
     * @brief Create a new average pooling operation
     * @param input pooling input
     * @param strides pooling strides
     * @param pads_begin pooling padding begin
     * @param pads_ends pooling padding end
     * @param kernel pooling kernel
     * @param exclude_pad exclude padding from the average calculation
     * @param rounding_type rounding type
     * @param auto_pad padding type
     * @return ov::op::Op*
     */
    ov::op::Op* average_pooling(ov::op::Op* input, std::vector<size_t> strides, std::vector<size_t> pads_begin,
                                std::vector<size_t> pads_ends, std::vector<size_t> kernel, bool exclude_pad = false,
                                ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
                                ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT) {
        auto pool = std::make_shared<ov::opset1::AvgPool>(input->output(0), ov::Strides(strides), pads_begin, pads_ends,
                                                          kernel, exclude_pad, rounding_type, auto_pad);
        operations.push_back(pool);
        return pool.get();
    }

    /**
     * @brief Create a new adaptive average pooling operation
     * @param input pooling input
     * @param output_shape output shape
     * @return ov::op::Op*
     */
    ov::op::Op* adaptive_average_pool(ov::op::Op* input, ov::op::Op* output_shape) {
        auto pool = std::make_shared<ov::opset8::AdaptiveAvgPool>(input->output(0), output_shape->output(0));
        operations.push_back(pool);
        return pool.get();
    }

    /**
     * @brief Create a new max pooling operation
     * @param input pooling input
     * @param strides pooling strides
     * @param pads_begin pooling padding begin
     * @param pads_ends pooling padding end
     * @param kernel pooling kernel
     * @param exclude_pad exclude padding from the max calculation
     * @param rounding_type rounding type
     * @param auto_pad padding type
     * @return ov::op::Op*
     */
    ov::op::Op* max_pooling(ov::op::Op* input, std::vector<size_t> strides, std::vector<size_t> pads_begin,
                            std::vector<size_t> pads_ends, std::vector<size_t> kernel,
                            ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
                            ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT) {
        auto pool = std::make_shared<ov::opset1::MaxPool>(input->output(0), ov::Strides(strides), pads_begin, pads_ends,
                                                          kernel, rounding_type, auto_pad);
        operations.push_back(pool);
        return pool.get();
    }

    /**
     * @brief Create a new adaptive max pooling operation
     * @param input pooling input
     * @param output_shape output shape
     * @return ov::op::Op*
     */
    ov::op::Op* adaptive_max_pool(ov::op::Op* input, ov::op::Op* output_shape) {
        auto pool = std::make_shared<ov::opset8::AdaptiveMaxPool>(input->output(0), output_shape->output(0),
                                                                  ov::element::i64);
        operations.push_back(pool);
        return pool.get();
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
     * @brief create a new strided slice
     *
     * @param input tensor to be strides.
     * @param begin tensor with begin indices for each dimension.
     * @param end tensor with end indices for each dimension.
     * @param strides tensor with strides for each dimension.
     * @param begin_mask mask for begin indices
     * @param end_mask mask for end indices
     * @return ov::op::Op*
     */
    ov::op::Op* slice(ov::op::Op* input, ov::op::Op* begin, ov::op::Op* end, ov::op::Op* strides,
                      const std::vector<int64_t> begin_mask, const std::vector<int64_t> end_mask) {
        auto reshape = std::make_shared<ov::opset1::StridedSlice>(input->output(0), begin->output(0), end->output(0),
                                                                  strides->output(0), begin_mask, end_mask);
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
     * @brief create a new squeeze operation
     *
     * @param input tensor to be squeezed.
     * @return ov::op::Op*
     */
    ov::op::Op* squeeze(ov::op::Op* input) {
        auto squeeze = std::make_shared<ov::opset1::Squeeze>(input->output(0));
        operations.push_back(squeeze);
        return squeeze.get();
    }

    /**
     * @brief create a new squeeze operation
     *
     * @param input tensor to be squeezed.
     * @param axis tensor with axes to unsqueeze
     * @return ov::op::Op*
     */
    ov::op::Op* unsqueeze(ov::op::Op* input, ov::op::Op* axis) {
        auto unsqueeze = std::make_shared<ov::opset1::Unsqueeze>(input->output(0), axis->output(0));
        operations.push_back(unsqueeze);
        return unsqueeze.get();
    }

    /**
     * @brief create a new concatenation operation
     *
     * @param x1 first concat input node
     * @param x2 second concat input node
     * @param axis axis along which to concatenate the input tensors
     * @return ov::op::Op*
     */
    ov::op::Op* concat(ov::op::Op* x1, ov::op::Op* x2, int64_t axis) {
        auto concat = std::make_shared<ov::opset1::Concat>(std::vector<OVNode>{x1->output(0), x2->output(0)}, axis);
        operations.push_back(concat);
        return concat.get();
    }

    /**
     * @brief create a new reduce max operation
     *
     * @param input operation's input node
     * @param reduction_axes the axis positions to be reduced
     * @param keep_dims if set to 1 it holds axes that are used for reduction
     * @return ov::op::Op*
     */
    ov::op::Op* reduce_max(ov::op::Op* input, ov::op::Op* reduction_axes, bool keep_dims) {
        auto reduce_max =
                std::make_shared<ov::opset1::ReduceMax>(input->output(0), reduction_axes->output(0), keep_dims);
        operations.push_back(reduce_max);
        return reduce_max.get();
    }

    /**
     * @brief create a new reduce mean operation
     *
     * @param input operation's input node
     * @param reduction_axes the axis positions to be reduced
     * @param keep_dims if set to 1 it holds axes that are used for reduction
     * @return ov::op::Op*
     */
    ov::op::Op* reduce_mean(ov::op::Op* input, ov::op::Op* reduction_axes, bool keep_dims) {
        auto reduce_mean =
                std::make_shared<ov::opset1::ReduceMean>(input->output(0), reduction_axes->output(0), keep_dims);
        operations.push_back(reduce_mean);
        return reduce_mean.get();
    }

    /**
     * @brief create a new reduce min operation
     *
     * @param input operation's input node
     * @param reduction_axes the axis positions to be reduced
     * @param keep_dims if set to 1 it holds axes that are used for reduction
     * @return ov::op::Op*
     */
    ov::op::Op* reduce_min(ov::op::Op* input, ov::op::Op* reduction_axes, bool keep_dims) {
        auto reduce_min =
                std::make_shared<ov::opset1::ReduceMin>(input->output(0), reduction_axes->output(0), keep_dims);
        operations.push_back(reduce_min);
        return reduce_min.get();
    }

    /**
     * @brief create a new reduce product operation
     *
     * @param input operation's input node
     * @param reduction_axes the axis positions to be reduced
     * @param keep_dims if set to 1 it holds axes that are used for reduction
     * @return ov::op::Op*
     */
    ov::op::Op* reduce_prod(ov::op::Op* input, ov::op::Op* reduction_axes, bool keep_dims) {
        auto reduce_prod =
                std::make_shared<ov::opset1::ReduceProd>(input->output(0), reduction_axes->output(0), keep_dims);
        operations.push_back(reduce_prod);
        return reduce_prod.get();
    }

    /**
     * @brief create a new reduce sum operation
     *
     * @param input operation's input node
     * @param reduction_axes the axis positions to be reduced
     * @param keep_dims if set to 1 it holds axes that are used for reduction
     * @return ov::op::Op*
     */
    ov::op::Op* reduce_sum(ov::op::Op* input, ov::op::Op* reduction_axes, bool keep_dims) {
        auto reduce_sum =
                std::make_shared<ov::opset1::ReduceSum>(input->output(0), reduction_axes->output(0), keep_dims);
        operations.push_back(reduce_sum);
        return reduce_sum.get();
    }

    /**
     * @brief Create a new absolute activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* abs(ov::op::Op* input) {
        auto abs = std::make_shared<ov::opset1::Abs>(input->output(0));
        operations.push_back(abs);
        return abs.get();
    }

    /**
     * @brief Create a new arccos activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* acos(ov::op::Op* input) {
        auto acos = std::make_shared<ov::opset1::Acos>(input->output(0));
        operations.push_back(acos);
        return acos.get();
    }

    /**
     * @brief Create a new arcsin activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* asin(ov::op::Op* input) {
        auto asin = std::make_shared<ov::opset1::Asin>(input->output(0));
        operations.push_back(asin);
        return asin.get();
    }

    /**
     * @brief Create a new arctan activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* atan(ov::op::Op* input) {
        auto atan = std::make_shared<ov::opset1::Atan>(input->output(0));
        operations.push_back(atan);
        return atan.get();
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
    ov::op::Op* cos(ov::op::Op* input) {
        auto cos = std::make_shared<ov::opset1::Cos>(input->output(0));
        operations.push_back(cos);
        return cos.get();
    }

    /**
     * @brief Create a new cosh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* cosh(ov::op::Op* input) {
        auto cosh = std::make_shared<ov::opset1::Cosh>(input->output(0));
        operations.push_back(cosh);
        return cosh.get();
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
    ov::op::Op* erf(ov::op::Op* input) {
        auto erf = std::make_shared<ov::opset1::Erf>(input->output(0));
        operations.push_back(erf);
        return erf.get();
    }

    /**
     * @brief Create a new exp activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* exp(ov::op::Op* input) {
        auto exp = std::make_shared<ov::opset1::Exp>(input->output(0));
        operations.push_back(exp);
        return exp.get();
    }

    /**
     * @brief Create a new floor activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* floor(ov::op::Op* input) {
        auto floor = std::make_shared<ov::opset1::Floor>(input->output(0));
        operations.push_back(floor);
        return floor.get();
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
    ov::op::Op* gelu(ov::op::Op* input, ov::op::GeluApproximationMode mode) {
        auto gelu = std::make_shared<ov::opset7::Gelu>(input->output(0), mode);
        operations.push_back(gelu);
        return gelu.get();
    }

    /**
     * @brief Create a new natural log operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* log(ov::op::Op* input) {
        auto log = std::make_shared<ov::opset1::Log>(input->output(0));
        operations.push_back(log);
        return log.get();
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
    ov::op::Op* sin(ov::op::Op* input) {
        auto sin = std::make_shared<ov::opset1::Sin>(input->output(0));
        operations.push_back(sin);
        return sin.get();
    }

    /**
     * @brief Create a new sinh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* sinh(ov::op::Op* input) {
        auto sinh = std::make_shared<ov::opset1::Sinh>(input->output(0));
        operations.push_back(sinh);
        return sinh.get();
    }

    /**
     * @brief Create a new sqrt activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* sqrt(ov::op::Op* input) {
        auto sqrt = std::make_shared<ov::opset1::Sqrt>(input->output(0));
        operations.push_back(sqrt);
        return sqrt.get();
    }

    /**
     * @brief Create a new tan activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* tan(ov::op::Op* input) {
        auto tan = std::make_shared<ov::opset1::Tan>(input->output(0));
        operations.push_back(tan);
        return tan.get();
    }

    /**
     * @brief Create a new tanh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* tanh(ov::op::Op* input) {
        auto tanh = std::make_shared<ov::opset1::Tanh>(input->output(0));
        operations.push_back(tanh);
        return tanh.get();
    }

    /**
     * @brief Create a new arccosh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* acosh(ov::op::Op* input) {
        auto acosh = std::make_shared<ov::opset4::Acosh>(input->output(0));
        operations.push_back(acosh);
        return acosh.get();
    }

    /**
     * @brief Create a new arcsinh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* asinh(ov::op::Op* input) {
        auto asinh = std::make_shared<ov::opset4::Asinh>(input->output(0));
        operations.push_back(asinh);
        return asinh.get();
    }

    /**
     * @brief Create a new arctanh activation operation
     *
     * @param input operation's input node
     * @return ov::op::Op*
     */
    ov::op::Op* atanh(ov::op::Op* input) {
        auto atanh = std::make_shared<ov::opset4::Atanh>(input->output(0));
        operations.push_back(atanh);
        return atanh.get();
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
    ov::op::Op* round(ov::op::Op* input) {
        auto round = std::make_shared<ov::opset5::Round>(input->output(0), ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        operations.push_back(round);
        return round.get();
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
    ov::op::Op* softmax(ov::op::Op* input, int64_t axis = -1) {
        auto smax = std::make_shared<ov::opset8::Softmax>(input->output(0), axis);
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
        if (attn_mask == nullptr) {
            auto sdpa = std::make_shared<ov::opset13::ScaledDotProductAttention>(query->output(0), key->output(0),
                                                                                 value->output(0), is_causal);

            operations.push_back(sdpa);
            return sdpa.get();
        } else {
            auto sdpa = std::make_shared<ov::opset13::ScaledDotProductAttention>(
                    query->output(0), key->output(0), value->output(0), attn_mask->output(0), is_causal);

            operations.push_back(sdpa);
            return sdpa.get();
        }
    }

    /**
     * @brief Create a new L2 normalization operation
     *
     * @param data operation's input node
     * @param axes node indicating axes along which reduction is calculated
     * @param eps the epsilon added to L2 norm
     * @return ov::op::Op*
     */
    ov::op::Op* normL2(ov::op::Op* data, ov::op::Op* axes, float eps) {
        auto normL2 =
                std::make_shared<ov::opset1::NormalizeL2>(data->output(0), axes->output(0), eps, ov::op::EpsMode::MAX);
        operations.push_back(normL2);
        return normL2.get();
    }

    /**
     * @brief Create a new power operation
     *
     * @param x1 operation's input node
     * @param x2 operation's input node of the exponent
     * @param auto_broadcast auto broadcast specification
     * @return ov::op::Op*
     */
    ov::op::Op* power(ov::op::Op* x1, ov::op::Op* x2, ov::op::AutoBroadcastType auto_broadcast) {
        auto power = std::make_shared<ov::opset1::Power>(x1->output(0), x2->output(0), auto_broadcast);
        operations.push_back(power);
        return power.get();
    }

    /**
     * @brief Create a new prelu operation
     *
     * @param x1 operation's input node
     * @param slope operation's slope
     * @return ov::op::Op*
     */
    ov::op::Op* prelu(ov::op::Op* x1, ov::op::Op* slope) {
        auto power = std::make_shared<ov::op::v0::PRelu>(x1->output(0), slope->output(0));
        operations.push_back(power);
        return power.get();
    }

    /**
     * @brief Create a new log softmax operation
     *
     * @param input operation's input node
     * @param axis the axis position on which to calculate the LogSoftmax
     * @return ov::op::Op*
     */
    ov::op::Op* log_softmax(ov::op::Op* input, int64_t axis) {
        auto log_softmax = std::make_shared<ov::opset5::LogSoftmax>(input->output(0), axis);
        operations.push_back(log_softmax);
        return log_softmax.get();
    }

    void result(ov::op::Op* op) {
        auto res = std::make_shared<ov::opset8::Result>(op->output(0));
        results.push_back(res);
    }

    /**
     * @brief Compile the model
     *
     * @param result the last operation in the network. Must have a [batch, output_channel] shape
     */
    void compile() {
        model = std::make_shared<ov::Model>(results, parameters, "NNFactory");

        compile_model(device);
    }
};

}  // namespace intel_npu_acceleration_library
