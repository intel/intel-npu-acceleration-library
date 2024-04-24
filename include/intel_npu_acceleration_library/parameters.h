//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <memory>
#include "intel_npu_acceleration_library/common.h"
#include "intel_npu_acceleration_library/conversion.h"

namespace intel_npu_acceleration_library {

/**
 * @brief A class representing a generic tensor shape
 *
 */
class Shape {
private:
    std::vector<size_t> dimensions;

public:
    /**
     * @brief Construct a new Shape object
     *
     * @param dims : a list of integers representing each dimension size
     */
    Shape(std::initializer_list<size_t> dims): dimensions(dims) {
    }

    /**
     * @brief Overload of the operator []. Return the dimension at index idx
     *
     * @param idx
     * @return const size_t&
     */
    const size_t& operator[](int idx) {
        return dimensions[idx];
    }

    /**
     * @brief Get the number of element of the tensor
     *
     * @return size_t
     */
    size_t get_size() const {
        return std::accumulate(std::begin(dimensions), std::end(dimensions), static_cast<size_t>(1),
                               std::multiplies<size_t>());
    }
};

/**
 * @brief The Parameter class represents a generic NN parameter
 *
 */
class Parameter {
protected:
    /// @brief Parameter shape
    Shape shape;

private:
    void* data;
    bool quantized;

public:
    /**
     * @brief Construct a new Parameter object
     *
     * @param shape parameter shape
     */
    Parameter(Shape shape): shape(shape), data(nullptr) {
    }

    /**
     * @brief Construct a new Parameter object from fp16 data pointer
     *
     * @param _data fp16 parameter data pointer
     * @param shape parameter shape
     */
    Parameter(half_ptr _data, Shape shape): shape(shape), quantized(false) {
        data = static_cast<void*>(_data);
    }

    /**
     * @brief Construct a new Parameter object from int8 data pointer
     *
     * @param _data int8 parameter data pointer
     * @param shape parameter shape
     */
    Parameter(int8_t* _data, Shape shape): shape(shape), quantized(true) {
        data = static_cast<void*>(_data);
    }

    /**
     * @brief Construct a new Parameter object from uint8 data pointer
     *
     * @param _data uint8 parameter data pointer
     * @param shape parameter shape
     */
    Parameter(uint8_t* _data, Shape shape): shape(shape), quantized(true) {
        data = static_cast<void*>(_data);
    }

    /**
     * @brief Get the size of the parameter
     *
     * @return size_t
     */
    size_t get_size() const {
        return shape.get_size();
    }

    /**
     * @brief Set the Parameter data to the memory location dst of size
     *
     * @param dst destination memory location
     * @param size destination memory location size
     */
    virtual void set_data(void* dst, size_t size) {
        memcpy(dst, data, size);
    }

    /**
     * @brief Destroy the Parameter object
     *
     */
    virtual ~Parameter() {
    }
};

/**
 * @brief The ParameterWithConversion represent a generic quantized NN parameter where the conversion to fp16 is
 * performed explicitly on CPU. The conversion equation is Y_float = Scale * float(data)
 *
 */
class ParameterWithConversion : public Parameter {
private:
    int8_t* data;
    float* scale;

public:
    /**
     * @brief Construct a new ParameterWithConversion object from int8 data, float scale and shape.
     *
     * @param data int8 data buffer
     * @param scale float per output channel scale
     * @param shape parameter shape
     */
    ParameterWithConversion(int8_t* data, float* scale, Shape shape): Parameter(shape), data(data), scale(scale) {
    }

    /**
     * @brief Set the Parameter data to the memory location dst of size. Here is where the conversion from int to float
     * is performed
     *
     * @param dst destination memory location
     * @param size destination memory location size
     */
    virtual void set_data(void* dst, size_t size) {
        (void)size;
        intel_npu_acceleration_library::to_fp16(data, scale, static_cast<half_ptr>(dst), shape[1], shape[0], 1);
    }
};

/**
 * @brief The class Parameters represents a list of NN parameter for a NPU kernel
 *
 */
class Parameters {
private:
    std::vector<std::shared_ptr<Parameter>> parameters;

public:
    /**
     * @brief Add a new float16 parameter
     *
     * @param data
     * @param shape
     * @return Parameters&
     */
    Parameters& add_parameter(half_ptr data, Shape shape) {
        parameters.push_back(std::make_shared<Parameter>(data, shape));
        return *this;
    }

    /**
     * @brief Add a new int8 parameter, provide also the scale
     *
     * @param data
     * @param scale
     * @param shape
     * @return Parameters&
     */
    Parameters& add_parameter(int8_t* data, half_ptr scale, Shape shape) {
        parameters.push_back(std::make_shared<Parameter>(data, shape));
        parameters.push_back(std::make_shared<Parameter>(scale, shape));
        return *this;
    }

    /**
     * @brief Add a new int4 parameter, provide also the scale
     *
     * @param data
     * @param scale
     * @param shape
     * @return Parameters&
     */
    Parameters& add_parameter(uint8_t* data, half_ptr scale, Shape shape) {
        parameters.push_back(std::make_shared<Parameter>(data, shape));
        parameters.push_back(std::make_shared<Parameter>(scale, shape));
        return *this;
    }

    /**
     * @brief Add a new int8 parameter with explicit CPU conversion
     *
     * @param data
     * @param scale
     * @param shape
     * @return Parameters&
     */
    Parameters& add_parameter(int8_t* data, float* scale, Shape shape) {
        parameters.push_back(std::make_shared<ParameterWithConversion>(data, scale, shape));
        return *this;
    }

    /**
     * @brief Get the parameters
     *
     * @return auto
     */
    auto& get_parameters() {
        return parameters;
    }
};

}  // namespace intel_npu_acceleration_library