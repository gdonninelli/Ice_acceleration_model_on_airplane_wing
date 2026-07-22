/**
 * @file FlattenLayer.cpp
 * @brief Implementation file for the FlattenLayer class.
 * Implements the FlattenLayer class, responsible for implementing flatten layers in the CNN model.
 */

#include "FlattenLayer.hpp"
#include <algorithm>
#include <stdexcept>

FlattenLayer::FlattenLayer() {
    _layer_name = "FlattenLayer";
}

std::shared_ptr<Tensor> FlattenLayer::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1 || !inputs[0]) {
        throw std::invalid_argument("FlattenLayer expects exactly one non-null input tensor.");
    }

    // Cache the input shape for use in the backward pass.
    const std::shared_ptr<Tensor>& input = inputs[0];
    _input_shape_cache = input->get_shape();

    if (_input_shape_cache.size() != 4) {
        throw std::invalid_argument("FlattenLayer forward expects a 4D tensor [batch, channels, height, width].");
    }

    // Extract dimensions from the input shape.
    const size_t batch_size = _input_shape_cache[0];
    const size_t channels = _input_shape_cache[1];
    const size_t height = _input_shape_cache[2];
    const size_t width = _input_shape_cache[3];
    // Calculate the total number of features after flattening.
    const size_t flat_features = channels * height * width;

    // Create the output tensor with shape [batch_size, flat_features]
    auto output = std::make_shared<Tensor>(std::vector<size_t>{batch_size, flat_features});

    // Copy data from the input tensor to the output tensor.
    float* in_ptr = input->get_data();
    float* out_ptr = output->get_data();
    
    // Since the flatten operation is a reshape, we can copy the data directly without any transformation.
    std::copy(in_ptr, in_ptr + input->size(), out_ptr);

    return output;
}

std::vector<std::shared_ptr<Tensor>> FlattenLayer::backward(std::shared_ptr<Tensor> grad_output) {
    if (!grad_output) {
        throw std::invalid_argument("FlattenLayer backward received null grad_output.");
    }

    if (_input_shape_cache.empty()) {
        throw std::invalid_argument("FlattenLayer backward called before forward.");
    }

    const std::vector<size_t> grad_shape = grad_output->get_shape();
    if (grad_shape.size() != 2) {
        throw std::invalid_argument("FlattenLayer backward expects a 2D gradient tensor.");
    }

    // Extract dimensions from the cached input shape.
    const size_t batch_size = _input_shape_cache[0];
    const size_t flat_features = _input_shape_cache[1] * _input_shape_cache[2] * _input_shape_cache[3];

    if (grad_shape[0] != batch_size || grad_shape[1] != flat_features) {
        throw std::invalid_argument("FlattenLayer backward gradient shape mismatch.");
    }

    // Create the input gradient tensor with the same shape as the cached input.
    auto grad_input = std::make_shared<Tensor>(_input_shape_cache);
    
    // Copy data from the grad_output tensor to the grad_input tensor.
    float* grad_out_ptr = grad_output->get_data();
    float* grad_in_ptr = grad_input->get_data();

    // Since the flatten operation is a reshape, we can copy the data directly without any transformation.
    std::copy(grad_out_ptr, grad_out_ptr + grad_output->size(), grad_in_ptr);

    return {grad_input};
}
