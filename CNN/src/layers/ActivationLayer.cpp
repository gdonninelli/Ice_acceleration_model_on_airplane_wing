/**
 * @file ActivationLayer.cpp
 * @brief Implementation file for the ActivationLayer class.
 * Implements the shared forward/backward machinery for all element-wise
 * activation layers in the CNN model.
 */

#include "ActivationLayer.hpp"
#include <stdexcept>

std::shared_ptr<Tensor> ActivationLayer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    if (inputs.size() != 1 || !inputs[0]) {
        throw std::invalid_argument(_layer_name + " expects exactly one non-null input tensor.");
    }

    // Cache the input tensor for use in the backward pass.
    _input_cache = inputs[0];
    const std::vector<size_t> &input_shape = _input_cache->get_shape();

    // Create an output tensor with the same shape as the input.
    auto output = std::make_shared<Tensor>(input_shape);

    float *in_ptr = _input_cache->get_data(); // shape: [batch_size, features]
    float *out_ptr = output->get_data(); // shape: [batch_size, features]

    // Apply the activation function element-wise -> out[i] = f(in[i])
    for (size_t i = 0; i < output->size(); ++i) { // iterates over all elements in the tensor
        out_ptr[i] = activate(in_ptr[i]);
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> ActivationLayer::backward(std::shared_ptr<Tensor> grad_output) {

    if (!grad_output || !_input_cache ||
        grad_output->get_shape() != _input_cache->get_shape()) {
        throw std::invalid_argument(
            _layer_name + " backward requires a matching gradient and prior forward pass.");
    }

    // Initialize the input gradient tensor with the same shape as the cached input.
    auto grad_input = std::make_shared<Tensor>(_input_cache->get_shape()); // shape: [batch_size, features]

    float *dout_ptr = grad_output->get_data(); // shape: [batch_size, features]
    float *din_ptr = grad_input->get_data(); // shape: [batch_size, features]
    float *cache_ptr = _input_cache->get_data(); // shape: [batch_size, features]

    // Apply the chain rule element-wise -> din[i] = dout[i] * f'(in[i])
    for (size_t i = 0; i < grad_input->size(); ++i) { // iterates over all elements in the tensor
        din_ptr[i] = dout_ptr[i] * derivative(cache_ptr[i]);
    }

    return {grad_input};
}
