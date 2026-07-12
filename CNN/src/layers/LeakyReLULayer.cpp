/**
 * @file LeakyReLULayer.cpp
 * @brief Implementation file for the LeakyReLULayer class.
 * Implements the LeakyReLULayer class, responsible for implementing Leaky ReLU activation layers in the CNN model.
 */

#include "LeakyReLULayer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

LeakyReLULayer::LeakyReLULayer(float alpha)
    : _alpha(alpha) {
    _layer_name = "LeakyReLULayer( " + std::to_string(alpha) + " )";
}

std::shared_ptr<Tensor> LeakyReLULayer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("LeakyReLULayer expects exactly one input tensor.");
    }

    // Cache the input tensor for use in the backward pass.
    _input_cache = inputs[0];
    const std::vector<size_t> &input_shape = _input_cache->get_shape();
    
    // Create an output tensor with the same shape as the input.
    auto output = std::make_shared<Tensor>(input_shape);

    float *in_ptr = _input_cache->get_data(); // shape: [batch_size, features]
    float *out_ptr = output->get_data(); // shape: [batch_size, features]

    // Apply the Leaky ReLU function -> f(x) = x if x > 0 else alpha * x
    for (size_t i = 0; i < output->size(); ++i) { // iterates over all elements in the tensor
        // - if the input value is greater than 0, the output is the same as the input (identity function).
        // - if the input value is less than or equal to 0, the output is
        out_ptr[i] = (in_ptr[i] > 0.0f) ? in_ptr[i] : _alpha * in_ptr[i];
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> LeakyReLULayer::backward(std::shared_ptr<Tensor> grad_output) {
    
    // Initialize the input gradient tensor with the same shape as the cached input.
    auto grad_input = std::make_shared<Tensor>(_input_cache->get_shape()); // shape: [batch_size, features]
    
    float *dout_ptr = grad_output->get_data(); // shape: [batch_size, features]
    float *din_ptr = grad_input->get_data(); // shape: [batch_size, features]
    float *cache_ptr = _input_cache->get_data(); // shape: [batch_size, features]

    // Apply the gradient of the Leaky ReLU function -> f'(x) = 1 if x > 0 else alpha
    for (size_t i = 0; i < grad_input->size(); ++i) { // iterates over all elements in the tensor
        // - if the cached input value is greater than 0, the gradient is passed through unchanged.
        // - if the cached input value is less than or equal to 0, the gradient is scaled by alpha.
        din_ptr[i] = dout_ptr[i] * ((cache_ptr[i] > 0.0f) ? 1.0f : _alpha);
    }

    return {grad_input};
}

void LeakyReLULayer::update_weights(std::shared_ptr<Optimizer> optimizer) {
    // No weights to update in LeakyReLULayer
}

std::pair<float*,float*> LeakyReLULayer::get_weights_and_grads() {
    // No weights in LeakyReLULayer, return null pointers
    return {nullptr, nullptr};
}

std::pair<float*,float*> LeakyReLULayer::get_biases_and_grads() {
    // No biases in LeakyReLULayer, return null pointers
    return {nullptr, nullptr};
}