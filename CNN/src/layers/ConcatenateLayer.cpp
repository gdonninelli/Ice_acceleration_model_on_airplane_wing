/**
 * @file ConcatenateLayer.cpp
 * @brief Implementation file for the ConcatenateLayer class.
 * Implements the ConcatenateLayer class, responsible for implementing concatenate layers in the CNN model.
 */

#include "ConcatenateLayer.hpp"
#include <algorithm>
#include <stdexcept>

ConcatenateLayer::ConcatenateLayer()
    : _batch_size_cache(0), _features_first_cache(0), _features_second_cache(0) {
    _layer_name = "ConcatenateLayer";
}

std::shared_ptr<Tensor> ConcatenateLayer::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 2 || !inputs[0] || !inputs[1]) {
        throw std::invalid_argument("ConcatenateLayer expects exactly two non-null input tensors.");
    }

    // Get shape of both input tensors
    const std::vector<size_t> shape_a = inputs[0]->get_shape(); // shape: [batch, features]
    const std::vector<size_t> shape_b = inputs[1]->get_shape(); // shape: [batch, 2]

    if (shape_a.size() != 2 || shape_b.size() != 2) {
        throw std::invalid_argument("ConcatenateLayer forward expects two 2D tensors.");
    }

    if (shape_a[0] != shape_b[0]) {
        throw std::invalid_argument("ConcatenateLayer requires matching batch sizes.");
    }

    if (shape_b[1] != 2) {
        throw std::invalid_argument("ConcatenateLayer expects metadata tensor with exactly 2 features (AoA, Re).");
    }

    // Store dimensions in cache for use in backward pass
    _batch_size_cache = shape_a[0];
    _features_first_cache = shape_a[1];
    _features_second_cache = shape_b[1];

    // Create output tensor with shape [batch, features_a + features_b]
    auto output = std::make_shared<Tensor>(
        std::vector<size_t>{_batch_size_cache, _features_first_cache + _features_second_cache});

    // Get raw pointers to the data for efficient copying
    float* out_ptr = output->get_data();
    float* a_ptr = inputs[0]->get_data();
    float* b_ptr = inputs[1]->get_data();

    // Concatenate the two input tensors along the feature dimension
    for (size_t n = 0; n < _batch_size_cache; ++n) { // iterate over batch
        // Calculate row offsets for input and output tensors.
        const size_t a_row_offset = n * _features_first_cache;
        const size_t b_row_offset = n * _features_second_cache;
        const size_t out_row_offset = n * (_features_first_cache + _features_second_cache);

        // Copy features from the first input tensor
        std::copy(a_ptr + a_row_offset,
                  a_ptr + a_row_offset + _features_first_cache,
                  out_ptr + out_row_offset);

        // Copy features from the second input tensor
        std::copy(b_ptr + b_row_offset,
                  b_ptr + b_row_offset + _features_second_cache,
                  out_ptr + out_row_offset + _features_first_cache);
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> ConcatenateLayer::backward(std::shared_ptr<Tensor> grad_output) {
    if (!grad_output) {
        throw std::invalid_argument("ConcatenateLayer backward received null grad_output.");
    }
    if (_batch_size_cache == 0 && (_features_first_cache + _features_second_cache) == 0) {
        throw std::invalid_argument("ConcatenateLayer backward called before forward.");
    }

    const std::vector<size_t> grad_shape = grad_output->get_shape();
    if (grad_shape.size() != 2) {
        throw std::invalid_argument("ConcatenateLayer backward expects a 2D gradient tensor.");
    }

    const size_t total_features = _features_first_cache + _features_second_cache;
    if (grad_shape[0] != _batch_size_cache || grad_shape[1] != total_features) {
        throw std::invalid_argument("ConcatenateLayer backward gradient shape mismatch.");
    }

    // Create gradient tensors for both inputs
    auto grad_first = std::make_shared<Tensor>(std::vector<size_t>{_batch_size_cache, _features_first_cache});
    auto grad_second = std::make_shared<Tensor>(std::vector<size_t>{_batch_size_cache, _features_second_cache});

    // Get raw pointers to the data for efficient copying
    float* grad_out_ptr = grad_output->get_data();
    float* grad_first_ptr = grad_first->get_data();
    float* grad_second_ptr = grad_second->get_data();

    // Gradient splitting operation
    for (size_t n = 0; n < _batch_size_cache; ++n) { // iterate over batch
        // Calculate row offsets for input, output, and gradient tensors.
        const size_t out_row_offset = n * total_features;
        const size_t first_row_offset = n * _features_first_cache;
        const size_t second_row_offset = n * _features_second_cache;

        // Copy gradients for the first input tensor
        std::copy(grad_out_ptr + out_row_offset,
                  grad_out_ptr + out_row_offset + _features_first_cache,
                  grad_first_ptr + first_row_offset);

        // Copy gradients for the second input tensor
        std::copy(grad_out_ptr + out_row_offset + _features_first_cache,
                  grad_out_ptr + out_row_offset + total_features,
                  grad_second_ptr + second_row_offset);
    }

    return {grad_first, grad_second};
}

void ConcatenateLayer::update_weights(std::shared_ptr<Optimizer> optimizer) {
    // No weights to update in the ConcatenateLayer.
}

std::pair<float*, float*> ConcatenateLayer::get_weights_and_grads() {
    // No weights in ConcatenateLayer, return null pointers.
    return {nullptr, nullptr};
}

std::pair<float*, float*> ConcatenateLayer::get_biases_and_grads() {
    // No biases in ConcatenateLayer, return null pointers.
    return {nullptr, nullptr};
}