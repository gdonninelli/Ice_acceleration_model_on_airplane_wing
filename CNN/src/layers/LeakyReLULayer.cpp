/**
 * @file LeakyReLULayer.cpp
 * @brief Implementation file for the LeakyReLULayer class.
 * Implements the LeakyReLULayer class, responsible for implementing Leaky ReLU activation layers in the CNN model.
 */

#include "LeakyReLULayer.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

LeakyReLULayer::LeakyReLULayer(float alpha)
    : _alpha(alpha) {
    if (!std::isfinite(_alpha) || _alpha < 0.0f) {
        throw std::invalid_argument(
            "LeakyReLU alpha must be finite and non-negative.");
    }
    _layer_name = "LeakyReLULayer( " + std::to_string(alpha) + " )";
}

float LeakyReLULayer::activate(float x) const {
    // - if the input value is greater than 0, the output is the same as the input (identity function).
    // - if the input value is less than or equal to 0, the output is scaled by alpha.
    return (x > 0.0f) ? x : _alpha * x;
}

float LeakyReLULayer::derivative(float x) const {
    // - if the input value is greater than 0, the gradient is passed through unchanged.
    // - if the input value is less than or equal to 0, the gradient is scaled by alpha.
    return (x > 0.0f) ? 1.0f : _alpha;
}
