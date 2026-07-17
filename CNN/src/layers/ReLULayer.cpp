/**
 * @file ReLULayer.cpp
 * @brief Implementation file for the ReLULayer class.
 * Implements the ReLULayer class, responsible for implementing ReLU activation layers in the CNN model.
 */

#include "ReLULayer.hpp"

ReLULayer::ReLULayer() {
    _layer_name = "ReLULayer";
}

float ReLULayer::activate(float x) const {
    // - if the input value is greater than 0, the output is the same as the input (identity function).
    // - if the input value is less than or equal to 0, the output is zero.
    return (x > 0.0f) ? x : 0.0f;
}

float ReLULayer::derivative(float x) const {
    // - if the input value is greater than 0, the gradient is passed through unchanged.
    // - if the input value is less than or equal to 0, the gradient is zeroed out.
    return (x > 0.0f) ? 1.0f : 0.0f;
}
