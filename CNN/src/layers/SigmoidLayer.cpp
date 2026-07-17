/**
 * @file SigmoidLayer.cpp
 * @brief Implementation file for the SigmoidLayer class.
 * Implements the SigmoidLayer class, responsible for implementing sigmoid activation layers in the CNN model.
 */

#include "SigmoidLayer.hpp"
#include <cmath>

SigmoidLayer::SigmoidLayer() {
    _layer_name = "SigmoidLayer";
}

float SigmoidLayer::activate(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

float SigmoidLayer::derivative(float x) const {
    // The derivative of the sigmoid is f(x) * (1 - f(x)).
    const float s = 1.0f / (1.0f + std::exp(-x));
    return s * (1.0f - s);
}
