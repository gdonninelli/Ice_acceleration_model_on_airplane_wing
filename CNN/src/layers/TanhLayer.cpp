/**
 * @file TanhLayer.cpp
 * @brief Implementation file for the TanhLayer class.
 * Implements the TanhLayer class, responsible for implementing hyperbolic tangent activation layers in the CNN model.
 */

#include "TanhLayer.hpp"
#include <cmath>

TanhLayer::TanhLayer() {
    _layer_name = "TanhLayer";
}

float TanhLayer::activate(float x) const {
    return std::tanh(x);
}

float TanhLayer::derivative(float x) const {
    // The derivative of tanh(x) is 1 - tanh(x)^2.
    const float t = std::tanh(x);
    return 1.0f - t * t;
}
