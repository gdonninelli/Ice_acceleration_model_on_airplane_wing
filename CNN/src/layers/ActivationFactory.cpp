/**
 * @file ActivationFactory.cpp
 * @brief Implementation file for the activation layer factory.
 * Implements the factory function that creates activation layers by name.
 */

#include "ActivationFactory.hpp"
#include "LeakyReLULayer.hpp"
#include "ReLULayer.hpp"
#include "SigmoidLayer.hpp"
#include "TanhLayer.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

std::unique_ptr<ActivationLayer> make_activation(const std::string &name, float alpha) {
    // Normalize the name to lowercase so the lookup is case-insensitive.
    std::string key = name;
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (key == "leakyrelu") {
        return std::make_unique<LeakyReLULayer>(alpha);
    }
    if (key == "relu") {
        return std::make_unique<ReLULayer>();
    }
    if (key == "tanh") {
        return std::make_unique<TanhLayer>();
    }
    if (key == "sigmoid") {
        return std::make_unique<SigmoidLayer>();
    }

    throw std::invalid_argument("Unknown activation '" + name +
                                "'. Valid options: leakyrelu, relu, tanh, sigmoid.");
}
