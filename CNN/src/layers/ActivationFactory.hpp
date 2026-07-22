/**
 * @file ActivationFactory.hpp
 * @brief Header file for the activation layer factory.
 * Declares a factory function that creates activation layers by name, allowing
 * the activation function to be selected at runtime (e.g. from the command line).
 */

#ifndef ACTIVATIONFACTORY_HPP
#define ACTIVATIONFACTORY_HPP

#include "ActivationLayer.hpp"
#include <memory>
#include <string>

/**
 * @brief Creates an activation layer from its name.
 *
 * Each call returns a new, independent layer instance. This matters because
 * activation layers cache their forward-pass input for the backward pass, so
 * the same instance must never be reused in multiple positions of the network.
 *
 * @param name The activation name (case-insensitive): "leakyrelu", "relu", "tanh" or "sigmoid".
 * @param alpha The negative slope, used only by "leakyrelu" (ignored by the others).
 * @return An owning pointer to the newly created activation layer.
 * @throws std::invalid_argument if the name does not match any known activation.
 */
std::unique_ptr<ActivationLayer> make_activation(const std::string &name, float alpha = 0.05f);

#endif // ACTIVATIONFACTORY_HPP
