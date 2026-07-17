/**
 * @file TanhLayer.hpp
 * @brief Header file for the TanhLayer class.
 * Defines the TanhLayer class, responsible for implementing hyperbolic tangent activation layers in the CNN model.
 */

#ifndef TANHLAYER_HPP
#define TANHLAYER_HPP

#include "ActivationLayer.hpp"

/**
 * @class TanhLayer
 * @brief Represents a hyperbolic tangent (Tanh) activation layer.
 * Applies f(x) = tanh(x), element-wise, squashing values into the range (-1, 1).
 */
class TanhLayer : public ActivationLayer {
    protected:
    /**
     * @brief Applies the Tanh function -> f(x) = tanh(x)
     * @param x The input value.
     * @return f(x).
     */
    float activate(float x) const override;

    /**
     * @brief Computes the Tanh derivative -> f'(x) = 1 - tanh(x)^2
     * @param x The input value.
     * @return f'(x).
     */
    float derivative(float x) const override;

    public:
    /**
     * @brief Constructor for the TanhLayer.
     */
    TanhLayer();
};

#endif // TANHLAYER_HPP
