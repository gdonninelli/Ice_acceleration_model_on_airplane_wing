/**
 * @file LeakyReLULayer.hpp
 * @brief Header file for the LeakyReLULayer class.
 * Defines the LeakyReLULayer class, responsible for implementing Leaky ReLU activation layers in the CNN model.
 */

#ifndef LEAKYRELULAYER_HPP
#define LEAKYRELULAYER_HPP

#include "ActivationLayer.hpp"

/**
 * @class LeakyReLULayer
 * @brief Represents a Leaky Rectified Linear Unit (Leaky ReLU) activation layer.
 * Applies f(x) = x if x > 0 else alpha * x, element-wise.
 */
class LeakyReLULayer : public ActivationLayer {
    private:
    float _alpha; // slope for negative input values

    protected:
    /**
     * @brief Applies the Leaky ReLU function -> f(x) = x if x > 0 else alpha * x
     * @param x The input value.
     * @return f(x).
     */
    float activate(float x) const override;

    /**
     * @brief Computes the Leaky ReLU derivative -> f'(x) = 1 if x > 0 else alpha
     * @param x The input value.
     * @return f'(x).
     */
    float derivative(float x) const override;

    public:
    /**
     * @brief Constructor for the LeakyReLULayer.
     * @param alpha The slope for negative input values.
     */
    LeakyReLULayer(float alpha = 0.05f);
};

#endif // LEAKYRELULAYER_HPP
