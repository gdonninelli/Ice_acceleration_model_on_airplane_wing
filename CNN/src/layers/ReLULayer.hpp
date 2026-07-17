/**
 * @file ReLULayer.hpp
 * @brief Header file for the ReLULayer class.
 * Defines the ReLULayer class, responsible for implementing ReLU activation layers in the CNN model.
 */

#ifndef RELULAYER_HPP
#define RELULAYER_HPP

#include "ActivationLayer.hpp"

/**
 * @class ReLULayer
 * @brief Represents a Rectified Linear Unit (ReLU) activation layer.
 * Applies f(x) = max(0, x), element-wise.
 */
class ReLULayer : public ActivationLayer {
    protected:
    /**
     * @brief Applies the ReLU function -> f(x) = max(0, x)
     * @param x The input value.
     * @return f(x).
     */
    float activate(float x) const override;

    /**
     * @brief Computes the ReLU derivative -> f'(x) = 1 if x > 0 else 0
     * @param x The input value.
     * @return f'(x).
     */
    float derivative(float x) const override;

    public:
    /**
     * @brief Constructor for the ReLULayer.
     */
    ReLULayer();
};

#endif // RELULAYER_HPP
