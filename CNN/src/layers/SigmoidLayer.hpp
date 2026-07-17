/**
 * @file SigmoidLayer.hpp
 * @brief Header file for the SigmoidLayer class.
 * Defines the SigmoidLayer class, responsible for implementing sigmoid activation layers in the CNN model.
 */

#ifndef SIGMOIDLAYER_HPP
#define SIGMOIDLAYER_HPP

#include "ActivationLayer.hpp"

/**
 * @class SigmoidLayer
 * @brief Represents a sigmoid activation layer.
 * Applies f(x) = 1 / (1 + exp(-x)), element-wise, squashing values into the range (0, 1).
 */
class SigmoidLayer : public ActivationLayer {
    protected:
    /**
     * @brief Applies the sigmoid function -> f(x) = 1 / (1 + exp(-x))
     * @param x The input value.
     * @return f(x).
     */
    float activate(float x) const override;

    /**
     * @brief Computes the sigmoid derivative -> f'(x) = f(x) * (1 - f(x))
     * @param x The input value.
     * @return f'(x).
     */
    float derivative(float x) const override;

    public:
    /**
     * @brief Constructor for the SigmoidLayer.
     */
    SigmoidLayer();
};

#endif // SIGMOIDLAYER_HPP
