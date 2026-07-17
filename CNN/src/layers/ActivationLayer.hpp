/**
 * @file ActivationLayer.hpp
 * @brief Header file for the ActivationLayer class.
 * Defines the ActivationLayer class, an abstract base class for all element-wise
 * activation layers in the CNN model. It implements the common forward/backward
 * machinery once, so that concrete activations only need to provide the scalar
 * function and its derivative.
 */

#ifndef ACTIVATIONLAYER_HPP
#define ACTIVATIONLAYER_HPP

#include "Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <memory>
#include <vector>

/**
 * @class ActivationLayer
 * @brief Abstract base class for element-wise activation layers.
 *
 * Concrete activations (e.g. ReLU, Leaky ReLU, Tanh, Sigmoid) derive from this
 * class and implement only activate() and derivative(). The forward pass, the
 * backward pass, the input caching and the no-op weight-related methods are
 * shared here, avoiding duplication across activation implementations.
 */
class ActivationLayer : public Layer {
    protected:
    // Cache the input tensor from the forward pass to use in the backward pass for gradient computation.
    std::shared_ptr<Tensor> _input_cache;

    /**
     * @brief Applies the activation function to a single value.
     * @param x The input value.
     * @return f(x).
     */
    virtual float activate(float x) const = 0;

    /**
     * @brief Computes the derivative of the activation function at a single value.
     * @param x The input value (from the cached forward-pass input).
     * @return f'(x).
     */
    virtual float derivative(float x) const = 0;

    public:
    /**
     * @brief Performs the forward pass by applying the activation element-wise.
     * @param inputs A vector containing a single input Tensor.
     * @return A shared pointer to the output Tensor after applying the activation.
     */
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override;

    /**
     * @brief Performs the backward pass (gradient computation) for the activation layer.
     * @param grad_output The gradient of the loss w.r.t. the layer's output.
     * @return A vector containing a single gradient tensor.
     */
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override;

    /**
     * @brief No-operation method for weight updates.
     * Since activation layers have no learnable parameters, this method does nothing.
     * It is implemented to satisfy the interface requirements of the Layer base class.
     * @param optimizer The optimizer to use for weight updates (not used in this layer).
     */
    void update_weights(std::shared_ptr<Optimizer> optimizer) override;

    /** @brief Gets the weights and their gradients.
     * @return Since activation layers have no learnable parameters, this method returns a pair of null pointers.
     */
    std::pair<float*,float*> get_weights_and_grads() override;

    /** @brief Gets the biases and their gradients.
     * @return Since activation layers have no learnable parameters, this method returns a pair of null pointers.
     */
    std::pair<float*,float*> get_biases_and_grads() override;
};

#endif // ACTIVATIONLAYER_HPP
