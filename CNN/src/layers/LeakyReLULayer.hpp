/**
 * @file LeakyReLULayer.hpp
 * @brief Header file for the LeakyReLULayer class.
 * Defines the LeakyReLULayer class, responsible for implementing Leaky ReLU activation layers in the CNN model.
 */

#ifndef LEAKYRELULAYER_HPP
#define LEAKYRELULAYER_HPP

#include "Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <memory> // TODO check if we need this include
#include <vector>

/**
 * @class LeakyReLULayer
 * @brief Represents a Leaky Rectified Linear Unit (Leaky ReLU) activation layer.
 */
class LeakyReLULayer : public Layer {
    private:
    float _alpha; // slope for negative input values

    // Cache the input tensor from the forward pass to use in the backward pass for gradient computation.
    std::shared_ptr<Tensor> _input_cache;

    public:
    /**
     * @brief Constructor for the LeakyReLULayer.
     * @param alpha The slope for negative input values.
     */
    LeakyReLULayer(float alpha = 0.05f);

    /**
     * @brief Performs the forward pass through the Leaky ReLU layer.
     * @param inputs  A vector containing a single input Tensor.
     * @return A shared pointer to the output Tensor after applying Leaky ReLU.
     */
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override;
    
    /**
     * @brief Performs the backward pass (gradient computation) for the Leaky ReLU layer.
     * @param grad_output The gradient of the loss w.r.t. the layer's output.
     * @return A vector containing a single gradient tensor.
     */
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override;

    /**
     * @brief No-operation method for weight updates.
     * Since the Leaky ReLU layer has no learnable parameters, this method does nothing.
     * It is implemented to satisfy the interface requirements of the Layer base class.
     * @param optimizer The optimizer to use for weight updates (not used in this layer).
     */
    void update_weights(std::shared_ptr<Optimizer> optimizer) override;
    
    /** @brief Gets the weights and their gradients.
     * @return Since the Leaky ReLU layer has no learnable parameters, this method returns a pair of null pointers.
     */
    std::pair<float*,float*> get_weights_and_grads() override;

    /** @brief Gets the biases and their gradients.
     * @return Since the Leaky ReLU layer has no learnable parameters, this method returns a pair of null pointers.
     */
    std::pair<float*,float*> get_biases_and_grads() override;
};

#endif // LEAKYRELULAYER_HPP