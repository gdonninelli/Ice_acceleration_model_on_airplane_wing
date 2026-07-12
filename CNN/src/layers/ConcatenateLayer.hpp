/**
 * @file ConcatenateLayer.hpp
 * @brief Header file for the ConcatenateLayer class.
 * Declares the ConcatenateLayer class, responsible for implementing concatenate layers in the CNN model.
 */

#ifndef CONCATENATELAYER_HPP
#define CONCATENATELAYER_HPP

#include "Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <memory>
#include <vector>

/**
 * @class ConcatenateLayer
 * @brief Implements a concatenate layer in the CNN model.
 */
class ConcatenateLayer : public Layer {
    private:
    // Cache variables to store dimensions for backward pass
    size_t _batch_size_cache;
    size_t _features_first_cache;
    size_t _features_second_cache;

    public:
    /**
     * @brief Constructor for the ConcatenateLayer.
     * Since it has no parameters, we can use the default constructor.
     */
    ConcatenateLayer();

    /**
     * @brief Performs the forward pass through the Concatenate layer.
     * @param inputs A vector containing exactly two input Tensors, both expected to be 2D.
     * @return A shared pointer to the concatenated output Tensor.
     */
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    
    /**
     * @brief Performs the backward pass (gradient computation) for the Concatenate layer.
     * @param grad_output The gradient of the loss w.r.t. the layer's output.
     * @return A vector containing two gradient tensors corresponding to the inputs of the layer.
     */
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override;

    /**
     * @brief No-operation method for weight updates.
     * Since the Concatenate layer has no learnable parameters, this method does nothing.
     * It is implemented to satisfy the interface requirements of the Layer base class.
     * @param optimizer The optimizer to use for weight updates (not used in this layer).
     */
    void update_weights(std::shared_ptr<Optimizer> optimizer) override;
    
    /** @brief Gets the weights and their gradients.
     * @return Since the Concatenate layer has no learnable parameters, this method returns a pair of null pointers.
     */
    std::pair<float*, float*> get_weights_and_grads() override;

    /** @brief Gets the biases and their gradients.
     * @return Since the Concatenate layer has no learnable parameters, this method returns a pair of null pointers.
     */
    std::pair<float*, float*> get_biases_and_grads() override;

};

#endif // CONCATENATELAYER_HPP