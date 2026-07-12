/**
 * @file FlattenLayer.hpp
 * @brief Header file for the FlattenLayer class.
 * Declares the FlattenLayer class, responsible for implementing flatten layers in the CNN model.
 */

#ifndef FLATTENLAYER_HPP
#define FLATTENLAYER_HPP

#include "Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <memory>
#include <vector>

/**
 * @class FlattenLayer
 * @brief Implements a flatten layer in the CNN model.
 */
class FlattenLayer : public Layer {
    private:
    std::vector<size_t> _input_shape_cache; // cache the input shape

    public:
    /**
     * @brief Constructor for the FlattenLayer.
     * Since it has no parameters, we can use the default constructor.
     */
    FlattenLayer();

    /**
     * @brief Performs the forward pass through the Flatten layer.
     * @param inputs A vector containing a single input Tensor with shape [batch, channels, height, width].
     * @return A shared pointer to the output Tensor with shape [batch, channels*height*width].
     */
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;

    /**
     * @brief Performs the backward pass (gradient computation) for the Flatten layer.
     * @param grad_output The gradient of the loss w.r.t. the layer's output with shape [batch, channels*height*width].
     * @return A vector containing a single gradient tensor with shape [batch, channels, height, width].
     */
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override;

    /**
     * @brief No-operation method for weight updates.
     * Since the Flatten layer has no learnable parameters, this method does nothing.
     * It is implemented to satisfy the interface requirements of the Layer base class.
     * @param optimizer The optimizer to use for weight updates (not used in this layer).
     */
    void update_weights(std::shared_ptr<Optimizer> optimizer) override;

    /** @brief Gets the weights and their gradients.
     * @return Since the Flatten layer has no learnable parameters, this method returns a pair of null pointers.
     */
    std::pair<float*, float*> get_weights_and_grads() override;

    /** @brief Gets the biases and their gradients.
     * @return Since the Flatten layer has no learnable parameters, this method returns a pair of null pointers.
     */
    std::pair<float*, float*> get_biases_and_grads() override;
};

#endif // FLATTENLAYER_HPP