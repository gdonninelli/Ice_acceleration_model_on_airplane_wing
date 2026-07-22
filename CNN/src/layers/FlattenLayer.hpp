/**
 * @file FlattenLayer.hpp
 * @brief Header file for the FlattenLayer class.
 * Declares the FlattenLayer class, responsible for implementing flatten layers in the CNN model.
 */

#ifndef FLATTENLAYER_HPP
#define FLATTENLAYER_HPP

#include "Layer.hpp"
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

};

#endif // FLATTENLAYER_HPP
