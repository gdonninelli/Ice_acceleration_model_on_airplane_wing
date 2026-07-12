/**
 * @file Conv2DLayer.hpp
 * @brief Implementation file for the Conv2DLayer class.
 * Implements the Conv2DLayer class, responsible for implementing 2D convolutional layers in the CNN model.
 */

#ifndef CONV2DLAYER_HPP
#define CONV2DLAYER_HPP

#include "Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <memory>
#include <utility>
#include <vector>

/**
 * @class Conv2DLayer
 * @brief Represents a 2D convolutional layer in a neural network.
 */
class Conv2DLayer : public Layer {
    private:
    int _in_channels;
    int _out_channels;
    int _kernel_size;
    int _stride;
    int _padding;

    // Tensor storing the learnable biases, one for each output channel.
    std::shared_ptr<Tensor> _weights; // shape: [out_channels, in_channels, kernel_size, kernel_size]
    std::shared_ptr<Tensor> _biases;  // shape: [out_channels]

    // Cache the input tensor from the forward pass to use in the backward pass for gradient computation.
    std::shared_ptr<Tensor> _input_cache; // shape: [batch, in_channels, height, width]

    public:
    /**
     * @brief Constructor for the Conv2DLayer.
     * @param in_channels The number of input channels.
     * @param out_channels The number of output channels (filters).
     * @param kernel_size The size of the square kernel (height and width).
     * @param stride The size of the square kernel (height and width).
     * @param padding The amount of zero-padding added to both sides of the input.
     */
    Conv2DLayer(int in_channels,
                int out_channels,
                int kernel_size,
                int stride = 1,
                int padding = 0);

    /**
     * @brief Performs the forward pass through the Conv2D layer.
     * @param inputs A vector containing a single input Tensor with shape [batch, in_channels, height, width].
     * @return A shared pointer to the output Tensor with shape [batch, out_channels, out_height, out_width].
     */
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    
    /**
     * @brief Performs the backward pass through the Conv2D layer.
     * @param grad_output The gradient of the loss with respect to the output of shape [batch, out_channels, out_height, out_width].
     * @return A vector containing the gradients of the loss with respect to the inputs.
     */
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override;

    /**
     * @brief Updates the weights of the layer using the provided optimizer.
     * @param optimizer A shared pointer to the optimizer used for updating the weights.
     */
    void update_weights(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Gets the weights and their gradients.
     * @return A pair of pointers to the weights and gradients arrays.
     */
    std::pair<float*, float*> get_weights_and_grads() override;

    /**
     * @brief Gets the biases and their gradients.
     * @return A pair of pointers to the biases and gradients arrays.
     */
    std::pair<float*, float*> get_biases_and_grads() override;

    /**
     * @brief Get the weights tensor of the layer.
     * @return A shared pointer to the weights Tensor.
     */
    std::shared_ptr<Tensor> get_weights_tensor();

    /**
     * @brief Get the biases tensor of the layer.
     * @return A shared pointer to the biases Tensor.
     */
    std::shared_ptr<Tensor> get_biases_tensor();
};

#endif // CONV2DLAYER_HPP