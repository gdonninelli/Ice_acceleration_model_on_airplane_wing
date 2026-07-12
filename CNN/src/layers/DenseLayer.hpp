/**
 * @file DenseLayer.hpp
 * @brief Header file for the DenseLayer class.
 * Defines the DenseLayer class, responsible for implementing dense (fully connected) layers in the CNN model.
 */

#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include "Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <memory> // TODO check if we need this include
#include <vector>
// TODO: check if we need these includes
// #include <utility>
// #include <string>

/** 
 * @class DenseLayer
 * @brief Represents a fully-connected (Dense) layer in a neural network.
 * It performs the linear transformation Y = X * W^T + b, where X is the input, W are the weights, and b are the biases.
 */
class DenseLayer : public Layer {
    private:
    int _input_features;
    int _output_features;

    // Tensor to store the learnable weights of the layer.
    std::shared_ptr<Tensor> _weights; // shape: [output_features, input_features]
    std::shared_ptr<Tensor> _biases; // shape: [output_features]

    // Cache the input tensor from the forward pass to use in the backward pass for gradient computation.
    std::shared_ptr<Tensor> _input_cache;

    public:
    /** 
     * @brief Constructor for the DenseLayer class.
     * @param input_features The number of input features.
     * @param output_features The number of output features.
    */
    DenseLayer(int input_features, int output_features);

    /** 
     * @brief Performs the forward pass through the DenseLayer.
     * It computes the output tensor as Y = X * W^T + b.
     * @param inputs A vector containing a single input Tensor (X).
     * @return A shared pointer to the output Tensor (Y).
     */
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override;

    /** 
     * @brief Performs the backward pass through the DenseLayer.
     * @param grad_output The gradient of the loss with respect to the output (dL/dY).
     * @return A vector containing a single shared pointer to the input gradient Tensor (dL/dX).
     */
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override;

    /** 
     * @brief Updates the weights of the layer using the provided optimizer.
     * @param optimizer A shared pointer to the optimizer.
     */
    void update_weights(std::shared_ptr<Optimizer> optimizer) override;
    
    /** 
     * @brief Retrieves pointers to the layer's weights and their corresponding gradients.
     * @return A pair of raw pointers: {weights_data_ptr, weights_grad_ptr}.
     */
    std::pair<float*,float*> get_weights_and_grads() override;

    /** 
     * @brief Retrieves pointers to the layer's biases and their corresponding gradients.
     * @return A pair of raw pointers: {biases_data_ptr, biases_grad_ptr}.
     */
    std::pair<float*,float*> get_biases_and_grads() override;

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

#endif // DENSELAYER_HPP