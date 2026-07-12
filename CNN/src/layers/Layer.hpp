/**
 * @file Layer.hpp
 * @brief Header file for the Layer class.
 * Defines the Layer class, responsible for implementing different types of layers in the CNN model. 
 * It uses the concept of polymorphism to provide a common interface for all layer types, allowing for
 * scalability in the design of the CNN architecture.
 */

#ifndef LAYER_HPP
#define LAYER_HPP

#include "core/Tensor.hpp"
#include <memory>
#include <string>

// Forward declaration of the Optimizer class to avoid circular dependency.
class Optimizer;

/**
 * @class Layer
 * @brief Abstract base class for all neural network layers, specifying the fundamental operations required for a layer.
 */
class Layer {
protected:
  std::string _layer_name; // name of the layer for identification

public:
  /**
   * @brief Virtual destructor for the Layer class.
   * It ensures that the destructor of the derived class is called when deleting an object through a pointer to Layer, 
   * allowing for proper cleanup of resources.
   */
  virtual ~Layer() = default;

  /**
   * @brief Pure virtual method for performing the forward pass of the layer.
   * @param inputs A vector of shared pointers to the input tensors for the layer.
   * @return A shared pointer to the output Tensor produced by the layer.
   */
  virtual std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>> &inputs) = 0;

  /**
   * @brief Pure virtual method for performing the backward pass of the layer.
   * @param grad_output A shared pointer to the Tensor representing the gradient of the loss with respect to the layer's output.
   * @return A vector of shared pointers to the Tensors representing the gradients of the loss with respect to the inputs of the layer.
   */
  virtual std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) = 0; 

  /**
   * @brief Pure virtual method for updating the weights of the layer using the provided optimizer.
   * @param optimizer A shared pointer to the optimizer used for updating the weights.
   */
  virtual void update_weights(std::shared_ptr<Optimizer> optimizer) = 0;

  /**
   * @brief Pure virtual method for retrieving pointers to the layer's weights and their corresponding gradients.
   * In this case, we don't add const since we want to allow modification of the weights and gradients.
   * @return A std::pair containing (pointer to weights, pointer to weight gradients).
   */
  virtual std::pair<float *, float *> get_weights_and_grads() = 0;
  
  /**
   * @brief Pure virtual method for retrieving pointers to the layer's biases and their corresponding gradients.
   * @return A std::pair containing (pointer to biases, pointer to bias gradients).
   */
  virtual std::pair<float *, float *> get_biases_and_grads() = 0;

  /**
   * @brief Get the name of the layer.
   * @return A constant reference to the layer's name string.
   */
  const std::string &get_layer_name() const { return _layer_name; }
};

#endif // LAYER_HPP