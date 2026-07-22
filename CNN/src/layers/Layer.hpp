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
#include <vector>

struct LayerParameter {
  std::string name;
  std::shared_ptr<Tensor> tensor;
};

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
   * @brief Returns every learnable parameter owned by this layer.
   * Parameter-free layers inherit the empty default implementation.
   */
  virtual std::vector<LayerParameter> parameters() const { return {}; }

  /**
   * @brief Get the name of the layer.
   * @return A constant reference to the layer's name string.
   */
  const std::string &get_layer_name() const { return _layer_name; }
};

#endif // LAYER_HPP
