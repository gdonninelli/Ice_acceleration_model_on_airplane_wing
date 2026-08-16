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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct LayerParameter {
  std::string name;
  std::shared_ptr<Tensor> tensor;
};

/**
 * @brief Stateless 64-bit mixing helpers for layers that need deterministic,
 * MPI-rank-invariant randomness (e.g. dropout masks). splitmix64 maps any
 * counter to a well-distributed value, so a mask can be a pure function of
 * (seed, sample, feature) instead of the state of a sequential generator.
 */
namespace layer_rng {
inline uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

inline uint64_t combine(uint64_t seed, uint64_t value) {
  return splitmix64(seed ^ splitmix64(value));
}

/** Maps a mixed hash to a float uniformly distributed in [0, 1). */
inline float to_unit_float(uint64_t hash) {
  return static_cast<float>(hash >> 40U) * 0x1.0p-24f;
}
} // namespace layer_rng

/**
 * @brief Per-forward-pass execution context shared with every layer.
 *
 * The default state describes inference: layers that behave differently in
 * training (dropout, and in the future batch normalization) stay inactive
 * until the trainer switches the context. `stream_seed` and `sample_offset`
 * make stochastic layers deterministic and independent of the MPI rank
 * layout: the seed is a pure function of (run seed, epoch, global batch) and
 * the offset locates this rank's first sample inside the global batch.
 */
struct LayerExecutionContext {
  bool training = false;
  uint64_t stream_seed = 0;
  size_t sample_offset = 0;
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
   * @brief Receives the execution context for the next forward pass.
   * Layers whose behavior does not depend on the training/inference
   * distinction inherit this no-op default.
   */
  virtual void set_execution_context(const LayerExecutionContext&) {}

  /**
   * @brief Get the name of the layer.
   * @return A constant reference to the layer's name string.
   */
  const std::string &get_layer_name() const { return _layer_name; }
};

#endif // LAYER_HPP
