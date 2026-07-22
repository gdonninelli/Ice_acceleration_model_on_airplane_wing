/**
 * @file DenseLayer.cpp
 * @brief Implementation file for the DenseLayer class.
 * Implements the DenseLayer class, responsible for implementing dense (fully connected) layers in the CNN model.
 */

#include "DenseLayer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

DenseLayer::DenseLayer(int input_features, int output_features, uint64_t seed)
    : _input_features(input_features), _output_features(output_features) {
    if (_input_features <= 0 || _output_features <= 0) {
      throw std::invalid_argument("DenseLayer feature counts must be positive.");
    }
    _layer_name = "DenseLayer( " + std::to_string(input_features) + "," + std::to_string(output_features) + " )";

    // Define the shape of the weights and biases tensors.
    std::vector<size_t> weight_shape = {(size_t)output_features, (size_t)input_features}; 
    _weights = std::make_shared<Tensor>(weight_shape); // shape: [output_features, input_features]
    _biases = std::make_shared<Tensor>(std::vector<size_t>{(size_t)output_features}); // shape: [output_features]

    // Weights initialization using Xavier/Glorot uniform initialization.
    float limit = std::sqrt(6.0f / (input_features + output_features));
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution(-limit, limit);

    // Initialize weights iterating through the data array of the weights tensor.
    float* w_ptr = _weights->get_data();
    for (size_t i = 0; i < _weights->size(); ++i) {
        w_ptr[i] = distribution(generator); // assign a random value
    }

    // Initialize biases to zero.
    float* b_ptr = _biases->get_data();
    for (size_t i = 0; i < _biases->size(); ++i) {
      b_ptr[i] = 0.0f;
    }
}

std::shared_ptr<Tensor>
DenseLayer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    _input_cache.reset();
    _output_shape_cache.clear();
    if (inputs.size() != 1 || !inputs[0]) {
        throw std::invalid_argument("DenseLayer expects exactly one non-null input tensor.");
    }

    const std::shared_ptr<Tensor>& input = inputs[0];
    const std::vector<size_t> input_shape = input->get_shape();
    if (input_shape.empty()) {
        throw std::invalid_argument("DenseLayer input shape must not be empty.");
    }
    
    // Check if the last dimension of the input matches the expected number of input features.
    if (input_shape.back() != (size_t)_input_features) {
        throw std::invalid_argument("Input feature size mismatch.");
    }

    // Calculate the batch size by multiplying all dimensions of the input shape except the last one.
    size_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        batch_size *= input_shape[i];
    }

    // The output shape is derived from the input shape by replacing the last dimension with the number of output features.
    std::vector<size_t> out_shape = input_shape;
    out_shape.back() = (size_t)_output_features;
    auto output = std::make_shared<Tensor>(out_shape);

    // Data pointer for input, weights, biases and output tensors.
    float *x_ptr = input->get_data(); // shape: [batch_size, input_features]
    float *w_ptr = _weights->get_data(); // shape: [output_features, input_features]
    float *b_ptr = _biases->get_data(); // shape: [output_features]
    float *y_ptr = output->get_data(); // shape: [batch_size, output_features]

    // Matrix Multiplication: Y = X * W^T + b
    // Y[b, j] = sum_i (X[b, i] * W[j, i]) + b[j]
    for (size_t b = 0; b < batch_size; ++b) { // iterates over batch samples
        for (size_t j = 0; j < (size_t)_output_features; ++j) { // iterates over output features.
          // Sum starts with the bias for the current output feature
          float sum = b_ptr[j];
            for (size_t i = 0; i < (size_t)_input_features; ++i) { // iterates over input features.
                sum += x_ptr[b * _input_features + i] * w_ptr[j * _input_features + i];
            }
            // Store the computed output value for the current batch and output feature.
            y_ptr[b * _output_features + j] = sum;
        }
    }

    _input_cache = input;
    _output_shape_cache = std::move(out_shape);
    return output;
}

std::vector<std::shared_ptr<Tensor>>
DenseLayer::backward(std::shared_ptr<Tensor> grad_output) {

  if (!grad_output || !_input_cache || _output_shape_cache.empty()) {
    throw std::invalid_argument("DenseLayer backward requires a prior forward pass and non-null gradient.");
  }
  if (grad_output->get_shape() != _output_shape_cache) {
    throw std::invalid_argument("DenseLayer backward gradient shape mismatch.");
  }

  // Calculate batch size from the shape of grad_output.
  const std::vector<size_t> &grad_shape = grad_output->get_shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < grad_shape.size() - 1; ++i) {
    batch_size *= grad_shape[i];
  }

  // Compute gradient with respect to input: dX = dY * W

  // Initialize the input gradient tensor with the same shape as the cached input.
  auto grad_input = std::make_shared<Tensor>(_input_cache->get_shape()); // shape: [batch_size, input_features]
  
  // Get pointers to the relevant data arrays.
  float *dy_ptr = grad_output->get_data(); // shape: [batch_size, output_features]
  float *w_ptr = _weights->get_data(); // shape: [output_features, input_features]
  float *dx_ptr = grad_input->get_data(); // shape: [batch_size, input_features]

  // Calculate dx[b, i] = sum_j (dY[b, j] * W[j, i])
  for (size_t b = 0; b < batch_size; ++b) { // iterates over batch samples
    for (size_t i = 0; i < (size_t)_input_features; ++i) { // iterates over input features
      float sum = 0.0f;
      for (size_t j = 0; j < (size_t)_output_features; ++j) { // iterates over output features
        sum += dy_ptr[b * _output_features + j] * w_ptr[j * _input_features + i];
      }
      // Store the computed gradient for the input feature
      dx_ptr[b * _input_features + i] = sum;
    }
  }

  // Compute gradient with respect to weights: dW = dY^T * X
  
  float *dw_ptr = _weights->get_grad(); // shape: [output_features, input_features]
  float *x_ptr = _input_cache->get_data(); // shape: [batch_size, input_features]

  // Calculate dw[j, i] = sum_b (dY[b, j] * X[b, i])
  for (size_t j = 0; j < (size_t)_output_features; ++j) { // iterates over output features
    for (size_t i = 0; i < (size_t)_input_features; ++i) { // iterates over input features
      float sum = 0.0f;
      for (size_t b = 0; b < batch_size; ++b) { // iterates over batch samples
        sum += dy_ptr[b * _output_features + j] * x_ptr[b * _input_features + i];
      }
      // Store the computed gradient for the weight
      dw_ptr[j * _input_features + i] += sum; // Accumulate gradients
    }
  }

  // Compute gradient with respect to biases: db = sum_b (dY[b, j])

  float *db_ptr = _biases->get_grad(); // shape: [output_features]

  // Calculate db[j] = sum_b (dY[b, j])
  for (size_t j = 0; j < (size_t)_output_features; ++j) { // iterates over output features
    float sum = 0.0f;
    for (size_t b = 0; b < batch_size; ++b) { // iterates over batch samples
      sum += dy_ptr[b * _output_features + j];
    }
    // Store the computed gradient for the bias
    db_ptr[j] += sum;
  }

  return {grad_input};
}

std::vector<LayerParameter> DenseLayer::parameters() const {
  return {{"weights", _weights}, {"biases", _biases}};
}

// Additional helper methods to retrieve the weights tensors.
std::shared_ptr<Tensor> DenseLayer::get_weights_tensor() { return _weights; }

// Additional helper method to retrieve the biases tensor.
std::shared_ptr<Tensor> DenseLayer::get_biases_tensor() { return _biases; }
