/**
 * @file Conv2DLayer.cpp
 * @brief Implementation file for the Conv2DLayer class.
 * Implements the Conv2DLayer class, responsible for implementing 2D convolutional layers in the CNN model.
 */

#include "Conv2DLayer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace {
    /**
     * @brief Calculates a 4D index for a tensor with dimensions [dim_b, dim_c, dim_d, dim_e].
     * 
     * @param a The index for the first dimension (batch).
     * @param b The index for the second dimension (channels).
     * @param c The index for the third dimension (height).
     * @param d The index for the fourth dimension (width).
     * @param dim_b The size of the second dimension (channels).
     * @param dim_c The size of the third dimension (height).
     * @param dim_d The size of the fourth dimension (width).
     * @return The calculated linear index into the flattened tensor data.
     */
    size_t idx4(size_t a, size_t b, size_t c, size_t d,
            size_t dim_b, size_t dim_c, size_t dim_d) {
            return ((a * dim_b + b) * dim_c + c) * dim_d + d;
        }
} // namespace

Conv2DLayer::Conv2DLayer(int in_channels,
                         int out_channels,
                         int kernel_size,
                         int stride,
                         int padding)
    : _in_channels(in_channels),
      _out_channels(out_channels),
      _kernel_size(kernel_size),
      _stride(stride),
      _padding(padding) {
    if (_in_channels <= 0 || _out_channels <= 0 || _kernel_size <= 0 || _stride <= 0 || _padding < 0) {
        throw std::invalid_argument("Conv2DLayer: invalid constructor arguments.");
    }

    _layer_name = "Conv2DLayer(" + std::to_string(in_channels) + "," +
                  std::to_string(out_channels) + "," + std::to_string(kernel_size) +
                  ", stride=" + std::to_string(stride) + ", padding=" + std::to_string(padding) + ")";

    // Define the shape of the weights and biases tensors.
    _weights = std::make_shared<Tensor>(
        std::vector<size_t>{static_cast<size_t>(_out_channels),
                            static_cast<size_t>(_in_channels),
                            static_cast<size_t>(_kernel_size),
                            static_cast<size_t>(_kernel_size)});
    _biases = std::make_shared<Tensor>(std::vector<size_t>{static_cast<size_t>(_out_channels)});

    // Glorot initialization uses a normal distribution with variance based on fan-in and fan-out.
    // Stddev = sqrt(2 / (fan_in + fan_out))
    const float fan_in = static_cast<float>(_in_channels * _kernel_size * _kernel_size);
    const float fan_out = static_cast<float>(_out_channels * _kernel_size * _kernel_size);
    const float stddev = std::sqrt(2.0f / (fan_in + fan_out));
    std::default_random_engine generator(1); // for reproducibility
    std::normal_distribution<float> distribution(0.0f, stddev);

    // Initialize weights iterating through the data array of the weights tensor.
    float* w_ptr = _weights->get_data();
    for (size_t i = 0; i < _weights->size(); ++i) {
        w_ptr[i] = distribution(generator); // assign a random value
    }

    // initialize biases to zero
    float* b_ptr = _biases->get_data();
    for (size_t i = 0; i < _biases->size(); ++i) {
        b_ptr[i] = 0.0f;
    }
}

std::shared_ptr<Tensor> Conv2DLayer::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1 || !inputs[0]) {
        throw std::invalid_argument("Conv2DLayer expects exactly one non-null input tensor.");
    }

    // Cache the input tensor for use in the backward pass.
    _input_cache = inputs[0];
    const std::vector<size_t> in_shape = _input_cache->get_shape();

    // Check if the input tensor has the expected 4D shape: [batch, channels, height, width].
    if (in_shape.size() != 4) {
        throw std::invalid_argument("Conv2DLayer input must be 4D: [batch, channels, height, width].");
    }

    // Extract dimensions from the input shape.
    const size_t batch_size = in_shape[0];
    const size_t in_channels = in_shape[1];
    const size_t in_h = in_shape[2];
    const size_t in_w = in_shape[3];

    // Check if the input channels match the expected number of input channels for the layer.
    if (in_channels != static_cast<size_t>(_in_channels)) {
        throw std::invalid_argument("Conv2DLayer input channel mismatch.");
    }

    // Check if the kernel size is compatible with the padded input dimensions.
    if (in_h + 2 * static_cast<size_t>(_padding) < static_cast<size_t>(_kernel_size) ||
        in_w + 2 * static_cast<size_t>(_padding) < static_cast<size_t>(_kernel_size)) {
        throw std::invalid_argument("Conv2DLayer kernel larger than padded input.");
    }

    // Output dimensions calculation using O = (I - K + 2P) / S + 1
    const size_t out_h = (in_h + 2 * static_cast<size_t>(_padding) - static_cast<size_t>(_kernel_size)) /
                             static_cast<size_t>(_stride) + 1;
    const size_t out_w = (in_w + 2 * static_cast<size_t>(_padding) - static_cast<size_t>(_kernel_size)) /
                             static_cast<size_t>(_stride) + 1;

    // Create the output tensor with the calculated shape.
    auto output = std::make_shared<Tensor>(
        std::vector<size_t>{batch_size, static_cast<size_t>(_out_channels), out_h, out_w});

    // Data pointers for input, weights, biases, and output.
    float* x_ptr = _input_cache->get_data(); // shape: [batch_size, in_channels, in_h, in_w]
    float* w_ptr = _weights->get_data(); // shape: [out_channels, in_channels, kernel_size, kernel_size]
    float* b_ptr = _biases->get_data(); // shape: [out_channels]
    float* y_ptr = output->get_data(); // shape: [batch_size, out_channels, out_h, out_w]

    // Get dimensions for indexing helper function and loops
    const size_t oc_dim = static_cast<size_t>(_out_channels);
    const size_t ic_dim = static_cast<size_t>(_in_channels);
    const size_t k_dim = static_cast<size_t>(_kernel_size);

    // Convolution operation
    // Y[n, oc, oh, ow] = sum_{ic, kh, kw} X[n, ic, ih, iw] * W[oc, ic, kh, kw] + b[oc]
    for (size_t n = 0; n < batch_size; ++n) { // iterates over batch samples
        for (size_t oc = 0; oc < oc_dim; ++oc) { // iterates over output channels (filters)
            for (size_t oh = 0; oh < out_h; ++oh) { // iterates over output height
                for (size_t ow = 0; ow < out_w; ++ow) { // iterates over output width
                    // Sum starts with the bias for the current output channel
                    float sum = b_ptr[oc]; 

                    for (size_t ic = 0; ic < ic_dim; ++ic) { // iterates over input channels
                        for (size_t kh = 0; kh < k_dim; ++kh) { // iterates over kernel height
                            for (size_t kw = 0; kw < k_dim; ++kw) { // iterates over kernel width

                                // Calculate the corresponding input height and width indices, accounting for stride and padding.
                                const int ih = static_cast<int>(oh * static_cast<size_t>(_stride) + kh) - _padding;
                                const int iw = static_cast<int>(ow * static_cast<size_t>(_stride) + kw) - _padding;

                                // If the calculated input indices are out of bounds (due to padding), skip this position.
                                if (ih < 0 || iw < 0 || ih >= static_cast<int>(in_h) || iw >= static_cast<int>(in_w)) {
                                    continue;
                                }
                                
                                // Calculate the linear indices for the input and weight tensors using the helper function.
                                const size_t in_index = idx4(n, ic, static_cast<size_t>(ih), static_cast<size_t>(iw),
                                                             ic_dim, in_h, in_w);
                                const size_t w_index = idx4(oc, ic, kh, kw, ic_dim, k_dim, k_dim);

                                // Accumulate the product of the input value and the corresponding weight into the sum for the output element.
                                sum += x_ptr[in_index] * w_ptr[w_index];
                            }
                        }
                    }
                    // Store the computed output value for the current batch, output channel, and spatial location.
                    const size_t out_index = idx4(n, oc, oh, ow, oc_dim, out_h, out_w);
                    y_ptr[out_index] = sum;
                }
            }
        }
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> Conv2DLayer::backward(std::shared_ptr<Tensor> grad_output) {
    if (!grad_output || !_input_cache) {
        throw std::invalid_argument("Conv2DLayer backward requires cached input and non-null grad_output.");
    }

    // Extract shapes and validate dimensions.
    const std::vector<size_t> in_shape = _input_cache->get_shape();
    const std::vector<size_t> gd_shape = grad_output->get_shape();

    if (in_shape.size() != 4 || gd_shape.size() != 4) {
        throw std::invalid_argument("Conv2DLayer backward expects 4D input and grad_output.");
    }

    // Extract dimensions from the input shape.
    const size_t batch_size = in_shape[0];
    const size_t in_channels = in_shape[1];
    const size_t in_h = in_shape[2];
    const size_t in_w = in_shape[3];

    // Extract dimensions from the grad_output shape.
    const size_t out_channels = gd_shape[1];
    const size_t out_h = gd_shape[2];
    const size_t out_w = gd_shape[3];

    // Check for shape compatibility between the cached input, grad_output, and layer parameters.
    if (in_channels != static_cast<size_t>(_in_channels) ||
        out_channels != static_cast<size_t>(_out_channels) ||
        gd_shape[0] != batch_size) {
        throw std::invalid_argument("Conv2DLayer backward shape mismatch.");
    }

    // Initialize the input gradient tensor with the same shape as the cached input.
    auto grad_input = std::make_shared<Tensor>(in_shape);

    // Data pointers to the relevant data arrays for input, grad_output, weights, weight gradients, bias gradients, and input gradients.
    float* x_ptr = _input_cache->get_data(); // shape: [batch_size, in_channels, in_h, in_w]
    float* dy_ptr = grad_output->get_data(); // shape: [batch_size, out_channels, out_h, out_w]
    float* w_ptr = _weights->get_data(); // shape: [out_channels, in_channels, kernel_size, kernel_size]
    float* dw_ptr = _weights->get_grad(); // shape: [out_channels, in_channels, kernel_size, kernel_size]
    float* db_ptr = _biases->get_grad(); // shape: [out_channels]
    float* dx_ptr = grad_input->get_data(); // shape: [batch_size, in_channels, in_h, in_w]

    // Get dimensions for indexing helper function and loops
    const size_t oc_dim = static_cast<size_t>(_out_channels);
    const size_t ic_dim = static_cast<size_t>(_in_channels);
    const size_t k_dim = static_cast<size_t>(_kernel_size);

    // Compute gradients with respect to weights and biases.
    for (size_t n = 0; n < batch_size; ++n) { // iterates over batch samples
        for (size_t oc = 0; oc < oc_dim; ++oc) { // iterates over output channels (filters)
            for (size_t oh = 0; oh < out_h; ++oh) { // iterates over output height
                for (size_t ow = 0; ow < out_w; ++ow) { // iterates over output width
                    // Calculate the linear index for the current position in the grad_output tensor.
                    const size_t out_index = idx4(n, oc, oh, ow, oc_dim, out_h, out_w);
                    const float grad_val = dy_ptr[out_index];
                    
                    // Accumulate bias gradient dB[oc] += dY[n, oc, oh, ow]
                    db_ptr[oc] += grad_val;

                    // Compute weight gradients dW[oc, ic, kh, kw] += dY[n, oc, oh, ow] * X[n, ic, ih, iw]
                    for (size_t ic = 0; ic < ic_dim; ++ic) { // loops over input channels
                        for (size_t kh = 0; kh < k_dim; ++kh) { // loops over kernel height
                            for (size_t kw = 0; kw < k_dim; ++kw) { // loops over kernel width
                                // Calculate the corresponding input height and width indices, accounting for stride and padding.
                                const int ih = static_cast<int>(oh * static_cast<size_t>(_stride) + kh) - _padding;
                                const int iw = static_cast<int>(ow * static_cast<size_t>(_stride) + kw) - _padding;
                                
                                // Check if the input coordinates are valid
                                if (ih < 0 || iw < 0 || ih >= static_cast<int>(in_h) || iw >= static_cast<int>(in_w)) {
                                    continue;
                                }
                                
                                // Calculate linear indices for input data and weights.
                                const size_t in_index = idx4(n, ic, static_cast<size_t>(ih), static_cast<size_t>(iw),
                                                             ic_dim, in_h, in_w);
                                const size_t w_index = idx4(oc, ic, kh, kw, ic_dim, k_dim, k_dim);

                                // Accumulate the gradient for the specific weight.
                                dw_ptr[w_index] += grad_val * x_ptr[in_index];
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute gradients with respect to inputs: dX[n, ic, ih, iw] += sum_{oc, kh, kw} dY[n, oc, oh, ow] * W[oc, ic, kh, kw]
    for (size_t n = 0; n < batch_size; ++n) { // iterates over batch samples
        for (size_t oc = 0; oc < oc_dim; ++oc) { // iterates over output channels (filters)
            for (size_t oh = 0; oh < out_h; ++oh) { // iterates over output height
                for (size_t ow = 0; ow < out_w; ++ow) { // iterates over output width
                    // Calculate the linear index for the current position in the grad_output tensor.
                    const size_t out_index = idx4(n, oc, oh, ow, oc_dim, out_h, out_w);
                    const float grad_val = dy_ptr[out_index];

                    for (size_t ic = 0; ic < ic_dim; ++ic) { // iterates over input channels
                        for (size_t kh = 0; kh < k_dim; ++kh) { // iterates over kernel height
                            for (size_t kw = 0; kw < k_dim; ++kw) { // iterates over kernel width
                                // Calculate the corresponding input height and width indices, accounting for stride and padding.
                                const size_t kh_orig = k_dim - 1 - kh;
                                const size_t kw_orig = k_dim - 1 - kw;

                                // Calculate the corresponding input height and width indices, accounting for stride and padding.
                                const int ih = static_cast<int>(oh * static_cast<size_t>(_stride) + kh_orig) - _padding;
                                const int iw = static_cast<int>(ow * static_cast<size_t>(_stride) + kw_orig) - _padding;

                                // Check if the input coordinates are valid
                                if (ih < 0 || iw < 0 || ih >= static_cast<int>(in_h) || iw >= static_cast<int>(in_w)) {
                                    continue;
                                }

                                // Calculate the linear indices for input data and weights.
                                const size_t in_index = idx4(n, ic, static_cast<size_t>(ih), static_cast<size_t>(iw),
                                                             ic_dim, in_h, in_w);
                                const size_t w_index = idx4(oc, ic, kh_orig, kw_orig, ic_dim, k_dim, k_dim);

                                // Accumulate the gradient for the specific input element.
                                dx_ptr[in_index] += grad_val * w_ptr[w_index];
                            }
                        }
                    }
                }
            }
        }
    }

    return {grad_input};
}

void Conv2DLayer::update_weights(std::shared_ptr<Optimizer> optimizer) {
    if (optimizer) {
        // Update weights and biases using the optimizer's apply_gradients method
        optimizer->apply_gradients(_weights->get_data(), _weights->get_grad(), _weights->size());
        optimizer->apply_gradients(_biases->get_data(), _biases->get_grad(), _biases->size());
    }
}

std::pair<float*, float*> Conv2DLayer::get_weights_and_grads() {
    // Direct access to the weights data and gradients for optimization purposes
    return {_weights->get_data(), _weights->get_grad()};
}

std::pair<float*, float*> Conv2DLayer::get_biases_and_grads() {
    // Direct access to the biases data and gradients for optimization purposes
    return {_biases->get_data(), _biases->get_grad()};
}

// Additional helper methods to retrieve the weights tensors.
std::shared_ptr<Tensor> Conv2DLayer::get_weights_tensor() { return _weights; }

// Additional helper method to retrieve the biases tensor.
std::shared_ptr<Tensor> Conv2DLayer::get_biases_tensor() { return _biases; }
