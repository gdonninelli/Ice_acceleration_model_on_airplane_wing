/**
 * @file AdamOptimizer.hpp
 * @brief Header file for the AdamOptimizer class.
 * Defines the AdamOptimizer class, responsible for implementing the Adam optimization algorithm.
 */

#ifndef ADAMOPTIMIZER_HPP
#define ADAMOPTIMIZER_HPP

#include "Optimizer.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

/**
 * @class AdamOptimizer
 * @brief Implements the Adam optimization algorithm for updating model weights during training.
 */
class AdamOptimizer : public Optimizer {
    private:
    /**
     * @brief Internal structure to hold the state for each parameter array being optimized.
     */
    struct AdamState {
        std::vector<float> m; // first moment vector (moving average of gradients)
        std::vector<float> v; // second moment vector (moving average of squared gradients)
        size_t timestep = 0; // timestep counter
    };

    // Hyperparameters for the Adam optimizer
    float _learning_rate;
    float _beta1;
    float _beta2;
    float _epsilon;
    float _weight_decay;

    // Map to store the Adam state for each parameter array, keyed by the raw pointer to the weights
    std::unordered_map<float*, AdamState> _state_by_param_ptr;

    public:
    /**
     * @brief Constructor for the AdamOptimizer class.
     * @param learning_rate The learning rate for the optimizer.
     * @param beta1 The exponential decay rate for the first moment estimates.
     * @param beta2 The exponential decay rate for the second moment estimates.
     * @param epsilon A small constant to prevent division by zero.
     * @param weight_decay The coefficient for L2 weight decay (regularization).
     */
    AdamOptimizer(float learning_rate = 1e-3f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f,
                  float weight_decay = 0.0f);

    /**
     * @brief Applies the Adam optimization algorithm to update the weights based on the provided gradients.
     * @param weights A raw pointer to the parameter data to be updated.
     * @param grads A raw pointer to the corresponding gradient.
     * @param size The total number of elements in the weights and grads arrays.
     */
    void apply_gradients(float* weights, float* grads, size_t size) override;
};

#endif // ADAMOPTIMIZER_HPP