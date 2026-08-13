/**
 * @file RMSpropOptimizer.hpp
 * @brief Header file for the RMSpropOptimizer class.
 * Defines the RMSpropOptimizer class, responsible for implementing the RMSprop optimization algorithm.
 */

#ifndef RMSPROPOPTIMIZER_HPP
#define RMSPROPOPTIMIZER_HPP

#include "Optimizer.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

/**
 * @class RMSpropOptimizer
 * @brief Implements the RMSprop optimization algorithm for updating model weights during training.
 */
class RMSpropOptimizer : public Optimizer {
    private:
    /**
     * @brief Internal structure to hold the state for each parameter array being optimized.
     */
    struct RMSpropState {
        std::vector<float> v; // exponentially decaying average of squared gradients (E[g^2])
    };

    // Hyperparameters for the RMSprop optimizer
    float _learning_rate;
    float _decay_rate; 
    float _epsilon;
    float _weight_decay;

    // Map to store the RMSprop state for each parameter array, keyed by the raw pointer to the weights
    std::unordered_map<float*, RMSpropState> _state_by_param_ptr;

    public:
    /**
     * @brief Constructor for the RMSpropOptimizer class.
     * @param learning_rate The learning rate for the optimizer (gamma).
     * @param decay_rate The decay rate for the moving average of squared gradients (mu).
     * @param epsilon A small constant to prevent division by zero.
     * @param weight_decay The coefficient for L2 weight decay (regularization).
     */
    RMSpropOptimizer(float learning_rate = 1e-3f,
                     float decay_rate = 0.9f,
                     float epsilon = 1e-8f,
                     float weight_decay = 0.0f);

    /**
     * @brief Applies the RMSprop optimization algorithm to update the weights based on the provided gradients.
     * @param weights A raw pointer to the parameter data to be updated.
     * @param grads A raw pointer to the corresponding gradient.
     * @param size The total number of elements in the weights and grads arrays.
     */
    void apply_gradients(float* weights, float* grads, size_t size) override;
    
    OptimizerMetadata metadata() const override;
};

#endif // RMSPROPOPTIMIZER_HPP
