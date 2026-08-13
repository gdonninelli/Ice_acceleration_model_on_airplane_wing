/**
 * @file SGDOptimizer.hpp
 * @brief Header file for the SGDOptimizer class.
 * Defines the SGDOptimizer class, responsible for implementing the SGD optimization algorithm.
 */

#ifndef SGDOPTIMIZER_HPP
#define SGDOPTIMIZER_HPP

#include "Optimizer.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

/**
 * @class SGDOptimizer
 * @brief Implements the Stochastic Gradient Descent (SGD) optimization algorithm for updating model weights during training.
 */
class SGDOptimizer : public Optimizer {
    private:
    /**
     * @brief Internal structure to hold the state for each parameter array being optimized.
     */
    struct SGDState {
        std::vector<float> velocity; // velocity vector for momentum
    };

    // Hyperparameters for the SGD optimizer
    float _learning_rate;
    float _momentum;
    float _weight_decay;

    // Map to store the SGD state for each parameter array, keyed by the raw pointer to the weights
    std::unordered_map<float*, SGDState> _state_by_param_ptr;

    public:
    /**
     * @brief Constructor for the SGDOptimizer class.
     * @param learning_rate The learning rate for the optimizer.
     * @param momentum The momentum factor.
     * @param weight_decay The coefficient for L2 weight decay (regularization).
     */
    SGDOptimizer(float learning_rate = 1e-3f,
                 float momentum = 0.0f,
                 float weight_decay = 0.0f);

    /**
     * @brief Applies the SGD optimization algorithm to update the weights based on the provided gradients.
     * @param weights A raw pointer to the parameter data to be updated.
     * @param grads A raw pointer to the corresponding gradient.
     * @param size The total number of elements in the weights and grads arrays.
     */
    void apply_gradients(float* weights, float* grads, size_t size) override;
    OptimizerMetadata metadata() const override;
};

#endif // SGDOPTIMIZER_HPP
