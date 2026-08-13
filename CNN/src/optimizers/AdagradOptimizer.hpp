/**
 * @file AdagradOptimizer.hpp
 * @brief Header file for the AdagradOptimizer class.
 * Defines the AdagradOptimizer class, responsible for implementing the Adagrad optimization algorithm.
 */

#ifndef ADAGRADOPTIMIZER_HPP
#define ADAGRADOPTIMIZER_HPP

#include "Optimizer.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

/**
 * @class AdagradOptimizer
 * @brief Implements the Adagrad optimization algorithm for updating model weights during training.
 */
class AdagradOptimizer : public Optimizer {
    private:
    /**
     * @brief Internal structure to hold the state for each parameter array being optimized.
     */
    struct AdagradState {
        std::vector<float> G; // sum of the squares of the gradients w.r.t to the parameter up to time step t
    };

    // Hyperparameters for the Adagrad optimizer
    float _learning_rate;
    float _epsilon;
    float _weight_decay;

    // Map to store the Adagrad state for each parameter array, keyed by the raw pointer to the weights
    std::unordered_map<float*, AdagradState> _state_by_param_ptr;

    public:
    /**
     * @brief Constructor for the AdagradOptimizer class.
     * @param learning_rate The learning rate for the optimizer (gamma).
     * @param epsilon A small constant to prevent division by zero.
     * @param weight_decay The coefficient for L2 weight decay (regularization).
     */
    AdagradOptimizer(float learning_rate = 1e-3f,
                     float epsilon = 1e-8f,
                     float weight_decay = 0.0f);

    /**
     * @brief Applies the Adagrad optimization algorithm to update the weights based on the provided gradients.
     * @param weights A raw pointer to the parameter data to be updated.
     * @param grads A raw pointer to the corresponding gradient.
     * @param size The total number of elements in the weights and grads arrays.
     */
    void apply_gradients(float* weights, float* grads, size_t size) override;
    
    OptimizerMetadata metadata() const override;
};

#endif // ADAGRADOPTIMIZER_HPP
