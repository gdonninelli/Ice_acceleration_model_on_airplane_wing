/**
 * @file Optimizer.hpp
 * @brief Header file for the Optimizer class.
 * Defines the Optimizer class, responsible for implementing different optimization algorithms for updating model weights.
 */

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cstddef>

/**
 * @class Optimizer
 * @brief Abstract base class for all optimizers, specifying the fundamental operation required for an optimizer. 
 */
class Optimizer {
    public:
    /**
     * @brief Virtual destructor for the Optimizer class.
     * Here the default one is sufficient.
     */
    virtual ~Optimizer() = default;

    /**
     * @brief Pure virtual method to apply gradients to parameters.
     * @param weights A raw pointer to the parameter (e.g., weight) data.
     * @param grads A raw pointer to the gradient data for the parameters.
     * @param size The total number of elements in the weights and grads arrays.
     */
    virtual void apply_gradients(float* weights, float* grads, size_t size) = 0;
};

#endif // OPTIMIZER_HPP