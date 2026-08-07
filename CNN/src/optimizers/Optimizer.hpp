/**
 * @file Optimizer.hpp
 * @brief Header file for the Optimizer class.
 * Defines the Optimizer class, responsible for implementing different optimization algorithms for updating model weights.
 */

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cstddef>
#include <limits>
#include <map>
#include <string>

struct OptimizerMetadata {
    std::string name = "optimizer";
    double configured_learning_rate = std::numeric_limits<double>::quiet_NaN();
    double effective_learning_rate = std::numeric_limits<double>::quiet_NaN();
    bool scheduled = false;
    std::map<std::string, double> settings;
};

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

    virtual OptimizerMetadata metadata() const { return {}; }
};

#endif // OPTIMIZER_HPP
