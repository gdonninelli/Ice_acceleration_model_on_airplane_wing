#include "SGDOptimizer.hpp"
#include <stdexcept>
#include <cmath>

SGDOptimizer::SGDOptimizer(float learning_rate,
                           float momentum,
                           float weight_decay)
    : _learning_rate(learning_rate),
      _momentum(momentum),
      _weight_decay(weight_decay) {

    // Hyperparameter validation
    if (!std::isfinite(_learning_rate) || _learning_rate <= 0.0f) {
        throw std::invalid_argument("SGD learning_rate must be finite and positive.");
    }
    if (!std::isfinite(_momentum) || _momentum < 0.0f || _momentum >= 1.0f) {
        throw std::invalid_argument("SGD momentum must be finite and in [0, 1).");
    }
    if (!std::isfinite(_weight_decay) || _weight_decay < 0.0f) {
        throw std::invalid_argument("SGD weight_decay must be finite and non-negative.");
    }
}

void SGDOptimizer::apply_gradients(float* weights, float* grads, size_t size) {
    // If pointers are null or size is zero, there's nothing to do.
    if (!weights || !grads || size == 0) {
        return;
    }

    // Get or create the SGDState for this specific parameter array pointer if momentum > 0
    SGDState* state_ptr = nullptr; // for the vanilla SGD case
    if (_momentum > 0.0f) {
        state_ptr = &_state_by_param_ptr[weights];
        // State initialization if the size of the state vectors does not match the current parameter size.
        if (state_ptr->velocity.size() != size) {
            state_ptr->velocity.assign(size, 0.0f); // initialize velocity vector to zeros
        }
    }

    // Update each parameter using the SGD update rule
    for (size_t i = 0; i < size; ++i) { // iterate over all elements in the parameter array
        float grad = grads[i];

        // Apply weight decay if specified (L2 regularization)
        if (_weight_decay != 0.0f) {
            grad += _weight_decay * weights[i];
        }

        if (_momentum > 0.0f && state_ptr != nullptr) {
            // Update velocity
            state_ptr->velocity[i] = _momentum * state_ptr->velocity[i] + grad;
            // Update the weights
            weights[i] -= _learning_rate * state_ptr->velocity[i];
        } else {
            // Update the weights directly
            weights[i] -= _learning_rate * grad;
        }
        
        // Reset the gradient to zero after applying the update
        grads[i] = 0.0f;
    }
}

OptimizerMetadata SGDOptimizer::metadata() const {
    return OptimizerMetadata{
        "sgd",
        _learning_rate,
        _learning_rate,
        false,
        {{"momentum", _momentum},
         {"weight_decay", _weight_decay}}};
}
