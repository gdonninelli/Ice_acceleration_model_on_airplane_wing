#include "AdagradOptimizer.hpp"
#include <stdexcept>
#include <cmath>

AdagradOptimizer::AdagradOptimizer(float learning_rate,
                                   float epsilon,
                                   float weight_decay)
    : _learning_rate(learning_rate),
      _epsilon(epsilon),
      _weight_decay(weight_decay) {

    // Hyperparameter validation
    if (!std::isfinite(_learning_rate) || _learning_rate <= 0.0f) {
        throw std::invalid_argument("Adagrad learning_rate must be finite and positive.");
    }
    if (!std::isfinite(_epsilon) || _epsilon <= 0.0f) {
        throw std::invalid_argument("Adagrad epsilon must be finite and positive.");
    }
    if (!std::isfinite(_weight_decay) || _weight_decay < 0.0f) {
        throw std::invalid_argument("Adagrad weight_decay must be finite and non-negative.");
    }
}

void AdagradOptimizer::apply_gradients(float* weights, float* grads, size_t size) {
    if (!weights || !grads || size == 0) {
        return;
    }

    // Get or create the AdagradState for this specific parameter array pointer
    AdagradState& state = _state_by_param_ptr[weights];
    if (state.G.size() != size) {
        state.G.assign(size, 0.0f); // initialize sum of squares to zeros
    }

    // Update each parameter using the Adagrad update rule
    for (size_t i = 0; i < size; ++i) { // iterate over all elements in the parameter array
        float grad = grads[i];

        // Apply weight decay if specified (L2 regularization)
        if (_weight_decay != 0.0f) {
            grad += _weight_decay * weights[i];
        }

        // Update the sum of squared gradients: G_{t,ii} = G_{t-1,ii} + g_{t,i}^2
        state.G[i] = state.G[i] + grad * grad;

        // Update the weights: w_{t+1,i} = w_{t,i} - (learning_rate / sqrt(G_{t,ii} + epsilon)) * g_{t,i}
        weights[i] -= (_learning_rate / std::sqrt(state.G[i] + _epsilon)) * grad;

        // Reset the gradient to zero after applying the update
        grads[i] = 0.0f;
    }
}

OptimizerMetadata AdagradOptimizer::metadata() const {
    return OptimizerMetadata{
        "adagrad",
        _learning_rate,
        _learning_rate,
        false,
        {{"epsilon", _epsilon},
         {"weight_decay", _weight_decay}}};
}
