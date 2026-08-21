#include "RMSpropOptimizer.hpp"
#include <stdexcept>
#include <cmath>

RMSpropOptimizer::RMSpropOptimizer(float learning_rate,
                                   float decay_rate,
                                   float epsilon,
                                   float weight_decay)
    : _learning_rate(learning_rate),
      _decay_rate(decay_rate),
      _epsilon(epsilon),
      _weight_decay(weight_decay) {

    // Hyperparameter validation
    if (!std::isfinite(_learning_rate) || _learning_rate <= 0.0f) {
        throw std::invalid_argument("RMSprop learning_rate must be finite and positive.");
    }
    if (!std::isfinite(_decay_rate) || _decay_rate < 0.0f || _decay_rate >= 1.0f) {
        throw std::invalid_argument("RMSprop decay_rate must be finite and in [0, 1).");
    }
    if (!std::isfinite(_epsilon) || _epsilon <= 0.0f) {
        throw std::invalid_argument("RMSprop epsilon must be finite and positive.");
    }
    if (!std::isfinite(_weight_decay) || _weight_decay < 0.0f) {
        throw std::invalid_argument("RMSprop weight_decay must be finite and non-negative.");
    }
}

void RMSpropOptimizer::apply_gradients(float* weights, float* grads, size_t size) {
    if (!weights || !grads || size == 0) {
        return;
    }

    // Get or create the RMSpropState for this specific parameter array pointer
    RMSpropState& state = _state_by_param_ptr[weights];
    if (state.v.size() != size) {
        state.v.assign(size, 0.0f); // initialize exponentially decaying average to zeros
    }

    // Update each parameter using the RMSprop update rule
    for (size_t i = 0; i < size; ++i) { // iterate over all elements in the parameter array
        float grad = grads[i];

        // Apply weight decay if specified (L2 regularization)
        if (_weight_decay != 0.0f) {
            grad += _weight_decay * weights[i];
        }

        // Update the moving average of squared gradients: E[g^2]_t = decay_rate * E[g^2]_{t-1} + (1 - decay_rate) * g_t^2
        state.v[i] = _decay_rate * state.v[i] + (1.0f - _decay_rate) * grad * grad;

        // Update the weights: w_{t+1} = w_t - (learning_rate / sqrt(E[g^2]_t + epsilon)) * g_t
        weights[i] -= (_learning_rate / std::sqrt(state.v[i] + _epsilon)) * grad;

        // Reset the gradient to zero after applying the update
        grads[i] = 0.0f;
    }
}

OptimizerMetadata RMSpropOptimizer::metadata() const {
    return OptimizerMetadata{
        "rmsprop",
        _learning_rate,
        _learning_rate,
        false,
        {{"decay_rate", _decay_rate},
         {"epsilon", _epsilon},
         {"weight_decay", _weight_decay}}};
}
