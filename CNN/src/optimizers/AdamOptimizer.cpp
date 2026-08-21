#include "AdamOptimizer.hpp"
#include <cmath>
#include <stdexcept>

AdamOptimizer::AdamOptimizer(float learning_rate,
                             float beta1,
                             float beta2,
                             float epsilon,
                             float weight_decay)
    : _learning_rate(learning_rate),
      _beta1(beta1),
      _beta2(beta2),
      _epsilon(epsilon),
      _weight_decay(weight_decay) {

    // Hyperparameter validation
    if (!std::isfinite(_learning_rate) || _learning_rate <= 0.0f) {
        throw std::invalid_argument("Adam learning_rate must be finite and positive.");
    }
    if (!std::isfinite(_beta1) || _beta1 < 0.0f || _beta1 >= 1.0f) {
        throw std::invalid_argument("Adam beta1 must be finite and in [0, 1).");
    }
    if (!std::isfinite(_beta2) || _beta2 < 0.0f || _beta2 >= 1.0f) {
        throw std::invalid_argument("Adam beta2 must be finite and in [0, 1).");
    }
    if (!std::isfinite(_epsilon) || _epsilon <= 0.0f) {
        throw std::invalid_argument("Adam epsilon must be finite and positive.");
    }
    if (!std::isfinite(_weight_decay) || _weight_decay < 0.0f) {
        throw std::invalid_argument("Adam weight_decay must be finite and non-negative.");
    }
}

void AdamOptimizer::apply_gradients(float* weights, float* grads, size_t size) {
    // If pointers are null or size is zero, there's nothing to do.
    if (!weights || !grads || size == 0) {
        return;
    }

    // Get or create the AdamState for this specific parameter array pointer.
    AdamState& state = _state_by_param_ptr[weights];

    // State initialization if the size of the state vectors does not match the current parameter size.
    if (state.m.size() != size || state.v.size() != size) {
        state.m.assign(size, 0.0f); // initialize first moment vector to zeros
        state.v.assign(size, 0.0f); // initialize second moment vector to zeros
        state.timestep = 0; // reset timestep for new parameter array
    }

    state.timestep += 1; // increment timestep

    // Increment the bias correction factors for the first and second moments
    const float bias_corr1 = 1.0f - std::pow(_beta1, static_cast<float>(state.timestep));
    const float bias_corr2 = 1.0f - std::pow(_beta2, static_cast<float>(state.timestep));

    // Update each parameter using the Adam update rule
    for (size_t i = 0; i < size; ++i) { // iterate over all elements in the parameter array
        float grad = grads[i];

        // Apply weight decay if specified (L2 regularization)
        if (_weight_decay != 0.0f) {
            grad += _weight_decay * weights[i];
        }

        // Update first moment using m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        state.m[i] = _beta1 * state.m[i] + (1.0f - _beta1) * grad;
        // Update second moment using v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        state.v[i] = _beta2 * state.v[i] + (1.0f - _beta2) * grad * grad;

        // Apply bias correction to the first and second moments
        const float m_hat = state.m[i] / bias_corr1;
        const float v_hat = state.v[i] / bias_corr2;

        // Update the weights using the Adam update rule: w_t = w_{t-1} - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        weights[i] -= _learning_rate * m_hat / (std::sqrt(v_hat) + _epsilon);
        
        // Reset the gradient to zero after applying the update
        grads[i] = 0.0f;
    }
}

OptimizerMetadata AdamOptimizer::metadata() const {
    return OptimizerMetadata{
        "adam",
        _learning_rate,
        _learning_rate,
        false,
        {{"beta1", _beta1},
         {"beta2", _beta2},
         {"epsilon", _epsilon},
         {"weight_decay", _weight_decay}}};
}
