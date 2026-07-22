/**
 * @file Loss.cpp
 * @brief Implementation file for the Loss functions.
 * Defines the loss functions for training the CNN model.
 */

#include "Loss.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace {
constexpr float kMaskLimitRad = 0.174533f; // approx. 10 degrees in radians
constexpr float kTwoPi = std::numbers::pi_v<float> * 2.0f;

void validate_simm_parameters(float lambda_val,
                              float target_mean,
                              float target_std) {
  if (!std::isfinite(lambda_val) || lambda_val < 0.0f) {
    throw std::invalid_argument("SIMM lambda must be finite and non-negative.");
  }
  if (!std::isfinite(target_mean) || !std::isfinite(target_std) ||
      target_std <= 0.0f) {
    throw std::invalid_argument(
        "SIMM target normalization must be finite with positive standard deviation.");
  }
}
} // namespace

namespace Loss {

float simm_forward(const std::shared_ptr<Tensor> &preds,
                   const std::shared_ptr<Tensor> &targets,
                   const std::shared_ptr<Tensor> &alphas,
                   float lambda_val,
                   float target_mean,
                   float target_std) {
  if (!preds || !targets || !alphas) {
    throw std::invalid_argument("Loss tensors must not be null.");
  }

  if (preds->size() != targets->size() || preds->size() != alphas->size()) {
    throw std::invalid_argument(
        "preds, targets and alphas must have the same flattened size.");
  }
  validate_simm_parameters(lambda_val, target_mean, target_std);

  const size_t n = preds->size(); // total number of elements in the tensors
  if (n == 0) {
    throw std::invalid_argument("Loss cannot be computed for empty tensors.");
  }

  // Get raw pointers to the data for efficient access
  float *pred_ptr = preds->get_data();
  float *target_ptr = targets->get_data();
  float *alpha_ptr = alphas->get_data();

  // Accumulate the data term and physics term separately
  float data_term = 0.0f;
  float physics_term = 0.0f;

  // Compute the loss terms
  for (size_t i = 0; i < n; ++i) { // iterate over all elements in the flattened tensors
    const float pred = pred_ptr[i];
    const float target = target_ptr[i];
    const float alpha = alpha_ptr[i];

    // Data term: MSE between prediction and target
    const float data_diff = pred - target;
    data_term += data_diff * data_diff;

    // Express the physical prior in the same standardized target space as the data term.
    const float mask = (std::abs(alpha) <= kMaskLimitRad) ? 1.0f : 0.0f;
    const float standardized_physics_target =
        ((kTwoPi * alpha) - target_mean) / target_std;
    const float physics_diff = pred - standardized_physics_target;
    physics_term += mask * physics_diff * physics_diff;
  }

  // Calculate the final loss as the average of the data term and the weighted physics term
  return (data_term / static_cast<float>(n)) +
         lambda_val * (physics_term / static_cast<float>(n));
}

std::shared_ptr<Tensor> simm_backward(const std::shared_ptr<Tensor> &preds,
                                      const std::shared_ptr<Tensor> &targets,
                                      const std::shared_ptr<Tensor> &alphas,
                                      float lambda_val,
                                      float target_mean,
                                      float target_std) {
  if (!preds || !targets || !alphas) {
    throw std::invalid_argument("Loss tensors must not be null.");
  }

  if (preds->size() != targets->size() || preds->size() != alphas->size()) {
    throw std::invalid_argument(
        "preds, targets and alphas must have the same flattened size.");
  }
  validate_simm_parameters(lambda_val, target_mean, target_std);

  const size_t n = preds->size(); // total number of elements in the tensors
  if (n == 0) {
    throw std::invalid_argument(
        "Loss gradient cannot be computed for empty tensors.");
  }

  // Create a new tensor to hold the gradients
  auto grad = std::make_shared<Tensor>(preds->get_shape());

  // Get raw pointers to the data for efficient access
  float *pred_ptr = preds->get_data();
  float *target_ptr = targets->get_data();
  float *alpha_ptr = alphas->get_data();
  float *grad_ptr = grad->get_data();

  // Pre-calculate the inverse of N for efficiency within the loop
  const float inv_n = 1.0f / static_cast<float>(n);

  // Compute the gradients for each element
  for (size_t i = 0; i < n; ++i) { // iterate over all elements in the flattened tensors
    const float pred = pred_ptr[i]; 
    const float target = target_ptr[i];
    const float alpha = alpha_ptr[i];

    // Mask to determine if the physics term should contribute to the gradient
    const float mask = (std::abs(alpha) <= kMaskLimitRad) ? 1.0f : 0.0f;

    // Gradient of the data term: d(data_term)/d(pred) = 2 * (pred - target) / N
    const float grad_data = 2.0f * inv_n * (pred - target);
    // Gradient of the physics term: d(physics_term)/d(pred) = 2 * lambda * mask * (pred - 2*pi*alpha) / N
    const float standardized_physics_target =
        ((kTwoPi * alpha) - target_mean) / target_std;
    const float grad_physics = 2.0f * lambda_val * mask * inv_n *
                               (pred - standardized_physics_target);
    grad_ptr[i] = grad_data + grad_physics;
  }

  return grad;
}

float physical_mse(const std::shared_ptr<Tensor>& preds,
                   const std::shared_ptr<Tensor>& targets,
                   float target_std) {
  if (!preds || !targets) {
    throw std::invalid_argument("MSE tensors must not be null.");
  }
  if (preds->size() != targets->size() || preds->size() == 0) {
    throw std::invalid_argument("MSE tensors must have the same non-zero size.");
  }
  if (!std::isfinite(target_std) || target_std <= 0.0f) {
    throw std::invalid_argument("MSE target standard deviation must be positive and finite.");
  }

  const float* pred_ptr = preds->get_data();
  const float* target_ptr = targets->get_data();
  double sum = 0.0;
  for (size_t i = 0; i < preds->size(); ++i) {
    const double difference =
        static_cast<double>(pred_ptr[i] - target_ptr[i]) * target_std;
    sum += difference * difference;
  }
  return static_cast<float>(sum / static_cast<double>(preds->size()));
}

} // namespace Loss
