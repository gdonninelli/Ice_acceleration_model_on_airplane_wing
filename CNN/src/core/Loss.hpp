/**
 * @file Loss.hpp
 * @brief Header file for the Loss functions.
 * Declares the loss functions for training the CNN model.
 */

#ifndef LOSS_HPP
#define LOSS_HPP

#include "Tensor.hpp"
#include <memory>

/**
 * @brief Namespace for loss functions used in training the CNN model.
 */
namespace Loss {

/**
 * @brief Computes the forward pass of the SIMM loss function, which combines a data term and a physics term.
 * @param preds A shared pointer to the Tensor containing the model's predictions.
 * @param targets  A shared pointer to the Tensor containing the real target values.
 * @param alphas A shared pointer to the Tensor containing the alpha values.
 * @param lambda_val A float scalar representing the regularization strength (weight) for the physics term.
 * @return float The computed scalar loss value (float).
 */
float simm_forward(const std::shared_ptr<Tensor> &preds,
                   const std::shared_ptr<Tensor> &targets,
                   const std::shared_ptr<Tensor> &alphas, 
                   float lambda_val);

/**
 * @brief Computes the backward pass of the SIMM loss function with respect to the predictions.
 * @param preds A shared pointer to the Tensor containing the model's predictions.  
 * @param targets A shared pointer to the Tensor containing the real target values.
 * @param alphas A shared pointer to the Tensor containing the alpha values.
 * @param lambda_val The regularization strength for the physics term.
 * @return A shared pointer to a Tensor containing the gradients of the loss w.r.t. `preds`.
 */
std::shared_ptr<Tensor> simm_backward(const std::shared_ptr<Tensor> &preds,
                                      const std::shared_ptr<Tensor> &targets,
                                      const std::shared_ptr<Tensor> &alphas,
                                      float lambda_val);

} // namespace Loss

#endif // LOSS_HPP