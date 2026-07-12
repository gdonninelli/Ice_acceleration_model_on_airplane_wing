/**
 * @file CNNModel.hpp
 * @brief Header file for the CNNModel class.
 * Defines the CNNModel class, responsible for implementing the overall structure and functionality of the CNN.
 */

#ifndef CNNMODEL_HPP
#define CNNMODEL_HPP

#include "layers/Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include "core/Tensor.hpp"
#include <memory>
#include <string>
#include <vector>

/** @class CNNModel
 * @brief Implements the overall structure and functionality of the CNN.
 */
class CNNModel {
    private:
    std::vector<std::shared_ptr<Layer>> conv_blocks; // stores convolutional layers
    std::vector<std::shared_ptr<Layer>> dense_blocks; // stores dense layers
    std::shared_ptr<Layer> concatenate_layer; // layer to concatenate convolutional and scalar features
    std::shared_ptr<Optimizer> optimizer; // optimizer for training
    float _lambda_val; // regularization term weight for the physics loss component

    public:
    /** 
     * @brief Constructor for the CNNModel class.
     * @param optimizer The optimizer to use for training.
     * @param lambda_val The weight for the physics loss component.
     */
    CNNModel(std::shared_ptr<Optimizer> optimizer, float lambda_val = 0.25f);

    /** 
     * @brief Adds a convolutional layer to the model.
     * @param layer A shared pointer to the Conv2DLayer to add.
     */
    void add_conv_layer(std::shared_ptr<Layer> layer);
    
    /** 
     * @brief Adds a dense layer to the model.
     * @param layer A shared pointer to the DenseLayer to add.
     */
    void add_dense_layer(std::shared_ptr<Layer> layer);
    
    /** 
     * @brief Sets the concatenate layer for the model.
     * @param layer A shared pointer to the concatenate layer to set.
     */
    void set_concatenate_layer(std::shared_ptr<Layer> layer);
    
    /** 
     * @brief Clears the concatenate layer from the model.
     */
    void clear_concatenate_layer();

    /** 
     * @brief Performs the forward pass through the model.
     * @param sdf_input A shared pointer to the Tensor containing the SDF input data.
     * @param scalar_input A shared pointer to the Tensor containing the scalar input data.
     * @return A shared pointer to the final output Tensor of the model.
     */
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> sdf_input,
                                    std::shared_ptr<Tensor> scalar_input);

    /** 
     * @brief Performs the backward pass through the model.
     * @param loss_grad A shared pointer to the Tensor containing the gradients of the loss function.
     */
    void backward(std::shared_ptr<Tensor> loss_grad);

    /** 
     * @brief Updates all learnable parameters in the model.
     */
    void update();

    /** 
     * @brief Zeros all learnable parameter gradients in the model.
     */
    void zero_grad();

    /** 
     * @brief Synchronizes gradients across all MPI processes.
     */
    void synchronize_gradients();

    /** 
     * @brief Broadcasts initial weights from the root process to all other processes.
     * @param root_rank The rank of the process that holds the initial weights to be broadcast.
     */
    void broadcast_initial_weights(int root_rank);

    /**
     * @brief Performs a single training step.
     * 
     * @param sdf_batch The batch of SDF inputs.
     * @param scalar_batch The batch of scalar inputs.
     * @param target_batch The batch of target outputs.
     * @return The loss value for the training step.
     */
    float train_step(std::shared_ptr<Tensor> sdf_batch,
                     std::shared_ptr<Tensor> scalar_batch,
                     std::shared_ptr<Tensor> target_batch);

    /** 
     * @brief Performs a prediction using the current model weights.
     * @param sdf_input A shared pointer to the Tensor containing the SDF input data.
     * @param scalar_input A shared pointer to the Tensor containing the scalar input data.
     * @return A shared pointer to the predicted output Tensor.
     */
    std::shared_ptr<Tensor> predict(std::shared_ptr<Tensor> sdf_input,
                                    std::shared_ptr<Tensor> scalar_input);

    /** 
     * @brief Exports the model's learnable parameters (weights and biases) to a file.
     * @param filepath The path to the file where weights will be saved.
     */
    void export_weights(const std::string& filepath) const;

    /** 
     * @briefImports learnable parameters (weights and biases) from a file.
     * @param filepath The path to the file from which weights will be loaded.
     */
    void import_weights(const std::string& filepath);
};

#endif // CNNMODEL_HPP