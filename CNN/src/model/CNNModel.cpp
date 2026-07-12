/**
 * @file CNNModel.cpp
 * @brief Implementation file for the CNNModel class.
 * Implements the methods for the CNNModel class, responsible for implementing the overall structure and functionality of the CNN.
 */

#include "CNNModel.hpp"
#include "layers/ConcatenateLayer.hpp"
#include "layers/Conv2DLayer.hpp"
#include "layers/DenseLayer.hpp"
#include "core/Loss.hpp"
#include <fstream>
#include <algorithm>
#include <limits>
#include <mpi.h>
#include <stdexcept>


namespace {
    /**
    * @brief Writes a Tensor's data and metadata to an output stream in binary format.
    * @param out The output file stream to write to.
    * @param tensor A shared pointer to the Tensor to write.
    */
    void write_tensor_data(std::ofstream& out, const std::shared_ptr<Tensor>& tensor) {
        const bool has_tensor = static_cast<bool>(tensor);
        out.write(reinterpret_cast<const char*>(&has_tensor), sizeof(has_tensor));
    
        if (!has_tensor) {
            return;
        }

        // Write the tensor's shape and data
        const std::vector<size_t> shape = tensor->get_shape();
        const size_t rank = shape.size(); // number of dimensions
        out.write(reinterpret_cast<const char*>(&rank), sizeof(rank)); // write the rank of the tensor
        out.write(reinterpret_cast<const char*>(shape.data()), sizeof(size_t) * rank); // write the shape dimensions

        // Serialize the tensor data as raw floats
        const size_t n = tensor->size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n)); // write the total count
        out.write(reinterpret_cast<const char*>(tensor->get_data()), sizeof(float) * n); // write the raw data
    }

    /**
     * @brief Writes a layer's parameters to an output stream.
     * @param out The output file stream to write to.
     * @param layer A shared pointer to the layer whose parameters to write.
     */
    void write_layer_params(std::ofstream& out, const std::shared_ptr<Layer>& layer) {
        // Get layer name and serialize its size and content
        const std::string layer_name = layer->get_layer_name();
        const size_t name_len = layer_name.size();
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(layer_name.c_str(), static_cast<std::streamsize>(name_len));

        // Pointers to the weights and biases tensors (if they exist)
        std::shared_ptr<Tensor> weights;
        std::shared_ptr<Tensor> biases;

        // Use dynamic_pointer_cast to check the layer type and get the corresponding weights and biases tensors
        if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
            weights = conv->get_weights_tensor();
            biases = conv->get_biases_tensor();
        } else if (auto dense = std::dynamic_pointer_cast<DenseLayer>(layer)) {
            weights = dense->get_weights_tensor();
            biases = dense->get_biases_tensor();
        }

        // Write the serialized weight and bias tensors
        write_tensor_data(out, weights);
        write_tensor_data(out, biases);
    }

    /**
     * @brief Helper struct to temporarily store tensor data read from a file.
     */
    struct TensorPayload {
        bool present = false; // flag indicating if this tensor was present in the file
        std::vector<size_t> shape;
        std::vector<float> data;
    };

    /**
     * @brief Reads a specific number of bytes from an input stream into a destination buffer.
     * 
     * @param in The input file stream
     * @param dest A pointer to the destination buffer.
     * @param bytes The number of bytes to read
     */
    void read_exact(std::ifstream& in, void* dest, size_t bytes) {
        in.read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(bytes));
        if (!in) {
            throw std::runtime_error("Failed to read from weights file.");
        }
    }

    /**
     * @brief Reads a Tensor's data and metadata from an input stream.
     * @param in The input file stream.
     * @return A TensorPayload struct containing the deserialized tensor data.
     */
    TensorPayload read_tensor_payload(std::ifstream& in) {
        bool has_tensor = false;
        // Read the presence flag for the tensor
        read_exact(in, &has_tensor, sizeof(has_tensor));
        if (!has_tensor) {
            return {};
        }

        // Read the tensor rank (number of dimensions).
        size_t rank = 0;
        read_exact(in, &rank, sizeof(rank));

        // Read the shape of the tensor (size of each dimension).
        std::vector<size_t> shape(rank);
        if (rank > 0) {
            read_exact(in, shape.data(), sizeof(size_t) * rank);
        }

        // Read the total number of elements in the tensor.
        size_t count = 0;
        read_exact(in, &count, sizeof(count));

        // Validate the count against the shape
        const size_t expected = Tensor::calculate_size(shape);
        if (expected != count) {
            throw std::runtime_error("Tensor element count mismatch while importing weights.");
        }

        // Read the tensor data buffer.
        std::vector<float> data(count);
        if (count > 0) {
            read_exact(in, data.data(), sizeof(float) * count);
        }

        // Populate and return the payload.
        TensorPayload payload;
        payload.present = true;
        // Use move for efficiency
        payload.shape = std::move(shape);
        payload.data = std::move(data);
        return payload;
    }

    /**
     * @brief Assigns deserialized tensor data to a target Tensor object.
     * @param payload The TensorPayload containing the data to assign.
     * @param tensor A shared pointer to the target Tensor object.
     */
    void assign_tensor_data(const TensorPayload& payload,
                            const std::shared_ptr<Tensor>& tensor) {
        
        if (!payload.present) {
            throw std::runtime_error("Missing tensor in weights file.");
        }
        
        if (!tensor) {
            throw std::runtime_error("Target tensor is null.");
        }

        if (tensor->get_shape() != payload.shape) {
            throw std::runtime_error("Shape mismatch while importing weights.");
        }
    
        if (tensor->size() != payload.data.size()) {
            throw std::runtime_error("Element count mismatch while importing weights.");
        }

        // Copy the data from the payload into the tensor's data buffer.
        std::copy(payload.data.begin(), payload.data.end(), tensor->get_data());
        // Reset gradients after loading weights, as they are not loaded from file.
        tensor->zero_grad();
    }


    /**
     * @brief Checks if the MPI environment has been initialized.
     * @return True if MPI is initialized, otherwise false.
     */
    bool mpi_ready() {
        int initialized = 0;
        MPI_Initialized(&initialized);
        return initialized != 0;
    }

    /**
     * @brief Gets the total number of MPI processes in the world communicator.
     * @return The number of processes. Returns 1 if MPI is not initialized.
     */
    int mpi_world_size() {
        int world_size = 1; // number of processors active
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        return world_size;
    }

    /**
     * @brief Converts a size_t count to an int for MPI calls, with a check to prevent overflow.
     * @param count The size_t count to check.
     * @return The count as an int if it fits, otherwise throws an exception.
     */
    int checked_mpi_count(size_t count) {
        if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("MPI buffer too large for int-based count.");
        }
        return static_cast<int>(count);
    }

    /**
     * @brief Performs an in-place Allreduce operation to average gradients across MPI processes.
     * @param grad Pointer to the gradient buffer to be averaged.
     * @param count Number of elements in the buffer.
     * @param world_size Total number of MPI processes.
     */
    void allreduce_average(float* grad, size_t count, int world_size) {
        if (!grad || count == 0 || world_size <= 1) {
            return;
        }
        const int mpi_count = checked_mpi_count(count);
        
        // Perform MPI_Allreduce to sum gradients in-place
        MPI_Allreduce(MPI_IN_PLACE, grad, mpi_count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); // sum in place
        
        // Average the summed gradients by dividing by the world size.
        const float inv_world = 1.0f / static_cast<float>(world_size);
        for (size_t i = 0; i < count; ++i) {
            grad[i] *= inv_world;
        }
    }

    /**
     * @brief Broadcasts a buffer of data from a root process to all other processes.
     * @param data Pointer to the data buffer to broadcast.
     * @param count Number of elements in the buffer.
     * @param root_rank Number of the process that will send the data.
     */
    void broadcast_buffer(float* data, size_t count, int root_rank) {
        if (!data || count == 0) {
            return;
        }
        const int mpi_count = checked_mpi_count(count);
        // Broadcast data from the root rank to all processes
        MPI_Bcast(data, mpi_count, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
    }
} // namespace

CNNModel::CNNModel(std::shared_ptr<Optimizer> optimizer_ptr, float lambda_val)
    : optimizer(optimizer_ptr), _lambda_val(lambda_val) {
    if (!optimizer) {
        throw std::invalid_argument("CNNModel requires a non-null optimizer.");
    }
    // Initialize the concatenate layer by default, but it can be replaced or cleared later.
    concatenate_layer = std::make_shared<ConcatenateLayer>();
}

void CNNModel::add_conv_layer(std::shared_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("add_conv_layer received null layer.");
    }
    // Add the layer to the end of the convolutional blocks vector
    conv_blocks.push_back(layer);
}

void CNNModel::add_dense_layer(std::shared_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("add_dense_layer received null layer.");
    }
    // Add the layer to the end of the dense blocks vector
    dense_blocks.push_back(layer);
}

void CNNModel::set_concatenate_layer(std::shared_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("set_concatenate_layer received null layer.");
    }
    // Assign the new concatenate layer
    concatenate_layer = layer;
}

void CNNModel::clear_concatenate_layer() {
    // Release the shared pointer, making the concatenate layer null
    concatenate_layer.reset();
}

std::shared_ptr<Tensor> CNNModel::forward(std::shared_ptr<Tensor> sdf_input,
                                          std::shared_ptr<Tensor> scalar_input) {
    if (!sdf_input) {
        throw std::invalid_argument("CNNModel forward received null input.");
    }

    // Start with the primary input
    auto x = sdf_input;
    // Pass through convolutional blocks
    for (const auto& layer : conv_blocks) {
        x = layer->forward({x}); // conv layers expect a single input tensor
    }

    // If a concatenate layer is set, concatenate the output of the conv blocks with the scalar input
    if (concatenate_layer) {
        if (!scalar_input) {
            throw std::invalid_argument("CNNModel forward received null scalar_input.");
        }
        x = concatenate_layer->forward({x, scalar_input}); // concatenate layer expects two inputs
    }

    // Pass through dense blocks
    for (const auto& layer : dense_blocks) {
        x = layer->forward({x}); // dense layers expect a single input tensor
    }

    return x;
}

void CNNModel::backward(std::shared_ptr<Tensor> loss_grad) {
    if (!loss_grad) {
        throw std::invalid_argument("CNNModel backward received null loss_grad.");
    }

    // Start backpropagation from the end of the network
    auto grad = loss_grad;

    // Backpropagate through dense blocks in reverse order
    for (auto it = dense_blocks.rbegin(); it != dense_blocks.rend(); ++it) {
        grad = (*it)->backward(grad)[0]; // expect only one input gradient for dense layers
    }

    // If a concatenate layer is present, backpropagate through it
    if (concatenate_layer) {
        const auto split_grads = concatenate_layer->backward(grad);
        if (split_grads.empty()) {
            throw std::runtime_error("CNNModel concatenate backward returned empty gradients.");
        }
        // The first gradient corresponds to the output of the conv blocks
        // The second corresponds to the scalar input
        grad = split_grads[0]; // we need only this to continue backprop through conv blocks
    }

    // Backpropagate through convolutional blocks in reverse order
    for (auto it = conv_blocks.rbegin(); it != conv_blocks.rend(); ++it) {
        grad = (*it)->backward(grad)[0];
    }
}

void CNNModel::update() {
    // Apply gradient clipping and weight updates to a single layer's parameters
    auto clip_layer = [](const std::shared_ptr<Layer>& layer) {
        auto clip_tensor = [](const std::shared_ptr<Tensor>& t) {
            
            if (t) { // if the tensor is null, there's nothing to clip.
                float* grad = t->get_grad();
                size_t size = t->size();
                // Clip each element of the gradient to be within [-1, 1]
                for (size_t i = 0; i < size; ++i) {
                    if (grad[i] > 1.0f) grad[i] = 1.0f;
                    else if (grad[i] < -1.0f) grad[i] = -1.0f;
                }
            }
        };

        // If the layer is a Conv2DLayer or DenseLayer, clip its weights and biases gradients
        if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
            clip_tensor(conv->get_weights_tensor());
            clip_tensor(conv->get_biases_tensor());
        } else if (auto dense = std::dynamic_pointer_cast<DenseLayer>(layer)) {
            clip_tensor(dense->get_weights_tensor());
            clip_tensor(dense->get_biases_tensor());
        }
    };

    for (auto& layer : conv_blocks) { // iterate over convolutional layers
        clip_layer(layer);
        layer->update_weights(optimizer);
    }

    if (concatenate_layer) { // if a concatenate layer is set, update its weights as well
        clip_layer(concatenate_layer);
        concatenate_layer->update_weights(optimizer);
    }

    for (auto& layer : dense_blocks) { // iterate over dense layers
        clip_layer(layer);
        layer->update_weights(optimizer);
    }
}

void CNNModel::zero_grad() {
    // Helper lambda to zero gradients for a single layer's parameters
    auto zero_layer = [](const std::shared_ptr<Layer>& layer) {
        if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
            if (auto w = conv->get_weights_tensor()) w->zero_grad();
            if (auto b = conv->get_biases_tensor()) b->zero_grad();
        } else if (auto dense = std::dynamic_pointer_cast<DenseLayer>(layer)) {
            if (auto w = dense->get_weights_tensor()) w->zero_grad();
            if (auto b = dense->get_biases_tensor()) b->zero_grad();
        }
    };

    for (const auto& layer : conv_blocks) { // iterate over convolutional layers
        zero_layer(layer);
    }

    for (const auto& layer : dense_blocks) { // iterate over dense layers
        zero_layer(layer);
    }
}

void CNNModel::synchronize_gradients() {
    // Check if MPI is ready and if there's more than one process
    if (!mpi_ready()) {
        return;
    }

    // Get the total number of processes in the MPI world communicator
    const int world_size = mpi_world_size();
    if (world_size <= 1) {
        return;
    }

    // Helper lambda to synchronize gradients for a single layer's parameters
    auto sync_layer = [&](const std::shared_ptr<Layer>& layer) {
        
        // Check if the layer is a Conv2DLayer or DenseLayer and perform allreduce on its weights and biases gradients
        if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
            auto weights = conv->get_weights_tensor();
            auto biases = conv->get_biases_tensor();
            if (weights) {
                allreduce_average(weights->get_grad(), weights->size(), world_size);
            }
            if (biases) {
                allreduce_average(biases->get_grad(), biases->size(), world_size);
            }
            return;
        }

        if (auto dense = std::dynamic_pointer_cast<DenseLayer>(layer)) {
            auto weights = dense->get_weights_tensor();
            auto biases = dense->get_biases_tensor();
            if (weights) {
                allreduce_average(weights->get_grad(), weights->size(), world_size);
            }
            if (biases) {
                allreduce_average(biases->get_grad(), biases->size(), world_size);
            }
        }
    };

    for (const auto& layer : conv_blocks) { // iterate over convolutional layers
        sync_layer(layer);
    }

    for (const auto& layer : dense_blocks) { // iterate over dense layers
        sync_layer(layer);
    }
}

void CNNModel::broadcast_initial_weights(int root_rank) {
    if (!mpi_ready()) {
        return;
    }

    // Get the total number of processes in the MPI world communicator
    const int world_size = mpi_world_size();
    if (world_size <= 1) {
        return;
    }
    
    if (root_rank < 0 || root_rank >= world_size) {
        throw std::invalid_argument("broadcast_initial_weights root_rank is out of range.");
    }

    // Helper lambda to broadcast initial weights for a single layer
    auto broadcast_layer = [&](const std::shared_ptr<Layer>& layer) {

        // Check if the layer is a Conv2DLayer or DenseLayer and broadcast its weights and biases from the root process to all other processes
        if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
            auto weights = conv->get_weights_tensor();
            auto biases = conv->get_biases_tensor();
            if (weights) {
                broadcast_buffer(weights->get_data(), weights->size(), root_rank);
            }
            if (biases) {
                broadcast_buffer(biases->get_data(), biases->size(), root_rank);
            }
            return;
        }

        if (auto dense = std::dynamic_pointer_cast<DenseLayer>(layer)) {
            auto weights = dense->get_weights_tensor();
            auto biases = dense->get_biases_tensor();
            if (weights) {
                broadcast_buffer(weights->get_data(), weights->size(), root_rank);
            }
            if (biases) {
                broadcast_buffer(biases->get_data(), biases->size(), root_rank);
            }
        }
    };

    for (const auto& layer : conv_blocks) { // iterate over convolutional layers
        broadcast_layer(layer);
    }
    for (const auto& layer : dense_blocks) { // iterate over dense layers
        broadcast_layer(layer);
    }
}

float CNNModel::train_step(std::shared_ptr<Tensor> sdf_batch,
                           std::shared_ptr<Tensor> scalar_batch,
                           std::shared_ptr<Tensor> target_batch) {
    if (!sdf_batch || !scalar_batch || !target_batch) {
        throw std::invalid_argument("CNNModel train_step received null batch tensor.");
    }

    // Get model predictions
    auto preds = forward(sdf_batch, scalar_batch);

    // Since the SIMM loss requires the alpha values (AoA) from the scalar input, we need to extract them before computing the loss.
    const std::vector<size_t> scalar_shape = scalar_batch->get_shape();
    if (scalar_shape.size() != 2 || scalar_shape[1] < 1) {
        throw std::invalid_argument("scalar_batch must be 2D with at least one feature (AoA in column 0).");
    }

    // Create a tensor for the alpha values (AoA) with the same batch size as the scalar input
    const size_t batch_size = scalar_shape[0];
    auto alphas = std::make_shared<Tensor>(std::vector<size_t>{batch_size, 1});

    // Extract the alpha values (AoA) from the first column of the scalar input and store them in the alphas tensor
    float* scalar_ptr = scalar_batch->get_data();
    float* alpha_ptr = alphas->get_data();
    const size_t scalar_features = scalar_shape[1];

    for (size_t n = 0; n < batch_size; ++n) {
        // Copy the first feature for each batch sample
        alpha_ptr[n] = scalar_ptr[n * scalar_features + 0];
    }

    // Compute the SIMM loss using predictions, targets, and extracted alphas
    const float loss = Loss::simm_forward(preds, target_batch, alphas, _lambda_val);
    // Calculate the gradient of the loss with respect to the predictions
    auto loss_grad = Loss::simm_backward(preds, target_batch, alphas, _lambda_val);
    // Zero all learnable parameter gradients before accumulating new ones in the backward pass
    zero_grad();
    // Propagate this gradient back through the network
    backward(loss_grad);
    
    synchronize_gradients();
    update();
    
    return loss;
}

std::shared_ptr<Tensor> CNNModel::predict(std::shared_ptr<Tensor> sdf_input,
                                          std::shared_ptr<Tensor> scalar_input) {
    return forward(sdf_input, scalar_input);
}

void CNNModel::export_weights(const std::string& filepath) const {
    // Open the file in binary mode for writing
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open export file: " + filepath);
    }

    // Determine the total number of layers that have parameters to export (conv + concatenate + dense)
    const size_t total_layers = conv_blocks.size() + (concatenate_layer ? 1 : 0) + dense_blocks.size();
    // Write the total layer count first
    out.write(reinterpret_cast<const char*>(&total_layers), sizeof(total_layers));

    for (const auto& layer : conv_blocks) { // iterate over convolutional layers
        write_layer_params(out, layer);
    }

    if (concatenate_layer) { // include concatenate layer if it exists
        write_layer_params(out, concatenate_layer);
    }

    for (const auto& layer : dense_blocks) { // iterate over dense layers
        write_layer_params(out, layer);
    }
}

void CNNModel::import_weights(const std::string& filepath) {
    // Open the file in binary mode for reading
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + filepath);
    }

    // Read the total number of layers (conv + concatenate + dense)
    size_t total_layers = 0;
    read_exact(in, &total_layers, sizeof(total_layers));

    // Reconstruct the ordered list of layers as they were exported
    std::vector<std::shared_ptr<Layer>> ordered_layers;
    ordered_layers.reserve(conv_blocks.size() + (concatenate_layer ? 1 : 0) + dense_blocks.size());
    
    for (const auto& layer : conv_blocks) { // iterate over convolutional layers
        ordered_layers.push_back(layer);
    }

    if (concatenate_layer) { // include concatenate layer if it exists
        ordered_layers.push_back(concatenate_layer);
    }

    for (const auto& layer : dense_blocks) { // iterate over dense layers
        ordered_layers.push_back(layer);
    }

    if (total_layers != ordered_layers.size()) {
        throw std::runtime_error("Layer count mismatch while importing weights.");
    }

    // Process each layer sequentially as read from the file
    for (const auto& layer : ordered_layers) {
        // Read layer name length and name
        size_t name_len = 0;
        read_exact(in, &name_len, sizeof(name_len));

        std::string layer_name(name_len, '\0'); // create string of correct size
        
        if (name_len > 0) {
            // Read the layer name from the file
            read_exact(in, &layer_name[0], name_len);
        }

        const std::string expected_name = layer->get_layer_name();
        if (layer_name != expected_name) {
            throw std::runtime_error("Layer name mismatch: expected " + expected_name + ", got " + layer_name);
        }

        // Read the serialized weights and biases tensors for this layer
        TensorPayload weights_payload = read_tensor_payload(in);
        TensorPayload biases_payload = read_tensor_payload(in);

        // Assign the loaded data to the layer's parameters
        if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
            assign_tensor_data(weights_payload, conv->get_weights_tensor());
            assign_tensor_data(biases_payload, conv->get_biases_tensor());
        } else if (auto dense = std::dynamic_pointer_cast<DenseLayer>(layer)) {
            assign_tensor_data(weights_payload, dense->get_weights_tensor());
            assign_tensor_data(biases_payload, dense->get_biases_tensor());
        }
    }
}