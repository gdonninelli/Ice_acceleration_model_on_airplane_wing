/**
 * @file main.cpp
 * @brief Main entry point for training the CNN model.
 * This file sets up the data loaders, model architecture, and training loop for the CNN model
 */

#include "src/layers/DenseLayer.hpp"
#include "src/layers/ActivationFactory.hpp"
#include "src/layers/Conv2DLayer.hpp"
#include "src/layers/FlattenLayer.hpp"
#include "src/layers/ConcatenateLayer.hpp"
#include "src/model/CNNModel.hpp"
#include "src/optimizers/AdamOptimizer.hpp"
#include "src/core/Loss.hpp"
#include "src/data/NPZDataLoader.hpp"
#include "src/core/Tensor.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <mpi.h>

namespace {
/**
 * @brief Calculates the output dimension of a convolution operation.
 * @param in_dim The input dimension (height or width).
 * @param kernel The kernel size.
 * @param stride The stride of the operation.
 * @param padding The padding applied to the input.
 * @return The calculated output dimension.
 */
size_t conv_out_dim(size_t in_dim, int kernel, int stride, int padding) {
    if (kernel <= 0 || stride <= 0 || padding < 0) {
        throw std::invalid_argument("Invalid convolution spec.");
    }
    const size_t k = static_cast<size_t>(kernel);
    const size_t s = static_cast<size_t>(stride);
    const size_t p = static_cast<size_t>(padding);

    if (in_dim + 2 * p < k) {
        throw std::invalid_argument("Convolution kernel larger than padded input.");
    }
    // Apply the formula to calculate the output dimension.
    return (in_dim + 2 * p - k) / s + 1;
}

/**
 * @brief Slices a batch tensor to extract a subset of samples.
 * @param batch A shared pointer to the input batch Tensor.
 * @param start The starting index (sample index) of the slice.
 * @param count The number of samples to extract.
 * @return A shared pointer to a new Tensor containing the sliced data.
 */
std::shared_ptr<Tensor> slice_batch_tensor(const std::shared_ptr<Tensor>& batch,
                                           size_t start,
                                           size_t count) {
    if (!batch) {
        throw std::invalid_argument("slice_batch_tensor received null tensor.");
    }

    const std::vector<size_t> shape = batch->get_shape();

    if (shape.empty()) {
        throw std::invalid_argument("slice_batch_tensor received empty shape.");
    }

    const size_t batch_size = shape[0]; // first dimension is assumed to be batch size

    if (start + count > batch_size) {
        throw std::invalid_argument("slice_batch_tensor range exceeds batch size.");
    }

    // Create the output tensor with the same shape but adjusted batch size.
    std::vector<size_t> out_shape = shape;
    out_shape[0] = count; // set the new batch size.
    auto output = std::make_shared<Tensor>(out_shape);

    if (count == 0) {
        return output;
    }

    // Calculate the stride (bytes or elements per sample) based on the total size and batch size.
    const size_t stride = batch->size() / batch_size;

    // Get raw data pointers for efficient copying.
    const float* src = batch->get_data() + start * stride; // source pointer.
    float* dst = output->get_data(); // destination pointer.

    // Copy the data for the requested slice.
    std::copy(src, src + count * stride, dst);
    return output;
}
} // namespace

/**
 * @brief Main function to drive the CNN model training.
 *
 * Orchestrates the entire training process:
 * 1. Initializes MPI environment.
 * 2. Loads training and validation datasets.
 * 3. Constructs the CNN model architecture by adding layers.
 * 4. Initializes the optimizer.
 * 5. Broadcasts initial model weights across MPI processes.
 * 6. Iterates through epochs, processing batches for training and validation.
 * 7. Reports training and validation loss.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return 0 on successful completion, 1 on error.
 */
int main(int argc, char** argv) {
    // Initialize MPI environment for distributed training.
    MPI_Init(&argc, &argv);
    int rank = 0; // rank of the current process
    int world_size = 1; // total number of MPI processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // get the total number of processes

    try {
        // Parse command-line options for the activation function.
        // Usage: cnn_executable [--activation leakyrelu|relu|tanh|sigmoid] [--alpha <value>]
        std::string activation_name = "leakyrelu"; // default activation
        float leaky_alpha = 0.05f; // negative slope, used only by leakyrelu
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--activation" && i + 1 < argc) {
                activation_name = argv[++i];
            } else if (arg == "--alpha" && i + 1 < argc) {
                leaky_alpha = std::stof(argv[++i]);
            } else {
                throw std::invalid_argument(
                    "Unknown or incomplete option '" + arg +
                    "'. Usage: cnn_executable [--activation leakyrelu|relu|tanh|sigmoid] [--alpha <value>]");
            }
        }

        if (!std::isfinite(leaky_alpha) || leaky_alpha < 0.0f) {
            throw std::invalid_argument("--alpha must be a finite, non-negative number.");
        }

        // Validate the activation name early (throws on unknown names) and report the choice.
        const std::string activation_label = make_activation(activation_name, leaky_alpha)->get_layer_name();
        if (rank == 0) {
            std::cout << "Using activation: " << activation_label << std::endl;
        }

        // Load the training and validation datasets from NPZ files.
        NPZDataLoader train_loader("dataset/cnn_dataset_train.npz");
        NPZDataLoader val_loader("dataset/cnn_dataset_test.npz");

        // Create an Adam optimizer instance with specified hyperparameters.
        auto optimizer = std::make_shared<AdamOptimizer>(1e-5f);
        // Instantiate the CNN model, passing the optimizer and lambda value for the loss.
        CNNModel model(optimizer, 0.25f);

        // 1. Convolutional layer: 1 input channel, 8 output channels, kernel 5, stride 5, padding 0
        model.add_conv_layer(std::make_shared<Conv2DLayer>(1, 8, 5, 5, 0));
        model.add_conv_layer(make_activation(activation_name, leaky_alpha));
        model.add_conv_layer(std::make_shared<FlattenLayer>()); // to prepare for dense layers

        // 2. Concatenate layer (merges with Re and Angle of Attack)
        model.set_concatenate_layer(std::make_shared<ConcatenateLayer>());

        // Calculate the expected flattened feature size after the convolutional block
        size_t current_h = conv_out_dim(150, 5, 5, 0); // SDF input dim is 150x150
        size_t current_w = conv_out_dim(150, 5, 5, 0);
        // The flattened features are (output_channels_from_conv) * height * width.
        const size_t flat_features = 8 * current_h * current_w;
        
        // 3. Feed forward network: 2 hidden layers of size 128 and 64 while output is 1
        model.add_dense_layer(std::make_shared<DenseLayer>(static_cast<int>(flat_features + 2), 128));
        model.add_dense_layer(make_activation(activation_name, leaky_alpha));

        model.add_dense_layer(std::make_shared<DenseLayer>(128, 64));
        model.add_dense_layer(make_activation(activation_name, leaky_alpha));
        
        model.add_dense_layer(std::make_shared<DenseLayer>(64, 1));

        // Distribuite the initial weights from the root process (rank 0) to all other processes.
        model.broadcast_initial_weights(0);

        // Define batch sizes for local and global processing.
        const size_t local_batch = 64; // number of samples processed per MPI process per step.
        const size_t global_batch = local_batch * static_cast<size_t>(world_size); // total samples processed per step across all processes.
        
        // Calculate the number of training and validation steps per epoch.
        const size_t train_steps_per_epoch = train_loader.num_samples() / global_batch;
        const size_t val_steps_per_epoch = val_loader.num_samples() / global_batch;

        if (train_steps_per_epoch == 0) {
            throw std::runtime_error("Batch size larger than dataset.");
        }

        // Define the total number of training epochs.
        const int epochs = 100;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Reset data loaders at the beginning of each epoch to iterate from the start.
            train_loader.reset();
            val_loader.reset();
            
            double local_train_loss = 0.0; // accumulator for the training loss on the local process.
            unsigned long long local_train_seen = 0; // counter for the number of training samples processed on the local process.

            for (size_t step = 0; step < train_steps_per_epoch; ++step) { // iterate through training steps for the epoch
                // Get a global batch of data.
                auto batch = train_loader.get_batch(global_batch);
                // Determine the start index for the current process's local batch slice.
                const size_t start = static_cast<size_t>(rank) * local_batch;

                // Slice the global batch into local batches for this process.
                auto sdf_batch = slice_batch_tensor(batch[0], start, local_batch);
                auto scalar_batch = slice_batch_tensor(batch[1], start, local_batch);
                auto target_batch = slice_batch_tensor(batch[2], start, local_batch);

                // Perform one training step (forward, loss, backward, update).
                const float loss = model.train_step(sdf_batch, scalar_batch, target_batch);

                // Accumulate local training loss and count processed samples.
                local_train_loss += static_cast<double>(loss) * static_cast<double>(local_batch);
                local_train_seen += static_cast<unsigned long long>(local_batch);
            }

            double local_val_loss = 0.0; // accumulator for the validation loss on the local process.
            unsigned long long local_val_seen = 0; // counter for the number of validation samples processed on the local process.
            
            for (size_t step = 0; step < val_steps_per_epoch; ++step) { // iterate through validation steps for the epoch
                // Get a global batch of data.
                auto batch = val_loader.get_batch(global_batch);
                // Determine the start index for the current process's local batch slice.
                const size_t start = static_cast<size_t>(rank) * local_batch;

                // Slice the global batch into local batches.
                auto sdf_batch = slice_batch_tensor(batch[0], start, local_batch);
                auto scalar_batch = slice_batch_tensor(batch[1], start, local_batch);
                auto target_batch = slice_batch_tensor(batch[2], start, local_batch);

                // Perform forward pass for validation.
                auto preds = model.forward(sdf_batch, scalar_batch);
                
                // Extract alpha values for the SIMM loss (assuming AoA is the first scalar feature).
                auto alphas = std::make_shared<Tensor>(std::vector<size_t>{local_batch, 1});
                float* scalar_ptr = scalar_batch->get_data();
                float* alpha_ptr = alphas->get_data();

                // Assuming the angle of attack is the feature at index 0.
                for (size_t n = 0; n < local_batch; ++n) {
                    alpha_ptr[n] = scalar_ptr[n * 2 + 0]; 
                }

                // Default lambda is 0 so simm_forward acts as MSE over the prediction.
                const float loss = Loss::simm_forward(preds, target_batch, alphas, 0.25f);

                // Accumulate local validation loss and count processed samples.
                local_val_loss += static_cast<double>(loss) * static_cast<double>(local_batch);
                local_val_seen += static_cast<unsigned long long>(local_batch);
            }

            // Use MPI_Reduce to sum up local loss and sample counts across all processes.
            double global_train_loss = 0.0;
            unsigned long long global_train_seen = 0;
            double global_val_loss = 0.0;
            unsigned long long global_val_seen = 0;

            // Sum up training losses and counts.
            MPI_Reduce(&local_train_loss, &global_train_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_train_seen, &global_train_seen, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            // Sum up validation losses and counts.
            MPI_Reduce(&local_val_loss, &global_val_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_val_seen, &global_val_seen, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0 && global_train_seen > 0) {
                // Calculate average losses if any samples were processed.
                const double avg_train_loss = global_train_loss / static_cast<double>(global_train_seen);
                
                // Print epoch progress and loss metrics.
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                          << " | Train Loss: " << avg_train_loss;
                
                if (global_val_seen > 0) {
                    const double avg_val_loss = global_val_loss / static_cast<double>(global_val_seen);
                    std::cout << " | Val Loss: " << avg_val_loss;
                } else {
                    std::cout << " | Val Loss: n/a"; // indicate if no validation data was processed.
                }
                std::cout << std::endl;
            }
        }
    } catch (const std::exception& e) {
        // Catch any exceptions thrown during execution, report error on all ranks, and exit.
        std::cerr << "Training failed with exception on rank " << rank
                  << ": " << e.what() << std::endl;
        MPI_Finalize(); // clean up MPI resources before exiting.
        return 1; // indicate an error exit.
    }

    // Clean up the MPI environment before exiting the program.
    MPI_Finalize();
    // Indicate successful program execution.
    return 0;
}
