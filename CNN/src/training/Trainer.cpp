#include "Trainer.hpp"
#include "core/Loss.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
bool mpi_ready() {
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Finalized(&finalized);
    }
    return initialized != 0 && finalized == 0;
}

struct DistributedInfo {
    int rank = 0;
    int size = 1;
};

DistributedInfo distributed_info(MPI_Comm communicator) {
    DistributedInfo info;
    if (mpi_ready()) {
        MPI_Comm_rank(communicator, &info.rank);
        MPI_Comm_size(communicator, &info.size);
    }
    return info;
}

struct Partition {
    size_t offset;
    size_t count;
};

Partition partition_batch(size_t batch_size, int rank, int world_size) {
    const size_t processes = static_cast<size_t>(world_size);
    const size_t base = batch_size / processes;
    const size_t remainder = batch_size % processes;
    const size_t rank_index = static_cast<size_t>(rank);
    return {
        rank_index * base + std::min(rank_index, remainder),
        base + (rank_index < remainder ? 1 : 0)};
}

void reduce_sum(double local_value,
                unsigned long long local_count,
                double& global_value,
                unsigned long long& global_count,
                MPI_Comm communicator) {
    if (!mpi_ready()) {
        global_value = local_value;
        global_count = local_count;
        return;
    }
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM,
                  communicator);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, communicator);
}

void throw_if_distributed_failure(const std::string& local_error,
                                  const std::string& phase,
                                  MPI_Comm communicator) {
    int local_failed = local_error.empty() ? 0 : 1;
    int any_failed = local_failed;
    if (mpi_ready()) {
        MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                      communicator);
    }
    if (any_failed != 0) {
        if (!local_error.empty()) {
            throw std::runtime_error(phase + ": " + local_error);
        }
        throw std::runtime_error(phase + " failed on another MPI rank.");
    }
}
} // namespace

Trainer::Trainer(MPI_Comm communicator) : _communicator(communicator) {}

TrainingResult Trainer::fit(CNNModel& model,
                            const Dataset& training_dataset,
                            std::span<const size_t> training_indices,
                            const Dataset& validation_dataset,
                            std::span<const size_t> validation_indices,
                            const NormalizationStats& normalization,
                            const LossConfig& loss_config,
                            const TrainingConfig& training_config,
                            uint64_t run_seed,
                            bool verbose) const {
    if (training_indices.empty() || validation_indices.empty()) {
        throw std::invalid_argument("Trainer requires non-empty training and validation sets.");
    }
    if (training_config.epochs == 0 || training_config.global_batch_size == 0) {
        throw std::invalid_argument("Training epochs and global batch size must be positive.");
    }

    const DistributedInfo distributed = distributed_info(_communicator);
    std::string parameter_error;
    try {
        model.parameters();
    } catch (const std::exception& error) {
        parameter_error = error.what();
    } catch (...) {
        parameter_error = "unknown parameter metadata error";
    }
    throw_if_distributed_failure(parameter_error, "Model parameter setup",
                                 _communicator);
    model.broadcast_initial_weights(0);
    std::vector<size_t> ordered_training(training_indices.begin(), training_indices.end());
    double final_training_objective = 0.0;
    std::vector<EpochMetrics> history;

    auto evaluate_validation = [&]() {
        double local_validation_sum = 0.0;
        unsigned long long local_validation_seen = 0;
        std::string validation_error;
        try {
            for (size_t batch_offset = 0;
                 batch_offset < validation_indices.size();
                 batch_offset += training_config.global_batch_size) {
                const size_t global_count = std::min(
                    training_config.global_batch_size,
                    validation_indices.size() - batch_offset);
                const Partition local = partition_batch(
                    global_count, distributed.rank, distributed.size);
                if (local.count == 0) {
                    continue;
                }

                const std::span<const size_t> local_indices(
                    validation_indices.data() + batch_offset + local.offset,
                    local.count);
                const DataBatch batch =
                    validation_dataset.make_batch(local_indices, normalization);
                const auto predictions = model.predict(batch.sdf, batch.scalars);
                const float mse = Loss::physical_mse(
                    predictions, batch.targets,
                    static_cast<float>(normalization.target_std));
                local_validation_sum += static_cast<double>(mse) * local.count;
                local_validation_seen +=
                    static_cast<unsigned long long>(local.count);
            }
        } catch (const std::exception& error) {
            validation_error = error.what();
        } catch (...) {
            validation_error = "unknown local validation error";
        }
        throw_if_distributed_failure(validation_error, "Validation",
                                     _communicator);

        double global_validation_sum = 0.0;
        unsigned long long global_validation_seen = 0;
        reduce_sum(local_validation_sum, local_validation_seen,
                   global_validation_sum, global_validation_seen,
                   _communicator);
        if (global_validation_seen == 0) {
            throw std::runtime_error("Validation processed no samples.");
        }
        return global_validation_sum /
               static_cast<double>(global_validation_seen);
    };

    for (size_t epoch = 0; epoch < training_config.epochs; ++epoch) {
        if (training_config.shuffle) {
            std::mt19937_64 generator(run_seed + epoch);
            std::shuffle(ordered_training.begin(), ordered_training.end(), generator);
        }

        double local_loss_sum = 0.0;
        unsigned long long local_seen = 0;
        for (size_t batch_offset = 0; batch_offset < ordered_training.size();
             batch_offset += training_config.global_batch_size) {
            const size_t global_count = std::min(
                training_config.global_batch_size,
                ordered_training.size() - batch_offset);
            const Partition local = partition_batch(
                global_count, distributed.rank, distributed.size);

            std::string local_error;
            try {
                model.zero_grad();
                if (local.count > 0) {
                    const std::span<const size_t> local_indices(
                        ordered_training.data() + batch_offset + local.offset,
                        local.count);
                    const DataBatch batch =
                        training_dataset.make_batch(local_indices, normalization);
                    const auto predictions = model.forward(batch.sdf, batch.scalars);
                    const float loss = Loss::simm_forward(
                        predictions, batch.targets, batch.alpha_radians,
                        loss_config.physics_weight,
                        static_cast<float>(normalization.target_mean),
                        static_cast<float>(normalization.target_std));
                    const auto gradient = Loss::simm_backward(
                        predictions, batch.targets, batch.alpha_radians,
                        loss_config.physics_weight,
                        static_cast<float>(normalization.target_mean),
                        static_cast<float>(normalization.target_std));
                    model.backward(gradient);
                    local_loss_sum += static_cast<double>(loss) * local.count;
                    local_seen += static_cast<unsigned long long>(local.count);
                }
            } catch (const std::exception& error) {
                local_error = error.what();
            } catch (...) {
                local_error = "unknown local training error";
            }
            throw_if_distributed_failure(local_error, "Training batch",
                                         _communicator);

            model.synchronize_gradients(local.count);
            local_error.clear();
            try {
                model.update();
            } catch (const std::exception& error) {
                local_error = error.what();
            } catch (...) {
                local_error = "unknown optimizer error";
            }
            throw_if_distributed_failure(local_error, "Optimizer update",
                                         _communicator);
        }

        double global_loss_sum = 0.0;
        unsigned long long global_seen = 0;
        reduce_sum(local_loss_sum, local_seen, global_loss_sum, global_seen,
                   _communicator);
        if (global_seen == 0) {
            throw std::runtime_error("Training epoch processed no samples.");
        }
        final_training_objective =
            global_loss_sum / static_cast<double>(global_seen);
        const size_t completed_epoch = epoch + 1;
        const bool history_checkpoint =
            completed_epoch % kHistoryIntervalEpochs == 0;
        double checkpoint_validation_mse = 0.0;
        if (history_checkpoint) {
            checkpoint_validation_mse = evaluate_validation();
            history.push_back(EpochMetrics{
                completed_epoch,
                final_training_objective,
                checkpoint_validation_mse});
        }

        if (verbose && distributed.rank == 0) {
            std::cout << "Epoch " << completed_epoch << "/"
                      << training_config.epochs
                      << " | Train objective: " << final_training_objective;
            if (history_checkpoint) {
                std::cout << " | Validation physical MSE: "
                          << checkpoint_validation_mse;
            }
            std::cout << std::endl;
        }
    }

    const bool final_epoch_is_checkpoint =
        training_config.epochs % kHistoryIntervalEpochs == 0;
    const double final_validation_mse = final_epoch_is_checkpoint
        ? history.back().validation_mse
        : evaluate_validation();

    return TrainingResult{
        final_training_objective,
        final_validation_mse,
        training_indices.size(),
        validation_indices.size(),
        std::move(history)};
}
