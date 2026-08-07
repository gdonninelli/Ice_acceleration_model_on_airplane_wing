#include "Trainer.hpp"
#include "core/Loss.hpp"
#include "training/TrainingDiagnostics.hpp"
#include <algorithm>
#include <bit>
#include <iostream>
#include <random>
#include <limits>
#include <memory>
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

class ActivationObserverGuard {
public:
    explicit ActivationObserverGuard(CNNModel& model) : _model(model) {}
    ~ActivationObserverGuard() { _model.clear_activation_observer(); }

private:
    CNNModel& _model;
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
                                  MPI_Comm communicator,
                                  bool diagnostics_failure = false) {
    int local_failed = local_error.empty() ? 0 : 1;
    int any_failed = local_failed;
    if (mpi_ready()) {
        MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                      communicator);
    }
    if (any_failed != 0) {
        int local_diagnostics_failed =
            !local_error.empty() && diagnostics_failure ? 1 : 0;
        int any_diagnostics_failed = local_diagnostics_failed;
        if (mpi_ready()) {
            MPI_Allreduce(&local_diagnostics_failed, &any_diagnostics_failed,
                          1, MPI_INT, MPI_MAX, communicator);
        }
        if (any_diagnostics_failed != 0) {
            throw DiagnosticsError(
                !local_error.empty()
                    ? phase + ": " + local_error
                    : phase + " failed on another MPI rank.");
        }
        if (!local_error.empty()) {
            throw std::runtime_error(phase + ": " + local_error);
        }
        throw std::runtime_error(phase + " failed on another MPI rank.");
    }
}

void verify_diagnostics_agreement(const TrainingConfig& config,
                                  MPI_Comm communicator) {
    if (!mpi_ready()) {
        return;
    }
    uint64_t hash = 1469598103934665603ULL;
    auto append = [&](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    auto append_string = [&](const std::string& value) {
        append(value.size());
        for (unsigned char character : value) append(character);
    };
    append(config.validation_interval);
    append(config.diagnostics.enabled ? 1 : 0);
    append(config.diagnostics.histogram_bins);
    append(std::bit_cast<uint64_t>(config.diagnostics.histogram_min));
    append(std::bit_cast<uint64_t>(config.diagnostics.histogram_max));
    append_string(config.diagnostics.results_root);
    append_string(config.diagnostics.experiment_name);
    append_string(config.diagnostics.run_name);
    append_string(config.diagnostics.training_dataset_path);
    append_string(config.diagnostics.validation_dataset_path);

    const unsigned long long local_hash = hash;
    unsigned long long minimum_hash = 0;
    unsigned long long maximum_hash = 0;
    MPI_Allreduce(&local_hash, &minimum_hash, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, communicator);
    MPI_Allreduce(&local_hash, &maximum_hash, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, communicator);
    if (minimum_hash != maximum_hash) {
        throw std::runtime_error(
            "Training diagnostics configuration differs across MPI ranks.");
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
                             bool verbose,
                             const TrainingRunContext& run_context,
                             const TrialConfig* trial_config) const {
    if (training_indices.empty() || validation_indices.empty()) {
        throw std::invalid_argument("Trainer requires non-empty training and validation sets.");
    }
    if (training_config.epochs == 0 || training_config.global_batch_size == 0 ||
        training_config.validation_interval == 0) {
        throw std::invalid_argument(
            "Training epochs, global batch size, and validation interval must be positive.");
    }
    verify_diagnostics_agreement(training_config, _communicator);

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
    std::unique_ptr<TrainingDiagnosticsRecorder> diagnostics;
    std::string diagnostics_error;
    if (training_config.diagnostics.enabled) {
        try {
            if (!trial_config) {
                throw std::invalid_argument(
                    "Enabled diagnostics require the complete TrialConfig metadata.");
            }
            diagnostics = std::make_unique<TrainingDiagnosticsRecorder>(
                training_config.diagnostics, run_context, *trial_config, model,
                _communicator);
        } catch (const std::exception& error) {
            diagnostics_error = error.what();
        } catch (...) {
            diagnostics_error = "unknown diagnostics setup error";
        }
    }
    throw_if_distributed_failure(diagnostics_error, "Diagnostics setup",
                                 _communicator, true);
    ActivationObserverGuard observer_guard(model);
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
        const size_t completed_epoch = epoch + 1;
        if (diagnostics) {
            diagnostics->begin_epoch(completed_epoch);
            model.set_activation_observer(
                [&](size_t layer_index, const std::string& layer_name,
                    const Tensor& pre, const Tensor& post) {
                    diagnostics->capture_activation(layer_index, layer_name,
                                                    pre, post);
                });
        }
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
            bool local_diagnostics_failure = false;
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
            } catch (const DiagnosticsError& error) {
                local_error = error.what();
                local_diagnostics_failure = true;
            } catch (const std::exception& error) {
                local_error = error.what();
            } catch (...) {
                local_error = "unknown local training error";
            }
            throw_if_distributed_failure(local_error, "Training batch",
                                         _communicator,
                                         local_diagnostics_failure);

            model.synchronize_gradients(local.count);
            local_error.clear();
            try {
                if (diagnostics) {
                    diagnostics->before_optimizer_step(model);
                }
            } catch (const std::exception& error) {
                local_error = error.what();
            } catch (...) {
                local_error = "unknown pre-update diagnostics error";
            }
            throw_if_distributed_failure(local_error, "Pre-update diagnostics",
                                         _communicator, true);

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

            local_error.clear();
            try {
                if (diagnostics) {
                    diagnostics->after_optimizer_step(model);
                }
            } catch (const std::exception& error) {
                local_error = error.what();
            } catch (...) {
                local_error = "unknown post-update diagnostics error";
            }
            throw_if_distributed_failure(local_error, "Post-update diagnostics",
                                         _communicator, true);
        }

        model.clear_activation_observer();

        double global_loss_sum = 0.0;
        unsigned long long global_seen = 0;
        reduce_sum(local_loss_sum, local_seen, global_loss_sum, global_seen,
                   _communicator);
        if (global_seen == 0) {
            throw std::runtime_error("Training epoch processed no samples.");
        }
        final_training_objective =
            global_loss_sum / static_cast<double>(global_seen);
        const bool history_checkpoint =
            completed_epoch % training_config.validation_interval == 0;
        double checkpoint_validation_mse =
            std::numeric_limits<double>::quiet_NaN();
        if (history_checkpoint) {
            checkpoint_validation_mse = evaluate_validation();
            history.push_back(EpochMetrics{
                completed_epoch,
                final_training_objective,
                checkpoint_validation_mse});
        }

        EpochDiagnosticsSummary diagnostics_summary;
        std::string epoch_diagnostics_error;
        try {
            if (diagnostics) {
                const size_t batches =
                    (ordered_training.size() + training_config.global_batch_size - 1) /
                    training_config.global_batch_size;
                diagnostics_summary = diagnostics->finish_epoch(
                    completed_epoch, final_training_objective,
                    checkpoint_validation_mse,
                    static_cast<size_t>(global_seen), batches);
            }
        } catch (const std::exception& error) {
            epoch_diagnostics_error = error.what();
        } catch (...) {
            epoch_diagnostics_error = "unknown epoch diagnostics error";
        }
        throw_if_distributed_failure(epoch_diagnostics_error, "Epoch diagnostics",
                                     _communicator, true);

        if (verbose && distributed.rank == 0) {
            std::cout << "Epoch " << completed_epoch << "/"
                      << training_config.epochs
                      << " | samples=" << global_seen
                      << " | steps="
                      << ((ordered_training.size() +
                           training_config.global_batch_size - 1) /
                          training_config.global_batch_size)
                      << " | train=" << final_training_objective;
            if (history_checkpoint) {
                std::cout << " | validation_mse="
                          << checkpoint_validation_mse;
            } else {
                std::cout << " | validation_mse=missing";
            }
            std::cout << " | lr="
                      << model.optimizer_metadata().effective_learning_rate;
            if (diagnostics) {
                std::cout << " | grad_rms="
                          << diagnostics_summary.gradient_rms_summary
                          << " | activation_var(mean/min/max)="
                          << diagnostics_summary.activated_variance_mean << '/'
                          << diagnostics_summary.activated_variance_min << '/'
                          << diagnostics_summary.activated_variance_max;
            }
            std::cout << std::endl;
        }
    }

    const bool final_epoch_is_checkpoint =
        training_config.epochs % training_config.validation_interval == 0;
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
