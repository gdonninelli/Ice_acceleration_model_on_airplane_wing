#include "Trainer.hpp"
#include "core/Loss.hpp"
#include "optimizers/LRScheduler.hpp"
#include "training/TrainingDiagnostics.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
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

struct BatchRange {
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

size_t balanced_batch_count(size_t sample_count, size_t maximum_batch_size) {
    if (maximum_batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive.");
    }
    if (sample_count == 0) {
        return 0;
    }
    return sample_count / maximum_batch_size +
           (sample_count % maximum_batch_size == 0 ? 0 : 1);
}

size_t training_batch_count(size_t sample_count,
                            size_t maximum_batch_size,
                            BatchConstruction construction) {
    if (construction == BatchConstruction::RangeTail) {
        return balanced_batch_count(sample_count, maximum_batch_size);
    }
    return balanced_batch_count(sample_count, maximum_batch_size);
}

BatchRange balanced_batch_range(size_t sample_count,
                                size_t maximum_batch_size,
                                size_t batch_index) {
    const size_t batches =
        balanced_batch_count(sample_count, maximum_batch_size);
    if (batch_index >= batches) {
        throw std::out_of_range("Batch index is outside the training range.");
    }

    // Treat the configured batch size as an upper bound and distribute the
    // remainder across batches. For the canonical 1542-sample split, a
    // global batch size of 257 produces six equal batches and no tiny tail.
    const size_t base_size = sample_count / batches;
    const size_t remainder = sample_count % batches;
    return {
        batch_index * base_size + std::min(batch_index, remainder),
        base_size + (batch_index < remainder ? 1 : 0)};
}

BatchRange training_batch_range(size_t sample_count,
                                size_t maximum_batch_size,
                                size_t batch_index,
                                BatchConstruction construction) {
    if (construction == BatchConstruction::Balanced) {
        return balanced_batch_range(sample_count, maximum_batch_size,
                                    batch_index);
    }
    const size_t batches = training_batch_count(
        sample_count, maximum_batch_size, construction);
    if (batch_index >= batches) {
        throw std::out_of_range("Batch index is outside the training range.");
    }
    const size_t offset = batch_index * maximum_batch_size;
    return {offset, std::min(maximum_batch_size, sample_count - offset)};
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
    append(config.early_stopping ? 1 : 0);
    append(config.early_stopping_min_epochs);
    append(config.early_stopping_patience);
    append(std::bit_cast<uint64_t>(config.max_overfit_ratio));
    append(config.restore_best_weights ? 1 : 0);
    append(static_cast<uint64_t>(config.batch_construction));
    append(static_cast<uint64_t>(config.early_stopping_policy));

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

// Weight regularization is applied to weight tensors only. Biases are left
// untouched because penalizing them shifts the learned output offset without
// constraining model capacity. DenseLayer and Conv2DLayer are the only layers
// exposing parameters, and both name them "weights" and "biases".
constexpr char kWeightParameterName[] = "weights";

bool is_weight_tensor(const LayerParameter& parameter) {
    return parameter.tensor && parameter.name == kWeightParameterName;
}

// Subgradient of |w| with sign(0) = 0, so a weight driven exactly to zero
// receives no further push. std::copysign is unusable here: it returns +/-1
// for zero.
float weight_sign(float value) {
    if (value > 0.0f) {
        return 1.0f;
    }
    return (value < 0.0f) ? -1.0f : 0.0f;
}

// R_L2(W) = lambda * sum(w^2), summed over every weight tensor. The sum is
// deliberately not normalized by the parameter count or the batch size, so
// lambda keeps a fixed meaning across topologies and batch sizes.
double l2_penalty(const std::vector<LayerParameter>& parameters,
                  float lambda) {
    if (lambda == 0.0f) {
        return 0.0;
    }
    double squared_sum = 0.0;
    for (const auto& parameter : parameters) {
        if (!is_weight_tensor(parameter)) {
            continue;
        }
        const float* weights = parameter.tensor->get_data();
        for (size_t i = 0; i < parameter.tensor->size(); ++i) {
            squared_sum += static_cast<double>(weights[i]) * weights[i];
        }
    }
    return static_cast<double>(lambda) * squared_sum;
}

// R_L1(W) = lambda * sum(|w|), with the same non-normalized convention as
// l2_penalty.
double l1_penalty(const std::vector<LayerParameter>& parameters,
                  float lambda) {
    if (lambda == 0.0f) {
        return 0.0;
    }
    double absolute_sum = 0.0;
    for (const auto& parameter : parameters) {
        if (!is_weight_tensor(parameter)) {
            continue;
        }
        const float* weights = parameter.tensor->get_data();
        for (size_t i = 0; i < parameter.tensor->size(); ++i) {
            absolute_sum += std::abs(static_cast<double>(weights[i]));
        }
    }
    return static_cast<double>(lambda) * absolute_sum;
}

// Adds d(R_L2)/dw = 2 * lambda2 * w and d(R_L1)/dw = lambda1 * sign(w) to the
// already synchronized gradient.
void add_regularization_gradient(const std::vector<LayerParameter>& parameters,
                                 float l1_weight,
                                 float l2_weight) {
    if (l1_weight == 0.0f && l2_weight == 0.0f) {
        return;
    }
    for (const auto& parameter : parameters) {
        if (!is_weight_tensor(parameter)) {
            continue;
        }
        const float* weights = parameter.tensor->get_data();
        float* gradient = parameter.tensor->get_grad();
        for (size_t i = 0; i < parameter.tensor->size(); ++i) {
            gradient[i] += 2.0f * l2_weight * weights[i] +
                           l1_weight * weight_sign(weights[i]);
        }
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
    if (!std::isfinite(training_config.max_overfit_ratio) ||
        training_config.max_overfit_ratio < 0.0) {
        throw std::invalid_argument(
            "Maximum overfitting ratio must be finite and non-negative.");
    }
    if (training_config.early_stopping &&
        training_config.early_stopping_patience == 0) {
        throw std::invalid_argument(
            "Early-stopping patience must be positive when early stopping is enabled.");
    }
    verify_diagnostics_agreement(training_config, _communicator);
    if (!std::isfinite(loss_config.l1_weight) ||
        loss_config.l1_weight < 0.0f ||
        !std::isfinite(loss_config.l2_weight) ||
        loss_config.l2_weight < 0.0f) {
        throw std::invalid_argument(
            "L1/L2 weights must be finite and non-negative.");
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
    const size_t epoch_batch_count = training_batch_count(
        ordered_training.size(), training_config.global_batch_size,
        training_config.batch_construction);
    double final_training_objective = 0.0;
    double final_training_physical_mse = 0.0;
    std::vector<EpochMetrics> history;
    std::vector<std::vector<float>> best_parameters;
    double best_training_objective = std::numeric_limits<double>::infinity();
    double best_validation_physical_mse =
        std::numeric_limits<double>::infinity();
    size_t best_epoch = 0;
    size_t overfit_streak = 0;
    size_t epochs_completed = 0;
    bool stopped_early = false;

    std::unique_ptr<LRScheduler> scheduler;
    if (trial_config && trial_config->scheduler.has_scheduler()) {
        scheduler = trial_config->scheduler.build();
    }

    auto evaluate_physical_mse = [&](const Dataset& dataset,
                                     std::span<const size_t> indices,
                                     const std::string& phase) {
        double local_mse_sum = 0.0;
        unsigned long long local_seen = 0;
        std::string evaluation_error;
        try {
            for (size_t batch_offset = 0;
                 batch_offset < indices.size();
                 batch_offset += training_config.global_batch_size) {
                const size_t global_count = std::min(
                    training_config.global_batch_size,
                    indices.size() - batch_offset);
                const Partition local = partition_batch(
                    global_count, distributed.rank, distributed.size);
                if (local.count == 0) {
                    continue;
                }

                const std::span<const size_t> local_indices(
                    indices.data() + batch_offset + local.offset,
                    local.count);
                const DataBatch batch =
                    dataset.make_batch(local_indices, normalization);
                const auto predictions = model.predict(batch.sdf, batch.scalars);
                const float mse = Loss::physical_mse(
                    predictions, batch.targets,
                    static_cast<float>(normalization.target_std));
                local_mse_sum += static_cast<double>(mse) * local.count;
                local_seen += static_cast<unsigned long long>(local.count);
            }
        } catch (const std::exception& error) {
            evaluation_error = error.what();
        } catch (...) {
            evaluation_error = "unknown local physical-MSE evaluation error";
        }
        throw_if_distributed_failure(evaluation_error, phase, _communicator);

        double global_mse_sum = 0.0;
        unsigned long long global_seen = 0;
        reduce_sum(local_mse_sum, local_seen, global_mse_sum, global_seen,
                   _communicator);
        if (global_seen == 0) {
            throw std::runtime_error(phase + " processed no samples.");
        }
        return global_mse_sum / static_cast<double>(global_seen);
    };

    auto evaluate_training = [&]() {
        return evaluate_physical_mse(training_dataset, training_indices,
                                     "Training physical MSE");
    };
    auto evaluate_validation = [&]() {
        return evaluate_physical_mse(validation_dataset, validation_indices,
                                     "Validation physical MSE");
    };

    for (size_t epoch = 0; epoch < training_config.epochs; ++epoch) {
        if (scheduler) {
            const float scheduled_lr =
                scheduler->get_rate(epoch, training_config.epochs);
            model.set_learning_rate(scheduled_lr);
        }
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
        for (size_t batch_index = 0; batch_index < epoch_batch_count;
             ++batch_index) {
            const BatchRange batch = training_batch_range(
                ordered_training.size(), training_config.global_batch_size,
                batch_index, training_config.batch_construction);
            const size_t batch_offset = batch.offset;
            const size_t global_count = batch.count;
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
                    // Training context for stochastic layers (dropout). The
                    // stream seed depends only on globally agreed values and
                    // the sample offset locates this rank's slice inside the
                    // global batch, so masks are identical for any rank count.
                    LayerExecutionContext execution_context;
                    execution_context.training = true;
                    execution_context.stream_seed = layer_rng::combine(
                        layer_rng::combine(run_seed,
                                           static_cast<uint64_t>(epoch)),
                        static_cast<uint64_t>(batch_offset));
                    execution_context.sample_offset = local.offset;
                    model.set_execution_context(execution_context);
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
            // The gradient is globally averaged here and not yet clipped or
            // consumed. Every rank holds identical weights, so every rank adds
            // the identical penalty gradient and the replicas stay in step.
            add_regularization_gradient(model.parameters(),
                                        loss_config.l1_weight,
                                        loss_config.l2_weight);
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
        final_training_physical_mse = evaluate_training();
        if (!std::isfinite(final_training_objective) ||
            !std::isfinite(final_training_physical_mse)) {
            throw std::runtime_error("Training produced a non-finite loss.");
        }
        const bool history_checkpoint =
            completed_epoch % training_config.validation_interval == 0;
        const double checkpoint_training_physical_mse =
            final_training_physical_mse;
        // Validation is a reporting metric for every epoch. The configured
        // interval still controls the public history checkpoints used by CV.
        const double checkpoint_validation_physical_mse = evaluate_validation();
        if (!std::isfinite(checkpoint_validation_physical_mse)) {
            throw std::runtime_error(
                "Validation produced a non-finite physical MSE.");
        }
        if (history_checkpoint) {
            history.push_back(EpochMetrics{
                completed_epoch,
                final_training_objective,
                checkpoint_validation_physical_mse});
        }

        bool overfit = false;
        if (training_config.early_stopping) {
            const bool validation_improved =
                checkpoint_validation_physical_mse < best_validation_physical_mse;
            if (validation_improved) {
                best_validation_physical_mse =
                    checkpoint_validation_physical_mse;
                best_training_objective = final_training_objective;
                best_epoch = completed_epoch;
                best_parameters.clear();
                for (const auto& parameter : model.parameters()) {
                    const float* values = parameter.tensor->get_data();
                    best_parameters.emplace_back(
                        values, values + parameter.tensor->size());
                }
            }

            bool ratio_exceeded = false;
            if (checkpoint_training_physical_mse == 0.0) {
                ratio_exceeded = checkpoint_validation_physical_mse > 0.0;
            } else {
                ratio_exceeded = checkpoint_validation_physical_mse >
                                 checkpoint_training_physical_mse *
                                     (1.0 + training_config.max_overfit_ratio);
            }

            // A single noisy validation excursion should not stop training.
            // Require a post-warm-up streak, and only count epochs that are
            // both above the configured gap and worse than the best checkpoint.
            if (training_config.early_stopping_policy ==
                EarlyStoppingPolicy::FirstRatioExceeded) {
                overfit = ratio_exceeded;
            } else {
                if (completed_epoch < training_config.early_stopping_min_epochs ||
                    validation_improved || !ratio_exceeded) {
                    overfit_streak = 0;
                } else {
                    ++overfit_streak;
                }
                overfit =
                    overfit_streak >= training_config.early_stopping_patience &&
                    completed_epoch >= training_config.early_stopping_min_epochs;
            }
        }

        if (training_config.early_stopping && overfit) {
            stopped_early = true;
        }
        epochs_completed = completed_epoch;

        EpochDiagnosticsSummary diagnostics_summary;
        std::string epoch_diagnostics_error;
        try {
            if (diagnostics) {
                diagnostics_summary = diagnostics->finish_epoch(
                    completed_epoch, final_training_objective,
                    final_training_physical_mse,
                    checkpoint_validation_physical_mse,
                    static_cast<size_t>(global_seen), epoch_batch_count);
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
                      << " | steps=" << epoch_batch_count
                      << " | train_objective=" << final_training_objective
                      << " | training_physical_mse="
                      << final_training_physical_mse
                      << " | validation_physical_mse="
                      << checkpoint_validation_physical_mse;
            // Reported separately from the training objective so that the
            // objective stays comparable across regularization strengths.
            if (loss_config.l2_weight != 0.0f) {
                std::cout << " | L2 penalty: "
                          << l2_penalty(model.parameters(),
                                        loss_config.l2_weight);
            }
            if (loss_config.l1_weight != 0.0f) {
                std::cout << " | L1 penalty: "
                          << l1_penalty(model.parameters(),
                                        loss_config.l1_weight);
            }
            const OptimizerMetadata optimizer = model.optimizer_metadata();
            std::cout << " | lr=" << optimizer.effective_learning_rate
                      << " | configured_lr="
                      << optimizer.configured_learning_rate;
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

        if (stopped_early) {
            if (distributed.rank == 0 && verbose) {
                std::cout << "Early stopping at epoch " << completed_epoch;
                if (training_config.early_stopping_policy ==
                    EarlyStoppingPolicy::FirstRatioExceeded) {
                    std::cout << ": validation MSE exceeded training MSE by more than ";
                } else {
                    std::cout << " after "
                              << training_config.early_stopping_patience
                              << " consecutive epochs with validation MSE more than ";
                }
                std::cout << training_config.max_overfit_ratio * 100.0
                          << "% above training MSE";
                if (training_config.early_stopping_policy ==
                    EarlyStoppingPolicy::Patience) {
                    std::cout << " without improvement";
                }
                std::cout << '.' << std::endl;
            }
            break;
        }
    }

    bool restored_best_weights = false;
    if (training_config.early_stopping && training_config.restore_best_weights &&
        !best_parameters.empty()) {
        const auto& parameters = model.parameters();
        if (parameters.size() != best_parameters.size()) {
            throw std::runtime_error("Best model snapshot has an incompatible parameter count.");
        }
        for (size_t parameter_index = 0;
             parameter_index < parameters.size(); ++parameter_index) {
            auto& parameter = parameters[parameter_index];
            if (parameter.tensor->size() != best_parameters[parameter_index].size()) {
                throw std::runtime_error(
                    "Best model snapshot has an incompatible parameter shape.");
            }
            std::copy(best_parameters[parameter_index].begin(),
                      best_parameters[parameter_index].end(),
                      parameter.tensor->get_data());
            parameter.tensor->zero_grad();
        }
        model.broadcast_initial_weights(0);
        final_training_objective = best_training_objective;
        restored_best_weights = true;
    }

    const double final_validation_physical_mse = evaluate_validation();
    if (restored_best_weights) {
        final_training_physical_mse = evaluate_training();
    }
    if (!std::isfinite(final_training_physical_mse) ||
        !std::isfinite(final_validation_physical_mse)) {
        throw std::runtime_error(
            "Final model produced a non-finite physical MSE.");
    }

    return TrainingResult{
        final_training_objective,
        final_validation_physical_mse,
        training_indices.size(),
        validation_indices.size(),
        std::move(history),
        final_training_physical_mse,
        epochs_completed,
        best_epoch,
        stopped_early};
}
