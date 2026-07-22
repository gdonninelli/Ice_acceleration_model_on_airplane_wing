#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "data/Dataset.hpp"
#include "model/CNNModel.hpp"
#include "tuning/TrialConfig.hpp"
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <span>
#include <vector>

struct EpochMetrics {
    size_t epoch = 0;
    double training_objective = 0.0;
    double validation_mse = 0.0;
};

struct TrainingResult {
    double training_objective = 0.0;
    double validation_mse = 0.0;
    size_t training_samples = 0;
    size_t validation_samples = 0;
    std::vector<EpochMetrics> history;
};

class Trainer {
public:
    static constexpr size_t kHistoryIntervalEpochs = 10;

    explicit Trainer(MPI_Comm communicator = MPI_COMM_WORLD);

    TrainingResult fit(CNNModel& model,
                       const Dataset& training_dataset,
                       std::span<const size_t> training_indices,
                       const Dataset& validation_dataset,
                       std::span<const size_t> validation_indices,
                       const NormalizationStats& normalization,
                       const LossConfig& loss_config,
                       const TrainingConfig& training_config,
                       uint64_t run_seed,
                       bool verbose = false) const;

private:
    MPI_Comm _communicator;
};

#endif // TRAINER_HPP
