#ifndef CROSSVALIDATOR_HPP
#define CROSSVALIDATOR_HPP

#include "SearchSpace.hpp"
#include "data/Dataset.hpp"
#include "model/ModelFactory.hpp"
#include "training/Trainer.hpp"
#include <cstdint>
#include <limits>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

struct FoldIndices {
    std::vector<size_t> training;
    std::vector<size_t> validation;
};

class FoldSplitter {
public:
    virtual ~FoldSplitter() = default;
    virtual std::vector<FoldIndices> split(size_t sample_count) const = 0;
};

class RandomKFold : public FoldSplitter {
public:
    RandomKFold(size_t folds, bool shuffle = true, uint64_t seed = 42);
    std::vector<FoldIndices> split(size_t sample_count) const override;

private:
    size_t _folds;
    bool _shuffle;
    uint64_t _seed;
};

struct FoldMetrics {
    size_t fold = 0;
    double training_objective = 0.0;
    double validation_mse = 0.0;
    size_t training_samples = 0;
    size_t validation_samples = 0;
    std::vector<EpochMetrics> history;
};

struct CandidateResult {
    explicit CandidateResult(TrialConfig trial_config)
        : config(std::move(trial_config)) {}

    TrialConfig config;
    std::vector<FoldMetrics> folds;
    double mean_training_objective = 0.0;
    double mean_validation_mse = std::numeric_limits<double>::infinity();
    double validation_stddev = 0.0;
    bool success = false;
    std::string error;
};

struct SearchResult {
    std::vector<CandidateResult> candidates;
    size_t best_index = std::numeric_limits<size_t>::max();

    const CandidateResult& best() const;
};

class TrialRunner {
public:
    struct Context {
        size_t candidate_index = 0;
        size_t candidate_count = 1;
        size_t fold_count = 0;
    };

    virtual ~TrialRunner() = default;
    virtual FoldMetrics run(const TrialConfig& config,
                            const Dataset& dataset,
                             const FoldIndices& fold,
                             size_t fold_index) const = 0;
    virtual FoldMetrics run(const TrialConfig& config,
                            const Dataset& dataset,
                            const FoldIndices& fold,
                            size_t fold_index,
                            const Context&) const {
        return run(config, dataset, fold, fold_index);
    }
};

class CNNTrialRunner : public TrialRunner {
public:
    explicit CNNTrialRunner(MPI_Comm communicator = MPI_COMM_WORLD);

    FoldMetrics run(const TrialConfig& config,
                    const Dataset& dataset,
                     const FoldIndices& fold,
                     size_t fold_index) const override;
    FoldMetrics run(const TrialConfig& config,
                    const Dataset& dataset,
                    const FoldIndices& fold,
                    size_t fold_index,
                    const Context& context) const override;

private:
    MPI_Comm _communicator;
    ModelFactory _model_factory;
    Trainer _trainer;
};

class CrossValidator {
public:
    CrossValidator(const Dataset& dataset,
                   std::shared_ptr<const FoldSplitter> splitter,
                   std::shared_ptr<const TrialRunner> runner,
                   MPI_Comm communicator = MPI_COMM_WORLD,
                   bool verbose = true);

    CandidateResult evaluate(const TrialConfig& config) const;
    SearchResult tune(const CandidateSource& candidates) const;

private:
    const Dataset& _dataset;
    std::shared_ptr<const FoldSplitter> _splitter;
    std::shared_ptr<const TrialRunner> _runner;
    MPI_Comm _communicator;
    bool _verbose;

    int rank() const;
    std::vector<FoldIndices> make_fold_plan() const;
    CandidateResult evaluate_with_folds(
        const TrialConfig& config,
        const std::vector<FoldIndices>& folds,
        size_t candidate_index,
        size_t candidate_count) const;
};

#endif // CROSSVALIDATOR_HPP
