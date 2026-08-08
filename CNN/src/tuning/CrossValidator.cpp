#include "CrossValidator.hpp"
#include "training/TrainingDiagnostics.hpp"
#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

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

uint64_t fold_seed(uint64_t base_seed, size_t fold_index) {
    constexpr uint64_t golden_ratio = 0x9e3779b97f4a7c15ULL;
    return base_seed ^ (golden_ratio + static_cast<uint64_t>(fold_index) +
                        (base_seed << 6U) + (base_seed >> 2U));
}

void validate_folds(const std::vector<FoldIndices>& folds, size_t sample_count) {
    if (folds.size() < 2) {
        throw std::invalid_argument("Cross-validation requires at least two folds.");
    }
    std::vector<size_t> validation_coverage(sample_count, 0);
    for (const auto& fold : folds) {
        if (fold.training.empty() || fold.validation.empty()) {
            throw std::invalid_argument("Every fold must have training and validation samples.");
        }
        if (fold.training.size() + fold.validation.size() != sample_count) {
            throw std::invalid_argument(
                "Each fold must partition the complete dataset.");
        }

        std::unordered_set<size_t> training;
        training.reserve(fold.training.size());
        for (size_t index : fold.training) {
            if (index >= sample_count || !training.insert(index).second) {
                throw std::invalid_argument(
                    "Fold training indices must be unique and in range.");
            }
        }

        std::unordered_set<size_t> validation;
        validation.reserve(fold.validation.size());
        for (size_t index : fold.validation) {
            if (index >= sample_count || !validation.insert(index).second ||
                training.contains(index)) {
                throw std::invalid_argument(
                    "Fold validation indices must be unique, in range, and disjoint.");
            }
            ++validation_coverage[index];
        }
    }
    if (std::any_of(validation_coverage.begin(), validation_coverage.end(),
                    [](size_t count) { return count != 1; })) {
        throw std::invalid_argument(
            "Validation folds must cover every sample exactly once.");
    }
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

uint64_t fold_plan_hash(const std::vector<FoldIndices>& folds) {
    constexpr uint64_t offset_basis = 1469598103934665603ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset_basis;
    auto append = [&](size_t value) {
        hash ^= static_cast<uint64_t>(value);
        hash *= prime;
    };
    append(folds.size());
    for (const auto& fold : folds) {
        append(fold.training.size());
        for (size_t index : fold.training) {
            append(index);
        }
        append(fold.validation.size());
        for (size_t index : fold.validation) {
            append(index);
        }
    }
    return hash;
}

bool metric_close(double first, double second) {
    const double scale = std::max({1.0, std::abs(first), std::abs(second)});
    return std::abs(first - second) <= 1e-12 * scale;
}

uint64_t trial_config_hash(const TrialConfig& config) {
    constexpr uint64_t offset_basis = 1469598103934665603ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset_basis;
    auto append_integer = [&](uint64_t value) {
        hash ^= value;
        hash *= prime;
    };
    auto append_string = [&](const std::string& value) {
        append_integer(value.size());
        for (unsigned char character : value) {
            append_integer(character);
        }
    };

    append_string(config.name);
    append_integer(config.model.concatenate_scalars ? 1 : 0);
    append_integer(config.model.scalar_features);
    append_integer(config.model.feature_layers.size());
    for (const auto& recipe : config.model.feature_layers) {
        append_string(recipe.description());
    }
    append_integer(config.model.head_layers.size());
    for (const auto& recipe : config.model.head_layers) {
        append_string(recipe.description());
    }
    append_string(config.optimizer.description());
    append_integer(std::bit_cast<uint32_t>(config.loss.physics_weight));
    append_integer(std::bit_cast<uint32_t>(config.loss.l1_weight));
    append_integer(std::bit_cast<uint32_t>(config.loss.l2_weight));
    append_integer(config.training.epochs);
    append_integer(config.training.global_batch_size);
    append_integer(std::bit_cast<uint32_t>(config.training.gradient_clip));
    append_integer(config.training.seed);
    append_integer(config.training.shuffle ? 1 : 0);
    append_integer(config.training.validation_interval);
    append_integer(config.training.diagnostics.enabled ? 1 : 0);
    append_string(config.training.diagnostics.results_root);
    append_string(config.training.diagnostics.experiment_name);
    append_string(config.training.diagnostics.run_name);
    append_integer(config.training.diagnostics.histogram_bins);
    append_integer(std::bit_cast<uint64_t>(
        config.training.diagnostics.histogram_min));
    append_integer(std::bit_cast<uint64_t>(
        config.training.diagnostics.histogram_max));
    append_string(config.training.diagnostics.training_dataset_path);
    append_string(config.training.diagnostics.validation_dataset_path);
    for (const auto& [name, value] : config.selected_parameters) {
        append_string(name);
        append_string(value);
    }
    return hash;
}

void verify_trial_config_agreement(const TrialConfig& config,
                                   MPI_Comm communicator) {
    if (!mpi_ready()) {
        return;
    }
    const unsigned long long local_hash =
        static_cast<unsigned long long>(trial_config_hash(config));
    unsigned long long minimum_hash = 0;
    unsigned long long maximum_hash = 0;
    MPI_Allreduce(&local_hash, &minimum_hash, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, communicator);
    MPI_Allreduce(&local_hash, &maximum_hash, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, communicator);
    if (minimum_hash != maximum_hash) {
        throw std::runtime_error(
            "Trial configuration differs across MPI ranks.");
    }
}

std::string csv_field(const std::string& value) {
    std::string output = "\"";
    for (char character : value) {
        if (character == '"') output.push_back('"');
        output.push_back(character);
    }
    output.push_back('"');
    return output;
}

std::string json_field(const std::string& value) {
    std::string output = "\"";
    constexpr char hexadecimal[] = "0123456789abcdef";
    for (unsigned char character : value) {
        switch (character) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (character < 0x20) {
                output += "\\u00";
                output.push_back(hexadecimal[character >> 4U]);
                output.push_back(hexadecimal[character & 0x0fU]);
            } else {
                output.push_back(static_cast<char>(character));
            }
        }
    }
    output.push_back('"');
    return output;
}

void write_cv_artifacts(const std::vector<const CandidateResult*>& candidates,
                        MPI_Comm communicator) {
    const auto enabled_candidate = std::find_if(
        candidates.begin(), candidates.end(),
        [](const CandidateResult* candidate) {
            return candidate && candidate->config.training.diagnostics.enabled;
        });
    if (enabled_candidate == candidates.end()) {
        return;
    }
    int rank = 0;
    if (mpi_ready()) MPI_Comm_rank(communicator, &rank);
    std::string local_error;
    if (rank == 0) {
        try {
            const auto& diagnostics =
                (*enabled_candidate)->config.training.diagnostics;
            const auto directory = diagnostics_session_directory(diagnostics);
            std::filesystem::create_directories(directory);
            const auto summary_path = directory / "cv_summary.csv";
            std::ofstream summary(summary_path, std::ios::trunc);
            if (!summary) {
                throw std::runtime_error("Failed to open " + summary_path.string());
            }
            summary << "candidate_index,candidate_name,success,fold,training_objective,validation_physical_mse,training_samples,validation_samples,mean_validation_physical_mse,validation_stddev,error\n";
            for (size_t candidate_index = 0;
                 candidate_index < candidates.size(); ++candidate_index) {
                const CandidateResult& candidate = *candidates[candidate_index];
                if (candidate.folds.empty()) {
                    summary << candidate_index << ','
                            << csv_field(candidate.config.name) << ','
                            << (candidate.success ? 1 : 0)
                            << ",,,,,," << candidate.mean_validation_mse << ','
                            << candidate.validation_stddev << ','
                            << csv_field(candidate.error) << '\n';
                }
                for (const auto& fold : candidate.folds) {
                    summary << candidate_index << ','
                            << csv_field(candidate.config.name) << ','
                            << (candidate.success ? 1 : 0) << ',' << fold.fold
                            << ',' << std::setprecision(17)
                            << fold.training_objective << ','
                            << fold.validation_mse << ','
                            << fold.training_samples << ','
                            << fold.validation_samples << ','
                            << candidate.mean_validation_mse << ','
                            << candidate.validation_stddev << ','
                            << csv_field(candidate.error) << '\n';
                }
            }
            summary.flush();
            if (!summary) {
                throw std::runtime_error("Failed to write " + summary_path.string());
            }

            const auto metadata_path = directory / "metadata.json";
            std::ofstream metadata(metadata_path, std::ios::trunc);
            if (!metadata) {
                throw std::runtime_error("Failed to open " + metadata_path.string());
            }
            metadata << "{\n  \"format_version\": 1,\n"
                     << "  \"mode\": \"cross_validation\",\n"
                     << "  \"experiment_name\": "
                     << json_field(diagnostics.experiment_name) << ",\n"
                     << "  \"run_name\": "
                     << json_field(diagnostics.run_name) << ",\n"
                     << "  \"candidate_count\": " << candidates.size()
                     << "\n}\n";
            metadata.flush();
            if (!metadata) {
                throw std::runtime_error("Failed to write " +
                                         metadata_path.string());
            }
        } catch (const std::exception& error) {
            local_error = error.what();
        }
    }
    throw_if_distributed_failure(local_error, "Cross-validation diagnostics",
                                 communicator, true);
}
} // namespace

RandomKFold::RandomKFold(size_t folds, bool shuffle, uint64_t seed)
    : _folds(folds), _shuffle(shuffle), _seed(seed) {
    if (_folds < 2) {
        throw std::invalid_argument("RandomKFold requires at least two folds.");
    }
}

std::vector<FoldIndices> RandomKFold::split(size_t sample_count) const {
    if (sample_count < _folds) {
        throw std::invalid_argument("Fold count cannot exceed the sample count.");
    }

    std::vector<size_t> indices(sample_count);
    std::iota(indices.begin(), indices.end(), size_t{0});
    if (_shuffle) {
        std::mt19937_64 generator(_seed);
        std::shuffle(indices.begin(), indices.end(), generator);
    }

    std::vector<FoldIndices> folds;
    folds.reserve(_folds);
    size_t validation_start = 0;
    for (size_t fold = 0; fold < _folds; ++fold) {
        const size_t validation_size =
            sample_count / _folds + (fold < sample_count % _folds ? 1 : 0);
        const size_t validation_end = validation_start + validation_size;

        FoldIndices split;
        split.validation.insert(split.validation.end(),
                                indices.begin() + validation_start,
                                indices.begin() + validation_end);
        split.training.reserve(sample_count - validation_size);
        split.training.insert(split.training.end(), indices.begin(),
                              indices.begin() + validation_start);
        split.training.insert(split.training.end(),
                              indices.begin() + validation_end, indices.end());
        folds.push_back(std::move(split));
        validation_start = validation_end;
    }
    return folds;
}

const CandidateResult& SearchResult::best() const {
    if (best_index >= candidates.size()) {
        throw std::runtime_error("Search did not produce a successful candidate.");
    }
    return candidates[best_index];
}

CNNTrialRunner::CNNTrialRunner(MPI_Comm communicator)
    : _communicator(communicator), _trainer(communicator) {}

FoldMetrics CNNTrialRunner::run(const TrialConfig& config,
                                const Dataset& dataset,
                                const FoldIndices& fold,
                                size_t fold_index) const {
    return run(config, dataset, fold, fold_index,
               Context{0, 1, fold_index + 1});
}

FoldMetrics CNNTrialRunner::run(const TrialConfig& config,
                                const Dataset& dataset,
                                const FoldIndices& fold,
                                size_t fold_index,
                                const Context& context) const {
    const uint64_t seed = fold_seed(config.training.seed, fold_index);
    NormalizationStats normalization;
    std::unique_ptr<CNNModel> model;
    std::string local_error;
    try {
        normalization = dataset.fit_normalization(fold.training);
        model = _model_factory.build(config, dataset.sdf_height(),
                                     dataset.sdf_width(), dataset.scalar_features(),
                                     seed, _communicator);
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown pipeline construction error";
    }
    throw_if_distributed_failure(local_error, "Trial setup", _communicator);

    TrainingRunContext run_context;
    run_context.mode = "cross_validation";
    run_context.candidate_index = context.candidate_index;
    run_context.candidate_count = context.candidate_count;
    run_context.fold_index = fold_index;
    run_context.fold_count = context.fold_count;
    run_context.random_seed = seed;
    run_context.training_dataset_path =
        config.training.diagnostics.training_dataset_path;
    run_context.validation_dataset_path =
        config.training.diagnostics.training_dataset_path;
    const TrainingResult result = _trainer.fit(
        *model, dataset, fold.training, dataset, fold.validation, normalization,
        config.loss, config.training, seed, false, run_context, &config);
    return FoldMetrics{fold_index,
                       result.training_objective,
                       result.validation_mse,
                       result.training_samples,
                       result.validation_samples,
                       result.history};
}

CrossValidator::CrossValidator(const Dataset& dataset,
                               std::shared_ptr<const FoldSplitter> splitter,
                               std::shared_ptr<const TrialRunner> runner,
                               MPI_Comm communicator,
                               bool verbose)
    : _dataset(dataset),
      _splitter(std::move(splitter)),
      _runner(std::move(runner)),
      _communicator(communicator),
      _verbose(verbose) {
    if (!_splitter || !_runner) {
        throw std::invalid_argument("CrossValidator requires a splitter and trial runner.");
    }
}

int CrossValidator::rank() const {
    int value = 0;
    if (mpi_ready()) {
        MPI_Comm_rank(_communicator, &value);
    }
    return value;
}

std::vector<FoldIndices> CrossValidator::make_fold_plan() const {
    std::vector<FoldIndices> folds;
    std::string local_error;
    try {
        folds = _splitter->split(_dataset.num_samples());
        validate_folds(folds, _dataset.num_samples());
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown fold generation error";
    }
    throw_if_distributed_failure(local_error, "Fold generation", _communicator);

    if (mpi_ready()) {
        const unsigned long long local_hash =
            static_cast<unsigned long long>(fold_plan_hash(folds));
        unsigned long long minimum_hash = 0;
        unsigned long long maximum_hash = 0;
        MPI_Allreduce(&local_hash, &minimum_hash, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MIN, _communicator);
        MPI_Allreduce(&local_hash, &maximum_hash, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, _communicator);
        if (minimum_hash != maximum_hash) {
            throw std::runtime_error(
                "Fold splitter produced different plans across MPI ranks.");
        }
    }
    return folds;
}

CandidateResult CrossValidator::evaluate_with_folds(
    const TrialConfig& config,
    const std::vector<FoldIndices>& folds,
    size_t candidate_index,
    size_t candidate_count) const {
    if (config.training.validation_interval == 0) {
        throw std::invalid_argument("Validation interval must be positive.");
    }
    CandidateResult result(config);
    double weighted_training_sum = 0.0;
    double weighted_validation_sum = 0.0;
    size_t training_count = 0;
    size_t validation_count = 0;

    for (size_t fold_index = 0; fold_index < folds.size(); ++fold_index) {
        FoldMetrics metrics;
        std::string local_error;
        bool local_diagnostics_failure = false;
        try {
            metrics = _runner->run(
                config, _dataset, folds[fold_index], fold_index,
                TrialRunner::Context{candidate_index, candidate_count,
                                     folds.size()});
            if (metrics.fold != fold_index ||
                metrics.training_samples != folds[fold_index].training.size() ||
                metrics.validation_samples != folds[fold_index].validation.size()) {
                throw std::runtime_error(
                    "Trial runner returned the wrong fold or sample counts.");
            }
            if (!std::isfinite(metrics.training_objective) ||
                metrics.training_objective < 0.0 ||
                !std::isfinite(metrics.validation_mse) ||
                metrics.validation_mse < 0.0) {
                throw std::runtime_error(
                    "Trial runner returned invalid objective metrics.");
            }
            const size_t expected_history_size =
                config.training.epochs / config.training.validation_interval;
            if (metrics.history.size() != expected_history_size) {
                throw std::runtime_error(
                    "Trial runner returned an incomplete epoch history.");
            }
            for (size_t history_index = 0;
                 history_index < metrics.history.size(); ++history_index) {
                const EpochMetrics& point = metrics.history[history_index];
                const size_t expected_epoch =
                    (history_index + 1) * config.training.validation_interval;
                if (point.epoch != expected_epoch ||
                    !std::isfinite(point.training_objective) ||
                    point.training_objective < 0.0 ||
                    !std::isfinite(point.validation_mse) ||
                    point.validation_mse < 0.0) {
                    throw std::runtime_error(
                        "Trial runner returned invalid epoch history metrics.");
                }
            }
        } catch (const DiagnosticsError& error) {
            local_error = error.what();
            local_diagnostics_failure = true;
        } catch (const std::exception& error) {
            local_error = error.what();
        } catch (...) {
            local_error = "unknown trial runner error";
        }
        throw_if_distributed_failure(local_error, "Fold evaluation", _communicator,
                                     local_diagnostics_failure);

        if (mpi_ready()) {
            unsigned long long integer_metrics[3]{
                static_cast<unsigned long long>(metrics.fold),
                static_cast<unsigned long long>(metrics.training_samples),
                static_cast<unsigned long long>(metrics.validation_samples)};
            double floating_metrics[2]{metrics.training_objective,
                                       metrics.validation_mse};
            unsigned long long canonical_integers[3];
            double canonical_floats[2];
            std::copy(std::begin(integer_metrics), std::end(integer_metrics),
                      std::begin(canonical_integers));
            std::copy(std::begin(floating_metrics), std::end(floating_metrics),
                      std::begin(canonical_floats));
            MPI_Bcast(canonical_integers, 3, MPI_UNSIGNED_LONG_LONG, 0,
                      _communicator);
            MPI_Bcast(canonical_floats, 2, MPI_DOUBLE, 0, _communicator);

            std::vector<EpochMetrics> canonical_history = metrics.history;
            for (size_t history_index = 0;
                 history_index < canonical_history.size(); ++history_index) {
                unsigned long long canonical_epoch =
                    static_cast<unsigned long long>(
                        canonical_history[history_index].epoch);
                double canonical_values[2]{
                    canonical_history[history_index].training_objective,
                    canonical_history[history_index].validation_mse};
                MPI_Bcast(&canonical_epoch, 1, MPI_UNSIGNED_LONG_LONG, 0,
                          _communicator);
                MPI_Bcast(canonical_values, 2, MPI_DOUBLE, 0,
                          _communicator);

                const EpochMetrics& local_point = metrics.history[history_index];
                if (local_point.epoch != static_cast<size_t>(canonical_epoch) ||
                    !metric_close(local_point.training_objective,
                                  canonical_values[0]) ||
                    !metric_close(local_point.validation_mse,
                                  canonical_values[1])) {
                    local_error =
                        "trial runner history differs across MPI ranks";
                }
                canonical_history[history_index] = EpochMetrics{
                    static_cast<size_t>(canonical_epoch),
                    canonical_values[0],
                    canonical_values[1]};
            }

            if (!std::equal(std::begin(integer_metrics), std::end(integer_metrics),
                            std::begin(canonical_integers)) ||
                !metric_close(floating_metrics[0], canonical_floats[0]) ||
                !metric_close(floating_metrics[1], canonical_floats[1])) {
                local_error = "trial runner metrics differ across MPI ranks";
            }
            throw_if_distributed_failure(local_error, "Metric agreement",
                                         _communicator);
            metrics = FoldMetrics{
                static_cast<size_t>(canonical_integers[0]),
                canonical_floats[0],
                canonical_floats[1],
                static_cast<size_t>(canonical_integers[1]),
                static_cast<size_t>(canonical_integers[2]),
                std::move(canonical_history)};
        }

        result.folds.push_back(metrics);
        weighted_training_sum +=
            metrics.training_objective * metrics.training_samples;
        weighted_validation_sum += metrics.validation_mse * metrics.validation_samples;
        training_count += metrics.training_samples;
        validation_count += metrics.validation_samples;

        if (_verbose && rank() == 0) {
            std::cout << "  Fold " << (fold_index + 1) << "/" << folds.size()
                      << " | validation physical MSE: "
                      << metrics.validation_mse << std::endl;
        }
    }

    if (training_count == 0 || validation_count == 0) {
        throw std::runtime_error("Cross-validation produced an empty aggregate.");
    }
    result.mean_training_objective =
        weighted_training_sum / static_cast<double>(training_count);
    result.mean_validation_mse =
        weighted_validation_sum / static_cast<double>(validation_count);

    double weighted_variance = 0.0;
    for (const auto& fold : result.folds) {
        const double difference =
            fold.validation_mse - result.mean_validation_mse;
        weighted_variance += difference * difference * fold.validation_samples;
    }
    result.validation_stddev =
        std::sqrt(weighted_variance / static_cast<double>(validation_count));
    result.success = true;
    return result;
}

CandidateResult CrossValidator::evaluate(const TrialConfig& config) const {
    verify_trial_config_agreement(config, _communicator);
    const auto folds = make_fold_plan();
    CandidateResult result = evaluate_with_folds(config, folds, 0, 1);
    write_cv_artifacts({&result}, _communicator);
    return result;
}

SearchResult CrossValidator::tune(const CandidateSource& candidate_source) const {
    std::vector<TrialConfig> configurations;
    std::string local_error;
    try {
        configurations = candidate_source.candidates();
        if (configurations.empty()) {
            throw std::invalid_argument(
                "Hyperparameter search contains no candidates.");
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown candidate generation error";
    }
    throw_if_distributed_failure(local_error, "Candidate generation",
                                 _communicator);

    if (mpi_ready()) {
        const unsigned long long local_count = configurations.size();
        unsigned long long minimum_count = 0;
        unsigned long long maximum_count = 0;
        MPI_Allreduce(&local_count, &minimum_count, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MIN, _communicator);
        MPI_Allreduce(&local_count, &maximum_count, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, _communicator);
        if (minimum_count != maximum_count) {
            throw std::runtime_error(
                "Candidate source produced different counts across MPI ranks.");
        }
        for (const auto& configuration : configurations) {
            verify_trial_config_agreement(configuration, _communicator);
        }
    }
    const auto folds = make_fold_plan();

    SearchResult search;
    search.candidates.reserve(configurations.size());
    double best_score = std::numeric_limits<double>::infinity();
    for (size_t index = 0; index < configurations.size(); ++index) {
        if (_verbose && rank() == 0) {
            std::cout << "Candidate " << (index + 1) << "/"
                      << configurations.size() << ": "
                      << configurations[index].name << std::endl;
        }

        try {
            CandidateResult result =
                evaluate_with_folds(configurations[index], folds, index,
                                    configurations.size());
            if (_verbose && rank() == 0) {
                std::cout << "  Mean validation physical MSE: "
                          << result.mean_validation_mse << " +/- "
                          << result.validation_stddev << std::endl;
            }
            if (result.mean_validation_mse < best_score) {
                best_score = result.mean_validation_mse;
                search.best_index = search.candidates.size();
            }
            search.candidates.push_back(std::move(result));
        } catch (const DiagnosticsError&) {
            throw;
        } catch (const std::exception& error) {
            CandidateResult failed(configurations[index]);
            failed.error = error.what();
            if (_verbose && rank() == 0) {
                std::cout << "  Candidate failed: " << failed.error << std::endl;
            }
            search.candidates.push_back(std::move(failed));
        }
    }

    std::vector<const CandidateResult*> candidate_pointers;
    candidate_pointers.reserve(search.candidates.size());
    for (const auto& candidate : search.candidates) {
        candidate_pointers.push_back(&candidate);
    }
    write_cv_artifacts(candidate_pointers, _communicator);

    return search;
}
