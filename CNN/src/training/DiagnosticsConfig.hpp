#ifndef DIAGNOSTICSCONFIG_HPP
#define DIAGNOSTICSCONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

struct TrainingDiagnosticsConfig {
    bool enabled = false;
    std::string results_root = "results";
    std::string experiment_name = "cnn";
    std::string run_name = "run";
    size_t histogram_bins = 64;
    double histogram_min = -10.0;
    double histogram_max = 10.0;
    std::string training_dataset_path;
    std::string validation_dataset_path;
};

struct TrainingRunContext {
    static constexpr size_t no_index = std::numeric_limits<size_t>::max();

    std::string mode = "final_training";
    size_t candidate_index = no_index;
    size_t candidate_count = 0;
    size_t fold_index = no_index;
    size_t fold_count = 0;
    bool final_subdirectory = false;
    uint64_t random_seed = 0;
    std::string training_dataset_path;
    std::string validation_dataset_path;
};

#endif // DIAGNOSTICSCONFIG_HPP
