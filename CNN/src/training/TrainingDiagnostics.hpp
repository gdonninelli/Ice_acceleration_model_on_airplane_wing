#ifndef TRAININGDIAGNOSTICS_HPP
#define TRAININGDIAGNOSTICS_HPP

#include "DiagnosticsConfig.hpp"
#include "model/CNNModel.hpp"
#include "tuning/TrialConfig.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <string>
#include <stdexcept>

class DiagnosticsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class RunningStatistics {
public:
    void add(double value);
    void merge(const RunningStatistics& other);

    uint64_t count() const { return _count; }
    long double sum() const { return _sum; }
    long double squared_sum() const { return _squared_sum; }
    double mean() const;
    double population_variance() const;
    double minimum() const;
    double maximum() const;

private:
    uint64_t _count = 0;
    long double _sum = 0.0L;
    long double _squared_sum = 0.0L;
    double _minimum = 0.0;
    double _maximum = 0.0;
};

struct UpdateMeasurement {
    double pre_update_norm = 0.0;
    double update_norm = 0.0;
    double ratio = 0.0;
    bool denominator_near_zero = false;
};

double l2_norm(const float* values, size_t count);
UpdateMeasurement measure_update(const float* before,
                                 const float* after,
                                 size_t count,
                                 double epsilon = 1e-12);

struct EpochDiagnosticsSummary {
    double gradient_rms_summary = 0.0;
    double activated_variance_mean = 0.0;
    double activated_variance_min = 0.0;
    double activated_variance_max = 0.0;
};

std::string diagnostics_path_component(const std::string& value);
std::filesystem::path diagnostics_session_directory(
    const TrainingDiagnosticsConfig& config);
std::filesystem::path diagnostics_run_directory(
    const TrainingDiagnosticsConfig& config,
    const TrainingRunContext& context);

class TrainingDiagnosticsRecorder {
public:
    static constexpr double kUpdateRatioEpsilon = 1e-12;

    TrainingDiagnosticsRecorder(const TrainingDiagnosticsConfig& config,
                                const TrainingRunContext& context,
                                const TrialConfig& trial,
                                const CNNModel& model,
                                MPI_Comm communicator = MPI_COMM_WORLD);
    ~TrainingDiagnosticsRecorder();

    TrainingDiagnosticsRecorder(const TrainingDiagnosticsRecorder&) = delete;
    TrainingDiagnosticsRecorder& operator=(const TrainingDiagnosticsRecorder&) = delete;

    void begin_epoch(size_t epoch);
    void capture_activation(size_t layer_index,
                            const std::string& layer_name,
                            const Tensor& pre_activation,
                            const Tensor& post_activation);
    void before_optimizer_step(const CNNModel& model);
    void after_optimizer_step(const CNNModel& model);
    EpochDiagnosticsSummary finish_epoch(size_t epoch,
                                         double training_objective,
                                         double validation_mse,
                                         size_t samples,
                                         size_t batches);

    const std::filesystem::path& output_directory() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

#endif // TRAININGDIAGNOSTICS_HPP
