#include "TrainingDiagnostics.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef CNN_SOURCE_REVISION
#define CNN_SOURCE_REVISION "unknown"
#endif

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

std::string csv_field(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

std::string json_string(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<int>(character)
                       << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
    return output.str();
}

std::string json_number(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

std::string padded_index(size_t value) {
    std::ostringstream output;
    output << std::setw(3) << std::setfill('0') << value;
    return output.str();
}

uint64_t stable_hash(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char character : value) {
        hash ^= character;
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct EpochAggregate {
    uint64_t count = 0;
    long double sum = 0.0L;
    long double squared_sum = 0.0L;
    double maximum = 0.0;
    double last = 0.0;

    void add(double value) {
        if (count == 0 || value > maximum) {
            maximum = value;
        }
        ++count;
        sum += value;
        squared_sum += static_cast<long double>(value) * value;
        last = value;
    }

    double mean() const {
        return count == 0 ? 0.0 : static_cast<double>(sum / count);
    }

    double rms() const {
        return count == 0 ? 0.0
                          : std::sqrt(std::max(
                                0.0, static_cast<double>(squared_sum / count)));
    }
};

double tensors_norm(const std::vector<std::shared_ptr<Tensor>>& tensors,
                    bool gradients) {
    long double squared = 0.0L;
    for (const auto& tensor : tensors) {
        if (!tensor) {
            continue;
        }
        const float* values = gradients ? tensor->get_grad() : tensor->get_data();
        for (size_t index = 0; index < tensor->size(); ++index) {
            squared += static_cast<long double>(values[index]) * values[index];
        }
    }
    return std::sqrt(static_cast<double>(squared));
}

void require_output(std::ofstream& output, const std::filesystem::path& path) {
    if (!output) {
        throw std::runtime_error("Failed to write diagnostics file: " +
                                 path.string());
    }
}
} // namespace

void RunningStatistics::add(double value) {
    if (_count == 0) {
        _minimum = value;
        _maximum = value;
    } else {
        _minimum = std::min(_minimum, value);
        _maximum = std::max(_maximum, value);
    }
    ++_count;
    _sum += value;
    _squared_sum += static_cast<long double>(value) * value;
}

void RunningStatistics::merge(const RunningStatistics& other) {
    if (other._count == 0) {
        return;
    }
    if (_count == 0) {
        _minimum = other._minimum;
        _maximum = other._maximum;
    } else {
        _minimum = std::min(_minimum, other._minimum);
        _maximum = std::max(_maximum, other._maximum);
    }
    _count += other._count;
    _sum += other._sum;
    _squared_sum += other._squared_sum;
}

double RunningStatistics::mean() const {
    return _count == 0 ? std::numeric_limits<double>::quiet_NaN()
                       : static_cast<double>(_sum / _count);
}

double RunningStatistics::population_variance() const {
    if (_count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const long double average = _sum / _count;
    const long double variance = _squared_sum / _count - average * average;
    return static_cast<double>(std::max(0.0L, variance));
}

double RunningStatistics::minimum() const {
    return _count == 0 ? std::numeric_limits<double>::quiet_NaN() : _minimum;
}

double RunningStatistics::maximum() const {
    return _count == 0 ? std::numeric_limits<double>::quiet_NaN() : _maximum;
}

double l2_norm(const float* values, size_t count) {
    if (!values && count != 0) {
        throw std::invalid_argument("l2_norm received a null buffer.");
    }
    long double squared = 0.0L;
    for (size_t index = 0; index < count; ++index) {
        squared += static_cast<long double>(values[index]) * values[index];
    }
    return std::sqrt(static_cast<double>(squared));
}

UpdateMeasurement measure_update(const float* before,
                                 const float* after,
                                 size_t count,
                                 double epsilon) {
    if ((!before || !after) && count != 0) {
        throw std::invalid_argument("measure_update received a null buffer.");
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("Update-ratio epsilon must be positive.");
    }
    long double pre_squared = 0.0L;
    long double update_squared = 0.0L;
    for (size_t index = 0; index < count; ++index) {
        pre_squared += static_cast<long double>(before[index]) * before[index];
        const long double difference =
            static_cast<long double>(after[index]) - before[index];
        update_squared += difference * difference;
    }
    const double pre_norm = std::sqrt(static_cast<double>(pre_squared));
    const double update_norm = std::sqrt(static_cast<double>(update_squared));
    return UpdateMeasurement{pre_norm,
                             update_norm,
                             update_norm / std::max(pre_norm, epsilon),
                             pre_norm <= epsilon};
}

std::string diagnostics_path_component(const std::string& value) {
    std::string output;
    output.reserve(value.size());
    for (unsigned char character : value) {
        if (std::isalnum(character) || character == '-' || character == '_') {
            output.push_back(static_cast<char>(character));
        } else {
            output.push_back('_');
        }
    }
    if (output.empty()) {
        output = "run";
    }
    if (output != value) {
        std::ostringstream suffix;
        suffix << '_' << std::hex << std::setw(8) << std::setfill('0')
               << static_cast<uint32_t>(stable_hash(value));
        output += suffix.str();
    }
    return output;
}

std::filesystem::path diagnostics_session_directory(
    const TrainingDiagnosticsConfig& config) {
    return std::filesystem::path(config.results_root) /
           diagnostics_path_component(config.experiment_name) /
           diagnostics_path_component(config.run_name);
}

std::filesystem::path diagnostics_run_directory(
    const TrainingDiagnosticsConfig& config,
    const TrainingRunContext& context) {
    auto output = diagnostics_session_directory(config);
    if (context.mode == "cross_validation") {
        output /= "candidate_" + padded_index(context.candidate_index);
        output /= "fold_" + padded_index(context.fold_index);
    } else if (context.final_subdirectory) {
        output /= "final";
    }
    return output;
}

struct TrainingDiagnosticsRecorder::Impl {
    struct ScopeState {
        size_t layer_index = 0;
        std::string layer_name;
        std::string scope;
        std::vector<std::shared_ptr<Tensor>> tensors;
        EpochAggregate gradients;
        EpochAggregate pre_norms;
        EpochAggregate update_norms;
        EpochAggregate ratios;
        uint64_t near_zero_steps = 0;
    };

    struct ActivationState {
        size_t layer_index = 0;
        std::string layer_name;
        std::string phase;
        RunningStatistics statistics;
        std::vector<unsigned long long> histogram;
    };

    TrainingDiagnosticsConfig config;
    TrainingRunContext context;
    const TrialConfig& trial;
    MPI_Comm communicator;
    int rank = 0;
    int world_size = 1;
    std::filesystem::path directory;
    std::vector<ModelLayerInfo> layers;
    OptimizerMetadata optimizer;
    std::vector<ScopeState> scopes;
    std::vector<ActivationState> activations;
    std::unordered_map<const Tensor*, std::vector<float>> before_values;
    EpochAggregate learning_rates;
    size_t current_epoch = 0;
    std::ofstream epoch_file;
    std::ofstream gradient_file;
    std::ofstream update_file;
    std::ofstream activation_file;
    std::ofstream histogram_file;
    std::ofstream learning_rate_step_file;
    std::vector<double> effective_learning_rates;

    Impl(const TrainingDiagnosticsConfig& diagnostics_config,
         const TrainingRunContext& run_context,
         const TrialConfig& trial_config,
         const CNNModel& model,
         MPI_Comm mpi_communicator)
        : config(diagnostics_config),
          context(run_context),
          trial(trial_config),
          communicator(mpi_communicator),
          directory(diagnostics_run_directory(config, context)),
          layers(model.layer_info()),
          optimizer(model.optimizer_metadata()) {
        if (config.histogram_bins == 0 ||
            !std::isfinite(config.histogram_min) ||
            !std::isfinite(config.histogram_max) ||
            config.histogram_min >= config.histogram_max ||
            config.histogram_bins >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Invalid diagnostics histogram configuration.");
        }
        if (mpi_ready()) {
            MPI_Comm_rank(communicator, &rank);
            MPI_Comm_size(communicator, &world_size);
        }

        for (const auto& layer : layers) {
            if (!layer.parameters.empty()) {
                ScopeState all;
                all.layer_index = layer.index;
                all.layer_name = layer.name;
                all.scope = "all";
                for (const auto& parameter : layer.parameters) {
                    all.tensors.push_back(parameter.tensor);
                }
                scopes.push_back(std::move(all));
                for (const auto& parameter : layer.parameters) {
                    scopes.push_back(ScopeState{layer.index,
                                                layer.name,
                                                parameter.name,
                                                {parameter.tensor}});
                }
            }
            if (layer.activation) {
                activations.push_back(ActivationState{
                    layer.index, layer.name, "pre_activation", {},
                    std::vector<unsigned long long>(config.histogram_bins)});
                activations.push_back(ActivationState{
                    layer.index, layer.name, "post_activation", {},
                    std::vector<unsigned long long>(config.histogram_bins)});
            }
        }

        if (rank == 0) {
            std::filesystem::create_directories(directory);
            open_files();
            write_metadata(model);
        }
    }

    void open_files() {
        const auto epoch_path = directory / "epoch_metrics.csv";
        const auto gradient_path = directory / "gradient_norms.csv";
        const auto update_path = directory / "parameter_update_ratios.csv";
        const auto activation_path = directory / "activation_statistics.csv";
        const auto histogram_path = directory / "activation_histograms.csv";
        const auto learning_rate_step_path =
            directory / "learning_rate_steps.csv";
        epoch_file.open(epoch_path, std::ios::trunc);
        gradient_file.open(gradient_path, std::ios::trunc);
        update_file.open(update_path, std::ios::trunc);
        activation_file.open(activation_path, std::ios::trunc);
        histogram_file.open(histogram_path, std::ios::trunc);
        learning_rate_step_file.open(learning_rate_step_path, std::ios::trunc);
        require_output(epoch_file, epoch_path);
        require_output(gradient_file, gradient_path);
        require_output(update_file, update_path);
        require_output(activation_file, activation_path);
        require_output(histogram_file, histogram_path);
        require_output(learning_rate_step_file, learning_rate_step_path);
        epoch_file << "epoch,train_objective,validation_physical_mse,samples,batches,configured_learning_rate,effective_learning_rate\n";
        gradient_file << "epoch,layer_index,layer_name,parameter_scope,mean_norm,rms_norm,maximum_norm,last_norm,optimizer_steps\n";
        update_file << "epoch,layer_index,layer_name,parameter_scope,mean_pre_update_norm,mean_update_norm,mean_ratio,rms_ratio,maximum_ratio,last_ratio,near_zero_denominator_steps,optimizer_steps,epsilon\n";
        activation_file << "epoch,layer_index,layer_name,phase,count,mean,variance,minimum,maximum\n";
        histogram_file << "epoch,layer_index,layer_name,phase,bin_edges,counts,total_population\n";
        learning_rate_step_file <<
            "epoch,optimizer_step,effective_learning_rate\n";
    }

    void write_metadata(const CNNModel& model) {
        const auto path = directory / "metadata.json";
        std::ofstream output(path, std::ios::trunc);
        require_output(output, path);
        const std::string training_path = context.training_dataset_path.empty()
            ? config.training_dataset_path : context.training_dataset_path;
        const std::string validation_path = context.validation_dataset_path.empty()
            ? config.validation_dataset_path : context.validation_dataset_path;

        output << "{\n"
               << "  \"format_version\": 1,\n"
               << "  \"mode\": " << json_string(context.mode) << ",\n"
               << "  \"experiment_name\": " << json_string(config.experiment_name) << ",\n"
               << "  \"run_name\": " << json_string(config.run_name) << ",\n"
               << "  \"candidate_index\": ";
        if (context.candidate_index == TrainingRunContext::no_index) output << "null";
        else output << context.candidate_index;
        output << ",\n  \"candidate_count\": " << context.candidate_count
               << ",\n  \"candidate_name\": " << json_string(trial.name)
               << ",\n  \"fold_index\": ";
        if (context.fold_index == TrainingRunContext::no_index) output << "null";
        else output << context.fold_index;
        output << ",\n  \"fold_count\": " << context.fold_count
               << ",\n  \"random_seed\": " << context.random_seed
               << ",\n  \"epoch_count\": " << trial.training.epochs
               << ",\n  \"global_batch_size\": " << trial.training.global_batch_size
               << ",\n  \"validation_interval\": " << trial.training.validation_interval
               << ",\n  \"gradient_clip\": " << json_number(model.gradient_clip())
               << ",\n  \"physics_weight\": "
               << json_number(trial.loss.physics_weight)
               << ",\n  \"mpi_world_size\": " << world_size
               << ",\n  \"training_dataset_path\": " << json_string(training_path)
               << ",\n  \"validation_dataset_path\": " << json_string(validation_path)
               << ",\n  \"source_revision\": " << json_string(CNN_SOURCE_REVISION)
               << ",\n  \"optimizer\": {\"name\": " << json_string(optimizer.name)
               << ", \"configured_learning_rate\": "
               << json_number(optimizer.configured_learning_rate)
               << ", \"effective_learning_rate\": "
               << json_number(optimizer.effective_learning_rate)
               << ", \"scheduled\": " << (optimizer.scheduled ? "true" : "false")
               << ", \"settings\": {";
        bool first = true;
        for (const auto& [name, value] : optimizer.settings) {
            if (!first) output << ", ";
            output << json_string(name) << ": " << json_number(value);
            first = false;
        }
        output << "}},\n  \"selected_hyperparameters\": {";
        first = true;
        for (const auto& [name, value] : trial.selected_parameters) {
            if (!first) output << ", ";
            output << json_string(name) << ": " << json_string(value);
            first = false;
        }
        output << "},\n  \"model_layers\": [\n";
        for (size_t index = 0; index < layers.size(); ++index) {
            output << "    {\"index\": " << layers[index].index
                   << ", \"name\": " << json_string(layers[index].name)
                   << ", \"activation\": "
                   << (layers[index].activation ? "true" : "false")
                   << ", \"parameters\": [";
            for (size_t parameter = 0;
                 parameter < layers[index].parameters.size(); ++parameter) {
                if (parameter != 0) output << ", ";
                output << json_string(layers[index].parameters[parameter].name);
            }
            output << "]}" << (index + 1 == layers.size() ? "\n" : ",\n");
        }
        output << "  ],\n  \"model_recipe\": {\"feature_layers\": [";
        for (size_t index = 0; index < trial.model.feature_layers.size(); ++index) {
            if (index != 0) output << ", ";
            output << json_string(trial.model.feature_layers[index].description());
        }
        output << "], \"head_layers\": [";
        for (size_t index = 0; index < trial.model.head_layers.size(); ++index) {
            if (index != 0) output << ", ";
            output << json_string(trial.model.head_layers[index].description());
        }
        output << "], \"concatenate_scalars\": "
               << (trial.model.concatenate_scalars ? "true" : "false")
               << "},\n  \"histogram\": {\"bins\": " << config.histogram_bins
               << ", \"minimum\": " << json_number(config.histogram_min)
               << ", \"maximum\": " << json_number(config.histogram_max)
               << ", \"out_of_range_policy\": \"clamp_to_edge_bins\"},\n"
               << "  \"update_ratio_epsilon\": "
               << json_number(TrainingDiagnosticsRecorder::kUpdateRatioEpsilon)
               << "\n}\n";
        output.flush();
        require_output(output, path);
    }

    void reset_epoch(size_t epoch) {
        current_epoch = epoch;
        learning_rates = {};
        before_values.clear();
        effective_learning_rates.clear();
        for (auto& scope : scopes) {
            scope.gradients = {};
            scope.pre_norms = {};
            scope.update_norms = {};
            scope.ratios = {};
            scope.near_zero_steps = 0;
        }
        for (auto& activation : activations) {
            activation.statistics = {};
            std::fill(activation.histogram.begin(), activation.histogram.end(), 0);
        }
    }

    void add_tensor(ActivationState& state, const Tensor& tensor) {
        const float* values = tensor.get_data();
        const double width = config.histogram_max - config.histogram_min;
        for (size_t index = 0; index < tensor.size(); ++index) {
            const double value = values[index];
            state.statistics.add(value);
            size_t bin = 0;
            if (std::isfinite(value) && value > config.histogram_min) {
                if (value >= config.histogram_max) {
                    bin = config.histogram_bins - 1;
                } else {
                    bin = std::min(
                        config.histogram_bins - 1,
                        static_cast<size_t>((value - config.histogram_min) /
                                            width * config.histogram_bins));
                }
            }
            ++state.histogram[bin];
        }
    }

    ActivationState& activation_state(size_t layer_index,
                                      const std::string& phase) {
        auto iterator = std::find_if(
            activations.begin(), activations.end(),
            [&](const ActivationState& state) {
                return state.layer_index == layer_index && state.phase == phase;
            });
        if (iterator == activations.end()) {
            throw DiagnosticsError("Unknown activation layer in diagnostics.");
        }
        return *iterator;
    }

};

TrainingDiagnosticsRecorder::TrainingDiagnosticsRecorder(
    const TrainingDiagnosticsConfig& config,
    const TrainingRunContext& context,
    const TrialConfig& trial,
    const CNNModel& model,
    MPI_Comm communicator)
    : _impl(std::make_unique<Impl>(config, context, trial, model, communicator)) {}

TrainingDiagnosticsRecorder::~TrainingDiagnosticsRecorder() = default;

void TrainingDiagnosticsRecorder::begin_epoch(size_t epoch) {
    _impl->reset_epoch(epoch);
}

void TrainingDiagnosticsRecorder::capture_activation(
    size_t layer_index,
    const std::string& layer_name,
    const Tensor& pre_activation,
    const Tensor& post_activation) {
    auto& pre = _impl->activation_state(layer_index, "pre_activation");
    auto& post = _impl->activation_state(layer_index, "post_activation");
    if (pre.layer_name != layer_name || post.layer_name != layer_name) {
            throw DiagnosticsError("Activation layer ordering changed during training.");
    }
    _impl->add_tensor(pre, pre_activation);
    _impl->add_tensor(post, post_activation);
}

void TrainingDiagnosticsRecorder::before_optimizer_step(const CNNModel& model) {
    if (_impl->rank != 0) {
        return;
    }
    for (auto& scope : _impl->scopes) {
        scope.gradients.add(tensors_norm(scope.tensors, true));
    }
    _impl->before_values.clear();
    for (const auto& parameter : model.parameters()) {
        const float* values = parameter.tensor->get_data();
        _impl->before_values.emplace(
            parameter.tensor.get(),
            std::vector<float>(values, values + parameter.tensor->size()));
    }
    const double effective_learning_rate =
        model.optimizer_metadata().effective_learning_rate;
    _impl->learning_rates.add(effective_learning_rate);
    _impl->effective_learning_rates.push_back(effective_learning_rate);
}

void TrainingDiagnosticsRecorder::after_optimizer_step(const CNNModel&) {
    if (_impl->rank != 0) {
        return;
    }
    for (auto& scope : _impl->scopes) {
        long double pre_squared = 0.0L;
        long double update_squared = 0.0L;
        for (const auto& tensor : scope.tensors) {
            const auto found = _impl->before_values.find(tensor.get());
            if (found == _impl->before_values.end()) {
                throw DiagnosticsError("Missing pre-update parameter snapshot.");
            }
            const float* after = tensor->get_data();
            for (size_t index = 0; index < tensor->size(); ++index) {
                pre_squared += static_cast<long double>(found->second[index]) *
                               found->second[index];
                const long double delta =
                    static_cast<long double>(after[index]) - found->second[index];
                update_squared += delta * delta;
            }
        }
        const double pre_norm = std::sqrt(static_cast<double>(pre_squared));
        const double update_norm = std::sqrt(static_cast<double>(update_squared));
        const double ratio = update_norm /
            std::max(pre_norm, kUpdateRatioEpsilon);
        scope.pre_norms.add(pre_norm);
        scope.update_norms.add(update_norm);
        scope.ratios.add(ratio);
        if (pre_norm <= kUpdateRatioEpsilon) {
            ++scope.near_zero_steps;
        }
    }
    _impl->before_values.clear();
}

EpochDiagnosticsSummary TrainingDiagnosticsRecorder::finish_epoch(
    size_t epoch,
    double training_objective,
    double validation_mse,
    size_t samples,
    size_t batches) {
    if (epoch != _impl->current_epoch) {
        throw DiagnosticsError("Diagnostics epoch lifecycle mismatch.");
    }

    struct GlobalActivation {
        const Impl::ActivationState* local = nullptr;
        unsigned long long count = 0;
        long double sum = 0.0L;
        long double squared_sum = 0.0L;
        double minimum = 0.0;
        double maximum = 0.0;
        std::vector<unsigned long long> histogram;
    };
    std::vector<GlobalActivation> global_activations;
    global_activations.reserve(_impl->activations.size());
    for (const auto& state : _impl->activations) {
        GlobalActivation global;
        global.local = &state;
        global.count = state.statistics.count();
        global.sum = state.statistics.sum();
        global.squared_sum = state.statistics.squared_sum();
        global.minimum = global.count == 0
            ? std::numeric_limits<double>::infinity()
            : state.statistics.minimum();
        global.maximum = global.count == 0
            ? -std::numeric_limits<double>::infinity()
            : state.statistics.maximum();
        global.histogram = state.histogram;
        if (mpi_ready() && _impl->world_size > 1) {
            unsigned long long reduced_count = 0;
            long double local_sums[2]{global.sum, global.squared_sum};
            long double reduced_sums[2]{};
            double reduced_min = 0.0;
            double reduced_max = 0.0;
            MPI_Allreduce(&global.count, &reduced_count, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, _impl->communicator);
            MPI_Allreduce(local_sums, reduced_sums, 2, MPI_LONG_DOUBLE,
                          MPI_SUM, _impl->communicator);
            MPI_Allreduce(&global.minimum, &reduced_min, 1, MPI_DOUBLE,
                          MPI_MIN, _impl->communicator);
            MPI_Allreduce(&global.maximum, &reduced_max, 1, MPI_DOUBLE,
                          MPI_MAX, _impl->communicator);
            MPI_Allreduce(MPI_IN_PLACE, global.histogram.data(),
                          static_cast<int>(global.histogram.size()),
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, _impl->communicator);
            global.count = reduced_count;
            global.sum = reduced_sums[0];
            global.squared_sum = reduced_sums[1];
            global.minimum = reduced_min;
            global.maximum = reduced_max;
        }
        global_activations.push_back(std::move(global));
    }

    EpochDiagnosticsSummary summary;
    if (_impl->rank == 0) {
        const double configured_lr =
            _impl->optimizer.configured_learning_rate;
        _impl->epoch_file << epoch << ',' << std::setprecision(17)
                          << training_objective << ',';
        if (std::isfinite(validation_mse)) {
            _impl->epoch_file << validation_mse;
        }
        _impl->epoch_file << ',' << samples << ',' << batches << ','
                          << configured_lr << ','
                          << _impl->learning_rates.mean() << '\n';
        if (_impl->optimizer.scheduled) {
            for (size_t step = 0;
                 step < _impl->effective_learning_rates.size(); ++step) {
                _impl->learning_rate_step_file
                    << epoch << ',' << (step + 1) << ','
                    << std::setprecision(17)
                    << _impl->effective_learning_rates[step] << '\n';
            }
        }

        long double gradient_summary_squared = 0.0L;
        for (const auto& scope : _impl->scopes) {
            _impl->gradient_file
                << epoch << ',' << scope.layer_index << ','
                << csv_field(scope.layer_name) << ',' << csv_field(scope.scope)
                << ',' << std::setprecision(17) << scope.gradients.mean()
                << ',' << scope.gradients.rms() << ','
                << scope.gradients.maximum << ',' << scope.gradients.last
                << ',' << scope.gradients.count << '\n';
            _impl->update_file
                << epoch << ',' << scope.layer_index << ','
                << csv_field(scope.layer_name) << ',' << csv_field(scope.scope)
                << ',' << std::setprecision(17) << scope.pre_norms.mean()
                << ',' << scope.update_norms.mean() << ','
                << scope.ratios.mean() << ',' << scope.ratios.rms() << ','
                << scope.ratios.maximum << ',' << scope.ratios.last << ','
                << scope.near_zero_steps << ',' << scope.ratios.count << ','
                << kUpdateRatioEpsilon << '\n';
            if (scope.scope == "all") {
                const double rms = scope.gradients.rms();
                gradient_summary_squared += rms * rms;
            }
        }
        summary.gradient_rms_summary =
            std::sqrt(static_cast<double>(gradient_summary_squared));

        std::vector<double> post_variances;
        for (const auto& global : global_activations) {
            if (global.count == 0) {
                continue;
            }
            const long double mean = global.sum / global.count;
            const double variance = static_cast<double>(std::max(
                0.0L, global.squared_sum / global.count - mean * mean));
            _impl->activation_file
                << epoch << ',' << global.local->layer_index << ','
                << csv_field(global.local->layer_name) << ','
                << global.local->phase << ',' << global.count << ','
                << std::setprecision(17) << static_cast<double>(mean) << ','
                << variance << ',' << global.minimum << ',' << global.maximum
                << '\n';

            std::ostringstream edges;
            std::ostringstream counts;
            const double width =
                (_impl->config.histogram_max - _impl->config.histogram_min) /
                _impl->config.histogram_bins;
            for (size_t bin = 0; bin <= _impl->config.histogram_bins; ++bin) {
                if (bin != 0) edges << ';';
                edges << std::setprecision(17)
                      << _impl->config.histogram_min + width * bin;
            }
            for (size_t bin = 0; bin < global.histogram.size(); ++bin) {
                if (bin != 0) counts << ';';
                counts << global.histogram[bin];
            }
            _impl->histogram_file
                << epoch << ',' << global.local->layer_index << ','
                << csv_field(global.local->layer_name) << ','
                << global.local->phase << ',' << csv_field(edges.str()) << ','
                << csv_field(counts.str()) << ',' << global.count << '\n';
            if (global.local->phase == "post_activation") {
                post_variances.push_back(variance);
            }
        }
        if (!post_variances.empty()) {
            summary.activated_variance_mean =
                std::accumulate(post_variances.begin(), post_variances.end(), 0.0) /
                post_variances.size();
            const auto [minimum, maximum] = std::minmax_element(
                post_variances.begin(), post_variances.end());
            summary.activated_variance_min = *minimum;
            summary.activated_variance_max = *maximum;
        }
        _impl->epoch_file.flush();
        _impl->gradient_file.flush();
        _impl->update_file.flush();
        _impl->activation_file.flush();
        _impl->histogram_file.flush();
        _impl->learning_rate_step_file.flush();
        require_output(_impl->epoch_file,
                       _impl->directory / "epoch_metrics.csv");
        require_output(_impl->gradient_file,
                       _impl->directory / "gradient_norms.csv");
        require_output(_impl->update_file,
                       _impl->directory / "parameter_update_ratios.csv");
        require_output(_impl->activation_file,
                       _impl->directory / "activation_statistics.csv");
        require_output(_impl->histogram_file,
                       _impl->directory / "activation_histograms.csv");
        require_output(_impl->learning_rate_step_file,
                       _impl->directory / "learning_rate_steps.csv");
    }
    return summary;
}

const std::filesystem::path& TrainingDiagnosticsRecorder::output_directory() const {
    return _impl->directory;
}
