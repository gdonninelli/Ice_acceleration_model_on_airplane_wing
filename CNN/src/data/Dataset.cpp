#include "Dataset.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace {
double safe_std(double variance) {
    const double value = std::sqrt(std::max(variance, 0.0));
    return value < 1e-8 ? 1.0 : value;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}
} // namespace

size_t DataBatch::size() const {
    if (!targets) {
        return 0;
    }
    const auto shape = targets->get_shape();
    return shape.empty() ? 0 : shape[0];
}

Dataset::Dataset(const std::string& npz_path)
    : _sdf_height(0), _sdf_width(0) {
    _sdf_shape = load_shape(npz_path, "X_sdf");
    _scalar_shape = load_shape(npz_path, "X_scalars");
    _target_shape = load_shape(npz_path, "Y_cl");
    if (_sdf_shape.size() == 3) {
        _sdf_height = _sdf_shape[1];
        _sdf_width = _sdf_shape[2];
    } else if (_sdf_shape.size() == 4 && _sdf_shape[1] == 1) {
        _sdf_height = _sdf_shape[2];
        _sdf_width = _sdf_shape[3];
    }
    _sdf = load_array(npz_path, "X_sdf");
    _scalars = load_array(npz_path, "X_scalars");
    _targets = load_array(npz_path, "Y_cl");
    validate();
}

Dataset::Dataset(std::vector<float> sdf,
                 std::vector<float> scalars,
                 std::vector<float> targets,
                 size_t sdf_height,
                 size_t sdf_width)
    : _sdf_height(sdf_height),
      _sdf_width(sdf_width),
      _sdf(std::move(sdf)),
      _scalars(std::move(scalars)),
      _targets(std::move(targets)) {
    _sdf_shape = {_targets.size(), _sdf_height, _sdf_width};
    _scalar_shape = {_targets.size(), kScalarFeatures};
    _target_shape = {_targets.size()};
    validate();
}

void Dataset::validate() const {
    if (_sdf_height == 0 || _sdf_width == 0) {
        throw std::invalid_argument("Dataset SDF dimensions must be positive.");
    }
    if (_targets.empty()) {
        throw std::invalid_argument("Dataset must contain at least one sample.");
    }

    const size_t samples = _targets.size();
    const bool valid_sdf_shape =
        _sdf_shape == std::vector<size_t>{samples, _sdf_height, _sdf_width} ||
        _sdf_shape ==
            std::vector<size_t>{samples, 1, _sdf_height, _sdf_width};
    if (!valid_sdf_shape) {
        throw std::invalid_argument(
            "Dataset X_sdf shape must be [samples, height, width] or [samples, 1, height, width].");
    }
    if (_scalar_shape != std::vector<size_t>{samples, kScalarFeatures}) {
        throw std::invalid_argument("Dataset X_scalars shape must be [samples, 2].");
    }
    if (_target_shape != std::vector<size_t>{samples} &&
        _target_shape != std::vector<size_t>{samples, 1}) {
        throw std::invalid_argument("Dataset Y_cl shape must be [samples] or [samples, 1].");
    }

    const size_t sdf_values_per_sample = _sdf_height * _sdf_width;
    if (_sdf.size() != samples * sdf_values_per_sample) {
        throw std::invalid_argument("Dataset X_sdf size does not match target count.");
    }
    if (_scalars.size() != samples * kScalarFeatures) {
        throw std::invalid_argument("Dataset X_scalars must contain [Reynolds, AoA] per sample.");
    }
    const auto finite = [](float value) { return std::isfinite(value); };
    if (!std::all_of(_sdf.begin(), _sdf.end(), finite) ||
        !std::all_of(_scalars.begin(), _scalars.end(), finite) ||
        !std::all_of(_targets.begin(), _targets.end(), finite)) {
        throw std::invalid_argument("Dataset arrays must contain only finite values.");
    }
}

void Dataset::validate_indices(std::span<const size_t> indices) const {
    if (indices.empty()) {
        throw std::invalid_argument("Dataset index selection must not be empty.");
    }
    for (size_t index : indices) {
        if (index >= num_samples()) {
            throw std::out_of_range("Dataset sample index is out of range.");
        }
    }
}

std::vector<size_t> Dataset::all_indices() const {
    std::vector<size_t> indices(num_samples());
    std::iota(indices.begin(), indices.end(), size_t{0});
    return indices;
}

NormalizationStats Dataset::fit_normalization(std::span<const size_t> indices) const {
    validate_indices(indices);
    NormalizationStats stats;
    const size_t sdf_values_per_sample = _sdf_height * _sdf_width;
    const double sdf_count = static_cast<double>(indices.size() * sdf_values_per_sample);

    for (size_t index : indices) {
        const size_t sdf_offset = index * sdf_values_per_sample;
        for (size_t j = 0; j < sdf_values_per_sample; ++j) {
            stats.sdf_mean += _sdf[sdf_offset + j];
        }
        for (size_t feature = 0; feature < kScalarFeatures; ++feature) {
            stats.scalar_mean[feature] += _scalars[index * kScalarFeatures + feature];
        }
        stats.target_mean += _targets[index];
    }

    stats.sdf_mean /= sdf_count;
    for (size_t feature = 0; feature < kScalarFeatures; ++feature) {
        stats.scalar_mean[feature] /= static_cast<double>(indices.size());
    }
    stats.target_mean /= static_cast<double>(indices.size());

    double sdf_variance = 0.0;
    std::array<double, kScalarFeatures> scalar_variance{0.0, 0.0};
    double target_variance = 0.0;
    for (size_t index : indices) {
        const size_t sdf_offset = index * sdf_values_per_sample;
        for (size_t j = 0; j < sdf_values_per_sample; ++j) {
            const double difference = _sdf[sdf_offset + j] - stats.sdf_mean;
            sdf_variance += difference * difference;
        }
        for (size_t feature = 0; feature < kScalarFeatures; ++feature) {
            const double difference =
                _scalars[index * kScalarFeatures + feature] - stats.scalar_mean[feature];
            scalar_variance[feature] += difference * difference;
        }
        const double target_difference = _targets[index] - stats.target_mean;
        target_variance += target_difference * target_difference;
    }

    stats.sdf_std = safe_std(sdf_variance / sdf_count);
    for (size_t feature = 0; feature < kScalarFeatures; ++feature) {
        stats.scalar_std[feature] = safe_std(
            scalar_variance[feature] / static_cast<double>(indices.size()));
    }
    stats.target_std = safe_std(target_variance / static_cast<double>(indices.size()));
    return stats;
}

DataBatch Dataset::make_batch(std::span<const size_t> indices,
                              const NormalizationStats& stats) const {
    validate_indices(indices);
    const size_t batch_size = indices.size();
    const size_t sdf_values_per_sample = _sdf_height * _sdf_width;

    DataBatch batch{
        std::make_shared<Tensor>(
            std::vector<size_t>{batch_size, 1, _sdf_height, _sdf_width}),
        std::make_shared<Tensor>(
            std::vector<size_t>{batch_size, kScalarFeatures}),
        std::make_shared<Tensor>(std::vector<size_t>{batch_size, 1}),
        std::make_shared<Tensor>(std::vector<size_t>{batch_size, 1})};

    float* sdf_output = batch.sdf->get_data();
    float* scalar_output = batch.scalars->get_data();
    float* target_output = batch.targets->get_data();
    float* alpha_output = batch.alpha_radians->get_data();
    constexpr double degrees_to_radians = std::numbers::pi / 180.0;

    for (size_t row = 0; row < batch_size; ++row) {
        const size_t index = indices[row];
        const size_t source_sdf_offset = index * sdf_values_per_sample;
        const size_t output_sdf_offset = row * sdf_values_per_sample;
        for (size_t j = 0; j < sdf_values_per_sample; ++j) {
            sdf_output[output_sdf_offset + j] = static_cast<float>(
                (_sdf[source_sdf_offset + j] - stats.sdf_mean) / stats.sdf_std);
        }

        for (size_t feature = 0; feature < kScalarFeatures; ++feature) {
            const float raw = _scalars[index * kScalarFeatures + feature];
            scalar_output[row * kScalarFeatures + feature] = static_cast<float>(
                (raw - stats.scalar_mean[feature]) / stats.scalar_std[feature]);
        }

        target_output[row] = static_cast<float>(
            (_targets[index] - stats.target_mean) / stats.target_std);
        alpha_output[row] = static_cast<float>(
            _scalars[index * kScalarFeatures + kAoAColumn] * degrees_to_radians);
    }

    return batch;
}

std::vector<float> Dataset::load_array(const std::string& npz_path,
                                       const std::string& key) {
    const std::string command =
        "python3 -c \"import sys, numpy as np; "
        "d=np.load(sys.argv[1]); "
        "sys.stdout.buffer.write(d[sys.argv[2]].astype(np.float32).tobytes())\" " +
        shell_quote(npz_path) + " " + shell_quote(key);

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to start Python NPZ reader.");
    }

    std::vector<float> output;
    float buffer[1024];
    while (const size_t count = std::fread(buffer, sizeof(float), 1024, pipe)) {
        output.insert(output.end(), buffer, buffer + count);
    }

    const int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Failed to load array '" + key + "' from " + npz_path);
    }
    return output;
}

std::vector<size_t> Dataset::load_shape(const std::string& npz_path,
                                        const std::string& key) {
    const std::string command =
        "python3 -c \"import sys, numpy as np; "
        "a=np.load(sys.argv[1])[sys.argv[2]]; "
        "print(','.join(str(v) for v in a.shape))\" " +
        shell_quote(npz_path) + " " + shell_quote(key);

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to start Python NPZ shape reader.");
    }

    std::string text;
    char buffer[256];
    while (std::fgets(buffer, sizeof(buffer), pipe)) {
        text += buffer;
    }
    const int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Failed to load shape for array '" + key +
                                 "' from " + npz_path);
    }

    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    if (text.empty()) {
        return {};
    }

    std::vector<size_t> shape;
    std::stringstream stream(text);
    std::string dimension;
    while (std::getline(stream, dimension, ',')) {
        size_t consumed = 0;
        const unsigned long long value = std::stoull(dimension, &consumed);
        if (consumed != dimension.size()) {
            throw std::runtime_error("Invalid NPZ array shape returned by Python.");
        }
        shape.push_back(static_cast<size_t>(value));
    }
    return shape;
}
