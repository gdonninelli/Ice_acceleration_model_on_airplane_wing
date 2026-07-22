#include "CNNModel.hpp"
#include "layers/ConcatenateLayer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {
constexpr char kWeightMagic[] = "CNNWGT2";

bool mpi_ready() {
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Finalized(&finalized);
    }
    return initialized != 0 && finalized == 0;
}

int checked_mpi_count(size_t count) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MPI buffer exceeds the supported element count.");
    }
    return static_cast<int>(count);
}

void write_exact(std::ofstream& output, const void* data, size_t bytes) {
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error("Failed to write model weights.");
    }
}

void read_exact(std::ifstream& input, void* data, size_t bytes) {
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("Failed to read model weights.");
    }
}

void write_string(std::ofstream& output, const std::string& value) {
    const size_t size = value.size();
    write_exact(output, &size, sizeof(size));
    if (size > 0) {
        write_exact(output, value.data(), size);
    }
}

std::string read_string(std::ifstream& input) {
    size_t size = 0;
    read_exact(input, &size, sizeof(size));
    std::string value(size, '\0');
    if (size > 0) {
        read_exact(input, value.data(), size);
    }
    return value;
}

void write_tensor(std::ofstream& output, const Tensor& tensor) {
    const auto shape = tensor.get_shape();
    const size_t rank = shape.size();
    write_exact(output, &rank, sizeof(rank));
    if (rank > 0) {
        write_exact(output, shape.data(), rank * sizeof(size_t));
    }
    const size_t count = tensor.size();
    write_exact(output, &count, sizeof(count));
    if (count > 0) {
        write_exact(output, tensor.get_data(), count * sizeof(float));
    }
}

void read_tensor(std::ifstream& input, Tensor& tensor) {
    size_t rank = 0;
    read_exact(input, &rank, sizeof(rank));
    std::vector<size_t> shape(rank);
    if (rank > 0) {
        read_exact(input, shape.data(), rank * sizeof(size_t));
    }

    size_t count = 0;
    read_exact(input, &count, sizeof(count));
    if (shape != tensor.get_shape() || count != tensor.size()) {
        throw std::runtime_error("Parameter shape mismatch while importing weights.");
    }
    if (count > 0) {
        read_exact(input, tensor.get_data(), count * sizeof(float));
    }
    tensor.zero_grad();
}
} // namespace

CNNModel::CNNModel(std::unique_ptr<Optimizer> optimizer,
                   MPI_Comm communicator,
                   float gradient_clip)
    : _optimizer(std::move(optimizer)),
      _communicator(communicator),
      _gradient_clip(gradient_clip) {
    if (!_optimizer) {
        throw std::invalid_argument("CNNModel requires a non-null optimizer.");
    }
    if (!std::isfinite(_gradient_clip) || _gradient_clip < 0.0f) {
        throw std::invalid_argument("Gradient clip must be finite and non-negative.");
    }
}

void CNNModel::add_conv_layer(std::unique_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("add_conv_layer received a null layer.");
    }
    _feature_layers.push_back(std::move(layer));
    _parameters_dirty = true;
}

void CNNModel::add_dense_layer(std::unique_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("add_dense_layer received a null layer.");
    }
    _head_layers.push_back(std::move(layer));
    _parameters_dirty = true;
}

void CNNModel::set_concatenate_layer(std::unique_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("set_concatenate_layer received a null layer.");
    }
    _concatenate_layer = std::move(layer);
    _parameters_dirty = true;
}

void CNNModel::clear_concatenate_layer() {
    _concatenate_layer.reset();
    _parameters_dirty = true;
}

std::shared_ptr<Tensor> CNNModel::forward(std::shared_ptr<Tensor> sdf_input,
                                         std::shared_ptr<Tensor> scalar_input) {
    if (!sdf_input) {
        throw std::invalid_argument("CNNModel forward received a null SDF tensor.");
    }

    auto output = std::move(sdf_input);
    for (const auto& layer : _feature_layers) {
        output = layer->forward({output});
    }

    if (_concatenate_layer) {
        if (!scalar_input) {
            throw std::invalid_argument("CNNModel forward received a null scalar tensor.");
        }
        output = _concatenate_layer->forward({output, scalar_input});
    }

    for (const auto& layer : _head_layers) {
        output = layer->forward({output});
    }
    return output;
}

void CNNModel::backward(std::shared_ptr<Tensor> loss_grad) {
    if (!loss_grad) {
        throw std::invalid_argument("CNNModel backward received a null gradient.");
    }

    auto gradient = std::move(loss_grad);
    for (auto iterator = _head_layers.rbegin(); iterator != _head_layers.rend(); ++iterator) {
        auto gradients = (*iterator)->backward(gradient);
        if (gradients.size() != 1 || !gradients[0]) {
            throw std::runtime_error("Head layer must return one non-null input gradient.");
        }
        gradient = std::move(gradients[0]);
    }

    if (_concatenate_layer) {
        auto gradients = _concatenate_layer->backward(gradient);
        if (gradients.size() != 2 || !gradients[0]) {
            throw std::runtime_error("Concatenate layer must return two input gradients.");
        }
        gradient = std::move(gradients[0]);
    }

    for (auto iterator = _feature_layers.rbegin(); iterator != _feature_layers.rend(); ++iterator) {
        auto gradients = (*iterator)->backward(gradient);
        if (gradients.size() != 1 || !gradients[0]) {
            throw std::runtime_error("Feature layer must return one non-null input gradient.");
        }
        gradient = std::move(gradients[0]);
    }
}

std::vector<Layer*> CNNModel::ordered_layers() {
    std::vector<Layer*> layers;
    layers.reserve(_feature_layers.size() + _head_layers.size() +
                   (_concatenate_layer ? 1 : 0));
    for (const auto& layer : _feature_layers) {
        layers.push_back(layer.get());
    }
    if (_concatenate_layer) {
        layers.push_back(_concatenate_layer.get());
    }
    for (const auto& layer : _head_layers) {
        layers.push_back(layer.get());
    }
    return layers;
}

std::vector<const Layer*> CNNModel::ordered_layers() const {
    std::vector<const Layer*> layers;
    layers.reserve(_feature_layers.size() + _head_layers.size() +
                   (_concatenate_layer ? 1 : 0));
    for (const auto& layer : _feature_layers) {
        layers.push_back(layer.get());
    }
    if (_concatenate_layer) {
        layers.push_back(_concatenate_layer.get());
    }
    for (const auto& layer : _head_layers) {
        layers.push_back(layer.get());
    }
    return layers;
}

const std::vector<LayerParameter>& CNNModel::parameters() const {
    if (!_parameters_dirty) {
        return _parameter_cache;
    }

    std::vector<LayerParameter> output;
    std::unordered_set<const Tensor*> identities;
    for (const auto& layer : ordered_layers()) {
        auto layer_parameters = layer->parameters();
        for (auto& parameter : layer_parameters) {
            if (!parameter.tensor) {
                throw std::runtime_error("Layer exposed a null parameter tensor.");
            }
            if (!identities.insert(parameter.tensor.get()).second) {
                throw std::runtime_error(
                    "The same parameter tensor was exposed more than once.");
            }
            output.push_back(std::move(parameter));
        }
    }
    _parameter_cache = std::move(output);
    _parameters_dirty = false;
    return _parameter_cache;
}

void CNNModel::zero_grad() {
    for (const auto& parameter : parameters()) {
        if (parameter.tensor) {
            parameter.tensor->zero_grad();
        }
    }
}

void CNNModel::synchronize_gradients(size_t local_sample_count) {
    if (!mpi_ready()) {
        return;
    }

    int world_size = 1;
    MPI_Comm_size(_communicator, &world_size);
    if (world_size <= 1) {
        return;
    }

    const unsigned long long local_count =
        static_cast<unsigned long long>(local_sample_count);
    unsigned long long global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, _communicator);
    if (global_count == 0) {
        throw std::runtime_error("Cannot synchronize gradients for an empty global batch.");
    }

    for (const auto& parameter : parameters()) {
        if (!parameter.tensor || parameter.tensor->size() == 0) {
            continue;
        }
        float* gradient = parameter.tensor->get_grad();
        for (size_t i = 0; i < parameter.tensor->size(); ++i) {
            gradient[i] *= static_cast<float>(local_sample_count);
        }
        MPI_Allreduce(MPI_IN_PLACE, gradient,
                      checked_mpi_count(parameter.tensor->size()), MPI_FLOAT,
                      MPI_SUM, _communicator);
        const float inverse_count = 1.0f / static_cast<float>(global_count);
        for (size_t i = 0; i < parameter.tensor->size(); ++i) {
            gradient[i] *= inverse_count;
        }
    }
}

void CNNModel::update() {
    for (const auto& parameter : parameters()) {
        if (!parameter.tensor) {
            continue;
        }
        float* gradient = parameter.tensor->get_grad();
        if (_gradient_clip > 0.0f) {
            for (size_t i = 0; i < parameter.tensor->size(); ++i) {
                gradient[i] = std::clamp(
                    gradient[i], -_gradient_clip, _gradient_clip);
            }
        }
        _optimizer->apply_gradients(parameter.tensor->get_data(), gradient,
                                    parameter.tensor->size());
    }
}

void CNNModel::broadcast_initial_weights(int root_rank) {
    if (!mpi_ready()) {
        return;
    }
    int world_size = 1;
    MPI_Comm_size(_communicator, &world_size);
    if (root_rank < 0 || root_rank >= world_size) {
        throw std::invalid_argument("Broadcast root rank is out of range.");
    }
    if (world_size <= 1) {
        return;
    }

    for (const auto& parameter : parameters()) {
        if (parameter.tensor && parameter.tensor->size() > 0) {
            MPI_Bcast(parameter.tensor->get_data(),
                      checked_mpi_count(parameter.tensor->size()), MPI_FLOAT,
                      root_rank, _communicator);
        }
    }
}

std::shared_ptr<Tensor> CNNModel::predict(std::shared_ptr<Tensor> sdf_input,
                                         std::shared_ptr<Tensor> scalar_input) {
    return forward(std::move(sdf_input), std::move(scalar_input));
}

void CNNModel::export_weights(const std::string& filepath) const {
    parameters();
    std::ofstream output(filepath, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open weights file for export: " + filepath);
    }

    write_exact(output, kWeightMagic, sizeof(kWeightMagic));
    const auto layers = ordered_layers();
    const size_t layer_count = layers.size();
    write_exact(output, &layer_count, sizeof(layer_count));
    for (const auto& layer : layers) {
        write_string(output, layer->get_layer_name());
        const auto layer_parameters = layer->parameters();
        const size_t parameter_count = layer_parameters.size();
        write_exact(output, &parameter_count, sizeof(parameter_count));
        for (const auto& parameter : layer_parameters) {
            if (!parameter.tensor) {
                throw std::runtime_error("Layer exposed a null parameter tensor.");
            }
            write_string(output, parameter.name);
            write_tensor(output, *parameter.tensor);
        }
    }
}

void CNNModel::import_weights(const std::string& filepath) {
    parameters();
    std::ifstream input(filepath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open weights file for import: " + filepath);
    }

    char magic[sizeof(kWeightMagic)]{};
    read_exact(input, magic, sizeof(magic));
    if (std::memcmp(magic, kWeightMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("Unsupported model weights format.");
    }

    size_t layer_count = 0;
    read_exact(input, &layer_count, sizeof(layer_count));
    const auto layers = ordered_layers();
    if (layer_count != layers.size()) {
        throw std::runtime_error("Layer count mismatch while importing weights.");
    }

    for (const auto& layer : layers) {
        const std::string layer_name = read_string(input);
        if (layer_name != layer->get_layer_name()) {
            throw std::runtime_error("Layer mismatch while importing weights: expected " +
                                     layer->get_layer_name() + ", got " + layer_name);
        }

        size_t parameter_count = 0;
        read_exact(input, &parameter_count, sizeof(parameter_count));
        const auto layer_parameters = layer->parameters();
        if (parameter_count != layer_parameters.size()) {
            throw std::runtime_error("Parameter count mismatch while importing weights.");
        }

        for (const auto& parameter : layer_parameters) {
            const std::string parameter_name = read_string(input);
            if (parameter_name != parameter.name || !parameter.tensor) {
                throw std::runtime_error("Parameter mismatch while importing weights.");
            }
            read_tensor(input, *parameter.tensor);
        }
    }
}
