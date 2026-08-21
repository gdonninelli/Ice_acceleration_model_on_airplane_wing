#include "TrialConfig.hpp"
#include "layers/ActivationFactory.hpp"
#include "layers/Conv2DLayer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/DropoutLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "optimizers/AdagradOptimizer.hpp"
#include "optimizers/AdamOptimizer.hpp"
#include "optimizers/RMSpropOptimizer.hpp"
#include "optimizers/SGDOptimizer.hpp"
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
int checked_int(size_t value, const std::string& label) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(label + " exceeds the supported integer range.");
    }
    return static_cast<int>(value);
}

} // namespace

LayerRecipe::LayerRecipe(std::string description,
                         ShapeFunction infer_shape,
                         BuildFunction build)
    : _description(std::move(description)),
      _infer_shape(std::move(infer_shape)),
      _build(std::move(build)) {
    if (_description.empty() || !_infer_shape || !_build) {
        throw std::invalid_argument("LayerRecipe requires a description and build functions.");
    }
}

TensorShape LayerRecipe::infer_shape(const TensorShape& input_shape) const {
    TensorShape output = _infer_shape(input_shape);
    if (output.empty()) {
        throw std::invalid_argument("Layer recipe '" + _description +
                                    "' produced an empty output shape.");
    }
    for (size_t dimension : output) {
        if (dimension == 0) {
            throw std::invalid_argument("Layer recipe '" + _description +
                                        "' produced a zero-sized dimension.");
        }
    }
    return output;
}

std::unique_ptr<Layer> LayerRecipe::build(const TensorShape& input_shape,
                                          uint64_t seed) const {
    auto layer = _build(input_shape, seed);
    if (!layer) {
        throw std::runtime_error("Layer recipe '" + _description +
                                 "' returned a null layer.");
    }
    return layer;
}

OptimizerRecipe::OptimizerRecipe(std::string description, BuildFunction build)
    : _description(std::move(description)), _build(std::move(build)) {
    if (_description.empty() || !_build) {
        throw std::invalid_argument("OptimizerRecipe requires a description and builder.");
    }
}

std::unique_ptr<Optimizer> OptimizerRecipe::build() const {
    auto optimizer = _build();
    if (!optimizer) {
        throw std::runtime_error("Optimizer recipe returned a null optimizer.");
    }
    return optimizer;
}

namespace Recipes {
LayerRecipe conv2d(int output_channels,
                   int kernel_size,
                   int stride,
                   int padding) {
    std::ostringstream description;
    description << "conv2d(out=" << output_channels << ",kernel=" << kernel_size
                << ",stride=" << stride << ",padding=" << padding << ")";
    return LayerRecipe(
        description.str(),
        [=](const TensorShape& input) {
            if (input.size() != 3) {
                throw std::invalid_argument("Conv2D recipe expects [channels, height, width].");
            }
            if (output_channels <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0) {
                throw std::invalid_argument("Conv2D recipe contains invalid parameters.");
            }
            const size_t kernel = static_cast<size_t>(kernel_size);
            const size_t step = static_cast<size_t>(stride);
            const size_t pad = static_cast<size_t>(padding);
            if (input[1] + 2 * pad < kernel || input[2] + 2 * pad < kernel) {
                throw std::invalid_argument("Conv2D kernel exceeds the padded input.");
            }
            return TensorShape{
                static_cast<size_t>(output_channels),
                (input[1] + 2 * pad - kernel) / step + 1,
                (input[2] + 2 * pad - kernel) / step + 1};
        },
        [=](const TensorShape& input, uint64_t seed) {
            if (input.size() != 3) {
                throw std::invalid_argument("Conv2D build expects a 3D input shape.");
            }
            return std::make_unique<Conv2DLayer>(
                checked_int(input[0], "Conv2D input channels"), output_channels,
                kernel_size, stride, padding, seed);
        });
}

LayerRecipe activation(const std::string& name, float leaky_alpha) {
    const std::string normalized_name =
        make_activation(name, leaky_alpha)->get_layer_name();
    std::ostringstream description;
    description << "activation(" << normalized_name << ",alpha="
                << std::setprecision(std::numeric_limits<float>::max_digits10)
                << leaky_alpha << ")";
    return LayerRecipe(
        description.str(),
        [](const TensorShape& input) { return input; },
        [=](const TensorShape&, uint64_t) {
            return make_activation(name, leaky_alpha);
        });
}

LayerRecipe flatten() {
    return LayerRecipe(
        "flatten",
        [](const TensorShape& input) {
            if (input.size() != 3) {
                throw std::invalid_argument(
                    "Flatten recipe expects [channels, height, width].");
            }
            size_t features = 1;
            for (size_t dimension : input) {
                if (dimension > std::numeric_limits<size_t>::max() / features) {
                    throw std::overflow_error("Flatten feature count overflow.");
                }
                features *= dimension;
            }
            return TensorShape{features};
        },
        [](const TensorShape&, uint64_t) {
            return std::make_unique<FlattenLayer>();
        });
}

LayerRecipe dense(int output_features) {
    return LayerRecipe(
        "dense(out=" + std::to_string(output_features) + ")",
        [=](const TensorShape& input) {
            if (input.size() != 1 || output_features <= 0) {
                throw std::invalid_argument(
                    "Dense recipe expects one input dimension and positive output size.");
            }
            return TensorShape{static_cast<size_t>(output_features)};
        },
        [=](const TensorShape& input, uint64_t seed) {
            if (input.size() != 1) {
                throw std::invalid_argument("Dense build expects one input dimension.");
            }
            return std::make_unique<DenseLayer>(
                checked_int(input[0], "Dense input features"), output_features, seed);
        });
}

LayerRecipe dropout(float rate) {
    // Construct a probe layer so an invalid rate fails here, when the recipe
    // (or a search grid) is assembled, instead of deep inside a training run.
    const DropoutLayer validation_probe(rate, 0);
    (void)validation_probe;

    std::ostringstream description;
    description << "dropout(rate="
                << std::setprecision(std::numeric_limits<float>::max_digits10)
                << rate << ")";
    return LayerRecipe(
        description.str(),
        [](const TensorShape& input) { return input; },
        [=](const TensorShape&, uint64_t seed) {
            return std::make_unique<DropoutLayer>(rate, seed);
        });
}

OptimizerRecipe sgd(float learning_rate,
                    float momentum,
                    float weight_decay) {
    std::ostringstream description;
    description << std::setprecision(std::numeric_limits<float>::max_digits10);
    description << "sgd(lr=" << learning_rate << ",momentum=" << momentum
                << ",weight_decay=" << weight_decay << ")";
    return OptimizerRecipe(description.str(), [=] {
        return std::make_unique<SGDOptimizer>(learning_rate, momentum,
                                              weight_decay);
    });
}

OptimizerRecipe sgd_momentum(float learning_rate,
                             float momentum,
                             float weight_decay) {
    std::ostringstream description;
    description << std::setprecision(std::numeric_limits<float>::max_digits10);
    description << "sgd_momentum(lr=" << learning_rate << ",momentum=" << momentum
                << ",weight_decay=" << weight_decay << ")";
    return OptimizerRecipe(description.str(), [=] {
        return std::make_unique<SGDOptimizer>(learning_rate, momentum,
                                              weight_decay);
    });
}

OptimizerRecipe adagrad(float learning_rate,
                        float epsilon,
                        float weight_decay) {
    std::ostringstream description;
    description << std::setprecision(std::numeric_limits<float>::max_digits10);
    description << "adagrad(lr=" << learning_rate << ",eps=" << epsilon
                << ",weight_decay=" << weight_decay << ")";
    return OptimizerRecipe(description.str(), [=] {
        return std::make_unique<AdagradOptimizer>(learning_rate, epsilon,
                                                  weight_decay);
    });
}

OptimizerRecipe rmsprop(float learning_rate,
                        float decay_rate,
                        float epsilon,
                        float weight_decay) {
    std::ostringstream description;
    description << std::setprecision(std::numeric_limits<float>::max_digits10);
    description << "rmsprop(lr=" << learning_rate << ",decay=" << decay_rate
                << ",eps=" << epsilon << ",weight_decay=" << weight_decay << ")";
    return OptimizerRecipe(description.str(), [=] {
        return std::make_unique<RMSpropOptimizer>(learning_rate, decay_rate,
                                                  epsilon, weight_decay);
    });
}

OptimizerRecipe adam(float learning_rate,
                     float beta1,
                     float beta2,
                     float epsilon,
                     float weight_decay) {
    std::ostringstream description;
    description << std::setprecision(std::numeric_limits<float>::max_digits10);
    description << "adam(lr=" << learning_rate << ",beta1=" << beta1
                << ",beta2=" << beta2 << ",eps=" << epsilon
                << ",weight_decay=" << weight_decay << ")";
    return OptimizerRecipe(description.str(), [=] {
        return std::make_unique<AdamOptimizer>(learning_rate, beta1, beta2,
                                               epsilon, weight_decay);
    });
}

} // namespace Recipes
