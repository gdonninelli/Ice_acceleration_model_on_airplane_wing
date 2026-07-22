#include "ModelFactory.hpp"
#include "layers/ConcatenateLayer.hpp"
#include <stdexcept>

std::unique_ptr<CNNModel> ModelFactory::build(const TrialConfig& config,
                                              size_t sdf_height,
                                              size_t sdf_width,
                                              size_t scalar_features,
                                              uint64_t seed,
                                              MPI_Comm communicator) const {
    if (sdf_height == 0 || sdf_width == 0) {
        throw std::invalid_argument("ModelFactory requires positive SDF dimensions.");
    }
    if (config.model.feature_layers.empty() || config.model.head_layers.empty()) {
        throw std::invalid_argument("Model blueprint requires feature and head layers.");
    }
    if (config.model.concatenate_scalars && config.model.scalar_features == 0) {
        throw std::invalid_argument("Scalar fusion requires at least one scalar feature.");
    }
    if (config.model.concatenate_scalars &&
        config.model.scalar_features != scalar_features) {
        throw std::invalid_argument(
            "Model scalar feature count does not match the dataset schema.");
    }

    auto model = std::make_unique<CNNModel>(
        config.optimizer.build(), communicator, config.training.gradient_clip);
    TensorShape shape{1, sdf_height, sdf_width};
    uint64_t layer_seed = seed;

    for (const auto& recipe : config.model.feature_layers) {
        const TensorShape output_shape = recipe.infer_shape(shape);
        model->add_conv_layer(recipe.build(shape, layer_seed++));
        shape = output_shape;
    }

    if (config.model.concatenate_scalars) {
        if (shape.size() != 1) {
            throw std::invalid_argument(
                "Feature extractor must produce a flat feature vector before scalar fusion.");
        }
        model->set_concatenate_layer(
            std::make_unique<ConcatenateLayer>(config.model.scalar_features));
        shape[0] += config.model.scalar_features;
    } else {
        model->clear_concatenate_layer();
    }

    for (const auto& recipe : config.model.head_layers) {
        const TensorShape output_shape = recipe.infer_shape(shape);
        model->add_dense_layer(recipe.build(shape, layer_seed++));
        shape = output_shape;
    }

    if (shape != TensorShape{1}) {
        throw std::invalid_argument(
            "Regression model head must produce exactly one output feature.");
    }
    return model;
}
