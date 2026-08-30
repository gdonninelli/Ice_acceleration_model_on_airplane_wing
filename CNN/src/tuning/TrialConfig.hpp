#ifndef TRIALCONFIG_HPP
#define TRIALCONFIG_HPP

#include "layers/Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include "training/DiagnosticsConfig.hpp"
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using TensorShape = std::vector<size_t>;

class LayerRecipe {
public:
    using ShapeFunction = std::function<TensorShape(const TensorShape&)>;
    using BuildFunction =
        std::function<std::unique_ptr<Layer>(const TensorShape&, uint64_t)>;

    LayerRecipe(std::string description,
                ShapeFunction infer_shape,
                BuildFunction build);

    const std::string& description() const { return _description; }
    TensorShape infer_shape(const TensorShape& input_shape) const;
    std::unique_ptr<Layer> build(const TensorShape& input_shape,
                                 uint64_t seed) const;

private:
    std::string _description;
    ShapeFunction _infer_shape;
    BuildFunction _build;
};

class OptimizerRecipe {
public:
    using BuildFunction = std::function<std::unique_ptr<Optimizer>()>;

    OptimizerRecipe(std::string description, BuildFunction build);

    const std::string& description() const { return _description; }
    std::unique_ptr<Optimizer> build() const;

private:
    std::string _description;
    BuildFunction _build;
};

class LRScheduler;

class LRSchedulerRecipe {
public:
    using BuildFunction = std::function<std::unique_ptr<LRScheduler>()>;

    LRSchedulerRecipe() = default;
    LRSchedulerRecipe(std::string description, BuildFunction build);

    const std::string& description() const { return _description; }
    std::unique_ptr<LRScheduler> build() const;
    bool has_scheduler() const { return static_cast<bool>(_build); }

private:
    std::string _description = "none";
    BuildFunction _build;
};

struct ModelBlueprint {
    std::vector<LayerRecipe> feature_layers;
    std::vector<LayerRecipe> head_layers;
    bool concatenate_scalars = true;
    size_t scalar_features = 2;
};

struct LossConfig {
    float physics_weight = 0.25f;
    float l1_weight = 0.0f;
    float l2_weight = 0.0f;
};

struct TrainingConfig {
    size_t epochs = 100;
    size_t global_batch_size = 64;
    float gradient_clip = 1.0f;
    uint64_t seed = 42;
    bool shuffle = true;
    size_t validation_interval = 10;
    TrainingDiagnosticsConfig diagnostics;
    bool early_stopping = false;
    double max_overfit_ratio = 0.15;
    bool restore_best_weights = true;
};

struct TrialConfig {
    std::string name = "trial";
    ModelBlueprint model;
    OptimizerRecipe optimizer;
    LossConfig loss;
    TrainingConfig training;
    std::map<std::string, std::string> selected_parameters;
    LRSchedulerRecipe scheduler;
};

namespace Recipes {
LayerRecipe conv2d(int output_channels,
                   int kernel_size,
                   int stride = 1,
                   int padding = 0);
LayerRecipe activation(const std::string& name, float leaky_alpha = 0.05f);
LayerRecipe flatten();
LayerRecipe dense(int output_features);
LayerRecipe dropout(float rate);

OptimizerRecipe sgd(float learning_rate = 1e-5f,
                    float momentum = 0.0f,
                    float weight_decay = 0.0f);
OptimizerRecipe sgd_momentum(float learning_rate = 1e-5f,
                             float momentum = 0.9f,
                             float weight_decay = 0.0f);
OptimizerRecipe adagrad(float learning_rate = 1e-5f,
                        float epsilon = 1e-8f,
                        float weight_decay = 0.0f);
OptimizerRecipe rmsprop(float learning_rate = 1e-5f,
                        float decay_rate = 0.9f,
                        float epsilon = 1e-8f,
                        float weight_decay = 0.0f);
OptimizerRecipe adam(float learning_rate = 1e-5f,
                     float beta1 = 0.9f,
                     float beta2 = 0.999f,
                     float epsilon = 1e-8f,
                     float weight_decay = 0.0f);

LRSchedulerRecipe constant_lr(float learning_rate);
LRSchedulerRecipe step_lr(float initial_lr, size_t step_size, float gamma = 0.1f);
LRSchedulerRecipe cosine_lr(float min_lr, float max_lr);
LRSchedulerRecipe warmup_cosine_lr(float start_lr,
                                   float peak_lr,
                                   float min_lr,
                                   size_t warmup_epochs);
} // namespace Recipes

#endif // TRIALCONFIG_HPP
