#ifndef CNNMODEL_HPP
#define CNNMODEL_HPP

#include "layers/Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include "core/Tensor.hpp"
#include <cstddef>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * Runtime CNN graph with a feature trunk, optional scalar fusion, and a head.
 * Runtime instances are deliberately non-copyable because layers cache forward
 * state and optimizers own per-parameter state.
 */
class CNNModel {
public:
    explicit CNNModel(std::unique_ptr<Optimizer> optimizer,
                      MPI_Comm communicator = MPI_COMM_WORLD,
                      float gradient_clip = 1.0f);

    CNNModel(const CNNModel&) = delete;
    CNNModel& operator=(const CNNModel&) = delete;
    CNNModel(CNNModel&&) = default;
    CNNModel& operator=(CNNModel&&) = default;

    void add_conv_layer(std::unique_ptr<Layer> layer);
    void add_dense_layer(std::unique_ptr<Layer> layer);
    void set_concatenate_layer(std::unique_ptr<Layer> layer);
    void clear_concatenate_layer();

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> sdf_input,
                                    std::shared_ptr<Tensor> scalar_input);
    void backward(std::shared_ptr<Tensor> loss_grad);

    void zero_grad();
    void synchronize_gradients(size_t local_sample_count);
    void update();
    void broadcast_initial_weights(int root_rank = 0);

    std::shared_ptr<Tensor> predict(std::shared_ptr<Tensor> sdf_input,
                                    std::shared_ptr<Tensor> scalar_input);

    const std::vector<LayerParameter>& parameters() const;
    void export_weights(const std::string& filepath) const;
    void import_weights(const std::string& filepath);

private:
    std::vector<std::unique_ptr<Layer>> _feature_layers;
    std::vector<std::unique_ptr<Layer>> _head_layers;
    std::unique_ptr<Layer> _concatenate_layer;
    std::unique_ptr<Optimizer> _optimizer;
    MPI_Comm _communicator;
    float _gradient_clip;
    mutable std::vector<LayerParameter> _parameter_cache;
    mutable bool _parameters_dirty = true;

    std::vector<Layer*> ordered_layers();
    std::vector<const Layer*> ordered_layers() const;
};

#endif // CNNMODEL_HPP
