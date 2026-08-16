/**
 * @file DropoutLayer.cpp
 * @brief Implementation file for the DropoutLayer class.
 * Implements inverted dropout with deterministic, MPI-rank-invariant masks.
 */

#include "DropoutLayer.hpp"
#include <cmath>
#include <stdexcept>

DropoutLayer::DropoutLayer(float rate, uint64_t salt)
    : _rate(rate), _inverse_keep(1.0f), _salt(salt) {
    if (!std::isfinite(rate) || rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument(
            "Dropout rate must be finite and inside [0, 1).");
    }
    _inverse_keep = 1.0f / (1.0f - rate);
    _layer_name = "dropout";
}

void DropoutLayer::set_execution_context(const LayerExecutionContext& context) {
    _context = context;
}

std::shared_ptr<Tensor> DropoutLayer::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1 || !inputs[0]) {
        throw std::invalid_argument(
            "DropoutLayer expects exactly one non-null input tensor.");
    }

    const std::shared_ptr<Tensor>& input = inputs[0];
    const std::vector<size_t> shape = input->get_shape();
    if (shape.empty() || input->size() == 0) {
        throw std::invalid_argument(
            "DropoutLayer input tensor must not be empty.");
    }

    // Inference, or a zero rate: the layer is exactly the identity. The mask
    // cache is cleared so the backward pass is a passthrough as well.
    if (!_context.training || _rate == 0.0f) {
        _mask_cache.reset();
        auto output = std::make_shared<Tensor>(shape);
        const float* in_ptr = input->get_data();
        float* out_ptr = output->get_data();
        for (size_t i = 0; i < input->size(); ++i) {
            out_ptr[i] = in_ptr[i];
        }
        return output;
    }

    const size_t rows = shape[0];
    const size_t features = input->size() / rows;

    // The per-layer stream mixes the trainer-provided batch seed with this
    // layer's salt, so distinct dropout layers use decorrelated masks.
    const uint64_t layer_stream =
        layer_rng::combine(_context.stream_seed, _salt);

    auto output = std::make_shared<Tensor>(shape);
    auto mask = std::make_shared<Tensor>(shape);
    const float* in_ptr = input->get_data();
    float* out_ptr = output->get_data();
    float* mask_ptr = mask->get_data();

    for (size_t row = 0; row < rows; ++row) {
        // Identify the sample by its position in the global batch, not in the
        // rank-local slice, so the mask does not depend on the rank layout.
        const uint64_t global_row =
            static_cast<uint64_t>(_context.sample_offset + row);
        const uint64_t row_stream = layer_rng::combine(layer_stream, global_row);
        for (size_t j = 0; j < features; ++j) {
            const uint64_t element_hash =
                layer_rng::combine(row_stream, static_cast<uint64_t>(j));
            const bool keep = layer_rng::to_unit_float(element_hash) >= _rate;
            const float multiplier = keep ? _inverse_keep : 0.0f;
            const size_t index = row * features + j;
            mask_ptr[index] = multiplier;
            out_ptr[index] = in_ptr[index] * multiplier;
        }
    }

    _mask_cache = std::move(mask);
    return output;
}

std::vector<std::shared_ptr<Tensor>> DropoutLayer::backward(
    std::shared_ptr<Tensor> grad_output) {
    if (!grad_output) {
        throw std::invalid_argument(
            "DropoutLayer backward received a null gradient tensor.");
    }

    // Identity forward pass -> identity backward pass.
    if (!_mask_cache) {
        auto grad_input = std::make_shared<Tensor>(grad_output->get_shape());
        const float* grad_out_ptr = grad_output->get_data();
        float* grad_in_ptr = grad_input->get_data();
        for (size_t i = 0; i < grad_output->size(); ++i) {
            grad_in_ptr[i] = grad_out_ptr[i];
        }
        return {grad_input};
    }

    if (grad_output->size() != _mask_cache->size()) {
        throw std::invalid_argument(
            "DropoutLayer backward gradient does not match the forward mask.");
    }

    auto grad_input = std::make_shared<Tensor>(grad_output->get_shape());
    const float* grad_out_ptr = grad_output->get_data();
    const float* mask_ptr = _mask_cache->get_data();
    float* grad_in_ptr = grad_input->get_data();
    for (size_t i = 0; i < grad_output->size(); ++i) {
        grad_in_ptr[i] = grad_out_ptr[i] * mask_ptr[i];
    }
    return {grad_input};
}
