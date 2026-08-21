/**
 * @file DropoutLayer.hpp
 * @brief Header file for the DropoutLayer class.
 * Defines the DropoutLayer class, responsible for implementing inverted
 * dropout regularization in the CNN model.
 */

#ifndef DROPOUTLAYER_HPP
#define DROPOUTLAYER_HPP

#include "Layer.hpp"

/**
 * @class DropoutLayer
 * @brief Inverted dropout: during training each element is zeroed with
 * probability `rate` and the surviving elements are scaled by 1/(1-rate), so
 * the expected activation is unchanged and inference needs no rescaling.
 * During inference the layer is the identity.
 *
 * The mask is not drawn from a sequential random generator: element (row, j)
 * of the local batch is kept or dropped according to a stateless hash of
 * (stream seed, layer salt, global sample row, feature index) provided
 * through LayerExecutionContext. The decision therefore depends only on the
 * sample's position inside the *global* batch, never on which MPI rank
 * processes it, which preserves the project guarantee that training results
 * do not change with the rank count.
 */
class DropoutLayer : public Layer {
    private:
    float _rate;                            // drop probability in [0, 1)
    float _inverse_keep;                    // 1 / (1 - rate), applied at training time
    uint64_t _salt;                         // per-layer-position salt, from the build seed
    LayerExecutionContext _context;         // inference by default
    std::shared_ptr<Tensor> _mask_cache;    // scaled keep mask for the backward pass
    std::vector<size_t> _forward_shape_cache;
    bool _forward_cache_valid = false;

    public:
    /**
     * @brief Constructor for the DropoutLayer.
     * @param rate Probability of dropping each element; must be finite and in [0, 1).
     * @param salt Per-layer salt that decorrelates masks between dropout layers.
     */
    explicit DropoutLayer(float rate, uint64_t salt = 0);

    /** @brief Stores the context that the next forward pass will use. */
    void set_execution_context(const LayerExecutionContext& context) override;

    /**
     * @brief Applies dropout during training, or the identity during inference.
     * @param inputs A vector containing a single input Tensor of shape [batch, features...].
     * @return A shared pointer to the output Tensor.
     */
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;

    /**
     * @brief Propagates the gradient through the mask used in the forward pass.
     * @param grad_output The gradient of the loss w.r.t. the layer's output.
     * @return A vector containing a single gradient tensor.
     */
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override;

    float rate() const { return _rate; }
};

#endif // DROPOUTLAYER_HPP
