#ifndef DATASET_HPP
#define DATASET_HPP

#include "core/Tensor.hpp"
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct NormalizationStats {
    double sdf_mean = 0.0;
    double sdf_std = 1.0;
    std::array<double, 2> scalar_mean{0.0, 0.0};
    std::array<double, 2> scalar_std{1.0, 1.0};
    double target_mean = 0.0;
    double target_std = 1.0;
};

struct DataBatch {
    std::shared_ptr<Tensor> sdf;
    std::shared_ptr<Tensor> scalars;
    std::shared_ptr<Tensor> targets;
    std::shared_ptr<Tensor> alpha_radians;

    size_t size() const;
};

/**
 * Immutable, random-access dataset used to construct fold-specific batches.
 * Scalar columns follow the dataset schema [Reynolds, AoA in degrees].
 */
class Dataset {
public:
    static constexpr size_t kSdfHeight = 150;
    static constexpr size_t kSdfWidth = 150;
    static constexpr size_t kScalarFeatures = 2;
    static constexpr size_t kReynoldsColumn = 0;
    static constexpr size_t kAoAColumn = 1;

    explicit Dataset(const std::string& npz_path);

    Dataset(std::vector<float> sdf,
            std::vector<float> scalars,
            std::vector<float> targets,
            size_t sdf_height = kSdfHeight,
            size_t sdf_width = kSdfWidth);

    size_t num_samples() const { return _targets.size(); }
    size_t sdf_height() const { return _sdf_height; }
    size_t sdf_width() const { return _sdf_width; }
    size_t scalar_features() const { return kScalarFeatures; }

    std::vector<size_t> all_indices() const;
    NormalizationStats fit_normalization(std::span<const size_t> indices) const;
    DataBatch make_batch(std::span<const size_t> indices,
                         const NormalizationStats& stats) const;

private:
    size_t _sdf_height;
    size_t _sdf_width;
    std::vector<float> _sdf;
    std::vector<float> _scalars;
    std::vector<float> _targets;
    std::vector<size_t> _sdf_shape;
    std::vector<size_t> _scalar_shape;
    std::vector<size_t> _target_shape;

    void validate() const;
    void validate_indices(std::span<const size_t> indices) const;
    static std::vector<float> load_array(const std::string& npz_path,
                                         const std::string& key);
    static std::vector<size_t> load_shape(const std::string& npz_path,
                                          const std::string& key);
};

#endif // DATASET_HPP
