/**
 * @file NPZDataLoader.hpp
 * @brief Header file for the NPZDataLoader class.
 * Defines the NPZDataLoader class, responsible for loading data from .npz files.
 */

#ifndef NPZDATALOADER_HPP
#define NPZDATALOADER_HPP

#include "core/Tensor.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/**
 * @class NPZDataLoader 
 * @brief A class for loading data from .npz files.
 */
class NPZDataLoader {
public:
    /**
     * @brief Construct a new NPZDataLoader object
     * @param npz_path The path to the NPZ file containing the dataset.
     */
    NPZDataLoader(const std::string& npz_path);

    /**
     * @brief Returns the total number of samples in the dataset.
     * @return The number of samples loaded from the NPZ file.
     */
    size_t num_samples() const { return _num_samples; }

    /**
     * @brief Reset the data loader to the beginning of the dataset.
     */
    void reset();

    /**
     * @brief Get a batch of samples from the dataset.
     * @param batch_size The number of samples to include in the batch.
     * @return A vector of shared pointers to Tensor objects representing the batch.
     */
    std::vector<std::shared_ptr<Tensor>> get_batch(size_t batch_size);

private:
    size_t _num_samples = 0; // total number of samples from the dataset
    size_t _cursor = 0; // current index pointing to the next sample to be retrieved

    // Data storage for the dataset loaded from the NPZ file
    std::vector<float> _x_sdf;
    std::vector<float> _x_scalars;
    std::vector<float> _y_cl;

    /**
     * @brief Loads a specific array (by key) from an NPZ file
     * @param npz_path The path to the NPZ file.
     * @param key The key of the array to load within the NPZ file (e.g., "X_sdf").
     * @param out_data A reference to the vector where the loaded data will be stored.
     */
    void load_array(const std::string& npz_path, const std::string& key, std::vector<float>& out_data);
};

#endif // NPZDATALOADER_HPP
