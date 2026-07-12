/**
 * @file NPZDataLoader.cpp
 * @brief Implementation file for the NPZDataLoader class.
 * Implements the methods for loading data from .npz files.
 */

#include "NPZDataLoader.hpp"
#include <stdexcept>
#include <cstdio>
#include <iostream>
#include <cmath>

NPZDataLoader::NPZDataLoader(const std::string& npz_path) {
    // Use the load_array helper function to load each array from the NPZ file
    load_array(npz_path, "X_sdf", _x_sdf);
    load_array(npz_path, "X_scalars", _x_scalars);
    load_array(npz_path, "Y_cl", _y_cl);

    // Set the total number of samples based on the target array's size
    _num_samples = _y_cl.size();

    if (_num_samples == 0) {
        throw std::runtime_error("Failed to load dataset or empty dataset: " + npz_path);
    }

    if (_x_sdf.size() != _num_samples * 150 * 150) {
        throw std::runtime_error("X_sdf dimension mismatch in dataset: " + npz_path);
    }

    if (_x_scalars.size() != _num_samples * 2) {
        throw std::runtime_error("X_scalars dimension mismatch in dataset: " + npz_path);
    }

    // Normalize _x_sdf data.
    if (!_x_sdf.empty()) {
        // Calculate mean and standard deviation for standardization.
        double sum = 0.0;
        for (float v : _x_sdf) sum += v;
        double mean = sum / _x_sdf.size();

        // Calculate variance and standard deviation.
        double var_sum = 0.0;
        for (float v : _x_sdf) var_sum += (v - mean) * (v - mean);
        double std_dev = std::sqrt(var_sum / _x_sdf.size());

        if (std_dev < 1e-8) std_dev = 1.0; // avoid division by zero in case of zero variance

        // Standardize each element: z = (x - mean) / std_dev.
        for (float& v : _x_sdf) v = static_cast<float>((v - mean) / std_dev);
    }

    // Normalize _x_scalars (column 0 and 1 separately)
    if (!_x_scalars.empty()) {
        double sum0 = 0.0, sum1 = 0.0;

        // Calculate sums for the two features separately.
        for (size_t i = 0; i < _x_scalars.size(); i += 2) {
            sum0 += _x_scalars[i]; // first feature
            sum1 += _x_scalars[i+1]; // second feature
        }
        double mean0 = sum0 / _num_samples; // mean of first feature
        double mean1 = sum1 / _num_samples; // mean of second feature

        double var0 = 0.0, var1 = 0.0;

        // Calculate sum of squared differences from mean for each feature.
        for (size_t i = 0; i < _x_scalars.size(); i += 2) {
            var0 += (_x_scalars[i] - mean0) * (_x_scalars[i] - mean0);
            var1 += (_x_scalars[i+1] - mean1) * (_x_scalars[i+1] - mean1);
        }
        // Calculate standard deviation for each feature.
        double std0 = std::sqrt(var0 / _num_samples);
        double std1 = std::sqrt(var1 / _num_samples);

        // Avoid division by zero for standard deviation.
        if (std0 < 1e-8) std0 = 1.0;
        if (std1 < 1e-8) std1 = 1.0;
        
        // Standardize each feature column separately.
        for (size_t i = 0; i < _x_scalars.size(); i += 2) {
            _x_scalars[i] = static_cast<float>((_x_scalars[i] - mean0) / std0);
            _x_scalars[i+1] = static_cast<float>((_x_scalars[i+1] - mean1) / std1);
        }
    }

    // Normalize _y_cl
    if (!_y_cl.empty()) {

        // Calculate mean and standard deviation for standardization.
        double sum = 0.0;
        for (float v : _y_cl) sum += v;
        double mean = sum / _y_cl.size();

        double var_sum = 0.0;
        for (float v : _y_cl) var_sum += (v - mean) * (v - mean);
        double std_dev = std::sqrt(var_sum / _y_cl.size());
        
        // Avoid division by zero for standard deviation.
        if (std_dev < 1e-8) std_dev = 1.0;

        // Standardize each element: z = (x - mean) / std_dev.
        for (float& v : _y_cl) v = static_cast<float>((v - mean) / std_dev);
    }

    reset();
}

void NPZDataLoader::load_array(const std::string& npz_path, const std::string& key, std::vector<float>& out_data) {
    // Construct a command to run a Python script that loads the specified array from the NPZ file and writes it to stdout as bytes.
    std::string cmd = "python3 -c \"import sys, numpy as np; "
                      "d=np.load('" + npz_path + "'); "
                      "sys.stdout.buffer.write(d['" + key + "'].astype(np.float32).tobytes())\"";
    
    // Open a pipe to the command and read the output directly into the out_data vector.
    FILE* pipe = popen(cmd.c_str(), "r"); // r is read mode
    if (!pipe) {
        throw std::runtime_error("popen() failed to run python script.");
    }

    // Read data from the pipe in chunks into a temporary buffer.
    float buffer[1024];
    while (size_t bytes_read = fread(buffer, sizeof(float), 1024, pipe)) {
        // Append the read data to the out_data vector.
        out_data.insert(out_data.end(), buffer, buffer + bytes_read);
    }
    
    // Close the pipe and check the exit status of the Python script.
    int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Python script failed while reading " + key + " from " + npz_path);
    }
}

void NPZDataLoader::reset() {
    // Reset the cursor to the beginning of the dataset.
    _cursor = 0;
}

std::vector<std::shared_ptr<Tensor>> NPZDataLoader::get_batch(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("batch_size must be > 0.");
    }
    if (_cursor + batch_size > _num_samples) {
        throw std::runtime_error("Batch exceeds available samples. Call reset().");
    }

    // Create Tensor objects for the batch with the expected shapes.
    // X_sdf: Expected shape [batch_size, 1, 150, 150] for CNN input.
    auto sdf_tensor = std::make_shared<Tensor>(std::vector<size_t>{batch_size, 1, 150, 150});
    // X_scalars: Expected shape [batch_size, 2] for scalar inputs (e.g., AoA, Re).
    auto scalar_tensor = std::make_shared<Tensor>(std::vector<size_t>{batch_size, 2});
    // Y_cl: Expected shape [batch_size, 1] for target output.
    auto target_tensor = std::make_shared<Tensor>(std::vector<size_t>{batch_size, 1});

    // Data pointers for copying data into the tensors.
    float* sdf_ptr = sdf_tensor->get_data();
    float* scalar_ptr = scalar_tensor->get_data();
    float* target_ptr = target_tensor->get_data();

    // Copy the relevant slice of data from the loaded flattened vectors into the respective Tensors.
    std::copy(_x_sdf.begin() + _cursor * 150 * 150, 
              _x_sdf.begin() + (_cursor + batch_size) * 150 * 150, 
              sdf_ptr);
              
    std::copy(_x_scalars.begin() + _cursor * 2, 
              _x_scalars.begin() + (_cursor + batch_size) * 2, 
              scalar_ptr);
              
    std::copy(_y_cl.begin() + _cursor, 
              _y_cl.begin() + _cursor + batch_size, 
              target_ptr);

    // Advance the cursor to the next batch's starting position.
    _cursor += batch_size;
    
    return {sdf_tensor, scalar_tensor, target_tensor};
}
