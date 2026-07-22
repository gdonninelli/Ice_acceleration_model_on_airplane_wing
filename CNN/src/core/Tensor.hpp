/**
 * @file Tensor.hpp
 * @brief Header file for the Tensor class.
 * Defines the Tensor class, responsible for holding multi-dimensional arrays to
 * store data and gradients for the CNN model.
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>

/**
 * @class Tensor
 * @brief Tensor class to hold multi-dimensional array to store data and gradients for the CNN model.
 */
class Tensor {
private:
  std::vector<size_t> _shape;
  // Using single-precision floats in order to save memory since this is sufficient for Neural Networks.
  std::vector<float> _data;
  std::vector<float> _grad;
  size_t _total_size; // Product of dimensions in _shape.

public:
  /**
   * @brief Construct a new Tensor object.
   * @param shape The shape of the tensor.
   */
  Tensor(std::vector<size_t> shape);

  /**
   * @brief Get the shape of the tensor.
   * @return A constant reference to the vector representing the shape of the tensor.
   */
  std::vector<size_t> get_shape() const { return _shape; }

  /**
   * @brief Gets a pointer to the begininning of the data array.
   * @return A pointer to the `float` data array.
   */
  float *get_data() { return _data.data(); }
  const float *get_data() const { return _data.data(); }

  /**
   * @brief Gets a pointer to the beginning of the gradient array.
   * @return A pointer to the `float` gradient array.
   */
  float *get_grad() { return _grad.data(); }
  const float *get_grad() const { return _grad.data(); }

  /**
   * @brief Resets all gradient values in the tensor to zero.
   *
   */
  void zero_grad();

  /**
   * @brief Get the total number of elements in the tensor.
   * @return The total size of the tensor (product of dimensions).
   */
  size_t size() const { return _total_size; }

  /**
   * @brief Overloaded operator[] for non-constant access to tensor elements.
   * @param index The 1D index of the element to access.
   * @return A reference to the `float` element at the specified index.
   */
  float &operator[](size_t index) {
    return _data[index];
  } // non-const version for writing.

  /**
   * @brief Overloaded operator[] for constant access to tensor elements.
   * @param index The 1D index of the element to access.
   * @return A reference to the `float` element at the specified index.
   */
  const float &operator[](size_t index) const {
    return _data[index];
  } // const version for reading.

  /**
   * @brief Calculates the total number of elements in a tensor given its shape.
   * Since it is static, it can be called without an instance of the Tensor
   * class.
   * @param shape The shape of the tensor. This is passed by reference to avoid
   * unnecessary copying.
   * @return The total size of the tensor (product of dimensions).
   */
  static size_t
  calculate_size(const std::vector<size_t>
                     &shape); // I pass by reference to avoid unnecessary
                              // copying of the shape vector

  /**
   * @brief Prints the metadata of the tensor (useful for debugging).
   */
  void print_metadata() const;
};

#endif // TENSOR_HPP
