/**
 * @file Tensor.cpp
 * @brief Implementation file for the Tensor class.
 * It relies on the declarations in Tensor.hpp.
 */

#include "Tensor.hpp"
#include <algorithm>
#include <iostream>

Tensor::Tensor(std::vector<size_t> shape)
    : _shape(shape), _total_size(calculate_size(_shape)) {
  // Initializes all elements to 0.0f.
  _data.resize(_total_size, 0.0f);
  _grad.resize(_total_size, 0.0f);
}

void Tensor::zero_grad() {
  // Resets all gradient values to zero.
  std::fill(_grad.begin(), _grad.end(), 0.0f);
}

size_t Tensor::calculate_size(const std::vector<size_t> &shape) {
  // If the shape vector is empty, the tensor has no elements.
  if (shape.empty())
    return 0;

  size_t res = 1;
  // Iterate through the dimensions in the shape vector and multiply them
  // together to get the total size of the tensor.
  for (auto dim : shape) {
    res *= dim;
  }
  return res;
}

void Tensor::print_metadata() const {
  std::cout << "Shape: ";
  // Iterate through each dimension in the shape vector and print it.
  for (auto dim : _shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;
  std::cout << "Total size: " << _total_size << std::endl;
}