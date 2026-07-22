#include "csllm/tensor.hpp"

#include <string>

namespace csllm {

std::string Shape::str() const {
  std::string s = "(";
  for (int i = 0; i < ndim; ++i) {
    if (i) s += ", ";
    s += std::to_string(dims[i]);
  }
  s += ")";
  return s;
}

}  // namespace csllm
