#include "csllm/serialize.hpp"

#include <sys/mman.h>

namespace csllm {

// Phase 2 — the writer and the header parser. The accessor and teardown below
// are already correct and are what the rest of the engine will call.

void save_checkpoint(const std::string&, const ModelConfig&, const std::vector<TensorEntry>&,
                     const void*, std::size_t) {
  CSLLM_NOT_IMPLEMENTED();
}

MappedCheckpoint::MappedCheckpoint(const std::string&) { CSLLM_NOT_IMPLEMENTED(); }

MappedCheckpoint::~MappedCheckpoint() {
  if (map_ != nullptr) ::munmap(map_, map_bytes_);
}

const void* MappedCheckpoint::tensor_data(const std::string& name) const {
  CSLLM_CHECK(payload_ != nullptr, "checkpoint is not mapped");
  for (const auto& entry : header_.tensors) {
    if (entry.name == name) return payload_ + entry.offset;
  }
  throw Error("tensor not found in checkpoint: " + name);
}

}  // namespace csllm
