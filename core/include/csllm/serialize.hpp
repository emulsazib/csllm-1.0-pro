#pragma once
//
// The .csllm checkpoint format — self-describing and mmap-able, so the gateway
// starts near-instantly and worker processes share pages via the page cache.
//
//   offset 0  : magic "CSLLM\0\0\0"           (8 bytes)
//   offset 8  : version   uint32              (little-endian)
//   offset 12 : header_len uint32
//   offset 16 : JSON header {
//                 "config":  { ...ModelConfig... },
//                 "tensors": [{"name","dtype","shape","offset"}, ...]
//               }
//   then      : 64-byte-aligned fp32 payload
//
#include <string>
#include <vector>

#include "csllm/common.hpp"
#include "csllm/model.hpp"
#include "csllm/tensor.hpp"

namespace csllm {

inline constexpr char kMagic[8] = {'C', 'S', 'L', 'L', 'M', '\0', '\0', '\0'};
inline constexpr u32 kFormatVersion = 1;

struct TensorEntry {
  std::string name;
  std::string dtype;      // "f32"
  std::vector<i64> shape;
  u64 offset;             // byte offset into the payload
};

struct CheckpointHeader {
  ModelConfig config;
  std::vector<TensorEntry> tensors;
  u32 version = kFormatVersion;
};

void save_checkpoint(const std::string& path, const ModelConfig& cfg,
                     const std::vector<TensorEntry>& entries, const void* payload,
                     std::size_t payload_bytes);

// Memory-maps the file; the returned pointer stays valid for the mapping's life.
class MappedCheckpoint {
 public:
  explicit MappedCheckpoint(const std::string& path);
  ~MappedCheckpoint();

  MappedCheckpoint(const MappedCheckpoint&) = delete;
  MappedCheckpoint& operator=(const MappedCheckpoint&) = delete;

  const CheckpointHeader& header() const noexcept { return header_; }
  const void* tensor_data(const std::string& name) const;

 private:
  CheckpointHeader header_;
  void* map_ = nullptr;
  std::size_t map_bytes_ = 0;
  const std::byte* payload_ = nullptr;
};

}  // namespace csllm
