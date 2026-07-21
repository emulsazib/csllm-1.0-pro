#include "csllm/serialize.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

#include "csllm/json.hpp"

namespace csllm {
namespace {

constexpr std::size_t kAlign = 64;

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) & ~(a - 1); }

std::string build_header(const ModelConfig& cfg, const std::vector<TensorEntry>& entries) {
  std::string s = "{\"config\":" + cfg.to_json() + ",\"tensors\":[";
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    if (i) s += ",";
    s += "{\"name\":" + json::quote(e.name);
    s += ",\"dtype\":" + json::quote(e.dtype);
    s += ",\"shape\":[";
    for (std::size_t d = 0; d < e.shape.size(); ++d) {
      if (d) s += ",";
      s += std::to_string(e.shape[d]);
    }
    s += "],\"offset\":" + std::to_string(e.offset) + "}";
  }
  s += "]}";
  return s;
}

}  // namespace

void save_checkpoint(const std::string& path, const ModelConfig& cfg,
                     const std::vector<TensorEntry>& entries, const void* payload,
                     std::size_t payload_bytes) {
  const std::string header = build_header(cfg, entries);
  const auto header_len = static_cast<u32>(header.size());
  const std::size_t data_offset = align_up(16 + header.size(), kAlign);

  std::FILE* f = std::fopen(path.c_str(), "wb");
  CSLLM_CHECK(f != nullptr, "cannot open '" + path + "' for writing");

  auto write_or_throw = [&](const void* p, std::size_t n) {
    if (n == 0) return;
    if (std::fwrite(p, 1, n, f) != n) {
      std::fclose(f);
      throw Error("short write to '" + path + "'");
    }
  };

  const u32 version = kFormatVersion;
  write_or_throw(kMagic, sizeof(kMagic));
  write_or_throw(&version, sizeof(version));
  write_or_throw(&header_len, sizeof(header_len));
  write_or_throw(header.data(), header.size());

  // Pad so the fp32 payload starts 64-byte aligned — that is what lets the
  // mapped tensors be read directly without a bounce buffer.
  const std::size_t pad = data_offset - (16 + header.size());
  const std::vector<char> zeros(pad, 0);
  write_or_throw(zeros.data(), pad);
  write_or_throw(payload, payload_bytes);

  std::fclose(f);
}

MappedCheckpoint::MappedCheckpoint(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  CSLLM_CHECK(fd >= 0, "cannot open checkpoint '" + path + "'");

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw Error("cannot stat checkpoint '" + path + "'");
  }
  map_bytes_ = static_cast<std::size_t>(st.st_size);
  CSLLM_CHECK(map_bytes_ >= 16, "checkpoint '" + path + "' is truncated");

  void* m = ::mmap(nullptr, map_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);  // the mapping keeps its own reference
  CSLLM_CHECK(m != MAP_FAILED, "cannot mmap checkpoint '" + path + "'");
  map_ = m;

  const auto* base = static_cast<const std::byte*>(map_);
  if (std::memcmp(base, kMagic, sizeof(kMagic)) != 0) {
    ::munmap(map_, map_bytes_);
    map_ = nullptr;
    throw Error("'" + path + "' is not a .csllm checkpoint (bad magic)");
  }

  u32 version = 0, header_len = 0;
  std::memcpy(&version, base + 8, sizeof(version));
  std::memcpy(&header_len, base + 12, sizeof(header_len));
  if (version != kFormatVersion) {
    ::munmap(map_, map_bytes_);
    map_ = nullptr;
    throw Error("checkpoint version " + std::to_string(version) + " != supported " +
                std::to_string(kFormatVersion));
  }
  if (16 + static_cast<std::size_t>(header_len) > map_bytes_) {
    ::munmap(map_, map_bytes_);
    map_ = nullptr;
    throw Error("checkpoint header runs past end of file");
  }

  const std::string text(reinterpret_cast<const char*>(base + 16), header_len);
  const json::Value root = json::parse(text);

  header_.version = version;

  const json::Value& cfg = root["config"];
  ModelConfig mc;
  mc.vocab_size = cfg.integer("vocab_size");
  mc.n_layer = cfg.integer("n_layer");
  mc.n_head = cfg.integer("n_head");
  mc.n_embd = cfg.integer("n_embd");
  mc.block_size = cfg.integer("block_size");
  mc.ffn_hidden = cfg.integer("ffn_hidden");
  if (cfg.has("rope_theta")) mc.rope_theta = static_cast<f32>(cfg.real("rope_theta"));
  if (cfg.has("norm_eps")) mc.norm_eps = static_cast<f32>(cfg.real("norm_eps"));
  mc.validate();
  header_.config = mc;

  const json::Value& tensors = root["tensors"];
  CSLLM_CHECK(tensors.type == json::Value::Type::Array, "checkpoint 'tensors' must be an array");
  for (const auto& t : tensors.array) {
    TensorEntry e;
    e.name = t.text("name");
    e.dtype = t.text("dtype");
    e.offset = static_cast<u64>(t.integer("offset"));
    for (const auto& d : t["shape"].array) e.shape.push_back(static_cast<i64>(d.number));
    header_.tensors.push_back(std::move(e));
  }

  payload_ = base + align_up(16 + static_cast<std::size_t>(header_len), kAlign);
}

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
