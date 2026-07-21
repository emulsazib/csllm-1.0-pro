#pragma once
//
// A minimal JSON reader/writer — just enough for ModelConfig and the .csllm
// tensor manifest.
//
// Exists so checkpoints stay human-inspectable (`head -c 400 model.csllm` shows
// the config) without taking on a third-party dependency. Objects keep
// insertion order so round-tripping is stable.
//
#include <string>
#include <utility>
#include <vector>

#include "csllm/common.hpp"

namespace csllm::json {

struct Value;
// std::vector supports incomplete types (C++17); std::map does not, which is
// why the object is an ordered vector of pairs rather than a map.
using Members = std::vector<std::pair<std::string, Value>>;
using Array = std::vector<Value>;

struct Value {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  Array array;
  Members members;

  bool has(const std::string& key) const;
  const Value& operator[](const std::string& key) const;   // throws if absent
  i64 integer(const std::string& key) const;
  double real(const std::string& key) const;
  std::string text(const std::string& key) const;
};

Value parse(const std::string& text);

// Writer helpers.
std::string quote(const std::string& s);
std::string number_to_string(double v);

}  // namespace csllm::json
