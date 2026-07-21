#include "csllm/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace csllm::json {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}

  Value parse_document() {
    skip_ws();
    Value v = parse_value();
    skip_ws();
    CSLLM_CHECK(i_ >= s_.size(), "JSON: trailing characters at offset " + std::to_string(i_));
    return v;
  }

 private:
  void skip_ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) {
      ++i_;
    }
  }

  char peek() const {
    CSLLM_CHECK(i_ < s_.size(), "JSON: unexpected end of input");
    return s_[i_];
  }

  void expect(char c) {
    CSLLM_CHECK(i_ < s_.size() && s_[i_] == c,
                std::string("JSON: expected '") + c + "' at offset " + std::to_string(i_));
    ++i_;
  }

  bool literal(const char* lit) {
    const std::size_t n = std::char_traits<char>::length(lit);
    if (s_.compare(i_, n, lit) == 0) {
      i_ += n;
      return true;
    }
    return false;
  }

  Value parse_value() {
    skip_ws();
    switch (peek()) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': {
        Value v;
        v.type = Value::Type::String;
        v.string = parse_string();
        return v;
      }
      case 't':
      case 'f': {
        Value v;
        v.type = Value::Type::Bool;
        if (literal("true")) {
          v.boolean = true;
        } else {
          CSLLM_CHECK(literal("false"), "JSON: bad literal at offset " + std::to_string(i_));
          v.boolean = false;
        }
        return v;
      }
      case 'n': {
        CSLLM_CHECK(literal("null"), "JSON: bad literal at offset " + std::to_string(i_));
        return Value{};
      }
      default: return parse_number();
    }
  }

  Value parse_object() {
    expect('{');
    Value v;
    v.type = Value::Type::Object;
    skip_ws();
    if (peek() == '}') {
      ++i_;
      return v;
    }
    for (;;) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      v.members.emplace_back(std::move(key), parse_value());
      skip_ws();
      if (peek() == ',') {
        ++i_;
        continue;
      }
      expect('}');
      return v;
    }
  }

  Value parse_array() {
    expect('[');
    Value v;
    v.type = Value::Type::Array;
    skip_ws();
    if (peek() == ']') {
      ++i_;
      return v;
    }
    for (;;) {
      v.array.push_back(parse_value());
      skip_ws();
      if (peek() == ',') {
        ++i_;
        continue;
      }
      expect(']');
      return v;
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      CSLLM_CHECK(i_ < s_.size(), "JSON: unterminated escape");
      const char e = s_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          // Only the BMP subset we ever emit; enough for tensor names.
          CSLLM_CHECK(i_ + 4 <= s_.size(), "JSON: truncated \\u escape");
          const auto cp = static_cast<unsigned>(std::stoul(s_.substr(i_, 4), nullptr, 16));
          i_ += 4;
          if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
          } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default: throw Error("JSON: unknown escape \\" + std::string(1, e));
      }
    }
    expect('"');
    return out;
  }

  Value parse_number() {
    const std::size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    while (i_ < s_.size() && ((s_[i_] >= '0' && s_[i_] <= '9') || s_[i_] == '.' || s_[i_] == 'e' ||
                              s_[i_] == 'E' || s_[i_] == '-' || s_[i_] == '+')) {
      ++i_;
    }
    CSLLM_CHECK(i_ > start, "JSON: expected a number at offset " + std::to_string(start));
    Value v;
    v.type = Value::Type::Number;
    v.number = std::strtod(s_.substr(start, i_ - start).c_str(), nullptr);
    return v;
  }

  const std::string& s_;
  std::size_t i_ = 0;
};

}  // namespace

bool Value::has(const std::string& key) const {
  for (const auto& kv : members) {
    if (kv.first == key) return true;
  }
  return false;
}

const Value& Value::operator[](const std::string& key) const {
  for (const auto& kv : members) {
    if (kv.first == key) return kv.second;
  }
  throw Error("JSON: missing key '" + key + "'");
}

i64 Value::integer(const std::string& key) const {
  const Value& v = (*this)[key];
  CSLLM_CHECK(v.type == Value::Type::Number, "JSON: key '" + key + "' is not a number");
  return static_cast<i64>(std::llround(v.number));
}

double Value::real(const std::string& key) const {
  const Value& v = (*this)[key];
  CSLLM_CHECK(v.type == Value::Type::Number, "JSON: key '" + key + "' is not a number");
  return v.number;
}

std::string Value::text(const std::string& key) const {
  const Value& v = (*this)[key];
  CSLLM_CHECK(v.type == Value::Type::String, "JSON: key '" + key + "' is not a string");
  return v.string;
}

Value parse(const std::string& text) { return Parser(text).parse_document(); }

std::string quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  out += "\"";
  return out;
}

std::string number_to_string(double v) {
  // %.17g round-trips an IEEE double exactly.
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

}  // namespace csllm::json
