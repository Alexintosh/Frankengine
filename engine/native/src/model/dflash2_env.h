#pragma once

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ingine::model::dflash2_env {

inline float parse_float(const char* name, std::string_view value, bool allow_zero = false) {
  float result = 0.f;
  const char* end = value.data() + value.size();
  const auto parsed = std::from_chars(value.data(), end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(result) ||
      std::signbit(result) || (!allow_zero && result == 0.f))
    throw std::invalid_argument(std::string(name) + (allow_zero ? " requires a nonnegative finite float"
                                                                : " requires a positive finite float"));
  return result;
}

inline float read_float(const char* name, float fallback, bool allow_zero = false) {
  const char* value = std::getenv(name);
  return value ? parse_float(name, value, allow_zero) : fallback;
}

}  // namespace ingine::model::dflash2_env
