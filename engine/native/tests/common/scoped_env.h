#pragma once

#include "common/test_util.h"

#include <optional>
#include <string>

namespace ingine::test {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value = nullptr) : name_(name) {
    if (const char* old = std::getenv(name)) saved_.emplace(old);
    INGINE_CHECK((value ? setenv(name_, value, 1) : unsetenv(name_)) == 0);
  }
  ~ScopedEnv() {
    INGINE_CHECK((saved_ ? setenv(name_, saved_->c_str(), 1) : unsetenv(name_)) == 0);
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  const char* name_;
  std::optional<std::string> saved_;
};

}  // namespace ingine::test
