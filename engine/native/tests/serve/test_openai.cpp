#include "serve/openai.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace {

using ingine::serve::Json;
using ingine::serve::parse_request;
using ingine::text::ChatFormat;

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void test_thinking(const ChatFormat& fmt) {
  const Json messages = Json::array({{{"role", "user"}, {"content", "Hi"}}});
  const auto parse = [&](Json options) {
    options["messages"] = messages;
    return parse_request(options, fmt, 4096);
  };

  check(parse(Json::object()).sampling.thinking_enabled, "thinking should be enabled by default");
  check(parse({{"reasoning_effort", "low"}}).sampling.thinking_enabled, "positive effort disabled thinking");

  const auto disabled = parse({{"enable_thinking", false}, {"reasoning_effort", "high"}});
  check(!disabled.sampling.thinking_enabled && disabled.prefill_ids == fmt.no_thinking_prefill(),
        "enable_thinking=false should append the content prefix and disable thinking");
  const auto nested = parse({{"enable_thinking", true}, {"chat_template_kwargs", {{"enable_thinking", false}}}});
  check(!nested.sampling.thinking_enabled && nested.prefill_ids == fmt.no_thinking_prefill(),
        "chat_template_kwargs.enable_thinking should override the top-level flag");

  for (const Json& effort : {Json(" NONE "), Json(0.0), Json("0.0")}) {
    const auto zero = parse({{"enable_thinking", true}, {"reasoning_effort", effort}});
    check(!zero.sampling.thinking_enabled && zero.prefill_ids.empty(),
          "zero effort should disable thinking without adding the content prefix");
  }
  check(parse({{"reasoning_effort", "low"}, {"chat_template_kwargs", {{"reasoning_effort", "none"}}}})
            .sampling.thinking_enabled,
        "top-level reasoning_effort should override chat_template_kwargs");
  check(!parse({{"chat_template_kwargs", {{"reasoning_effort", "none"}}}}).sampling.thinking_enabled,
        "nested zero effort should disable thinking");
  bool rejected = false;
  try {
    parse({{"reasoning_effort", "off"}});
  } catch (const ingine::text::ChatError& e) {
    rejected = e.param == "reasoning_effort";
  }
  check(rejected, "unsupported reasoning_effort values should retain the existing validation");
}

}  // namespace

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : std::getenv("INGINE_MODEL_DIR");
  if (!dir || !std::filesystem::is_regular_file(std::filesystem::path(dir) / "tokenizer.json")) {
    std::puts("skip: set INGINE_MODEL_DIR or pass a model directory");
    return 77;
  }
  try {
    test_thinking(ChatFormat::load(dir));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
