#pragma once

#include "model/model_data.h"

#include <filesystem>
#include <memory>

namespace ingine::model {

// Single-device TP validation. Both rank-local branches run on one stream. Replicated residual
// and routing operations share the same inputs, and each rank owns its KV and convolution state.
enum class TpReduction { Reconstruct, Fp32, Bf16, ExpertOrdered };

class TpSimulation {
 public:
  TpSimulation(const std::filesystem::path& model_dir, int max_ctx, bool dense_fp8, bool kv_fp8,
               bool head_nvfp4, cudaStream_t stream);
  ~TpSimulation();

  void reset(cudaStream_t stream);
  const State& state(int rank) const;
  void attention(const Layer& reference, Buffers& b, int visit, bool single, bool history,
                 const int64_t* pos, TpReduction reduction, cudaStream_t stream);
  void mlp(const Layer& reference, Buffers& b, int visit, bool history, TpReduction reduction,
           cudaStream_t stream);
  void head(Buffers& b, cudaStream_t stream);

 private:
  struct Rank;
  std::array<std::unique_ptr<Rank>, 2> ranks_;
  DeviceBuffer sum_, replica_, replica_history_;
};

}  // namespace ingine::model
