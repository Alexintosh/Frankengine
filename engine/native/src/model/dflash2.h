#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/head/nvfp4_head.h"

#include <filesystem>
#include <memory>

namespace ingine::model {

// In-order committed context. Proposal KV is held separately from the ring.
bool dflash2_full_head();

class DFlash2Head;
class TpRuntime;

class DFlash2 {
 public:
  static constexpr int block = 8;
  static constexpr int mask_token = 200056;
  DFlash2(const std::filesystem::path& checkpoint, cudaStream_t stream, bool nvfp4);
  ~DFlash2();

  void bind_head(kernels::head::Nvfp4Matrix head, const __nv_bfloat16* embed,
                 const float* embed_norm);
  void bind_sharded_head(DFlash2Head& head, const __nv_bfloat16* embed, const float* embed_norm);
  void warmup();
  const int* propose(int anchor);
  const float* confidence() const;
  void reset();
  void commit(const __nv_bfloat16* taps, int n, int start, bool decode = false);
  const __nv_bfloat16* forward(const __nv_bfloat16* embeddings);
  const int* select(const int* candidates, const float* unary, const int* anchor);

 private:
  const int* propose(int anchor, bool warming_up);
  void propose_sharded_head();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class DFlash2Head {
 public:
  DFlash2Head(TpRuntime& tp, kernels::head::Nvfp4Matrix matrix, cudaStream_t stream);
  ~DFlash2Head();
  void run(const __nv_bfloat16* hidden = nullptr);
  const int* ids() const;
  const float* values() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ingine::model
