// Weights in the decoder's device layouts, read from the Hugging Face checkpoint: weights.load / load_layer,
// plus fast.FastDecoder._concat_projections (one [R, H] projection matrix per layer, the per-projection
// matrices are row views of it), _stack_layer_params (stacked norms and gate weights) and _fp8_dense
// (wproj, wo and the unembedding as FP8 rows + FP32 row scales), or lossless PICO12 packing, and
// model.Ablation.load / none.
//
// Large tensors are uploaded straight from the file with the core DeviceUploader; small per-channel
// parameters are read on the host, widened to FP32 (exact from BF16) and copied.
#include "model/runtime.h"

#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/dense/dense.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

namespace fs = std::filesystem;

float bf16_bits_to_float(std::uint16_t b) {
  const std::uint32_t w = std::uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

std::uint16_t float_to_bf16_bits(float f) {  // round to nearest even (torch .to(bfloat16))
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  if ((w & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((w >> 16) | 0x40);
  return std::uint16_t((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}

// Checkpoint values widened to FP32 on the host (BF16 -> FP32 exact, F32 as is).
std::vector<float> host_f32(const SafetensorsModel& m, const std::string& name) {
  const SafetensorsRef ref = m.at(name);
  const std::vector<std::byte> raw = m.read(name);
  const std::size_t n = std::size_t(ref->numel());
  std::vector<float> out(n);
  if (ref->dtype == "F32") {
    std::memcpy(out.data(), raw.data(), n * 4);
  } else if (ref->dtype == "BF16") {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      out[i] = bf16_bits_to_float(b);
    }
  } else {
    throw std::runtime_error("weights: " + name + " has unsupported dtype " + ref->dtype);
  }
  return out;
}

class Loader {
 public:
  Loader(const SafetensorsModel& m, DeviceUploader& up, cudaStream_t stream, std::vector<DeviceBuffer>& storage)
      : m_(m), up_(up), stream_(stream), storage_(storage) {}

  void* alloc(std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  }

  Tensor upload_f32(const std::vector<float>& host, std::initializer_list<std::int32_t> shape) {
    void* d = alloc(host.size() * 4);
    CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    return rm(d, DType::FP32, shape);
  }

  Tensor upload_bf16(const std::vector<float>& host, std::initializer_list<std::int32_t> shape) {
    std::vector<std::uint16_t> b(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) b[i] = float_to_bf16_bits(host[i]);
    void* d = alloc(b.size() * 2);
    CUDA_CHECK(cudaMemcpy(d, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    return rm(d, DType::BF16, shape);
  }

  // f32(name): checkpoint tensor as FP32 (weights.load_layer's `f32`), viewed with `shape`.
  Tensor f32(const std::string& name, std::initializer_list<std::int32_t> shape) {
    return upload_f32(host_f32(m_, name), shape);
  }

  void queue(const std::string& name, void* dst, std::int64_t rows = -1, std::int64_t first = 0) {
    const SafetensorsRef ref = m_.at(name);
    UploadRequest r = upload_request(ref, dst);
    if (rows >= 0) {
      const std::int64_t n = ref->shape.at(0);
      if (first < 0 || first > n || rows > n - first) throw std::invalid_argument("checkpoint slice outside tensor");
      const auto stride = r.bytes / n;
      r.offset += first * stride;
      r.bytes = rows * stride;
    }
    pending_.push_back(r);
  }

  // W13 stores interleaved gate/up pairs. Each expert's shard is a contiguous file range.
  Tensor expert_rows(const std::string& name, ExpertSlice slice, int cols) {
    const SafetensorsRef ref = m_.at(name);
    auto* dst = static_cast<char*>(alloc(std::size_t(EXPERTS) * 2 * slice.packed * cols));
    if (slice.packed != slice.width)
      CUDA_CHECK(cudaMemsetAsync(dst, 0, std::size_t(EXPERTS) * 2 * slice.packed * cols, stream_));
    for (int e = 0; e < EXPERTS; ++e) {
      UploadRequest r = upload_request(ref, dst + std::size_t(e) * 2 * slice.packed * cols);
      r.offset += std::size_t(e * 2 * INTER + 2 * slice.begin) * cols;
      r.bytes = std::size_t(2 * slice.width) * cols;
      pending_.push_back(r);
    }
    const DType dtype = ref->dtype == "U8" ? DType::U8 : DType::FP8_E4M3FN;
    return rm(dst, dtype, {EXPERTS, 2 * slice.packed, cols});
  }

  // Raw checkpoint tensor (any dtype) in new device memory, uploaded at flush().
  Tensor raw(const std::string& name) {
    const SafetensorsRef ref = m_.at(name);
    void* d = alloc(ref->bytes);
    queue(name, d);
    return ref->view(d);
  }

  void flush() {
    if (!pending_.empty()) up_.upload(pending_, stream_);
    pending_.clear();
  }

  const SafetensorsModel& model() const { return m_; }
  cudaStream_t stream() const { return stream_; }

 private:
  const SafetensorsModel& m_;
  DeviceUploader& up_;
  cudaStream_t stream_;
  std::vector<DeviceBuffer>& storage_;
  std::vector<UploadRequest> pending_;
};

std::array<ExpertFormat, LAYERS> expert_formats(const fs::path& model_dir) {
  std::ifstream in(model_dir / "hf_quant_config.json");
  if (!in) throw std::runtime_error("weights: cannot open " + (model_dir / "hf_quant_config.json").string());
  const nlohmann::json q = nlohmann::json::parse(in);
  std::array<ExpertFormat, LAYERS> fmt{};
  std::array<bool, LAYERS> seen{};
  const std::string suffix = "mlp.experts.w13_weight";
  for (const auto& group : q.at("model").at("parameter_groups")) {
    const std::string dt = group.at("compute_dtype").get<std::string>();
    for (const auto& t : group.at("tensors")) {
      const std::string name = t.get<std::string>();
      if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        continue;
      // model.llm.layers.<l>.mlp.experts.w13_weight
      const std::size_t a = std::string("model.llm.layers.").size();
      const int l = std::stoi(name.substr(a, name.find('.', a) - a));
      if (l < 0 || l >= LAYERS) continue;
      if (dt == "nvfp4") fmt[l] = ExpertFormat::NVFP4;
      else if (dt == "mxfp8") fmt[l] = ExpertFormat::MXFP8;
      else throw std::runtime_error("weights: unsupported expert format " + dt);
      seen[l] = true;
    }
  }
  for (int l = 0; l < LAYERS; ++l)
    if (!seen[l]) throw std::runtime_error("weights: no expert format for layer " + std::to_string(l));
  return fmt;
}

// FP8 rows + scales of a BF16 matrix (dense fp8_rows), into new storage.
Dense to_fp8(Loader& ld, const Tensor& bf16, int rows, int cols) {
  void* q = ld.alloc(std::size_t(rows) * cols);
  float* s = static_cast<float*>(ld.alloc(std::size_t(rows) * 4));
  kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(bf16.data), rows, cols, q, s, ld.stream());
  return Dense{rm(q, DType::FP8_E4M3FN, {rows, cols}), rm(s, DType::FP32, {rows})};
}

Dense to_pico12(Loader& ld, const void* bf16, int rows, int cols) {
  auto packed = kernels::pico12::pack_pico12_host(bf16, rows, cols);
  if (packed.bytes > std::size_t(std::numeric_limits<std::int32_t>::max()))
    throw std::length_error("weights: PICO12 blob exceeds tensor size limit");
  void* d = ld.alloc(packed.bytes);
  CUDA_CHECK(cudaMemcpyAsync(d, packed.data.get(), packed.bytes, cudaMemcpyHostToDevice, ld.stream()));
  CUDA_CHECK(cudaStreamSynchronize(ld.stream()));
  return {rm(d, DType::U8, {int(packed.bytes)}), {}, packed.layout, true};
}

void read_bf16_rows(SafetensorsRef ref, void* dst, int cols, int rows, int first) {
  const auto req = upload_request(ref, dst);
  const std::size_t stride = std::size_t(cols) * sizeof(std::uint16_t);
  ref.file->read_range(req.offset + std::size_t(first) * stride, std::size_t(rows) * stride, dst);
}

Dense row_view(const Dense& d, int begin, int end) {
  const int cols = d.cols();
  const std::size_t esize = d.fp8() ? 1 : 2;
  Dense v;
  v.w = rm(static_cast<char*>(d.w.data) + std::size_t(begin) * cols * esize, d.w.dtype, {end - begin, cols});
  if (d.fp8()) v.scale = rm(static_cast<float*>(d.scale.data) + begin, DType::FP32, {end - begin});
  return v;
}

void copy_columns(Loader& ld, void* dst, int dst_cols, const void* src, int src_cols,
                  int first, int cols, int rows, int element_size) {
  CUDA_CHECK(cudaMemcpy2DAsync(dst, std::size_t(dst_cols) * element_size,
                              static_cast<const char*>(src) + std::size_t(first) * element_size,
                              std::size_t(src_cols) * element_size, std::size_t(cols) * element_size,
                              rows, cudaMemcpyDeviceToDevice, ld.stream()));
}

Layer load_layer_into(Loader& ld, int l, ExpertFormat fmt, bool dense_fp8, const TpLayout& tp, bool dense_pico12) {
  const SafetensorsModel& m = ld.model();
  const std::string p = "model.llm.layers." + std::to_string(l) + ".";
  Layer layer;
  layer.idx = l;
  layer.full = is_full_layer(l);
  layer.loop = is_loop_layer(l);
  const int H = HIDDEN;

  if (layer.full) {
    layer.q_norm = ld.f32(p + "q_input_norm.weight", {H});
    layer.k_norm = ld.f32(p + "k_input_norm.weight", {H});
    layer.v_norm = ld.f32(p + "v_input_norm.weight", {H});
  } else {
    layer.attn_norm = ld.f32(p + "attn_norm.weight", {H});
  }
  layer.mlp_norm = ld.f32(p + "mlp_norm.weight", {H});

  // projections: one [R, H] BF16 matrix in SEG order (fast._concat_projections)
  static const char* seg_names[NUM_SEGS] = {"attn.wq_du.weight",        "attn.gate_attn.weight",
                                            "attn.wr_du.weight",        "attn.attn_sink_proj.weight",
                                            "attn.wk_dv.weight",        "attn.wv_dv.weight",
                                            "attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj"};
  const int R = tp.projection_rows(layer.full);
  const bool sharded = tp.tp_size > 1;
  DeviceBuffer proj_tmp, wo_tmp, w2_tmp, w2_scale_tmp;
  Tensor wproj_bf16, wo_bf16;
  auto bf16_matrix = [&](const std::string& name, int rows, int cols) {
    const SafetensorsRef ref = m.at(name);
    if (ref->dtype != "BF16" || ref->numel() != std::int64_t(rows) * cols)
      throw std::runtime_error("weights: " + name + " must be BF16 [" + std::to_string(rows) + ", " +
                               std::to_string(cols) + "]");
    return ref;
  };
  const std::string wo_name = p + "attn.wo_ud.weight";
  const auto wo_ref = bf16_matrix(wo_name, H, HEADS * HEAD_DIM);
  if (dense_pico12) {
    std::vector<std::uint16_t> proj(std::size_t(R) * H);
    for (int s = 0; s < num_segs(layer.full); ++s) {
      const auto ref = bf16_matrix(p + seg_names[s], SEG_ROWS[s], H);
      read_bf16_rows(ref, proj.data() + std::size_t(tp.weight_offset(s)) * H, H,
                     tp.segment_rows(s), tp.tp_rank * tp.segment_rows(s));
    }
    layer.wproj = to_pico12(ld, proj.data(), R, H);
    const int cols = HEADS * HEAD_DIM;
    std::vector<std::uint16_t> wo(std::size_t(H) * cols);
    read_bf16_rows(wo_ref, wo.data(), cols, H, 0);
    if (sharded) {
      const int first = tp.tp_rank * tp.q_dim_local;
      // Compact the rank's column slice in place, ahead of the rows still to read.
      for (int r = 0; r < H; ++r)
        std::memmove(wo.data() + std::size_t(r) * tp.q_dim_local, wo.data() + std::size_t(r) * cols + first,
                     std::size_t(tp.q_dim_local) * sizeof(wo[0]));
    }
    layer.wo = to_pico12(ld, wo.data(), H, tp.q_dim_local);
  } else {
    void* d = dense_fp8 ? (proj_tmp = DeviceBuffer(std::size_t(R) * H * 2)).p : ld.alloc(std::size_t(R) * H * 2);
    for (int s = 0; s < num_segs(layer.full); ++s) {
      const std::string name = p + seg_names[s];
      bf16_matrix(name, SEG_ROWS[s], H);
      ld.queue(name, static_cast<char*>(d) + std::size_t(tp.weight_offset(s)) * H * 2,
               tp.segment_rows(s), tp.tp_rank * tp.segment_rows(s));
    }
    wproj_bf16 = rm(d, DType::BF16, {R, H});
    void* o = dense_fp8 || sharded ? (wo_tmp = DeviceBuffer(std::size_t(H) * 3072 * 2)).p : ld.alloc(std::size_t(H) * 3072 * 2);
    ld.queue(wo_name, o, H, 0);
    wo_bf16 = rm(o, DType::BF16, {H, HEADS * HEAD_DIM});
  }

  auto f32_shard = [&](const std::string& name, int rows, int cols, bool columns) {
    if (!sharded) return ld.f32(p + name, {rows, cols});
    const auto src = host_f32(m, p + name);
    const int local_rows = columns ? rows : rows / tp.tp_size;
    const int local_cols = columns ? cols / tp.tp_size : cols;
    std::vector<float> shard(std::size_t(local_rows) * local_cols);
    const int first_row = columns ? 0 : tp.tp_rank * local_rows;
    const int first_col = columns ? tp.tp_rank * local_cols : 0;
    for (int r = 0; r < local_rows; ++r)
      std::copy_n(src.data() + std::size_t(first_row + r) * cols + first_col, local_cols,
                  shard.data() + std::size_t(r) * local_cols);
    return ld.upload_f32(shard, {local_rows, local_cols});
  };
  layer.q_scale = f32_shard("attn.q_l2_clip.scale", HEADS, HEAD_DIM, false);
  layer.r_scale = f32_shard("attn.rel_l2_clip.scale", HEADS, D_REL, false);
  layer.rel_proj = ld.f32(p + "attn.rel_logits_proj.proj", {D_REL, REL_EXTENT});
  if (layer.full) {
    layer.k_static = f32_shard("attn.k_sconv.static_w", CONV_K, KV_DIM, true);
    layer.v_static = f32_shard("attn.v_sconv.static_w", CONV_K, KV_DIM, true);
  } else {
    layer.k_conv = f32_shard("attn.k_sconv.weight", KV_DIM, CONV_K, false);
    layer.v_conv = f32_shard("attn.v_sconv.weight", KV_DIM, CONV_K, false);
  }
  layer.attn_conv = ld.f32(p + "attn_sconv.weight", {H, CONV_K});
  layer.mlp_conv = ld.f32(p + "mlp_sconv.weight", {H, CONV_K});
  layer.router = ld.f32(p + "mlp.gate.weight", {EXPERTS, H});
  layer.router_bias = ld.f32(p + "mlp.gate.bias", {EXPERTS});
  layer.router_bf16 = ld.upload_bf16(host_f32(m, p + "mlp.gate.weight"), {EXPERTS, H});  // exact
  if (l > 0) {
    const std::string qn = "model.llm.attn_res.attn_queries." + std::to_string(l - 1);
    const std::int32_t nq = std::int32_t(m.at(qn)->numel() / H);
    layer.attn_query = ld.f32(qn, {nq, H});
  }
  layer.mlp_query = ld.f32("model.llm.attn_res.mlp_queries." + std::to_string(l), {1, H});

  // stacked norm weights (fast._stack_layer_params)
  {
    std::vector<float> n3(std::size_t(3) * H);
    if (layer.full) {
      const char* names[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
      for (int g = 0; g < 3; ++g) {
        const std::vector<float> v = host_f32(m, p + names[g]);
        std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
      }
    } else {
      const std::vector<float> v = host_f32(m, p + "attn_norm.weight");
      for (int g = 0; g < 3; ++g) std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
    }
    layer.attn_norm3 = ld.upload_f32(n3, {3, H});
    layer.attn_norm_mix = layer.full ? layer.attn_norm3 : rm(layer.attn_norm3.data, DType::FP32, {1, H});
    layer.mlp_norm1 = rm(layer.mlp_norm.data, DType::FP32, {1, H});
  }

  if (layer.loop) {
    // checkpoint controllers: attn (q), k, v, mlp; stacks [q, k, v] and [mlp]
    const char* ckpt[4] = {"attn", "k", "v", "mlp"};
    std::vector<float> down[4], forget[4], gain[4];
    for (int g = 0; g < 4; ++g) {
      const std::string c = p + "controllers." + ckpt[g] + ".";
      down[g] = host_f32(m, c + "down_weight");
      forget[g] = host_f32(m, c + "forget_weight");
      gain[g] = host_f32(m, c + "gain_weight");
    }
    auto stack = [](std::vector<float>* parts, int n) {
      std::vector<float> out;
      for (int i = 0; i < n; ++i) out.insert(out.end(), parts[i].begin(), parts[i].end());
      return out;
    };
    const int D2 = 2 * H;
    layer.qkv_gate_down = ld.upload_bf16(stack(down, 3), {3, GATE_RANK, D2});  // exact (BF16 checkpoint)
    layer.qkv_gate_forget = ld.upload_f32(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain = ld.upload_f32(stack(gain, 3), {3, H, GATE_RANK});
    layer.qkv_gate_forget16 = ld.upload_bf16(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain16 = ld.upload_bf16(stack(gain, 3), {3, H, GATE_RANK});
    layer.mlp1_gate_down = ld.upload_bf16(down[3], {1, GATE_RANK, D2});
    layer.mlp1_gate_forget = ld.upload_f32(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain = ld.upload_f32(gain[3], {1, H, GATE_RANK});
    layer.mlp1_gate_forget16 = ld.upload_bf16(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain16 = ld.upload_bf16(gain[3], {1, H, GATE_RANK});
    auto at = [](const Tensor& stack, int g, DType dt, std::int32_t a, std::int32_t b) {
      const std::size_t es = dt == DType::FP32 ? 4 : 2;
      return rm(static_cast<char*>(stack.data) + std::size_t(g) * a * b * es, dt, {a, b});
    };
    Tensor* downs[3] = {&layer.q_gate_down, &layer.k_gate_down, &layer.v_gate_down};
    Tensor* forgets[3] = {&layer.q_gate_forget, &layer.k_gate_forget, &layer.v_gate_forget};
    Tensor* gains[3] = {&layer.q_gate_gain, &layer.k_gate_gain, &layer.v_gate_gain};
    for (int g = 0; g < 3; ++g) {
      *downs[g] = at(layer.qkv_gate_down, g, DType::BF16, GATE_RANK, D2);
      *forgets[g] = at(layer.qkv_gate_forget, g, DType::FP32, H, GATE_RANK);
      *gains[g] = at(layer.qkv_gate_gain, g, DType::FP32, H, GATE_RANK);
    }
    layer.mlp_gate_down = at(layer.mlp1_gate_down, 0, DType::BF16, GATE_RANK, D2);
    layer.mlp_gate_forget = at(layer.mlp1_gate_forget, 0, DType::FP32, H, GATE_RANK);
    layer.mlp_gate_gain = at(layer.mlp1_gate_gain, 0, DType::FP32, H, GATE_RANK);
  }

  // routed experts in checkpoint encoding
  const std::string e = p + "mlp.experts.";
  Experts& ex = layer.experts;
  ex.fmt = fmt;
  const bool nv = fmt == ExpertFormat::NVFP4;
  const ExpertSlice slice = tp.expert_slice(nv);
  const int code_block = nv ? 2 : 1, scale_block = nv ? 16 : 32;
  const std::string want_codes = nv ? "U8" : "F8_E4M3", want_scales = nv ? "F8_E4M3" : "U8";
  if (m.at(e + "w13_weight")->dtype != want_codes || m.at(e + "w13_weight_scale")->dtype != want_scales ||
      m.at(e + "w2_weight")->dtype != want_codes || m.at(e + "w2_weight_scale")->dtype != want_scales)
    throw std::runtime_error("weights: layer " + std::to_string(l) + " experts do not match their format");
  if (sharded) {
    ex.w13 = ld.expert_rows(e + "w13_weight", slice, H / code_block);
    ex.w13_scale = ld.expert_rows(e + "w13_weight_scale", slice, H / scale_block);
    auto pack_w2 = [&](const std::string& name, int block, DeviceBuffer& temporary) {
      const auto ref = m.at(name);
      temporary = DeviceBuffer(ref->bytes);
      ld.queue(name, temporary.p);
      const int cols = slice.packed / block;
      const auto dtype = ref->dtype == "U8" ? DType::U8 : DType::FP8_E4M3FN;
      return rm(ld.alloc(std::size_t(EXPERTS) * H * cols), dtype, {EXPERTS, H, cols});
    };
    ex.w2 = pack_w2(e + "w2_weight", code_block, w2_tmp);
    ex.w2_scale = pack_w2(e + "w2_weight_scale", scale_block, w2_scale_tmp);
  } else {
    ex.w13 = ld.raw(e + "w13_weight");
    ex.w13_scale = ld.raw(e + "w13_weight_scale");
    ex.w2 = ld.raw(e + "w2_weight");
    ex.w2_scale = ld.raw(e + "w2_weight_scale");
  }
  if (nv) {
    for (const char* w : {"w13", "w2"}) {
      const std::vector<float> amax = host_f32(m, e + w + "_weight_amax");
      const std::vector<float> den = host_f32(m, e + w + "_weight_outer_scale_denominator");
      std::vector<float> f(EXPERTS);
      for (int i = 0; i < EXPERTS; ++i) f[i] = amax[i] / den[den.size() == 1 ? 0 : i];  // FP32 division
      (std::string(w) == "w13" ? ex.w13_factor : ex.w2_factor) = ld.upload_f32(f, {EXPERTS});
    }
  }

  ld.flush();
  if (sharded) {
    auto copy_w2 = [&](const Tensor& dst, const DeviceBuffer& source, int block) {
      if (slice.packed != slice.width) CUDA_CHECK(cudaMemsetAsync(dst.data, 0, dst.bytes(), ld.stream()));
      copy_columns(ld, dst.data, slice.packed / block, source.p, INTER / block,
                   slice.begin / block, slice.width / block, EXPERTS * H, 1);
    };
    copy_w2(ex.w2, w2_tmp, code_block);
    copy_w2(ex.w2_scale, w2_scale_tmp, scale_block);
  }

  DeviceBuffer wo_codes_tmp;
  if (!dense_pico12) {
    layer.wproj = dense_fp8 ? to_fp8(ld, wproj_bf16, R, H) : Dense{wproj_bf16, {}};
    if (dense_fp8 && sharded) {
      // Quantize the full checkpoint rows first. Each rank retains the same FP32 row scales.
      wo_codes_tmp = DeviceBuffer(std::size_t(H) * HEADS * HEAD_DIM);
      auto* scales = static_cast<float*>(ld.alloc(std::size_t(H) * sizeof(float)));
      kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(wo_bf16.data), H, HEADS * HEAD_DIM,
                               wo_codes_tmp.p, scales, ld.stream());
      layer.wo = {rm(wo_codes_tmp.p, DType::FP8_E4M3FN, {H, HEADS * HEAD_DIM}), rm(scales, DType::FP32, {H})};
    } else {
      layer.wo = dense_fp8 ? to_fp8(ld, wo_bf16, H, HEADS * HEAD_DIM) : Dense{wo_bf16, {}};
    }
    if (sharded) {
      const int element_size = dense_fp8 ? 1 : 2;
      void* local = ld.alloc(std::size_t(H) * tp.q_dim_local * element_size);
      copy_columns(ld, local, tp.q_dim_local, layer.wo.w.data, HEADS * HEAD_DIM, tp.tp_rank * tp.q_dim_local,
                   tp.q_dim_local, H, element_size);
      layer.wo.w = rm(local, layer.wo.w.dtype, {H, tp.q_dim_local});
    }
    Dense* views[NUM_SEGS] = {&layer.wq, &layer.wgate, &layer.wr, &layer.wsink,
                              &layer.wk, &layer.wv,    &layer.wk_dyn, &layer.wv_dyn};
    for (int s = 0; s < num_segs(layer.full); ++s) *views[s] = row_view(layer.wproj, tp.weight_offset(s), tp.weight_offset(s + 1));
  }
  if (dense_fp8 || sharded) CUDA_CHECK(cudaStreamSynchronize(ld.stream()));  // temporaries
  return layer;
}

kernels::head::CertifiedHead load_cert_head(Loader& ld, const __nv_bfloat16* values, int rows) {
  using namespace kernels::head;
  constexpr int H = HIDDEN;
  const auto stream = ld.stream();
  const std::size_t elements = std::size_t(rows) * H;
  const float g = nvfp4_outer_scale(values, elements, stream);
  auto* codes = static_cast<uint8_t*>(ld.alloc(elements / 2));
  auto* scales = static_cast<uint8_t*>(ld.alloc(elements / 16));
  auto* bounds = static_cast<float*>(ld.alloc(std::size_t(rows) * sizeof(float)));
  nvfp4_pack_rows(values, rows, g, codes, scales, stream);
  const Nvfp4Matrix proxy{codes, scales, g, rows, 0};
  cert_head_bounds(values, proxy, bounds, stream);
  return {proxy, bounds};
}

HeadWeights load_head_into(Loader& ld, bool dense_fp8, bool head_nvfp4, const TpLayout& tp, bool dense_pico12,
                           bool cert_head = false) {
  const std::string name = "model.llm.unembed.weight";
  const auto ref = ld.model().at(name);
  const int H = HIDDEN, local = tp.v_end - tp.v_begin;
  if (ref->dtype != "BF16" || ref->shape.size() != 2 || ref->shape.at(1) != H || ref->shape.at(0) < UNPADDED_VOCAB)
    throw std::runtime_error("weights: invalid unembed tensor");

  const int rows = head_nvfp4 && tp.tp_size == 1 ? int(ref->shape.at(0)) : local;
  const std::size_t bytes = std::size_t(rows) * H * 2;
  HeadWeights out;
  out.v_begin = tp.v_begin;
  if (dense_pico12 && !head_nvfp4) {
    ld.flush();
    auto host = std::make_unique_for_overwrite<std::uint16_t[]>(std::size_t(rows) * H);
    read_bf16_rows(ref, host.get(), H, rows, tp.v_begin);
    out.dense = to_pico12(ld, host.get(), rows, H);
    if (cert_head) {
      DeviceBuffer original(bytes);
      original.copy_from_host(host.get(), bytes);
      out.certified = load_cert_head(ld, static_cast<const __nv_bfloat16*>(original.p), rows);
      CUDA_CHECK(cudaStreamSynchronize(ld.stream()));
    }
    return out;
  }
  DeviceBuffer tmp;
  void* d = dense_fp8 || head_nvfp4 ? (tmp = DeviceBuffer(bytes)).p : ld.alloc(bytes);
  ld.queue(name, d, rows, tp.v_begin);
  ld.flush();

  if (head_nvfp4) {
    auto stream = ld.stream();
    const auto* values = static_cast<const __nv_bfloat16*>(d);
    float g = 0;
    if (tp.tp_size == 1) {
      g = kernels::head::nvfp4_outer_scale(values, std::size_t(rows) * H, stream);
    } else {
      // Scan bounded batches for the checkpoint-wide calibration, including padded rows.
      // Keep only the rank's own BF16 rows for packing.
      constexpr int batch = 4096;
      DeviceBuffer scan(std::size_t(batch) * H * 2);
      const int total = int(ref->shape.at(0));
      for (const auto& [first, end] : {std::pair{0, tp.v_begin}, {tp.v_end, total}}) {
        for (int begin = first; begin < end; begin += batch) {
          const int count = std::min(batch, end - begin);
          ld.queue(name, scan.p, count, begin);
          ld.flush();
          g = std::max(g, kernels::head::nvfp4_outer_scale(
              static_cast<const __nv_bfloat16*>(scan.p), std::size_t(count) * H, stream));
        }
      }
      g = std::max(g, kernels::head::nvfp4_outer_scale(values, std::size_t(rows) * H, stream));
    }
    auto* codes = static_cast<uint8_t*>(ld.alloc(std::size_t(local) * H / 2));
    auto* scales = static_cast<uint8_t*>(ld.alloc(std::size_t(local) * H / 16));
    kernels::head::nvfp4_pack_rows(values, local, g, codes, scales, stream);
    out.nvfp4 = {codes, scales, g, local, tp.v_begin};
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    const Tensor bf = rm(d, DType::BF16, {local, H});
    out.dense = dense_fp8 ? to_fp8(ld, bf, local, H) : Dense{bf, {}};
    if (cert_head) out.certified = load_cert_head(ld, static_cast<const __nv_bfloat16*>(d), local);
    if (dense_fp8) CUDA_CHECK(cudaStreamSynchronize(ld.stream()));
  }
  return out;
}

}  // namespace

Weights load_weights(const fs::path& model_dir, bool dense_fp8, cudaStream_t stream, bool head_nvfp4,
                     const TpLayout& tp, bool dense_pico12, bool cert_head) {
  validate_dense_formats(dense_fp8, dense_pico12);
  const SafetensorsModel m(model_dir);
  DeviceUploader up;
  Weights w;
  w.dense_fp8 = dense_fp8;
  w.dense_pico12 = dense_pico12;
  w.expert_formats = expert_formats(model_dir);
  w.storage.reserve(4096);
  Loader ld(m, up, stream, w.storage);
  const int H = HIDDEN;
  for (int l = 0; l < LAYERS; ++l)
    w.layers.push_back(load_layer_into(ld, l, w.expert_formats[l], dense_fp8, tp, dense_pico12));

  // message_queries [12, 4, H] = cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1) with queries and
  // kv_queries [24, H] unflattened to [12, 2, H]
  {
    const std::vector<float> q = host_f32(m, "model.llm.middle_cycle_cross_pass_queries");
    const std::vector<float> kv = host_f32(m, "model.llm.middle_cycle_cross_pass_kv_queries");
    const int n = int(q.size() / H / 2);
    std::vector<float> mq(std::size_t(n) * 4 * H);
    for (int i = 0; i < n; ++i) {
      const float* rows[4] = {&q[std::size_t(2 * i) * H], &kv[std::size_t(2 * i) * H], &kv[std::size_t(2 * i + 1) * H],
                              &q[std::size_t(2 * i + 1) * H]};
      for (int j = 0; j < 4; ++j) std::copy(rows[j], rows[j] + H, mq.begin() + (std::ptrdiff_t(i) * 4 + j) * H);
    }
    w.message_queries = ld.upload_f32(mq, {n, 4, H});
  }
  w.embed = ld.raw("model.llm.embed.weight");
  if (w.embed.dtype != DType::BF16) throw std::runtime_error("weights: embed must be BF16");
  w.embed_norm = ld.f32("model.llm.embed_norm.weight", {H});
  w.final_query = ld.f32("model.llm.attn_res.output_query", {1, H});
  w.final_norm = ld.f32("model.llm.norm.weight", {H});
  const HeadWeights head = load_head_into(ld, dense_fp8, head_nvfp4, tp, dense_pico12,
                                          cert_head && tp.tp_size == 1 && !dense_fp8 && !head_nvfp4);
  w.unembed = head.dense;
  w.unembed_nvfp4 = head.nvfp4;
  w.unembed_certified = head.certified;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return w;
}

Layer load_layer(const fs::path& model_dir, int l, bool dense_fp8, cudaStream_t stream,
                 std::vector<DeviceBuffer>& storage, const TpLayout& tp) {
  if (l < 0 || l >= LAYERS) throw std::invalid_argument("load_layer: layer out of range");
  const SafetensorsModel m(model_dir);
  DeviceUploader up;
  Loader ld(m, up, stream, storage);
  Layer layer = load_layer_into(ld, l, expert_formats(model_dir)[l], dense_fp8, tp, false);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return layer;
}

HeadWeights load_head(const fs::path& model_dir, bool dense_fp8, bool head_nvfp4, cudaStream_t stream,
                      std::vector<DeviceBuffer>& storage, const TpLayout& tp, bool dense_pico12) {
  validate_dense_formats(dense_fp8, dense_pico12);
  const SafetensorsModel m(model_dir);
  DeviceUploader up;
  Loader ld(m, up, stream, storage);
  return load_head_into(ld, dense_fp8, head_nvfp4, tp, dense_pico12);
}

// File layouts (FP32 or BF16 tensors):
//   attn_dirs, mlp_dirs [48, H]; attn_w, mlp_w [48]                                  plain projection
//   attn_dirs, mlp_dirs [48, H]; attn_weight, mlp_weight, attn_offset, mlp_offset [48]; gate [1];
//       optionally attn_det, mlp_det [48, H] and attn_threshold, mlp_threshold [48]
// The second form removes w * r * (r . y - offset), limited to components above the offset when gate is
// non-zero and, with detectors, to writes with det . y above the threshold. Its weights carry other names so
// that engines that only know the first form reject the file.
Ablation load_ablation(const fs::path& path) {
  Ablation a;
  const int H = HIDDEN;
  const std::size_t nd = std::size_t(NUM_VISITS) * H;
  // rows 0..47: directions, rows 48..95: detectors (zero without them)
  std::vector<float> attn_dirs(2 * nd, 0.f), mlp_dirs(2 * nd, 0.f);
  std::vector<float> attn_w(std::size_t(NUM_VISITS) * 4, 0.f), mlp_w(attn_w.size(), 0.f);
  if (!path.empty()) {
    const SafetensorsModel m(path);
    auto read = [&](const std::string& name, std::size_t n) {
      std::vector<float> v = host_f32(m, name);
      if (v.size() != n) throw std::runtime_error("ablation: " + name + " has the wrong size");
      return v;
    };
    // F.normalize(dirs, dim=-1): x / max(||x||, 1e-12) in FP32
    auto normalize = [&](std::vector<float> v) {
      for (int r = 0; r < NUM_VISITS; ++r) {
        double ss = 0;
        for (int c = 0; c < H; ++c) ss += double(v[std::size_t(r) * H + c]) * v[std::size_t(r) * H + c];
        const float norm = std::max(float(std::sqrt(ss)), 1e-12f);
        for (int c = 0; c < H; ++c) v[std::size_t(r) * H + c] = v[std::size_t(r) * H + c] / norm;
      }
      return v;
    };
    const bool plain = m.contains("attn_w");
    a.detect = !plain && m.contains("attn_det");
    const float floor = !plain && read("gate", 1)[0] != 0.f ? 0.f : -std::numeric_limits<float>::max();
    auto load = [&](const std::string& kind, std::vector<float>& dirs, std::vector<float>& rows) {
      const std::vector<float> r = normalize(read(kind + "_dirs", nd));
      std::copy(r.begin(), r.end(), dirs.begin());
      const std::vector<float> w = read(kind + (plain ? "_w" : "_weight"), NUM_VISITS);
      const std::vector<float> c = plain ? std::vector<float>(NUM_VISITS, 0.f) : read(kind + "_offset", NUM_VISITS);
      std::vector<float> tau(NUM_VISITS, 0.f);
      if (a.detect) {
        const std::vector<float> d = read(kind + "_det", nd);  // as stored: the threshold is in its units
        std::copy(d.begin(), d.end(), dirs.begin() + std::ptrdiff_t(nd));
        tau = read(kind + "_threshold", NUM_VISITS);
      }
      for (int v = 0; v < NUM_VISITS; ++v)
        rows[4 * v] = w[v], rows[4 * v + 1] = c[v], rows[4 * v + 2] = floor, rows[4 * v + 3] = tau[v];
    };
    load("attn", attn_dirs, attn_w);
    load("mlp", mlp_dirs, mlp_w);
    a.active = true;
  }
  auto up = [&](const std::vector<float>& h, std::initializer_list<std::int32_t> shape) {
    a.storage.emplace_back(h.size() * 4);
    CUDA_CHECK(cudaMemcpy(a.storage.back().p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return rm(a.storage.back().p, DType::FP32, shape);
  };
  a.attn_dirs = up(attn_dirs, {2 * NUM_VISITS, H});
  a.attn_w = up(attn_w, {NUM_VISITS, 4});
  a.mlp_dirs = up(mlp_dirs, {2 * NUM_VISITS, H});
  a.mlp_w = up(mlp_w, {NUM_VISITS, 4});
  return a;
}

}  // namespace ingine::model
