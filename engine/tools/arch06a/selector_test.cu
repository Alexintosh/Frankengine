#include "kernels/draft/dflash2_selector.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}

template <typename T>
T* allocate(int count) {
  T* ptr = nullptr;
  check(cudaMallocManaged(&ptr, sizeof(T) * count));
  return ptr;
}

int main() {
  auto* ids = allocate<int>(7 * 16);
  auto* unary = allocate<float>(7 * 16);
  auto* projected = allocate<__nv_bfloat16>(7 * 256);
  auto* predecessor = allocate<__nv_bfloat16>(16 * 256);
  auto* successor = allocate<__nv_bfloat16>(16 * 256);
  auto* anchor = allocate<int>(1);
  auto* lattice = allocate<float>(7 * 16 * 16);
  auto* tokens = allocate<int>(7);
  *anchor = 3;
  for (int e = 0; e < 7; ++e) {
    for (int c = 0; c < 16; ++c) {
      ids[e * 16 + c] = c;
      unary[e * 16 + c] = float(c) / 16;
    }
    for (int r = 0; r < 256; ++r)
      projected[e * 256 + r] = __float2bfloat16_rn(r == 0 ? 1.0f : 0.0f);
  }
  for (int c = 0; c < 16; ++c) {
    for (int r = 0; r < 256; ++r) {
      predecessor[c * 256 + r] = __float2bfloat16_rn(c % 2 == 0 ? -1.0f : 1.0f);
      successor[c * 256 + r] = __float2bfloat16_rn(float(c) / 8);
    }
  }
  ingine::kernels::draft::dflash2_greedy_selector(
      ids, unary, projected, predecessor, successor, anchor, lattice, tokens, nullptr);
  check(cudaDeviceSynchronize());
  for (int e = 0; e < 7; ++e) {
    for (int p = 0; p < 16; ++p) {
      const float sign = (e == 0 || p % 2) ? 1.0f : -1.0f;
      for (int c = 0; c < 16; ++c) {
        const float expected = 30 * std::tanh((float(c) / 16 + sign * float(c) / 8) / 30);
        if (std::abs(lattice[(e * 16 + p) * 16 + c] - expected) > 1e-5f) return 2;
      }
    }
    if (tokens[e] != 15) return 3;
  }
  check(cudaFree(tokens));
  check(cudaFree(lattice));
  check(cudaFree(anchor));
  check(cudaFree(successor));
  check(cudaFree(predecessor));
  check(cudaFree(projected));
  check(cudaFree(unary));
  check(cudaFree(ids));
  std::puts("1792 lattice values and seven greedy tokens match");
}
