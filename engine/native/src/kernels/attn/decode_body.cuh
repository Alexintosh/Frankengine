// Shared device body for consecutive-token and independent-sequence attention kernels.
    constexpr model::TpLayout layout(model::KV_HEADS / HKV);
    constexpr int HQ = layout.hq_local, CONV_C = layout.kv_dim_local;
    constexpr int OFF_GATE = layout.activation_offset(model::SEG_GATE), OFF_REL = layout.activation_offset(model::SEG_REL),
                  OFF_SINK = layout.activation_offset(model::SEG_SINK), OFF_K = layout.activation_offset(model::SEG_K),
                  OFF_V = layout.activation_offset(model::SEG_V), OFF_KDYN = layout.activation_offset(model::SEG_K_DYN),
                  OFF_VDYN = layout.activation_offset(model::SEG_V_DYN);
    static_assert(FULL || (!F8 && !Q4), "quantized caches are full-layer only");
    static_assert(!F8 || !Q4);
    using CT = std::conditional_t<F8, __nv_fp8_storage_t, __nv_bfloat16>;  // cache element
    using KV = std::conditional_t<F8, uint2, uint4>;                        // 8 dims of a cache row
    constexpr int KW = 8, ROWS = GQA * M;  // KW: key-slice warps of the M = 1 path
    using L = Smem<CS, M, F8, Q4>;
    constexpr int OD = L::OD;
    extern __shared__ __align__(16) float sm[];
    __nv_bfloat16(*s_nkv)[M][D] = reinterpret_cast<__nv_bfloat16(*)[M][D]>(sm + L::NKV);

    auto* const new_q4 = reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(sm) + L::Q4_PACKED);
    auto* const new_scale = reinterpret_cast<__half*>(reinterpret_cast<char*>(sm) + L::Q4_SCALE);

    const int g = blockIdx.y, rank = blockIdx.x;  // cluster = the CS blocks of KV head g
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, half = lane >> 4, c = lane & 15;
    const int tok = warp / KW, kw = warp % KW;
    const bool owner = rank == 0;
    const int R = p.R;
    CT* const kc = static_cast<CT*>(p.k_cache);
    CT* const vc = static_cast<CT*>(p.v_cache);

    // ---- stage the inputs in shared memory with cp.async (one round trip, no registers held). Weights
    // first: they may be read before the previous kernel completes (programmatic dependent launch).
    for (int i = tid; i < GQA * D / 4; i += NT) cp16(sm + L::QS + 4 * i, p.q_scale + g * GQA * D + 4 * i);
    for (int i = tid; i < GQA * DREL / 4; i += NT) cp16(sm + L::RS + 4 * i, p.r_scale + g * GQA * DREL + 4 * i);
    if (owner) {
        for (int i = tid; i < 2 * D; i += NT) {  // sliding [kv][ch][4]; full [kv][lag][D]
            const int kv = i / D, q = i % D;
            const float* w = kv ? p.v_conv : p.k_conv;
            cp16(sm + L::W + 4 * i, FULL ? w + (q / (D / 4)) * CONV_C + g * D + 4 * (q % (D / 4)) : w + (g * D + q) * 4);
        }
        for (int i = tid; i < DREL * 2; i += NT)  // rel_proj[:, 0..7]: the new keys' distances
            cp16(sm + L::RELP0 + 4 * i, p.rel_proj + (i / 2) * EXT + 4 * (i % 2));
    }
    const int p0 = int(ld_s64(p.pos));
    if (FULL && p0 + M > p.cap) return;  // out of cache capacity (uniform: no block reaches a barrier)
    const int L_old = FULL ? p0 : WIN;
    const int P = p0 + tok;

    // ---- K/V streaming: two register buffers; the first two are issued now, K/V before the rel_proj
    // columns (whose addresses wait for `pos`). Keys go to the blocks in tiles of 2 STEP, round robin, so
    // the expensive relative-bias tail (distance < 1024) is spread over the whole cluster.
    constexpr int STEP = KW * 2 * U, TILE = 2 * STEP, STRIDE = CS * TILE;
    const int c0 = rank * TILE, old_end = L_old;
    const std::uint64_t kvpol = l2::policy<KV_L2>();
    auto load_kv = [&](Buf<KV>& b, const int base) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = base + (kw * U + u) * 2 + half;
            b.k[u] = b.v[u] = KV{};
            b.kp[u] = -1;
            if (j < old_end) {
                const size_t off = (size_t(j) * HKV + g) * D + c * 8;
                if constexpr (Q4) {
                    const size_t head = size_t(j) * HKV + g;
                    b.k[u] = detail::load_q4<false>(p.k_cache, p.k_scale, head, c, kvpol);
                    b.v[u] = detail::load_q4<true>(p.v_cache, p.v_scale, head, c, kvpol);
                } else if constexpr (F8) {
                    b.k[u] = ld_kv8(kc + off, kvpol);
                    b.v[u] = ld_kv8(vc + off, kvpol);
                } else {
                    b.k[u] = ld_kv(kc + off, kvpol);
                    b.v[u] = ld_kv(vc + off, kvpol);
                }
                b.kp[u] = FULL ? j : int(ld_s64(p.kpos + j));
            }
        }
    };
    auto load_rp = [&](Buf<KV>& b, const int base) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = base + (kw * U + u) * 2 + half;
            // distance if the ring is full and in order (exact for the linear cache)
            b.dp[u] = FULL ? P - j : tok + 1 + ((p0 - 1 - j) & (WIN - 1));
            b.rp[u] = j < old_end && b.dp[u] < EXT ? __ldg(p.rel_proj + c * EXT + b.dp[u]) : 0.f;
        }
    };
    auto load = [&](Buf<KV>& b, const int base) {
        load_kv(b, base);
        load_rp(b, base);
    };
    // ---- M > 1 key tiles: constants, layout and the cp.async issue of one tile (used by the M > 1 path)
    constexpr int KT = L::KT, NE = KT + M - 1, NRP = L::NRP, NTN = NRP / 8, KM = KT / 16;
    constexpr int PSTRIDE = L::PSTRIDE;
    __nv_bfloat16* kvs = reinterpret_cast<__nv_bfloat16*>(sm + L::KVS);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(sm + L::QB);
    int* kps = reinterpret_cast<int*>(sm + L::KPS);
    float* srp = sm + L::SRP;
    __half* ps = reinterpret_cast<__half*>(sm + L::PS);
    const int gid = lane >> 2, t4 = lane & 3;
    // 16-bit offset of (key, k|v, 16-byte chunk) in stage st; chunks XOR-swizzled by key
    auto kv_off = [&](int st, int key, int kv, int ch) { return ((st * KT + key) * 2 + kv) * D + ((ch ^ (key & 7)) * 8); };
    // distance of key base + KT - 1 for token 0: the rel_proj window starts there
    auto dbase_of = [&](const int base) {
        return FULL ? p0 - (base + KT - 1) : 1 + ((p0 - 1 - (base + KT - 1)) & (WIN - 1));
    };
    auto issue = [&](const int st, const int base) {  // BF16 caches (the E4M3 path streams raw tiles below)
        for (int i = tid; i < KT * 2 * (D / 8); i += NT) {
            const int key = i / (2 * (D / 8)), kv = (i / (D / 8)) % 2, ch = i % (D / 8), j = base + key;
            const bool ok = j < L_old;
            if constexpr (Q4) {
                uint4 packed{};
                if (ok) {
                    const size_t head = size_t(j) * HKV + g;
                    packed = kv ? detail::load_q4<true>(p.v_cache, p.v_scale, head, ch, kvpol)
                                : detail::load_q4<false>(p.k_cache, p.k_scale, head, ch, kvpol);
                }
                *reinterpret_cast<uint4*>(kvs + kv_off(st, key, kv, ch)) = packed;
            } else {
                cp16z(reinterpret_cast<float*>(kvs + kv_off(st, key, kv, ch)),
                      reinterpret_cast<const float*>((kv ? vc : kc) + (size_t(ok ? j : 0) * HKV + g) * D + ch * 8), ok,
                      kvpol);
            }
        }
        if constexpr (!FULL) {
            for (int i = tid; i < KT; i += NT)  // low word of the int64 position
                cp4z(reinterpret_cast<float*>(kps + st * KT + i),
                     reinterpret_cast<const float*>(p.kpos + (base + i < L_old ? base + i : 0)), base + i < L_old);
        }
        const int dbase = dbase_of(base);
        if (dbase < EXT) {
            for (int i = tid; i < DREL * NE; i += NT) {
                const int c2 = i / NE, e = i % NE, d = dbase + e;
                const bool ok = d >= 0 && d < EXT;
                cp4z(srp + (st * DREL + c2) * (KT + MAXM) + e, p.rel_proj + c2 * EXT + (ok ? d : 0), ok);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // E4M3 caches, M > 1: tile t of this block (keys rank KT + t CS KT ..) is copied as raw E4M3 rows into
    // slot t % NRAW of the stage region: K rows with their 16-byte chunks XOR-swizzled by key (read
    // straight into mma fragments), then V rows (widened to fp16 in the staging region, which is dead once
    // q / rel and the owner's convs are done). Every call commits a group (empty past the end), so the
    // group count per tile is fixed.
    constexpr int NRAW = 2, RAWB = KT * 2 * D;  // E4M3 tiles in flight, bytes per tile
    static_assert(!F8 || M == 1 || (NRAW * RAWB <= 2 * KT * 2 * D && KT * D * 2 <= L::UNION_END * 4),
                  "E4M3 tiles and the widened V tile must fit the stage and staging regions");
    char* const raw = reinterpret_cast<char*>(sm + L::KVS);
    __half* const wv = reinterpret_cast<__half*>(sm);  // E4M3: the widened V tile [KT][D], chunks swizzled
    auto issue_raw = [&](const int t) {
        const int base = rank * KT + t * CS * KT;
        if (base < L_old) {
            for (int i = tid; i < KT * 2 * (D / 16); i += NT) {
                const int key = i / (2 * (D / 16)), kv = (i / (D / 16)) % 2, ch = i % (D / 16), j = base + key;
                const bool ok = j < L_old;
                char* dst = raw + (t % NRAW) * RAWB + kv * KT * D + key * D + (kv ? ch : ch ^ (key & 7)) * 16;
                cp16z(reinterpret_cast<float*>(dst),
                      reinterpret_cast<const float*>((kv ? vc : kc) + (size_t(ok ? j : 0) * HKV + g) * D + ch * 16), ok,
                      kvpol);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // M = 1: the cached keys' K/V and kpos are issued before the wait. pos, kpos and the cache rows below
    // p0 were written two or more kernels back (see PDL above), and the staged inputs then come in one
    // round trip that is not queued behind the K/V burst (issued after the wait, ahead of the staged
    // inputs, the burst delays them by ~2 us; behind them it costs a round trip).
    Buf<KV> A, B;
    if constexpr (M == 1) {
        if (c0 < old_end) load_kv(A, c0);
        if (c0 + STEP < old_end) load_kv(B, c0 + STEP);
        if (c0 < old_end) load_rp(A, c0);
        if (c0 + STEP < old_end) load_rp(B, c0 + STEP);
    }
    // a loop ptxas cannot see through: it would sink the loads above below a plain wait
#pragma unroll 1
    for (int i = 0; i <= p.zero; ++i) asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = tid; i < M * GQA * D / 4; i += NT)       // q rows of heads 3g..3g+2 are contiguous
        cp16(sm + L::Q + 4 * i, p.proj + size_t(i / (GQA * D / 4)) * R + g * GQA * D + 4 * (i % (GQA * D / 4)));
    for (int i = tid; i < M * GQA * DREL / 4; i += NT)
        cp16(sm + L::REL + 4 * i,
             p.proj + size_t(i / (GQA * DREL / 4)) * R + OFF_REL + g * GQA * DREL + 4 * (i % (GQA * DREL / 4)));
    for (int i = tid; i < ROWS * OD / 4; i += NT) {
        const int row = i / (OD / 4), part = i % (OD / 4);
        cp16(sm + L::GATE + 4 * i,
             p.proj + size_t(row / GQA) * R + OFF_GATE + (g * GQA + row % GQA) * D + rank * OD + 4 * part);
    }
    for (int i = tid; i < ROWS; i += NT) cp4(sm + L::SINK + i, p.proj + size_t(i / GQA) * R + OFF_SINK + g * GQA + i % GQA);
    if (owner) {
        for (int i = tid; i < 3 * 2 * D / 4; i += NT) {  // [j][kv][D]
            const int j = i / (2 * D / 4), kv = (i / (D / 4)) % 2, q = i % (D / 4);
            cp16(sm + L::ST + 4 * i, (kv ? p.conv_v : p.conv_k) + j * CONV_C + g * D + 4 * q);
        }
        for (int i = tid; i < M * 2 * D / 4; i += NT) {  // [m][kv][D]
            const int m = i / (2 * D / 4), kv = (i / (D / 4)) % 2, q = i % (D / 4);
            cp16(sm + L::X + 4 * i, p.proj + size_t(m) * R + (kv ? OFF_V : OFF_K) + g * D + 4 * q);
        }
        if constexpr (FULL) {
            for (int i = tid; i < M * 2; i += NT)
                cp16(sm + L::DYN + 4 * i, p.proj + size_t(i / 2) * R + (i % 2 ? OFF_VDYN : OFF_KDYN) + g * 4);
        }
    }

    if constexpr (M > 1 && F8) {  // staged inputs as the oldest group, then the first NRAW tiles behind them
        asm volatile("cp.async.commit_group;" ::: "memory");
#pragma unroll 1
        for (int t = 0; t < NRAW; ++t) issue_raw(t);
        asm volatile("cp.async.wait_group %0;" ::"n"(NRAW) : "memory");
        __syncthreads();
    } else {
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        if constexpr (M > 1) {
            if (rank * KT < L_old) issue(0, rank * KT);  // in flight during q / rel and the owner's work
        }
    }
    // dependents (proj_conv) may launch now: their weight stream queues behind this block's K/V requests
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // ---- q and rel post-processing, once per block. q: one warp per row (lane = 4 dims); rel: one
    // half-warp per row. Fixed trip counts and clamped rows (no branches), so the reductions of a warp's
    // rows overlap. Results: scaled q (fp32, M = 1) or exact bf16 q (M > 1), bf16-valued rel.
    auto tau_of = [&](int t) {
        return p.log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p0 + t + 1) * (1.f / 8192.f), 1.f)) : 1.f;
    };
    if constexpr (M == 1) {  // 6 items, one per warp
        if (warp < 2 * GQA) {
            const int i = warp % GQA;
            const float tau = tau_of(0);
            if (warp < GQA) {
                float4* qa = reinterpret_cast<float4*>(sm + L::Q + i * D) + lane;
                const float4 a = *qa, s = reinterpret_cast<const float4*>(sm + L::QS + i * D)[lane];
                float x[4] = {a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w};
                const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    x[e] = bf16r(x[e] * f);
                    if (p.log_scaling) x[e] = bf16r(x[e] * tau);
                }
                *qa = make_float4(x[0] * QSCALE, x[1] * QSCALE, x[2] * QSCALE, x[3] * QSCALE);
            } else {
                float* ra = sm + L::REL + i * DREL + c;
                const float r = *ra * sm[L::RS + i * DREL + c];
                float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
                if (p.log_scaling) y = bf16r(y * tau);
                __syncwarp();
                if (half == 0) *ra = y;
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < (ROWS + 7) / 8; ++k) {
            const int rq = warp + 8 * k, row = min(rq, ROWS - 1), i = row % GQA;
            const float tau = tau_of(row / GQA);
            const float4 a = reinterpret_cast<const float4*>(sm + L::Q + row * D)[lane];
            const float4 s = reinterpret_cast<const float4*>(sm + L::QS + i * D)[lane];
            float x[4] = {a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w};
            const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                x[e] = bf16r(x[e] * f);
                if (p.log_scaling) x[e] = bf16r(x[e] * tau);
            }
            if (rq < ROWS) {  // exact bf16 q for the tensor cores (scale applied to the scores)
                __nv_bfloat16* qb = reinterpret_cast<__nv_bfloat16*>(sm + L::QB);
                reinterpret_cast<uint2*>(qb + row * D + (((lane >> 1) ^ (row & 7)) * 8) + (lane & 1) * 4)[0] =
                    make_uint2(pack2(x[0], x[1]), pack2(x[2], x[3]));
            }
        }
#pragma unroll
        for (int k = 0; k < (ROWS + 15) / 16; ++k) {
            const int rr = warp * 2 + half + 16 * k, row = min(rr, ROWS - 1), i = row % GQA;
            const float r = sm[L::REL + row * DREL + c] * sm[L::RS + i * DREL + c];
            float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
            if (p.log_scaling) y = bf16r(y * tau_of(row / GQA));
            if (rr < ROWS) sm[L::RELD + row * DREL + c] = y;
        }
    }
    if constexpr (M > 1) {  // padding rows: zero q and rel
        for (int i = tid; i < (L::NRP - ROWS) * D / 2; i += NT) sm[L::QB + ROWS * D / 2 + i] = 0.f;
        for (int i = tid; i < (L::NRP - ROWS) * DREL; i += NT) sm[L::RELD + ROWS * DREL + i] = 0.f;
    }
    __syncthreads();
    float qv[GQA][8], relv[GQA];
    if constexpr (M == 1) {
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            const float4* qa = reinterpret_cast<const float4*>(sm + L::Q + (tok * GQA + i) * D + c * 8);
            const float4 a0 = qa[0], a1 = qa[1];
            qv[i][0] = a0.x, qv[i][1] = a0.y, qv[i][2] = a0.z, qv[i][3] = a0.w;
            qv[i][4] = a1.x, qv[i][5] = a1.y, qv[i][6] = a1.z, qv[i][7] = a1.w;
            relv[i] = sm[L::REL + (tok * GQA + i) * DREL + c];
        }
    }

    // ---- owner: new keys / values of this KV head
    if (owner) {
        float* nkf = sm + L::NKF;  // [2][M][D]
        for (int idx = tid; idx < 2 * D; idx += NT) {
            const int kv = idx / D, ch = idx % D;
            float e[3 + M];  // [state, x_0..x_{M-1}]
#pragma unroll
            for (int j = 0; j < 3; ++j) e[j] = sm[L::ST + (j * 2 + kv) * D + ch];
#pragma unroll
            for (int m = 0; m < M; ++m) e[3 + m] = sm[L::X + (m * 2 + kv) * D + ch];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float y;
                if constexpr (FULL) {  // dynamic conv, weights by lag (0 = current), / sqrt(9)
                    float o = 0.f;
#pragma unroll
                    for (int lag = 0; lag < 4; ++lag) {
                        const float w = __fadd_rn(sm[L::W + (kv * 4 + lag) * D + ch], sm[L::DYN + (m * 2 + kv) * 4 + lag]);
                        o = __fadd_rn(o, __fmul_rn(w, e[m + 3 - lag]));
                    }
                    y = __fmul_rn(__fadd_rn(o, e[m + 3]), INV_3);
                } else {  // static conv, conv1d order (tap 3 = current), / sqrt(5)
                    const float* w = sm + L::W + (kv * D + ch) * 4;
                    float o = __fmul_rn(w[0], e[m]);
#pragma unroll
                    for (int j = 1; j < 4; ++j) o = __fadd_rn(o, __fmul_rn(w[j], e[m + j]));
                    y = __fmul_rn(__fadd_rn(o, e[m + 3]), INV_SQRT5);
                }
                nkf[(kv * M + m) * D + ch] = bf16r(y);
            }
#pragma unroll
            for (int i = 0; i < 3; ++i) sm[L::NST + (i * 2 + kv) * D + ch] = e[M + i];
        }
        __syncthreads();
        for (int pr = warp; pr < 2 * M; pr += NT / 32) {
            const int kv = pr / M, m = pr % M;
            const float4 x = reinterpret_cast<const float4*>(nkf + pr * D)[lane];
            const float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
            const float f = kv == 0 ? clip_factor(ss, CLIP_QK) : rsqrtf(ss / float(D) + 1e-6f);
            float y[4] = {x.x * f, x.y * f, x.z * f, x.w * f};
            if constexpr (F8) {  // the values the cache will hold: BF16, then E4M3 (exact in BF16)
#pragma unroll
                for (int e = 0; e < 4; ++e) y[e] = e4m3r(bf16r(y[e]));
            }
            uint2 v;
            if constexpr (Q4) {
#pragma unroll
                for (int e = 0; e < 4; ++e) y[e] = bf16r(y[e]);
                __half scale;
                const uint16_t bits = q4_g64::pack4(y, scale);
                new_q4[pr * (D / 4) + lane] = bits;
                if ((lane & 15) == 0) new_scale[pr * q4_g64::groups + (lane >> 4)] = scale;
                const float f = __half2float(scale);
                v = kv ? make_uint2(q4_g64::expand_pair<true>(bits, 0, f), q4_g64::expand_pair<true>(bits, 2, f))
                       : make_uint2(q4_g64::expand_pair<false>(bits, 0, f), q4_g64::expand_pair<false>(bits, 2, f));
            } else {
                v = make_uint2(pack2(y[0], y[1]), pack2(y[2], y[3]));
            }
            reinterpret_cast<uint2*>(&s_nkv[kv][m][0])[lane] = v;
        }
        __syncthreads();
    }

    // ---- attention over this block's keys
    float mrun[GQA], lrun[GQA], acc[GQA][8];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
        mrun[i] = NEG_INIT, lrun[i] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
    }
    auto compute = [&](const auto& b, const bool is_new) {  // Buf<uint4> (BF16) or Buf<uint2> (E4M3)
        float s[U][GQA];
        bool valid[U], need[U];
        int dist[U];
        bool any_need = false;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kp = b.kp[u], d = P - kp;
            // kp = -1 marks a missing key; old entries need kp < p0 (and the window on sliding layers)
            const bool ok = kp >= 0 && d >= 0 && (is_new || (kp < p0 && (FULL || d < WIN)));
            valid[u] = ok;
            need[u] = ok && d < EXT;
            dist[u] = d;
            any_need |= need[u];
            float kf[8];
            unpack8(b.k[u], kf);
#pragma unroll
            for (int i = 0; i < GQA; ++i) {
                float t = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) t = fmaf(qv[i][e], kf[e], t);
                s[u][i] = hsum16(t);
            }
        }
        if (__any_sync(0xffffffffu, any_need)) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                float rp = b.rp[u];
                if (need[u] && dist[u] != b.dp[u])
                    rp = is_new ? sm[L::RELP0 + c * 8 + dist[u]] : __ldg(p.rel_proj + c * EXT + dist[u]);
#pragma unroll
                for (int i = 0; i < GQA; ++i) {
                    const float bias = hsum16(relv[i] * rp);
                    if (need[u]) s[u][i] = fmaf(bf16r(bias), LOG2E, s[u][i]);  // one rounding (as M > 1)
                }
            }
        }
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            float mx = mrun[i];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (!valid[u]) s[u][i] = -INFINITY;
                mx = fmaxf(mx, s[u][i]);
            }
            const float sc = exp2f(mrun[i] - mx);
            mrun[i] = mx;
            lrun[i] *= sc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][e] *= sc;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            float vf[8];
            // a masked key's value may be anything (an unused cache slot): keep NaN/Inf out of 0 * v
            using V = std::decay_t<decltype(b.v[0])>;
            if constexpr (Q4) {
                const uint4 bits = valid[u] ? b.v[u] : V{};
                const uint32_t pairs[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    vf[2 * i] = __half2float(__ushort_as_half(static_cast<uint16_t>(pairs[i])));
                    vf[2 * i + 1] = __half2float(__ushort_as_half(static_cast<uint16_t>(pairs[i] >> 16)));
                }
            } else {
                unpack8(valid[u] ? b.v[u] : V{}, vf);
            }
#pragma unroll
            for (int i = 0; i < GQA; ++i) {
                const float pu = exp2f(s[u][i] - mrun[i]);
                lrun[i] += pu;
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(pu, vf[e], acc[i][e]);
            }
        }
    };

    if constexpr (M == 1) {
        if (owner) {  // the call's own key first, from shared memory, while the K/V stream is in flight
            Buf<uint4> N;  // BF16 (holding E4M3 values when F8)
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int m2 = (kw * U + u) * 2 + half;
                N.k[u] = N.v[u] = make_uint4(0u, 0u, 0u, 0u);
                N.kp[u] = -1, N.dp[u] = -1, N.rp[u] = 0.f;
                if (m2 < M) {
                    N.k[u] = *reinterpret_cast<const uint4*>(&s_nkv[0][m2][c * 8]);
                    N.v[u] = *reinterpret_cast<const uint4*>(&s_nkv[1][m2][c * 8]);
                    N.kp[u] = p0 + m2;
                }
            }
            compute(N, true);
        }
        for (int base = c0; base < old_end; base += STRIDE) {
            compute(A, false);
            if (base + STRIDE < old_end) load(A, base + STRIDE);
            if (base + STEP < old_end) {
                compute(B, false);
                if (base + STRIDE + STEP < old_end) load(B, base + STRIDE + STEP);
            }
        }

        // ---- block partial: merge the two half-warps, then the 8 warps
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            const float mo = __shfl_xor_sync(0xffffffffu, mrun[i], 16);
            const float lo = __shfl_xor_sync(0xffffffffu, lrun[i], 16);
            const float mx = fmaxf(mrun[i], mo);
            const float f1 = exp2f(mrun[i] - mx), f2 = exp2f(mo - mx);
            lrun[i] = lrun[i] * f1 + lo * f2;
#pragma unroll
            for (int e = 0; e < 8; ++e)
                acc[i][e] = acc[i][e] * f1 + __shfl_xor_sync(0xffffffffu, acc[i][e], 16) * f2;
            mrun[i] = mx;
        }
        float* red_o = sm + L::RED_O;
        float* red_ml = sm + L::RED_ML;
        __syncthreads();  // the staged inputs (aliased by red_*) are consumed
        if (half == 0) {
#pragma unroll
            for (int i = 0; i < GQA; ++i) {
                float4* dst = reinterpret_cast<float4*>(red_o + (warp * GQA + i) * D + c * 8);
                dst[0] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                dst[1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
                if (c == 0) red_ml[(warp * GQA + i) * 2] = mrun[i], red_ml[(warp * GQA + i) * 2 + 1] = lrun[i];
            }
        }
        __syncthreads();
        // this block's partial (unnormalised o, then max and sum per row) to the workspace (L2)
        float* part = p.part + size_t(g * CS + rank) * ROWS * PSTR;
        for (int idx = tid; idx < ROWS * (D / 4); idx += NT) {
            const int row = idx / (D / 4), d4 = idx % (D / 4);
            float mx = NEG_INIT;
#pragma unroll
            for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_ml[(k * GQA + row) * 2]);
            float l = 0.f;
            float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int k = 0; k < KW; ++k) {
                const int w = k * GQA + row;
                const float f = exp2f(red_ml[w * 2] - mx);
                l += red_ml[w * 2 + 1] * f;
                const float4 a = reinterpret_cast<const float4*>(red_o + w * D)[d4];
                o.x += a.x * f, o.y += a.y * f, o.z += a.z * f, o.w += a.w * f;
            }
            __stcg(reinterpret_cast<float4*>(part + row * PSTR) + d4, o);
            if (d4 == 0) __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(mx, l));
        }
    } else {
        // ---- M > 1 on tensor cores. Rows r = 3 token + head (NRP padded), tiles of KT keys loaded once
        // per block with cp.async (double-buffered, round robin over the cluster, with kpos and the
        // rel_proj window of the distances a full in-order ring / the linear cache implies).
        //   S^T (keys x rows) = K Q^T: mma m16n8k16 bf16 (exact bf16 q and k, fp32 sums), items (16 keys,
        //   8 rows) over the warps; mask, relative bias and the tile row max (shared memory).
        //   P = exp2(S - m) as two fp16 terms hi + lo (P - hi), so P V carries ~22 bits of P; V converted
        //   to fp16 in place (exact for these values: |v| <= sqrt(128), tiny ones lose < 3e-8).
        //   O^T (dims x rows) += V^T P^T: mma m16n8k16 fp16, warp w owns dims 16w..16w+15 of every row.
        // bf16 / e4m3 values -> fp16; anything not finite and in range (only possible in masked, unused slots) -> 0
        auto half2_of = [](float a, float b) {
            a = fabsf(a) <= 65504.f ? a : 0.f;
            b = fabsf(b) <= 65504.f ? b : 0.f;
            const __half2 h = __floats2half2_rn(a, b);
            return *reinterpret_cast<const uint32_t*>(&h);
        };
        auto to_half = [&](uint32_t w) { return half2_of(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
        // V rows of stage st to fp16: each thread converts the chunks it copied (its cp.async are done)
        auto convert_v = [&](const int st) {
            for (int i = tid; i < KT * 2 * (D / 8); i += NT) {
                const int key = i / (2 * (D / 8)), kv = (i / (D / 8)) % 2, ch = i % (D / 8);
                if (kv) {
                    uint4* q = reinterpret_cast<uint4*>(kvs + kv_off(st, key, 1, ch));
                    const uint4 u = *q;
                    *q = make_uint4(to_half(u.x), to_half(u.y), to_half(u.z), to_half(u.w));
                }
            }
        };
        // 16-bit offset of 16-byte chunk ch8 of row `key` in a [rows][D] tile, chunks XOR-swizzled by key
        auto roff = [](int key, int ch8) { return key * D + ((ch8 ^ (key & 7)) * 8); };
        // F8: the V rows of raw slot `slot` to fp16 in wv (exact for E4M3 values)
        auto expand_v = [&](const int slot) {
            constexpr int NCH = KT * (D / 16) / NT;  // 16-byte E4M3 chunks per thread
            static_assert(NCH * NT == KT * (D / 16), "chunks must divide over the block");
#pragma unroll
            for (int n = 0; n < NCH; ++n) {
                const int i = tid + n * NT, key = i / (D / 16), ch = i % (D / 16);
                const uint4 r = *reinterpret_cast<const uint4*>(raw + slot * RAWB + KT * D + key * D + ch * 16);
                const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
                for (int h = 0; h < 2; ++h) {  // dims 16 ch + 8 h .. + 7
                    uint32_t o[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = e4m3x2_f2((w[2 * h + e / 2] >> (16 * (e % 2))) & 0xffffu);
                        o[e] = half2_of(f.x, f.y);
                    }
                    *reinterpret_cast<uint4*>(wv + roff(key, 2 * ch + h)) = make_uint4(o[0], o[1], o[2], o[3]);
                }
            }
        };

        // this thread's output rows n*8 + 2 t4 (+1), dims 16 warp + gid (+8): accumulators and softmax state
        // (every warp tracks the same row statistics, so no shared running max is needed)
        float oacc[NTN][4], mrow[NTN][2], lrow[NTN][2];
#pragma unroll
        for (int n = 0; n < NTN; ++n) {
#pragma unroll
            for (int v = 0; v < 4; ++v) oacc[n][v] = 0.f;
            mrow[n][0] = mrow[n][1] = NEG_INIT, lrow[n][0] = lrow[n][1] = 0.f;
        }
        float2* itm = reinterpret_cast<float2*>(sm + L::ITM);  // per (key m-tile, row): item max, item sum

        // One tile: [S items (16 keys x 8 rows) over the warps, item max, P relative to it, item sums] |
        // barrier, mid() | [P V, rescaled per item]. The caller's barrier before the next tile protects P,
        // itm and the tile. K comes from kaddr(key, chunk) (16-bit rows, ldmatrix) or, kraw, straight from a
        // raw E4M3 tile (4 bytes per row and k-step, widened in registers; the k order inside an mma is
        // permuted, and q is read with the same permutation); V from vaddr(key, chunk) (fp16 rows).
        auto consume = [&](auto kaddr, auto vaddr, auto kraw_t, const char* kraw, const int st, const int base,
                           const bool is_new, auto&& mid) {
            constexpr bool KRAW = decltype(kraw_t)::value;
            const int nk = is_new ? 1 : KM;  // key m-tiles in use (the call's own keys fit in one)
            // items (key m-tile km, 8-row tile nn): warp w takes km = w % nk and the row tiles nn = w / nk +
            // j NW / nk, loading and widening each K fragment once for all of them
            constexpr int NW = NT / 32, NPW = (NTN + NW / KM - 1) / (NW / KM);
            const int km = warp % nk, nn0 = warp / nk, nstep = NW / nk;
            float a4[NPW][4];
#pragma unroll
            for (int j = 0; j < NPW; ++j) a4[j][0] = a4[j][1] = a4[j][2] = a4[j][3] = 0.f;
            if (nn0 < NTN) {
#pragma unroll
                for (int ks = 0; ks < D / 16; ++ks) {
                    uint32_t af[4];
                    if constexpr (KRAW) {  // a0/a2: dims 4 t4 .. + 3 of key gid; a1/a3: key gid + 8
                        const int key = km * 16 + gid;
                        const char* kr = kraw + key * D + ((ks ^ (key & 7)) * 16) + 4 * t4;
                        const uint32_t u0 = *reinterpret_cast<const uint32_t*>(kr);
                        const uint32_t u1 = *reinterpret_cast<const uint32_t*>(kr + 8 * D);
                        const float2 f0 = e4m3x2_f2(u0 & 0xffffu), f1 = e4m3x2_f2(u1 & 0xffffu);
                        const float2 f2 = e4m3x2_f2(u0 >> 16), f3 = e4m3x2_f2(u1 >> 16);
                        af[0] = pack2(f0.x, f0.y), af[1] = pack2(f1.x, f1.y);
                        af[2] = pack2(f2.x, f2.y), af[3] = pack2(f3.x, f3.y);
                    } else {
                        const int q = lane >> 3, key = km * 16 + (q & 1) * 8 + (lane & 7);
                        ldsm_x4(smem_u32(kaddr(key, ks * 2 + (q >> 1))), af);
                    }
#pragma unroll
                    for (int j = 0; j < NPW; ++j) {
                        const int nn = nn0 + j * nstep;
                        if (nn < NTN) {
                            uint32_t bfr[2];
                            if constexpr (KRAW) {  // the same k permutation as af
                                const int row = nn * 8 + gid, ch = 2 * ks + (t4 >> 1);
                                const uint2 q2 =
                                    *reinterpret_cast<const uint2*>(qb + row * D + ((ch ^ (row & 7)) * 8) + (t4 & 1) * 4);
                                bfr[0] = q2.x, bfr[1] = q2.y;
                            } else {
                                const int row = nn * 8 + (lane & 7), ch = ks * 2 + ((lane >> 3) & 1);
                                ldsm_x2(smem_u32(qb + row * D + ((ch ^ (row & 7)) * 8)), bfr);
                            }
                            mma_bf16(a4[j], af, bfr);
                        }
                    }
                }
            }
#pragma unroll
            for (int jt = 0; jt < NPW; ++jt) {
                const int nn = nn0 + jt * nstep, r0 = nn * 8 + 2 * t4;
                if (nn >= NTN) break;
                const int dbase = dbase_of(base);
                const bool staged = !F8 && !is_new && dbase < EXT;  // E4M3 tiles stage no rel_proj window
                float sv[4];
#pragma unroll
                for (int v = 0; v < 4; ++v) {  // (key gid | gid + 8, row 2 t4 | 2 t4 + 1)
                    const int key = km * 16 + gid + (v >> 1) * 8, row = r0 + (v & 1), t = row / GQA, j = base + key;
                    const int kp = is_new ? (key < M ? p0 + key : -1) : (j < L_old ? (FULL ? j : kps[st * KT + key]) : -1);
                    const int d = p0 + t - kp;
                    const bool ok = row < ROWS && kp >= 0 && d >= 0 && (is_new || (kp < p0 && (FULL || d < WIN)));
                    float s = a4[jt][v] * QSCALE;
                    if (ok && d < EXT) {  // bias bf16(rel[row] . rel_proj[:, d])
                        const float* rl = sm + L::RELD + row * DREL;
                        const int e = d - dbase;
                        float b = 0.f;
                        if (is_new) {
#pragma unroll
                            for (int k = 0; k < DREL; ++k) b = fmaf(rl[k], sm[L::RELP0 + k * 8 + d], b);
                        } else if (staged && e >= 0 && e < NE) {
#pragma unroll
                            for (int k = 0; k < DREL; ++k) b = fmaf(rl[k], srp[(st * DREL + k) * (KT + MAXM) + e], b);
                        } else {  // ring wrap point / out-of-order ring / E4M3 tile: read the column
#pragma unroll
                            for (int k = 0; k < DREL; ++k) b = fmaf(rl[k], __ldg(p.rel_proj + k * EXT + d), b);
                        }
                        s = fmaf(bf16r(b), LOG2E, s);  // one rounding at every M (as M = 1)
                    }
                    sv[v] = ok ? s : -INFINITY;
                }
                float mx0 = fmaxf(sv[0], sv[2]), mx1 = fmaxf(sv[1], sv[3]);
#pragma unroll
                for (int o = 4; o < 32; o <<= 1) {
                    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
                    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
                }
                mx0 = fmaxf(mx0, NEG_INIT), mx1 = fmaxf(mx1, NEG_INIT);  // fully masked item: P = 0, not NaN
                float s0 = 0.f, s1 = 0.f;
#pragma unroll
                for (int v = 0; v < 4; ++v) {  // P as fp16 hi + lo
                    const int key = km * 16 + gid + (v >> 1) * 8, row = r0 + (v & 1);
                    const float pf = exp2f(sv[v] - ((v & 1) ? mx1 : mx0));
                    const __half h = __float2half_rn(pf);
                    ps[row * PSTRIDE + key] = h;
                    ps[(NRP + row) * PSTRIDE + key] = __float2half_rn(pf - __half2float(h));
                    ((v & 1) ? s1 : s0) += pf;
                }
#pragma unroll
                for (int o = 4; o < 32; o <<= 1) {
                    s0 += __shfl_xor_sync(0xffffffffu, s0, o);
                    s1 += __shfl_xor_sync(0xffffffffu, s1, o);
                }
                if (gid == 0) itm[km * NRP + r0] = make_float2(mx0, s0), itm[km * NRP + r0 + 1] = make_float2(mx1, s1);
            }
            __syncthreads();
            mid();
            float f[KM][NTN][2];  // per item weight exp2(item max - new max) of each of this thread's rows
#pragma unroll
            for (int n = 0; n < NTN; ++n) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = n * 8 + 2 * t4 + h;
                    float2 it[KM];
                    float mn = mrow[n][h];
#pragma unroll
                    for (int k = 0; k < KM; ++k) {
                        it[k] = k < nk ? itm[k * NRP + r] : make_float2(NEG_INIT, 0.f);
                        mn = fmaxf(mn, it[k].x);
                    }
                    const float sc = exp2f(mrow[n][h] - mn);
                    float l = lrow[n][h] * sc;
#pragma unroll
                    for (int k = 0; k < KM; ++k) {
                        f[k][n][h] = exp2f(it[k].x - mn);
                        l += it[k].y * f[k][n][h];
                    }
                    oacc[n][h] *= sc, oacc[n][h + 2] *= sc;
                    mrow[n][h] = mn, lrow[n][h] = l;
                }
            }
#pragma unroll
            for (int ks = 0; ks < KM; ++ks) {
                if (ks >= nk) break;
                uint32_t af[4];
                {
                    const int q = lane >> 3, key = ks * 16 + (q >> 1) * 8 + (lane & 7);
                    ldsm_x4_t(smem_u32(vaddr(key, warp * 2 + (q & 1))), af);
                }
#pragma unroll
                for (int n = 0; n < NTN; ++n) {
                    uint32_t bfr[2];
                    float t[4] = {0.f, 0.f, 0.f, 0.f};
                    const __half* pr = ps + (n * 8 + (lane & 7)) * PSTRIDE + ks * 16 + ((lane >> 3) & 1) * 8;
                    ldsm_x2(smem_u32(pr), bfr);
                    mma_f16(t, af, bfr);
                    ldsm_x2(smem_u32(pr + NRP * PSTRIDE), bfr);
                    mma_f16(t, af, bfr);
#pragma unroll
                    for (int v = 0; v < 4; ++v) oacc[n][v] = fmaf(f[ks][n][v & 1], t[v], oacc[n][v]);
                }
            }
        };
        auto nothing = [] {};
        using RawK = std::true_type;
        using SmemK = std::false_type;

        // the call's own keys first (stage 1 of kvs; for E4M3 caches V rows 0..15 and K rows 32..47 of wv),
        // while the first tiles are in flight
        if (owner) {
            for (int i = tid; i < 16 * 2 * (D / 8); i += NT) {
                const int key = i / (2 * (D / 8)), kv = (i / (D / 8)) % 2, ch = i % (D / 8);
                uint4 u = make_uint4(0u, 0u, 0u, 0u);
                if (key < M) u = reinterpret_cast<const uint4*>(&s_nkv[kv][key][0])[ch];
                if (!Q4 && kv) u = make_uint4(to_half(u.x), to_half(u.y), to_half(u.z), to_half(u.w));
                uint4* dst = F8 ? reinterpret_cast<uint4*>(wv + (kv ? 0 : 32 * D) + roff(key, ch))
                                : reinterpret_cast<uint4*>(kvs + kv_off(1, key, kv, ch));
                *dst = u;
            }
            __syncthreads();
            if constexpr (F8)
                consume([&](int key, int ch) { return reinterpret_cast<const __nv_bfloat16*>(wv + 32 * D + roff(key, ch)); },
                        [&](int key, int ch) { return wv + roff(key, ch); }, SmemK{}, nullptr, 0, 0, true, nothing);
            else
                consume([&](int key, int ch) { return kvs + kv_off(1, key, 0, ch); },
                        [&](int key, int ch) { return kvs + kv_off(1, key, 1, ch); }, SmemK{}, nullptr, 1, 0, true, nothing);
        }
        if constexpr (F8) {
            int t = 0;
            for (int base = rank * KT; base < L_old; base += CS * KT, ++t) {
                asm volatile("cp.async.wait_group %0;" ::"n"(NRAW - 1) : "memory");  // this thread's part of tile t
                __syncthreads();  // tile t landed for everyone; the previous tile's P V is done with wv
                expand_v(t % NRAW);
                __syncthreads();
                consume([&](int key, int ch) { return kvs + kv_off(0, key, 0, ch); },  // (unused: K is raw)
                        [&](int key, int ch) { return wv + roff(key, ch); }, RawK{}, raw + (t % NRAW) * RAWB, 0, base,
                        false, [&] { issue_raw(t + NRAW); });  // after the S items, this tile's slot is free
            }
        } else {
            int st = 0;
            for (int base = rank * KT; base < L_old; base += CS * KT, st ^= 1) {
                asm volatile("cp.async.wait_group 0;" ::: "memory");  // tile `base` (the only one in flight)
                if constexpr (!Q4) convert_v(st);
                __syncthreads();  // tile landed for everyone; the previous tile's P V is done
                if (base + CS * KT < L_old) issue(st ^ 1, base + CS * KT);
                consume([&](int key, int ch) { return kvs + kv_off(st, key, 0, ch); },
                        [&](int key, int ch) { return kvs + kv_off(st, key, 1, ch); }, SmemK{}, nullptr, st, base, false,
                        nothing);
            }
        }

        // this block's partial (unnormalised o, then max and sum per row) to the workspace (L2)
        float* part = p.part + size_t(g * CS + rank) * ROWS * PSTR;
#pragma unroll
        for (int n = 0; n < NTN; ++n) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int row = n * 8 + 2 * t4 + (v & 1), dim = warp * 16 + gid + (v >> 1) * 8;
                if (row < ROWS) __stcg(part + row * PSTR + dim, oacc[n][v]);
            }
            if (warp == 0 && gid == 0) {
#pragma unroll
                for (int v = 0; v < 2; ++v) {
                    const int row = n * 8 + 2 * t4 + v;
                    if (row < ROWS) __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(mrow[n][v], lrow[n][v]));
                }
            }
        }
    }
    // every partial of the head is written, and every block of the head is done reading the ring
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (!FULL && M > 1 && owner && tid == NT - 1) {  // kpos is read by every head: the last head writes it
        __threadfence();
        if (atomicAdd(p.counter, 1) == HKV - 1) {
            __threadfence();
            for (int m = 0; m < M; ++m) p.kpos[(p0 + m) & (WIN - 1)] = p0 + m;
            *p.counter = 0;
        }
    }

    // Each block outputs its OD dims of every row.
    for (int i = tid; i < ROWS * OD; i += NT) {
        const int row = i / OD, dim = rank * OD + i % OD;
        const float* src = p.part + size_t(g * CS) * ROWS * PSTR + row * PSTR;
        float2 ml[CS];
        float o[CS];
#pragma unroll
        for (int q = 0; q < CS; ++q) {
            ml[q] = __ldcg(reinterpret_cast<const float2*>(src + q * ROWS * PSTR + D));
            o[q] = __ldcg(src + q * ROWS * PSTR + dim);
        }
        const float sink = sm[L::SINK + row] * LOG2E;
        float mx = sink;
#pragma unroll
        for (int q = 0; q < CS; ++q) mx = fmaxf(mx, ml[q].x);
        float l = exp2f(sink - mx), a = 0.f;
#pragma unroll
        for (int q = 0; q < CS; ++q) {
            const float f = exp2f(ml[q].x - mx);
            l += ml[q].y * f;
            a += o[q] * f;
        }
        const float gate = 1.f / (1.f + expf(-sm[L::GATE + i]));
        p.out[size_t(row / GQA) * HQ * D + (g * GQA + row % GQA) * D + dim] = __float2bfloat16_rn(bf16r(a / l) * gate);
    }

    // owner: the global writes, at the end so they delay nothing (and, for M > 1 on a ring, because the
    // overwritten slots were still being read)
    if (owner) {
        for (int idx = tid; idx < 2 * M * 16; idx += NT) {  // new K/V rows
            const int kv = idx / (M * 16), m = (idx / 16) % M, c8 = idx % 16;
            const int row = FULL ? p0 + m : (p0 + m) & (WIN - 1);
            const size_t head = size_t(row) * HKV + g;
            if constexpr (Q4) {
                auto* dst = static_cast<uint32_t*>(kv ? p.v_cache : p.k_cache);
                auto* scales = static_cast<__half*>(kv ? p.v_scale : p.k_scale);
                const int pr = kv * M + m;
                dst[head * (D / 8) + c8] = reinterpret_cast<const uint32_t*>(new_q4)[pr * (D / 8) + c8];
                if (c8 < q4_g64::groups)
                    scales[head * q4_g64::groups + c8] = new_scale[pr * q4_g64::groups + c8];
            } else if constexpr (F8) {  // exact: the values are E4M3 already
                float f[8];
                unpack8(reinterpret_cast<const uint4*>(&s_nkv[kv][m][0])[c8], f);
                auto* dst = reinterpret_cast<uint2*>((kv ? vc : kc) + head * D);
                dst[c8] = make_uint2(pack_e4m3x2(f[0], f[1]) | pack_e4m3x2(f[2], f[3]) << 16,
                                                               pack_e4m3x2(f[4], f[5]) | pack_e4m3x2(f[6], f[7]) << 16);
            } else {
                auto* dst = reinterpret_cast<uint4*>((kv ? vc : kc) + head * D);
                dst[c8] = reinterpret_cast<const uint4*>(&s_nkv[kv][m][0])[c8];
            }
        }
        for (int idx = tid; idx < 3 * 2 * D; idx += NT) {  // conv states of this head's channels
            const int i = idx / (2 * D), kv = (idx / D) % 2, ch = idx % D;
            (kv ? p.conv_v : p.conv_k)[i * CONV_C + g * D + ch] = sm[L::NST + idx];
        }
        if (!FULL && M == 1 && g == 0 && tid == 0) p.kpos[p0 & (WIN - 1)] = p0;
    }
