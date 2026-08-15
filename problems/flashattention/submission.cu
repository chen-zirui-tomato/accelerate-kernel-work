#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <stdexcept>

#define HEAD_DIM 128
#define WARPS_PER_BLOCK 8
#define BLOCK_N 32
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ float warp_sum(float v) {
  unsigned mask = 0xffffffffu;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return __shfl_sync(mask, v, 0);
}

__global__ void warp_row_flash_attention_kernel(
    const at::Half* __restrict__ Q, const at::Half* __restrict__ K,
    const at::Half* __restrict__ V, at::Half* __restrict__ O,
    const int total_rows, const int seq_len, const float scale) {
  __shared__ __align__(16) at::Half k_s[BLOCK_N * HEAD_DIM];
  __shared__ __align__(16) at::Half v_s[BLOCK_N * HEAD_DIM];

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row_in_seq = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;
  const int q_row = bh * seq_len + row_in_seq;
  const bool valid_q = row_in_seq < seq_len && q_row < total_rows;
  const int base = bh * seq_len * HEAD_DIM;

  float q_frag[4];
  #pragma unroll
  for (int t = 0; t < 4; ++t) {
    int d = lane + t * 32;
    q_frag[t] = valid_q ? static_cast<float>(Q[base + row_in_seq * HEAD_DIM + d])
                        : 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  float acc_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {
    const int valid_n = min(BLOCK_N, seq_len - kv_start);
    const int tile_elems = valid_n * HEAD_DIM;
    const int vec_elems = tile_elems / 8;

    const int4* k_src = reinterpret_cast<const int4*>(K + base + kv_start * HEAD_DIM);
    const int4* v_src = reinterpret_cast<const int4*>(V + base + kv_start * HEAD_DIM);
    int4* k_dst = reinterpret_cast<int4*>(k_s);
    int4* v_dst = reinterpret_cast<int4*>(v_s);

    for (int idx = threadIdx.x; idx < vec_elems; idx += blockDim.x) {
      k_dst[idx] = k_src[idx];
      v_dst[idx] = v_src[idx];
    }

    __syncthreads();

    if (valid_q) {
      for (int j = 0; j < valid_n; ++j) {
        float partial = 0.0f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
          int d = lane + t * 32;
          partial += q_frag[t] * static_cast<float>(k_s[j * HEAD_DIM + d]);
        }

        float score = warp_sum(partial) * scale;
        float m_new = fmaxf(m, score);
        float alpha = exp2f((m - m_new) * LOG2E);
        float beta = exp2f((score - m_new) * LOG2E);

        l = l * alpha + beta;

        #pragma unroll
        for (int t = 0; t < 4; ++t) {
          int d = lane + t * 32;
          acc_frag[t] =
              acc_frag[t] * alpha + beta * static_cast<float>(v_s[j * HEAD_DIM + d]);
        }

        m = m_new;
      }
    }

    __syncthreads();
  }

  if (valid_q) {
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      int d = lane + t * 32;
      O[base + row_in_seq * HEAD_DIM + d] = static_cast<at::Half>(acc_frag[t] / l);
    }
  }
}

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  if (!Q.is_cuda() || !K.is_cuda() || !V.is_cuda()) {
    throw std::runtime_error("Q, K, and V must be CUDA tensors");
  }
  if (Q.dim() != 4 || K.dim() != 4 || V.dim() != 4) {
    throw std::runtime_error("Q, K, and V must have shape [B, H, S, D]");
  }
  if (Q.scalar_type() != torch::kHalf || K.scalar_type() != torch::kHalf ||
      V.scalar_type() != torch::kHalf) {
    throw std::runtime_error("This kernel requires FP16 tensors");
  }
  if (Q.size(3) != HEAD_DIM) {
    throw std::runtime_error("This kernel requires head_dim == 128");
  }

  auto O = torch::empty_like(Q);

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int total_rows = batch_size * num_heads * seq_len;
  const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

  dim3 blocks((seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, // 一个warp写一行
              batch_size * num_heads);
  dim3 threads(WARPS_PER_BLOCK * 32);

  warp_row_flash_attention_kernel<<<blocks, threads>>>(
      Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(), V.data_ptr<at::Half>(),
      O.data_ptr<at::Half>(), total_rows, seq_len, scale);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return O;
}
