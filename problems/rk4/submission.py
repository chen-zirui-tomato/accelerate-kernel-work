import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

constexpr int kBlockSize = 256;

__device__ __forceinline__ bool is_interior_point(
    int x,
    int y,
    int z,
    int Nx,
    int Ny,
    int Nz
) {
    return x >= 4 && x < Nx - 4 &&
           y >= 4 && y < Ny - 4 &&
           z >= 4 && z < Nz - 4;
}

__device__ __forceinline__ float load_stage_value(
    const float* __restrict__ u,
    const float* __restrict__ k,
    int offset,
    int x,
    int y,
    int z,
    int Nx,
    int Ny,
    int Nz,
    float scale
) {
    const float base = u[offset];
    if (scale == 0.0f || !is_interior_point(x, y, z, Nx, Ny, Nz)) {
        return base;
    }
    return base + scale * k[offset];
}

__device__ __forceinline__ float load_stage_value_fast(
    const float* __restrict__ u,
    const float* __restrict__ k,
    int offset,
    float scale
) {
    return u[offset] + scale * k[offset];
}

__global__ void lap_u_cuda_kernel(
    const float* __restrict__ u,
    float* __restrict__ out,
    float* __restrict__ acc,
    int Nx,
    int Ny,
    int Nz,
    float alpha,
    float inv_hx2,
    float inv_hy2,
    float inv_hz2
) {
    const int interior_X = Nx - 8;
    const int interior_Y = Ny - 8;
    const int interior_Z = Nz - 8;
    const int total = interior_X * interior_Y * interior_Z;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int x = idx % interior_X + 4;
    const int y = (idx / interior_X) % interior_Y + 4;
    const int z = idx / (interior_X * interior_Y) + 4;

    constexpr float c0 = -205.0f / 72.0f;
    constexpr float c1 =    8.0f /  5.0f;
    constexpr float c2 =   -1.0f /  5.0f;
    constexpr float c3 =    8.0f / 315.0f;
    constexpr float c4 =   -1.0f / 560.0f;

    const int stride_y = Nx;
    const int stride_z = Ny * Nx;
    const int offset = z * stride_z + y * stride_y + x;
    const float uc = u[offset];

    const float u_xx = (
        c0 * uc +
        c1 * (u[offset + 1] + u[offset - 1]) +
        c2 * (u[offset + 2] + u[offset - 2]) +
        c3 * (u[offset + 3] + u[offset - 3]) +
        c4 * (u[offset + 4] + u[offset - 4])
    ) * inv_hx2;

    const float u_yy = (
        c0 * uc +
        c1 * (u[offset + stride_y] + u[offset - stride_y]) +
        c2 * (u[offset + 2 * stride_y] + u[offset - 2 * stride_y]) +
        c3 * (u[offset + 3 * stride_y] + u[offset - 3 * stride_y]) +
        c4 * (u[offset + 4 * stride_y] + u[offset - 4 * stride_y])
    ) * inv_hy2;

    const float u_zz = (
        c0 * uc +
        c1 * (u[offset + stride_z] + u[offset - stride_z]) +
        c2 * (u[offset + 2 * stride_z] + u[offset - 2 * stride_z]) +
        c3 * (u[offset + 3 * stride_z] + u[offset - 3 * stride_z]) +
        c4 * (u[offset + 4 * stride_z] + u[offset - 4 * stride_z])
    ) * inv_hz2;

    const float result = alpha * (u_xx + u_yy + u_zz);
    out[offset] = result;
    acc[offset] = result;
}

__device__ __forceinline__ float compute_stage_laplacian(
    const float* __restrict__ u,
    const float* __restrict__ k,
    int x,
    int y,
    int z,
    int Nx,
    int Ny,
    int Nz,
    float alpha,
    float inv_hx2,
    float inv_hy2,
    float inv_hz2,
    float scale
) {
    constexpr float c0 = -205.0f / 72.0f;
    constexpr float c1 =    8.0f /  5.0f;
    constexpr float c2 =   -1.0f /  5.0f;
    constexpr float c3 =    8.0f / 315.0f;
    constexpr float c4 =   -1.0f / 560.0f;

    const int stride_y = Nx;
    const int stride_z = Ny * Nx;
    const int offset = z * stride_z + y * stride_y + x;
    const bool full_stage_interior =
        x >= 8 && x < Nx - 8 &&
        y >= 8 && y < Ny - 8 &&
        z >= 8 && z < Nz - 8;

    if (full_stage_interior) {
        const float uc = load_stage_value_fast(u, k, offset, scale);
        const float u_xx = (
            c0 * uc +
            c1 * (
                load_stage_value_fast(u, k, offset + 1, scale) +
                load_stage_value_fast(u, k, offset - 1, scale)) +
            c2 * (
                load_stage_value_fast(u, k, offset + 2, scale) +
                load_stage_value_fast(u, k, offset - 2, scale)) +
            c3 * (
                load_stage_value_fast(u, k, offset + 3, scale) +
                load_stage_value_fast(u, k, offset - 3, scale)) +
            c4 * (
                load_stage_value_fast(u, k, offset + 4, scale) +
                load_stage_value_fast(u, k, offset - 4, scale))
        ) * inv_hx2;
        const float u_yy = (
            c0 * uc +
            c1 * (
                load_stage_value_fast(u, k, offset + stride_y, scale) +
                load_stage_value_fast(u, k, offset - stride_y, scale)) +
            c2 * (
                load_stage_value_fast(u, k, offset + 2 * stride_y, scale) +
                load_stage_value_fast(u, k, offset - 2 * stride_y, scale)) +
            c3 * (
                load_stage_value_fast(u, k, offset + 3 * stride_y, scale) +
                load_stage_value_fast(u, k, offset - 3 * stride_y, scale)) +
            c4 * (
                load_stage_value_fast(u, k, offset + 4 * stride_y, scale) +
                load_stage_value_fast(u, k, offset - 4 * stride_y, scale))
        ) * inv_hy2;
        const float u_zz = (
            c0 * uc +
            c1 * (
                load_stage_value_fast(u, k, offset + stride_z, scale) +
                load_stage_value_fast(u, k, offset - stride_z, scale)) +
            c2 * (
                load_stage_value_fast(u, k, offset + 2 * stride_z, scale) +
                load_stage_value_fast(u, k, offset - 2 * stride_z, scale)) +
            c3 * (
                load_stage_value_fast(u, k, offset + 3 * stride_z, scale) +
                load_stage_value_fast(u, k, offset - 3 * stride_z, scale)) +
            c4 * (
                load_stage_value_fast(u, k, offset + 4 * stride_z, scale) +
                load_stage_value_fast(u, k, offset - 4 * stride_z, scale))
        ) * inv_hz2;

        return alpha * (u_xx + u_yy + u_zz);
    }

    const float uc = load_stage_value(u, k, offset, x, y, z, Nx, Ny, Nz, scale);
    const float u_xx = (
        c0 * uc +
        c1 * (
            load_stage_value(u, k, offset + 1, x + 1, y, z, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 1, x - 1, y, z, Nx, Ny, Nz, scale)) +
        c2 * (
            load_stage_value(u, k, offset + 2, x + 2, y, z, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 2, x - 2, y, z, Nx, Ny, Nz, scale)) +
        c3 * (
            load_stage_value(u, k, offset + 3, x + 3, y, z, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 3, x - 3, y, z, Nx, Ny, Nz, scale)) +
        c4 * (
            load_stage_value(u, k, offset + 4, x + 4, y, z, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 4, x - 4, y, z, Nx, Ny, Nz, scale))
    ) * inv_hx2;
    const float u_yy = (
        c0 * uc +
        c1 * (
            load_stage_value(u, k, offset + stride_y, x, y + 1, z, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - stride_y, x, y - 1, z, Nx, Ny, Nz, scale)) +
        c2 * (
            load_stage_value(u, k, offset + 2 * stride_y, x, y + 2, z, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 2 * stride_y, x, y - 2, z, Nx, Ny, Nz, scale)) +
        c3 * (
            load_stage_value(u, k, offset + 3 * stride_y, x, y + 3, z, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 3 * stride_y, x, y - 3, z, Nx, Ny, Nz, scale)) +
        c4 * (
            load_stage_value(u, k, offset + 4 * stride_y, x, y + 4, z, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 4 * stride_y, x, y - 4, z, Nx, Ny, Nz, scale))
    ) * inv_hy2;
    const float u_zz = (
        c0 * uc +
        c1 * (
            load_stage_value(u, k, offset + stride_z, x, y, z + 1, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - stride_z, x, y, z - 1, Nx, Ny, Nz, scale)) +
        c2 * (
            load_stage_value(u, k, offset + 2 * stride_z, x, y, z + 2, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 2 * stride_z, x, y, z - 2, Nx, Ny, Nz, scale)) +
        c3 * (
            load_stage_value(u, k, offset + 3 * stride_z, x, y, z + 3, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 3 * stride_z, x, y, z - 3, Nx, Ny, Nz, scale)) +
        c4 * (
            load_stage_value(u, k, offset + 4 * stride_z, x, y, z + 4, Nx, Ny, Nz, scale) +
            load_stage_value(u, k, offset - 4 * stride_z, x, y, z - 4, Nx, Ny, Nz, scale))
    ) * inv_hz2;

    return alpha * (u_xx + u_yy + u_zz);
}

__global__ void lap_stage_no_acc_cuda_kernel(
    const float* __restrict__ u,
    const float* __restrict__ k,
    float* __restrict__ out,
    int Nx,
    int Ny,
    int Nz,
    float alpha,
    float inv_hx2,
    float inv_hy2,
    float inv_hz2,
    float scale
) {
    const int interior_X = Nx - 8;
    const int interior_Y = Ny - 8;
    const int interior_Z = Nz - 8;
    const int total = interior_X * interior_Y * interior_Z;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int x = idx % interior_X + 4;
    const int y = (idx / interior_X) % interior_Y + 4;
    const int z = idx / (interior_X * interior_Y) + 4;
    const int offset = z * Ny * Nx + y * Nx + x;

    out[offset] = compute_stage_laplacian(
        u, k, x, y, z, Nx, Ny, Nz,
        alpha, inv_hx2, inv_hy2, inv_hz2, scale);
}

__global__ void lap_stage_acc_cuda_kernel(
    const float* __restrict__ u,
    const float* __restrict__ k,
    float* __restrict__ out,
    float* __restrict__ acc,
    int Nx,
    int Ny,
    int Nz,
    float alpha,
    float inv_hx2,
    float inv_hy2,
    float inv_hz2,
    float scale
) {
    const int interior_X = Nx - 8;
    const int interior_Y = Ny - 8;
    const int interior_Z = Nz - 8;
    const int total = interior_X * interior_Y * interior_Z;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int x = idx % interior_X + 4;
    const int y = (idx / interior_X) % interior_Y + 4;
    const int z = idx / (interior_X * interior_Y) + 4;
    const int offset = z * Ny * Nx + y * Nx + x;
    const float result = compute_stage_laplacian(
        u, k, x, y, z, Nx, Ny, Nz,
        alpha, inv_hx2, inv_hy2, inv_hz2, scale);

    out[offset] = result;
    acc[offset] += 2.0f * result;
}

__global__ void combine_cuda_kernel(
    float* __restrict__ u,
    const float* __restrict__ k1,
    const float* __restrict__ acc,
    int Nx,
    int Ny,
    int Nz,
    float dt
) {
    const int interior_X = Nx - 8;
    const int interior_Y = Ny - 8;
    const int interior_Z = Nz - 8;
    const int total = interior_X * interior_Y * interior_Z;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int x = idx % interior_X + 4;
    const int y = (idx / interior_X) % interior_Y + 4;
    const int z = idx / (interior_X * interior_Y) + 4;
    const int offset = z * Ny * Nx + y * Nx + x;

    u[offset] = u[offset] + (dt / 6.0f) *
        (acc[offset] + k1[offset]);
}

torch::Tensor custom_kernel(
    torch::Tensor u0,
    float alpha,
    float hx,
    float hy,
    float hz,
    int n_steps
) {
    TORCH_CHECK(u0.device().is_cuda(), "Tensor u0 must be a CUDA tensor");
    TORCH_CHECK(u0.scalar_type() == torch::kFloat32, "u0 must be float32");
    TORCH_CHECK(u0.dim() == 3, "u0 must have shape (Nz, Ny, Nx)");

    const int Nz = u0.size(0);
    const int Ny = u0.size(1);
    const int Nx = u0.size(2);

    if (Nx < 9 || Ny < 9 || Nz < 9) {
        throw std::runtime_error("All dimensions must be >= 9 for radius-4 stencil.");
    }

    TORCH_CHECK(u0.is_contiguous(), "u0 must be contiguous");

    torch::Tensor u = u0.clone();
    torch::Tensor k1 = torch::empty_like(u0);
    torch::Tensor k2 = torch::empty_like(u0);
    torch::Tensor acc = torch::empty_like(u0);

    const int interior_total = (Nx - 8) * (Ny - 8) * (Nz - 8);
    const dim3 interior_grid((interior_total + kBlockSize - 1) / kBlockSize);
    const dim3 block(kBlockSize);

    const float inv_hx2 = 1.0f / (hx * hx);
    const float inv_hy2 = 1.0f / (hy * hy);
    const float inv_hz2 = 1.0f / (hz * hz);
    const float dt = 0.05f / (alpha * (inv_hx2 + inv_hy2 + inv_hz2));

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto graph_stream = at::cuda::getStreamFromPool(false, u0.device().index());
    cudaStream_t stream = graph_stream.stream();

    cudaEvent_t ready_event;
    cudaEvent_t done_event;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    cudaError_t err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaEventRecord(ready_event, current_stream.stream());
    if (err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        cudaEventDestroy(done_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaStreamWaitEvent(stream, ready_event, 0);
    if (err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        cudaEventDestroy(done_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        cudaEventDestroy(done_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    for (int step = 0; step < n_steps; ++step) {
        lap_u_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(), k1.data_ptr<float>(), acc.data_ptr<float>(),
            Nx, Ny, Nz, alpha, inv_hx2, inv_hy2, inv_hz2);

        lap_stage_acc_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(), k1.data_ptr<float>(), k2.data_ptr<float>(), acc.data_ptr<float>(),
            Nx, Ny, Nz, alpha, inv_hx2, inv_hy2, inv_hz2, 0.5f * dt);

        lap_stage_acc_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(), k2.data_ptr<float>(), k1.data_ptr<float>(), acc.data_ptr<float>(),
            Nx, Ny, Nz, alpha, inv_hx2, inv_hy2, inv_hz2, 0.5f * dt);

        lap_stage_no_acc_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(), k1.data_ptr<float>(), k2.data_ptr<float>(), Nx, Ny, Nz,
            alpha, inv_hx2, inv_hy2, inv_hz2, dt);

        combine_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(),
            k2.data_ptr<float>(),
            acc.data_ptr<float>(),
            Nx, Ny, Nz, dt);
    }

    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        cudaEventDestroy(done_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        cudaEventDestroy(ready_event);
        cudaEventDestroy(done_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaGraphLaunch(graph_exec, stream);
    if (err != cudaSuccess) {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        cudaEventDestroy(ready_event);
        cudaEventDestroy(done_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaEventRecord(done_event, stream);
    if (err != cudaSuccess) {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        cudaEventDestroy(ready_event);
        cudaEventDestroy(done_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaStreamWaitEvent(current_stream.stream(), done_event, 0);
    if (err != cudaSuccess) {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        cudaEventDestroy(ready_event);
        cudaEventDestroy(done_event);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaGetLastError();
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(ready_event);
    cudaEventDestroy(done_event);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return u;
}

"""

# C++ header declaration
cpp_source = """
#include <torch/extension.h>
torch::Tensor custom_kernel(torch::Tensor u0,
            float alpha,
            float hx,
            float hy,
            float hz,
            int n_steps);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

try:
    cuda_module = load_inline(
        name='submission_cuda_rk4_caffrey',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['custom_kernel'],
        verbose=True,
        # with_cuda=True,
        # build_directory=".",  # Cache compiled modules here
    )
except Exception as e:
    print("\n====== CUDA Extension Build Error ======\n")
    print(e)                     # short error
    print("\n====== Full Traceback ======\n")
    import traceback
    traceback.print_exc()        # full Python traceback
    raise  

def custom_kernel(data: input_t) -> output_t:
    # RK4 assignment input_t: (u0, alpha, hx, hy, hz, n_steps)
    u0, alpha, hx, hy, hz, n_steps = data

    def scalar(value):
        return value.item() if isinstance(value, torch.Tensor) else value

    return cuda_module.custom_kernel(
        u0,
        scalar(alpha),
        scalar(hx),
        scalar(hy),
        scalar(hz),
        n_steps,
    )
