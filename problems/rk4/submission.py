import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

constexpr int kBlockSize = 256;

__global__ void lap_cuda_kernel(
    const float* __restrict__ u,
    float* __restrict__ k,
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

    k[offset] = alpha * (u_xx + u_yy + u_zz);
}

__global__ void stage_cuda_kernel(
    const float* __restrict__ u,
    const float* __restrict__ k,
    float* __restrict__ u_stage,
    int Nx,
    int Ny,
    int Nz,
    float scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny * Nz;
    if (idx >= total) {
        return;
    }

    const int x = idx % Nx;
    const int y = (idx / Nx) % Ny;
    const int z = idx / (Nx * Ny);
    const bool is_interior =
        x >= 4 && x < Nx - 4 &&
        y >= 4 && y < Ny - 4 &&
        z >= 4 && z < Nz - 4;

    u_stage[idx] = is_interior ? u[idx] + scale * k[idx] : u[idx];
}

__global__ void combine_cuda_kernel(
    const float* __restrict__ u,
    const float* __restrict__ k1,
    const float* __restrict__ k2,
    const float* __restrict__ k3,
    const float* __restrict__ k4,
    float* __restrict__ u_next,
    int Nx,
    int Ny,
    int Nz,
    float dt
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny * Nz;
    if (idx >= total) {
        return;
    }

    const int x = idx % Nx;
    const int y = (idx / Nx) % Ny;
    const int z = idx / (Nx * Ny);
    const bool is_interior =
        x >= 4 && x < Nx - 4 &&
        y >= 4 && y < Ny - 4 &&
        z >= 4 && z < Nz - 4;

    if (is_interior) {
        u_next[idx] = u[idx] + (dt / 6.0f) *
            (k1[idx] + 2.0f * k2[idx] + 2.0f * k3[idx] + k4[idx]);
    } else {
        u_next[idx] = u[idx];
    }
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
    torch::Tensor u_next = torch::empty_like(u0);
    torch::Tensor u_stage = torch::empty_like(u0);
    torch::Tensor k1 = torch::empty_like(u0);
    torch::Tensor k2 = torch::empty_like(u0);
    torch::Tensor k3 = torch::empty_like(u0);
    torch::Tensor k4 = torch::empty_like(u0);

    const int interior_total = (Nx - 8) * (Ny - 8) * (Nz - 8);
    const int total_elements = Nx * Ny * Nz;
    const dim3 interior_grid((interior_total + kBlockSize - 1) / kBlockSize);
    const dim3 full_grid((total_elements + kBlockSize - 1) / kBlockSize);
    const dim3 block(kBlockSize);

    const float inv_hx2 = 1.0f / (hx * hx);
    const float inv_hy2 = 1.0f / (hy * hy);
    const float inv_hz2 = 1.0f / (hz * hz);
    const float dt = 0.05f / (alpha * (inv_hx2 + inv_hy2 + inv_hz2));

    for (int step = 0; step < n_steps; ++step) {
        lap_cuda_kernel<<<interior_grid, block>>>(
            u.data_ptr<float>(), k1.data_ptr<float>(), Nx, Ny, Nz, alpha,
            inv_hx2, inv_hy2, inv_hz2);

        stage_cuda_kernel<<<full_grid, block>>>(
            u.data_ptr<float>(), k1.data_ptr<float>(), u_stage.data_ptr<float>(),
            Nx, Ny, Nz, 0.5f * dt);
        lap_cuda_kernel<<<interior_grid, block>>>(
            u_stage.data_ptr<float>(), k2.data_ptr<float>(), Nx, Ny, Nz, alpha,
            inv_hx2, inv_hy2, inv_hz2);

        stage_cuda_kernel<<<full_grid, block>>>(
            u.data_ptr<float>(), k2.data_ptr<float>(), u_stage.data_ptr<float>(),
            Nx, Ny, Nz, 0.5f * dt);
        lap_cuda_kernel<<<interior_grid, block>>>(
            u_stage.data_ptr<float>(), k3.data_ptr<float>(), Nx, Ny, Nz, alpha,
            inv_hx2, inv_hy2, inv_hz2);

        stage_cuda_kernel<<<full_grid, block>>>(
            u.data_ptr<float>(), k3.data_ptr<float>(), u_stage.data_ptr<float>(),
            Nx, Ny, Nz, dt);
        lap_cuda_kernel<<<interior_grid, block>>>(
            u_stage.data_ptr<float>(), k4.data_ptr<float>(), Nx, Ny, Nz, alpha,
            inv_hx2, inv_hy2, inv_hz2);

        combine_cuda_kernel<<<full_grid, block>>>(
            u.data_ptr<float>(),
            k1.data_ptr<float>(),
            k2.data_ptr<float>(),
            k3.data_ptr<float>(),
            k4.data_ptr<float>(),
            u_next.data_ptr<float>(),
            Nx, Ny, Nz, dt);
        std::swap(u, u_next);
    }


    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
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
        name='submission_cuda_rk4_localtest',
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
