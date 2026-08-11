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

__global__ void lap_stage_cuda_kernel(
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
    float scale, 
    float* acc,
    float acc_weight
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

    const bool full_stage_interior =
        x >= 8 && x < Nx - 8 &&
        y >= 8 && y < Ny - 8 &&
        z >= 8 && z < Nz - 8;

    float uc;
    float ux_p1, ux_m1, ux_p2, ux_m2, ux_p3, ux_m3, ux_p4, ux_m4;
    float uy_p1, uy_m1, uy_p2, uy_m2, uy_p3, uy_m3, uy_p4, uy_m4;
    float uz_p1, uz_m1, uz_p2, uz_m2, uz_p3, uz_m3, uz_p4, uz_m4;

    if (scale == 0.0f) {
        uc = u[offset];

        ux_p1 = u[offset + 1];
        ux_m1 = u[offset - 1];
        ux_p2 = u[offset + 2];
        ux_m2 = u[offset - 2];
        ux_p3 = u[offset + 3];
        ux_m3 = u[offset - 3];
        ux_p4 = u[offset + 4];
        ux_m4 = u[offset - 4];

        uy_p1 = u[offset + stride_y];
        uy_m1 = u[offset - stride_y];
        uy_p2 = u[offset + 2 * stride_y];
        uy_m2 = u[offset - 2 * stride_y];
        uy_p3 = u[offset + 3 * stride_y];
        uy_m3 = u[offset - 3 * stride_y];
        uy_p4 = u[offset + 4 * stride_y];
        uy_m4 = u[offset - 4 * stride_y];

        uz_p1 = u[offset + stride_z];
        uz_m1 = u[offset - stride_z];
        uz_p2 = u[offset + 2 * stride_z];
        uz_m2 = u[offset - 2 * stride_z];
        uz_p3 = u[offset + 3 * stride_z];
        uz_m3 = u[offset - 3 * stride_z];
        uz_p4 = u[offset + 4 * stride_z];
        uz_m4 = u[offset - 4 * stride_z];
    } else if (full_stage_interior) {
        uc = load_stage_value_fast(u, k, offset, scale);

        ux_p1 = load_stage_value_fast(u, k, offset + 1, scale);
        ux_m1 = load_stage_value_fast(u, k, offset - 1, scale);
        ux_p2 = load_stage_value_fast(u, k, offset + 2, scale);
        ux_m2 = load_stage_value_fast(u, k, offset - 2, scale);
        ux_p3 = load_stage_value_fast(u, k, offset + 3, scale);
        ux_m3 = load_stage_value_fast(u, k, offset - 3, scale);
        ux_p4 = load_stage_value_fast(u, k, offset + 4, scale);
        ux_m4 = load_stage_value_fast(u, k, offset - 4, scale);

        uy_p1 = load_stage_value_fast(u, k, offset + stride_y, scale);
        uy_m1 = load_stage_value_fast(u, k, offset - stride_y, scale);
        uy_p2 = load_stage_value_fast(u, k, offset + 2 * stride_y, scale);
        uy_m2 = load_stage_value_fast(u, k, offset - 2 * stride_y, scale);
        uy_p3 = load_stage_value_fast(u, k, offset + 3 * stride_y, scale);
        uy_m3 = load_stage_value_fast(u, k, offset - 3 * stride_y, scale);
        uy_p4 = load_stage_value_fast(u, k, offset + 4 * stride_y, scale);
        uy_m4 = load_stage_value_fast(u, k, offset - 4 * stride_y, scale);

        uz_p1 = load_stage_value_fast(u, k, offset + stride_z, scale);
        uz_m1 = load_stage_value_fast(u, k, offset - stride_z, scale);
        uz_p2 = load_stage_value_fast(u, k, offset + 2 * stride_z, scale);
        uz_m2 = load_stage_value_fast(u, k, offset - 2 * stride_z, scale);
        uz_p3 = load_stage_value_fast(u, k, offset + 3 * stride_z, scale);
        uz_m3 = load_stage_value_fast(u, k, offset - 3 * stride_z, scale);
        uz_p4 = load_stage_value_fast(u, k, offset + 4 * stride_z, scale);
        uz_m4 = load_stage_value_fast(u, k, offset - 4 * stride_z, scale);
    } else {
        uc = load_stage_value(u, k, offset, x, y, z, Nx, Ny, Nz, scale);

        ux_p1 = load_stage_value(u, k, offset + 1, x + 1, y, z, Nx, Ny, Nz, scale);
        ux_m1 = load_stage_value(u, k, offset - 1, x - 1, y, z, Nx, Ny, Nz, scale);
        ux_p2 = load_stage_value(u, k, offset + 2, x + 2, y, z, Nx, Ny, Nz, scale);
        ux_m2 = load_stage_value(u, k, offset - 2, x - 2, y, z, Nx, Ny, Nz, scale);
        ux_p3 = load_stage_value(u, k, offset + 3, x + 3, y, z, Nx, Ny, Nz, scale);
        ux_m3 = load_stage_value(u, k, offset - 3, x - 3, y, z, Nx, Ny, Nz, scale);
        ux_p4 = load_stage_value(u, k, offset + 4, x + 4, y, z, Nx, Ny, Nz, scale);
        ux_m4 = load_stage_value(u, k, offset - 4, x - 4, y, z, Nx, Ny, Nz, scale);

        uy_p1 = load_stage_value(u, k, offset + stride_y, x, y + 1, z, Nx, Ny, Nz, scale);
        uy_m1 = load_stage_value(u, k, offset - stride_y, x, y - 1, z, Nx, Ny, Nz, scale);
        uy_p2 = load_stage_value(u, k, offset + 2 * stride_y, x, y + 2, z, Nx, Ny, Nz, scale);
        uy_m2 = load_stage_value(u, k, offset - 2 * stride_y, x, y - 2, z, Nx, Ny, Nz, scale);
        uy_p3 = load_stage_value(u, k, offset + 3 * stride_y, x, y + 3, z, Nx, Ny, Nz, scale);
        uy_m3 = load_stage_value(u, k, offset - 3 * stride_y, x, y - 3, z, Nx, Ny, Nz, scale);
        uy_p4 = load_stage_value(u, k, offset + 4 * stride_y, x, y + 4, z, Nx, Ny, Nz, scale);
        uy_m4 = load_stage_value(u, k, offset - 4 * stride_y, x, y - 4, z, Nx, Ny, Nz, scale);

        uz_p1 = load_stage_value(u, k, offset + stride_z, x, y, z + 1, Nx, Ny, Nz, scale);
        uz_m1 = load_stage_value(u, k, offset - stride_z, x, y, z - 1, Nx, Ny, Nz, scale);
        uz_p2 = load_stage_value(u, k, offset + 2 * stride_z, x, y, z + 2, Nx, Ny, Nz, scale);
        uz_m2 = load_stage_value(u, k, offset - 2 * stride_z, x, y, z - 2, Nx, Ny, Nz, scale);
        uz_p3 = load_stage_value(u, k, offset + 3 * stride_z, x, y, z + 3, Nx, Ny, Nz, scale);
        uz_m3 = load_stage_value(u, k, offset - 3 * stride_z, x, y, z - 3, Nx, Ny, Nz, scale);
        uz_p4 = load_stage_value(u, k, offset + 4 * stride_z, x, y, z + 4, Nx, Ny, Nz, scale);
        uz_m4 = load_stage_value(u, k, offset - 4 * stride_z, x, y, z - 4, Nx, Ny, Nz, scale);
    }

    const float u_xx = (
        c0 * uc +
        c1 * (ux_p1 + ux_m1) +
        c2 * (ux_p2 + ux_m2) +
        c3 * (ux_p3 + ux_m3) +
        c4 * (ux_p4 + ux_m4)
    ) * inv_hx2;

    const float u_yy = (
        c0 * uc +
        c1 * (uy_p1 + uy_m1) +
        c2 * (uy_p2 + uy_m2) +
        c3 * (uy_p3 + uy_m3) +
        c4 * (uy_p4 + uy_m4)
    ) * inv_hy2;

    const float u_zz = (
        c0 * uc +
        c1 * (uz_p1 + uz_m1) +
        c2 * (uz_p2 + uz_m2) +
        c3 * (uz_p3 + uz_m3) +
        c4 * (uz_p4 + uz_m4)
    ) * inv_hz2;

    float result = alpha * (u_xx + u_yy + u_zz);
    out[offset] = result;
    if (acc != nullptr) {
        if (acc_weight == 0.0f) {
            acc[offset] = result;
        } else {
            acc[offset] += acc_weight * result;
        }
    }
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

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    for (int step = 0; step < n_steps; ++step) {
        lap_stage_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(), nullptr, k1.data_ptr<float>(), Nx, Ny, Nz,
            alpha, inv_hx2, inv_hy2, inv_hz2, 0.0f, acc.data_ptr<float>(), 0.0f);

        lap_stage_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(), k1.data_ptr<float>(), k2.data_ptr<float>(), Nx, Ny, Nz,
            alpha, inv_hx2, inv_hy2, inv_hz2, 0.5f * dt, acc.data_ptr<float>(), 2.0f);

        lap_stage_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(), k2.data_ptr<float>(), k1.data_ptr<float>(), Nx, Ny, Nz,
            alpha, inv_hx2, inv_hy2, inv_hz2, 0.5f * dt, acc.data_ptr<float>(), 2.0f);

        lap_stage_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(), k1.data_ptr<float>(), k2.data_ptr<float>(), Nx, Ny, Nz,
            alpha, inv_hx2, inv_hy2, inv_hz2, dt, nullptr, -1.0f);

        combine_cuda_kernel<<<interior_grid, block, 0, stream>>>(
            u.data_ptr<float>(),
            k2.data_ptr<float>(),
            acc.data_ptr<float>(),
            Nx, Ny, Nz, dt);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return u;
}
