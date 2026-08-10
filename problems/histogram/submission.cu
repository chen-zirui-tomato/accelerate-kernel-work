#include <torch/extension.h>
#include <cuda_runtime.h>

#include <cstdint>

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockSize / kWarpSize;

template <int NUM_WARPS>
__global__ void histogram_cuda_kernel(
    const uint8_t* __restrict__ data,
    int* __restrict__ histogram,
    int length,
    int num_channels,
    int num_bins) {
    extern __shared__ int warp_hist[];

    const int channel = blockIdx.x;
    const int warp_id = threadIdx.x / kWarpSize;

    for (int bin = threadIdx.x; bin < num_bins * NUM_WARPS; bin += blockDim.x) {
        warp_hist[bin] = 0;
    }
    __syncthreads();

    for (int row = threadIdx.x; row < length; row += blockDim.x) {
        const uint8_t value = data[channel * length + row];
        atomicAdd(&warp_hist[warp_id * num_bins + static_cast<int>(value)], 1);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            sum += warp_hist[w * num_bins + bin];
        }
        histogram[num_bins * channel + bin] = sum;
    }
}

torch::Tensor histogram_kernel(torch::Tensor data, int num_bins) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    TORCH_CHECK(data.scalar_type() == torch::kUInt8, "Tensor data must be uint8");
    TORCH_CHECK(data.dim() == 2, "Tensor data must have shape [length, num_channels]");
    TORCH_CHECK(data.is_contiguous(), "Tensor data must be contiguous");

    const int length = data.size(0);
    const int num_channels = data.size(1);
    torch::Tensor data_t = data.transpose(0, 1).contiguous();

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);

    const dim3 grid(num_channels);
    const dim3 block(kBlockSize);
    const size_t shared_mem_size = kNumWarps * static_cast<size_t>(num_bins) * sizeof(int);

    histogram_cuda_kernel<kNumWarps><<<grid, block, shared_mem_size>>>(
        data_t.data_ptr<uint8_t>(),
        histogram.data_ptr<int>(),
        length,
        num_channels,
        num_bins);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return histogram;
}
