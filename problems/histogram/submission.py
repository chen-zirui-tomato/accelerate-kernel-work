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

"""

# C++ header declaration
cpp_source = """
#include <torch/extension.h>
torch::Tensor histogram_kernel(torch::Tensor data, int num_bins);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

cuda_module = load_inline(
    name='submission_cuda_histogram_caffrey',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['histogram_kernel'],
    verbose=True,  # Enable verbose to see compilation details
    # with_cuda=True,
    # build_directory=".",
)

def custom_kernel(data: input_t) -> output_t:
    """
    Wrapper function matching the required signature.
    
    Args:
        data: Tuple of (array, num_bins) where:
            array:    Tensor of shape [length, num_channels] with integer values in [0, num_bins-1]
            num_bins: Number of bins for the histogram
    
    Returns:
        histogram: Tensor of shape [num_channels, num_bins] containing histogram counts for each channel
    """

    array, num_bins = data
    
    if not array.is_cuda:
        array = array.cuda()
    
    # Call CUDA kernel
    histogram = cuda_module.histogram_kernel(array, num_bins)

    return histogram
