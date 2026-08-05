import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _histogram_kernel(
    array_t,
    histogram,
    length,
    num_channels,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    counts = tl.zeros((num_bins,), dtype=tl.int32)
    bins = tl.arange(0, num_bins)
    base = pid * length

    start = 0
    while start < length:
        offsets = start + tl.arange(0, BLOCK_N)
        mask = offsets < length
        values = tl.load(array_t + base + offsets, mask=mask, other=0).to(tl.int32)
        hits = values[:, None] == bins[None, :]
        counts += tl.sum(hits.to(tl.int32), axis=0)
        start += BLOCK_N

    tl.store(histogram + pid * num_bins + bins, counts)


def custom_kernel(data: input_t) -> output_t:
    array, num_bins = data
    length, num_channels = array.shape

    array_t = array.transpose(0, 1).contiguous()
    histogram = torch.empty((num_channels, num_bins), dtype=torch.int32, device=array.device)

    block_n = 128
    grid = (num_channels,)
    _histogram_kernel[grid](
        array_t,
        histogram,
        length,
        num_channels,
        num_bins=num_bins,
        BLOCK_N=block_n,
        num_warps=4,
    )

    return histogram
