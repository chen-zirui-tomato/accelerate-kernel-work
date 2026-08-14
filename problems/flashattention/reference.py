import torch
import torch.nn.functional as F

from task import input_t, output_t
from utils import make_match_reference


def ref_kernel(data: input_t) -> output_t:
    """
    PyTorch SDPA reference implementation.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

    Args:
        data: tuple of (Q, K, V)
            Q: (batch_size, num_heads, seq_len, head_dim)
            K: (batch_size, num_heads, seq_len, head_dim)
            V: (batch_size, num_heads, seq_len, head_dim)
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    q, k, v = data
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def generate_input(
    batch_size: int, num_heads: int, head_dim: int, seq_len: int, seed: int
) -> input_t:

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    dtype = torch.float16
    device = "cuda"

    q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        generator=gen,
    )
    k = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        generator=gen,
    )
    v = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        generator=gen,
    )

    # Ensure contiguous for the CUDA kernel
    return (q.contiguous(), k.contiguous(), v.contiguous())


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
