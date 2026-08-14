import torch
import triton
import triton.language as tl

from task import input_t, output_t


@triton.jit
def _flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    numheads,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seqlen_block_id = tl.program_id(0)
    numhead_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    seqlen_offsets = seqlen_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    headdim_offsets = tl.arange(0, BLOCK_D)

    seqlen_mask = seqlen_offsets < seq_len
    headdim_mask = headdim_offsets < head_dim

    base_offset = (batch_id * numheads + numhead_id) * seq_len * head_dim
    q_base = q_ptr + base_offset
    k_base = k_ptr + base_offset
    v_base = v_ptr + base_offset
    o_base = o_ptr + base_offset

    q = tl.load(
        q_base
        + seqlen_offsets[:, None] * head_dim
        + headdim_offsets[None, :],
        mask=seqlen_mask[:, None] & headdim_mask[None, :],
        other=0.0,
    )

    # Online softmax state for each query row.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        key_seqlen_offsets = start_n + tl.arange(0, BLOCK_N)
        key_mask = key_seqlen_offsets < seq_len

        k = tl.load(
            k_base
            + key_seqlen_offsets[:, None] * head_dim
            + headdim_offsets[None, :],
            mask=key_mask[:, None] & headdim_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_base
            + key_seqlen_offsets[:, None] * head_dim
            + headdim_offsets[None, :],
            mask=key_mask[:, None] & headdim_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(key_mask[None, :], scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.float16), v, out_dtype=tl.float32
        )

        m_i = m_new
        l_i = l_new

    output = acc / l_i[:, None]
    tl.store(
        o_base
        + seqlen_offsets[:, None] * head_dim
        + headdim_offsets[None, :],
        output,
        mask=seqlen_mask[:, None] & headdim_mask[None, :],
    )


def custom_kernel(data: input_t) -> output_t:
    q, k, v = data
    batch_size, num_heads, seq_len, head_dim = q.shape

    # The assignment uses head_dim=128; keep this explicit for a stable tile.
    block_m = 64
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)

    output = torch.empty_like(q)
    grid = (
        triton.cdiv(seq_len, block_m),
        num_heads,
        batch_size,
    )

    _flash_attention_kernel[grid](
        q,
        k,
        v,
        output,
        num_heads,
        seq_len,
        head_dim,
        1.0 / (head_dim**0.5),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return output
