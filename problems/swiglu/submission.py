import torch
import triton
import triton.language as tl

from task import input_t, output_t


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("medium")
except AttributeError:
    pass

@triton.jit
def _gemmAndFusedEpilogue_kernel(x, W, V, b, c, beta, out, M_total:tl.constexpr, N_total: tl.constexpr, K_total: tl.constexpr,
                                        BLOCK_M: tl.constexpr,
                                        BLOCK_N: tl.constexpr,
                                        BLOCK_K: tl.constexpr):
    # 计算过程 G = x @ W, value = x @ V, gate = G * sigmoid(beta * G), out = gate * value
    # 因为最后*都是逐元素乘，所以可以分块计算，节省HBM搬运
    # 每次搬运的tile大小为x_flat[M, K]和both[K, N], 其中both是G和value拼接在一起的矩阵
    # x_flat, both, b, c, beta = input


    # 二维grid，第一维代表现在处理x_flat的tile的首行，第二维代表现在处理both的tile的首列
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offset_row = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_col = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)

    # 每个kernel分配的tile可能大于BLOCK_K,所以需要循环计算
    out_W_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)    
    out_V_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_row = offset_row[:, None] < M_total
    mask_col = offset_col[None, :] < N_total

    for k0 in range(0, K_total, BLOCK_K):
        offset_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offset_k < K_total

        x_ptr = x + offset_row[:, None] * K_total + offset_k[None, :]
        W_ptr = W + offset_k[:, None] * N_total + offset_col[None, :]
        V_ptr = V + offset_k[:, None] * N_total + offset_col[None, :]

        x_tile = tl.load(x_ptr, mask = mask_row & mask_k[None, :], other=0.0)
        W_tile = tl.load(W_ptr, mask = mask_k[:, None] & mask_col, other=0.0)
        V_tile = tl.load(V_ptr, mask = mask_k[:, None] & mask_col, other=0.0)
        # 计算G和value的tile
        out_W_tile += tl.dot(x_tile, W_tile, input_precision="tf32")
        out_V_tile += tl.dot(x_tile, V_tile, input_precision="tf32")
    # 计算G和value的tile的gate和out
    b_ptr = b + offset_col[None, :]
    c_ptr = c + offset_col[None, :]
    out_W_tile = out_W_tile + tl.load(b_ptr, mask = mask_col, other=0.0)
    out_V_tile = out_V_tile + tl.load(c_ptr, mask = mask_col, other=0.0)
    gate_tile = out_W_tile * tl.sigmoid(beta * out_W_tile)
    out_tile = gate_tile * out_V_tile
    # 将out_tile写回到both中
    tl.store(out + offset_row[:, None]*N_total + offset_col[None, :], out_tile, mask = mask_row & mask_col)


@triton.jit
def _swiglu_kernel(both, b, c, out, total: tl.constexpr, hidden: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    row = offsets // hidden
    h = offsets - row * hidden
    base = row * (hidden * 2) + h

    g = tl.load(both + base, mask=mask, other=0.0) + tl.load(b + h, mask=mask, other=0.0)
    v = tl.load(both + base + hidden, mask=mask, other=0.0) + tl.load(c + h, mask=mask, other=0.0)
    y = g * tl.sigmoid(beta * g) * v
    tl.store(out + offsets, y, mask=mask)


def custom_kernel(data: input_t) -> output_t:
    x, W, V, b, c, beta = data

    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden = W.shape[1]
    x_flat = x.reshape(batch_size * seq_len, x.shape[2])

    M_total = x_flat.shape[0]
    N_total = W.shape[1]
    K_total = x_flat.shape[1]
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (
        triton.cdiv(M_total, BLOCK_M),
        triton.cdiv(N_total, BLOCK_N),
    )
    out = torch.empty((batch_size * seq_len, hidden), device=x.device, dtype=x.dtype)

    _gemmAndFusedEpilogue_kernel[grid](x_flat, W, V, b, c, beta, out, M_total, N_total, K_total, 
                                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)

    # WV = torch.cat((W, V), dim=1)
    # both = torch.empty((batch_size * seq_len, hidden * 2), device=x.device, dtype=x.dtype)
    # out = torch.empty((batch_size * seq_len, hidden), device=x.device, dtype=x.dtype)

    # torch.mm(x_flat, WV, out=both)

    # total = out.numel()
    # block = 1024
    # grid = (triton.cdiv(total, block),)
    # _swiglu_kernel[grid](both, b, c, out, total, hidden, float(beta), BLOCK=block, num_warps=4)

    return out.reshape(batch_size, seq_len, hidden)
