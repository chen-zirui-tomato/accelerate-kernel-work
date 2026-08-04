import torch
import triton
import triton.language as tl

from task import input_t, output_t


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


@triton.jit
def _swiglu_kernel(gate, value, b, c, out, total: tl.constexpr, hidden: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    h = offsets % hidden

    g = tl.load(gate + offsets, mask=mask, other=0.0) + tl.load(b + h, mask=mask, other=0.0)
    v = tl.load(value + offsets, mask=mask, other=0.0) + tl.load(c + h, mask=mask, other=0.0)
    y = g * tl.sigmoid(beta * g) * v
    tl.store(out + offsets, y, mask=mask)


def custom_kernel(data: input_t) -> output_t:
    x, W, V, b, c, beta = data

    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden = W.shape[1]
    x_flat = x.reshape(batch_size * seq_len, x.shape[2])

    gate = torch.empty((batch_size * seq_len, hidden), device=x.device, dtype=x.dtype)
    value = torch.empty_like(gate)
    out = torch.empty_like(gate)

    torch.mm(x_flat, W, out=gate)
    torch.mm(x_flat, V, out=value)

    total = gate.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)
    _swiglu_kernel[grid](gate, value, b, c, out, total, hidden, float(beta), BLOCK=block, num_warps=4)

    return out.reshape(batch_size, seq_len, hidden)
