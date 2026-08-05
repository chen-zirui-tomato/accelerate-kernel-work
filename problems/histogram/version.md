# Histogram 版本记录
每次改动后需要检查：
  - `python3 ../eval.py test test_cases/test.txt` 是否通过。
  - `python3 ../eval.py benchmark test_cases/test.txt` 的结果是否优于 reference。
  - 如果要继续提速，再看 `python3 ../eval.py profile test_cases/test.txt` 的结果，判断瓶颈是在读数据、bin 争用，还是程序启动开销。

## v0 - reference 基线

- 初始状态：`problems/histogram/submission.py` 直接调用 `ref_kernel(data)`。
- `reference.py` 的实现是：
  - 对每个 channel 单独取出一列 `array[:, c]`
  - 用 `torch.bincount` 统计 0 ~ `num_bins - 1` 的出现次数
  - 把结果写到 `histogram[c, :]`
- 这版的逻辑很清楚，但性能上是 Python 级别的 channel 循环，不能利用我们自己控制的 block 级并行。

## v1 - Triton 基线：一个 program 负责一个 channel

- 改动：
  - 把 `submission.py` 从 reference 调用改成真实 Triton kernel。
  - 先把输入转成 `array_t = array.transpose(0, 1).contiguous()`，让每个 channel 在内存里连续。
  - 用 `grid = (num_channels,)`，即一个 Triton program 负责一个 channel。
  - 每个 program 内部按 `BLOCK_N` 分块扫描该 channel 的 `length` 维。
  - 先在局部 `counts[num_bins]` 里累计，再在最后一次性写回 `histogram[c, :]`。
- 当前 kernel 的核心思路：
  - 对每个 chunk 读入 `BLOCK_N` 个值
  - 用 `values[:, None] == bins[None, :]` 做局部比较
  - `tl.sum(..., axis=0)` 得到这一块里每个 bin 的增量
  - 累加到本地 `counts`
  - 最后 `tl.store` 一次写回
- 这样做的好处：
  - 没有 per-chunk 的 global partial histogram
  - 没有跨 program 的原子累加
  - 先把实现做正确，后面再靠 profile 决定要不要拆 bin、拆 warp 或换 CUDA
- 当前风险：
  - 这个版本是偏“正确性优先”的基线，不是最终性能版。
  - `values[:, None] == bins[None, :]` 会带来比较开销，后续可能需要更细的并行拆法或 CUDA 版优化。
