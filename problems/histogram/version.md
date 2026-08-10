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

## v2 - CUDA shared-memory 基线：一个 block 负责一个 channel

- 改动：
  - 新增 `problems/histogram/submiss.cu` 的 CUDA 实现。
  - host 端直接从 CUDA tensor 取 device pointer：`data.data_ptr<uint8_t>()` 和 `histogram.data_ptr<int>()`
  - kernel 使用 `grid = (num_channels,)`，即每个 block 负责一个 channel。
  - block 内用动态 shared memory 保存当前 channel 的局部 histogram：
    - kernel 内写 `extern __shared__ int shared_hist[]`
    - launch 时通过第三个 kernel 配置参数传入 `num_bins * sizeof(int)`
  - 每个 block 先并行清零 `shared_hist`，再让线程沿 length 维扫描本 channel 的元素。
  - 对 shared histogram 使用 `atomicAdd(&shared_hist[value], 1)` 解决同一个 bin 被多个线程同时更新的问题。
  - 统计完成后，每个 block 把自己的 `shared_hist[bin]` 写回 `histogram[channel, bin]`。
- 当前 CUDA 版本又加了一步输入转置：
  - 原始输入是 `[length, num_channels]` contiguous，固定 channel 扫 length 时访问地址间隔是 `num_channels`。
  - host 端先执行 `data.transpose(0, 1).contiguous()`，得到 `[num_channels, length]` 的连续布局。
  - kernel 读入地址从 `data[row * num_channels + channel]` 改成 `data[channel * length + row]`。
  - 这样同一个 block 处理一个 channel 时，线程沿 row 方向读的是连续地址，更利于 coalesced global memory load。
- 这样做的好处：
  - 避免了每个输入元素都对 global histogram 做 atomic add。
  - global memory 写回规模为每个 channel 只写 `num_bins` 个结果。
  - 实现结构接近最直接的 CUDA baseline，便于后续 profile 和继续优化。
- 当前风险：
  - `transpose(0, 1).contiguous()` 本身会产生一次额外的 CUDA 内存重排，是否划算需要 benchmark 判断。
  - shared memory 上仍然有 atomic contention；如果输入值集中在少数 bin，冲突会明显。
  - 每个 channel 只有一个 block，单个 channel 的 length 维并行度只来自一个 block 内的 256 个线程，后续可能需要拆成多个 block 做 partial histogram 再归约。
  - 下一步可尝试 per-warp private histogram：每个 warp 在 shared memory 里维护一份局部计数，最后按 bin 合并，减少整个 block 共同抢同一个 bin 的 shared atomic 冲突。

## v3 - CUDA warp-private histogram：每个 warp 维护一份局部计数

- 改动：
  - 在 v2 的 CUDA shared-memory baseline 上，把单份 block-local histogram 改成 per-warp private histogram。
  - block size 固定为 `kBlockSize = 256`，warp size 固定为 `kWarpSize = 32`，因此每个 block 有 `kNumWarps = 8` 个 warp。
  - kernel 改成模板形式：`template <int NUM_WARPS> __global__ void histogram_cuda_kernel(...)`。
  - launch 时使用 `histogram_cuda_kernel<kNumWarps><<<...>>>(...)`，让 warp 数成为编译期常量。
  - 动态 shared memory 的布局从 `[num_bins]` 改成 `[NUM_WARPS, num_bins]`：
    - shared memory 大小为 `kNumWarps * num_bins * sizeof(int)`
    - 每个 warp 只更新自己的 `warp_hist[warp_id * num_bins + value]`
  - 初始化阶段并行清零整块 `warp_hist`，范围是 `num_bins * NUM_WARPS`。
  - 扫描输入时，每个线程仍沿 length 维处理本 channel 的一部分元素，但 atomic add 只发生在当前 warp 的 private histogram 内。
  - 统计完成后，每个 bin 由线程负责把所有 warp 的同一 bin 累加：
    - `sum += warp_hist[w * num_bins + bin]`
    - 最后写回 `histogram[channel, bin]`
  - reduction 循环加了 `#pragma unroll`；由于 `NUM_WARPS` 是模板参数，编译器更容易展开这段小循环。
- 这样做的好处：
  - 相比 v2 中整个 block 共同更新同一份 `shared_hist`，v3 把 shared atomic 争用范围缩小到单个 warp 内。
  - 对输入值集中在少数 bin 的情况，理论上可以降低多个 warp 同时抢同一 shared memory 地址的冲突。
  - block size、warp 数、shared memory 大小现在由统一常量推导，后续修改 block size 时不容易漏改。
- 当前风险：
  - shared memory 用量从 `num_bins * sizeof(int)` 增加到 `kNumWarps * num_bins * sizeof(int)`；在当前参数 `num_bins = 256`、`kNumWarps = 8` 下约为 8KB，通常可接受。
  - 合并阶段每个 bin 要额外累加 `NUM_WARPS` 次；当前 `NUM_WARPS = 8`，这部分开销较小，但仍需要 benchmark 验证。
  - 仍然保留 `data.transpose(0, 1).contiguous()`，转置拷贝成本是否被连续读收益抵消，需要和非转置版本对比。
  - 仍然是每个 channel 一个 block；如果单个 channel 的 length 很大，可能需要进一步拆成多个 block 做 partial histogram 再归约。
