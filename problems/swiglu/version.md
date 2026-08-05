# SwiGLU 版本记录
每次改动后
服务器上需要检查：
  - `python ../eval.py test test_cases/test.txt` 是否通过。
  - `python ../eval.py benchmark test_cases/test.txt` 的均值是否明显低于 reference 的约 31 ms。
  - `python ../eval.py profile test_cases/test.txt` 中 `ampere_sgemm_128x128_nn` 是否消失，或者总时间是否明显下降。
  - 因为目前服务器中无法使用ncu，所以我们只能先测试nsys
  ``` bash
  PYTHONPATH="$PWD:$PWD/.." nsys profile --trace=cuda,cublas,nvtx,osrt --stats=true --force-overwrite true -o ./results/swiglu-submission-nsys python -c "import torch; from reference import generate_input; from submission import custom_kernel; data=generate_input(batch_size=256,in_features=2048,hidden_size=4096,seed=8846,seq=64); torch.cuda.synchronize(); custom_kernel(data); torch.cuda.synchronize()"
  ```
  ``` bash
  nsys stats --force-export=true ./results/swiglu-submission-nsys.nsys-rep | tee ./results/swiglu-submission-nsys-stats-version.txt
  ```
## v0 - reference 基线

- 初始状态：`problems/swiglu/submission.py` 直接调用 `ref_kernel(data)`。
- 从已有的 `results/swiglu-reference-nsys-stats.txt` 看，reference 的 GPU 时间主要集中在两次 GEMM：
  - `ampere_sgemm_128x128_nn`：2 次，总计约 35.508 ms，占 GPU kernel 时间 93.1%。
  - 其余 PyTorch elementwise add、sigmoid、mul 总共约 2.27 ms。
- 结论：只融合后处理 kernel 能减少 launch 和显存读写，但不会真正解决主瓶颈；如果目标是降低 `ampere_sgemm` 的占比，必须让 GEMM 本身变快，或者让它走不同的 tensor core 路径。

## v1 - torch.mm + Triton 融合 epilogue + TF32 调用tensorcore

- 把 `submission.py` 从直接调用 reference 改成真实实现。
- 将 `x` 从 `[batch_size, seq_len, in_features]` 展平成 `[batch_size * seq_len, in_features]`。
- 使用两次 `torch.mm(..., out=...)` 计算：
  - `gate = x @ W`
  - `value = x @ V`
- 新增 Triton kernel，把以下后处理融合成一个 kernel：
  - 加 bias：`gate + b` 和 `value + c`
  - Swish：`gate * sigmoid(beta * gate)`
  - 最终逐元素乘法：`swish_gate * value`
- 在 `submission.py` 顶部开启 TF32 matmul：
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`
  - 如果 PyTorch 支持，则调用 `torch.set_float32_matmul_precision("high")`
- 将 Triton epilogue 的 block size 从 256 调到 1024，并设置 `num_warps=4`。

- 实验目的：先删掉 reference 中多个 PyTorch elementwise kernel，得到一个更干净的 profiling 起点。
- 预期：GPU profile 里 elementwise kernel 数量下降，但如果 GEMM 仍是 FP32 SGEMM，`ampere_sgemm` 的绝对时间不会明显下降。
- 题目 correctness 容差是 `rtol=1e-2, atol=1e-2`。
- 如果 TF32 的数值误差可以通过测试，那么两次大 GEMM 有机会从慢的 FP32 `ampere_sgemm_128x128_nn` 转到 tensor-core-backed 路径。

### 测试结果与分析
- 两次 GEMM 已经从 reference 的 `ampere_sgemm_128x128_nn` 切换为 `cutlass_80_tensorop_s1688gemm_128x256_32x3_nn_align4`，说明 TF32/tensor core 路线生效。
- `custom_kernel` 主体时间约为两次 GEMM `4.882 ms` 加 Triton epilogue `0.446 ms`。
- 7 个 distribution kernel 来自输入生成 `generate_input`，不属于 SwiGLU kernel 主体。
- 当前瓶颈仍然是 GEMM：即使 epilogue 完全免费，理论上也只能从约 `5.33 ms` 降到约 `4.88 ms`。所以下一步优先尝试更激进的 matmul 精度/路径，而不是先深挖 epilogue。

## v2 - 合并两次 GEMM 为一次大 GEMM
- 改动：
  - 将 `torch.set_float32_matmul_precision("high")` 改为 `torch.set_float32_matmul_precision("medium")`。
  - 用 `WV = torch.cat((W, V), dim=1)` 得到 `[in_features, 2 * hidden_size]` 的权重矩阵。
  - 用一次 `torch.mm(x_flat, WV, out=both)` 同时计算 `xW` 和 `xV`。
  - 修改 Triton epilogue，让它从 `both[:, :hidden]` 读取 gate，从 `both[:, hidden:]` 读取 value。
- 实验假设：
  - 原先两次 GEMM 各约 `2.44 ms`，合并后有机会减少一次 GEMM launch，并让 `x` 的读取/缓存复用更好。
  - 大 GEMM 的 shape 从两个 `[16384, 2048] x [2048, 4096]` 变成一个 `[16384, 2048] x [2048, 8192]`。
  - 如果 `torch.cat((W, V), dim=1)` 的额外 copy 成本不高，benchmark 可能低于 v2 的 `4.709 ms`。
- 风险：
  - `torch.cat` 每次调用都会额外分配并复制权重，可能抵消合并 GEMM 的收益。
  - cuBLAS/CUTLASS 对 `N=8192` 的 kernel 选择不一定比两个 `N=4096` GEMM 更快。
- 服务器上需要检查：
  - `PYTHONPATH="$PWD:$PWD/.." python ../eval.py test test_cases/test.txt`
  - `PYTHONPATH="$PWD:$PWD/.." python ../eval.py benchmark test_cases/test.txt`
  - 如果 benchmark 有收益，再用 nsys 看是否变成 1 个 GEMM kernel，以及 `torch.cat` 是否引入额外 copy kernel。


## v3 - 手写triton合并 gate + sigmoid + mul

- 改动：
  - 不再依赖 `torch.mm` 做主 matmul。
  - 直接在一个 Triton kernel 里完成 `x @ W` 和 `x @ V` 的 tile 级计算。
  - 对每个输出 tile，沿 `K_total` 维循环累加两个 accumulator：
    - `out_W_tile` 对应 gate 分支
    - `out_V_tile` 对应 value 分支
  - 在同一个 kernel 内完成：
    - `+ b`
    - `+ c`
    - `sigmoid(beta * gate)`
    - `gate * sigmoid(...) * value`
  - 只把最终 `out_tile` 写回 global memory。
- 目前这版的目的：
  - 验证一个真正的 fused matmul 能否进一步减少中间 tensor 写回。
  - 如果成功，理论上可以避免把完整 gate/value 中间结果落到 HBM。
- 当前设计假设：
  - `M_total = batch_size * seq_len`
  - `N_total = hidden_size`
  - `K_total = in_features`
  - `BLOCK_M/N/K` 是 Triton software tile，不是硬件固定值，但应尽量选 tensor-core 友好的倍数。
- 当前风险：
  - 一个 program 里同时维护 gate/value 两套 accumulator，寄存器压力会明显上升。
  - tile 过大可能 spill，反而比 `torch.mm` 慢。
  - `BLOCK_M/BLOCK_N/BLOCK_K` 的最佳值需要靠实际 benchmark 调。
- 下一步验证：
  - 先跑 `test` 确认 correctness。
  - 再跑 `benchmark` 看是否真的比 v2 更快。
  - 如果编译或性能不理想，优先调 `BLOCK_M/BLOCK_N/BLOCK_K`，再考虑把 kernel 拆回“两次 GEMM + epilogue”的版本。
