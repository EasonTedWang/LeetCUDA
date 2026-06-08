# LeetCUDA CUDA 学习大纲

> 这是一份会持续更新的 CUDA 学习路线文档。
> 后续我们会根据你的实际学习进度，随时补充笔记、标记完成项、修正学习顺序、记录踩坑和新的练习任务。

## 当前状态

- 仓库路径：`/home/eason/LeetCUDA`
- 学习目标：基于 LeetCUDA，从零开始系统学习 CUDA 编程。
- 已初始化的子仓库：
  - `HGEMM`
  - `ffpa-attn`
  - `third-party/cutlass`
- 推荐起点：`kernels/elementwise`
- 当前阶段：第 0 阶段 / 第 1 阶段准备

## 这个仓库适合怎样学习 CUDA

LeetCUDA 不是一个单一应用，而是一组 CUDA kernel 练习、优化和 benchmark 的集合。大多数目录都遵循类似流程：

```text
CUDA kernel 实现
-> PyTorch C++/CUDA extension 绑定
-> Python 正确性检查和性能测试
-> 对比 torch / cuBLAS / 其他 baseline
-> 逐步加入优化技巧
```

重要目录说明：

- `kernels/elementwise`：CUDA 基础、线程索引、向量化访存、`half`、`half2`。
- `kernels/relu`、`kernels/sigmoid`、`kernels/gelu`、`kernels/swish`：激活函数和逐元素算子变体。
- `kernels/reduce`：warp reduce、block reduce、shared memory、atomic。
- `kernels/dot-product`：把 reduce 用到点积场景。
- `kernels/softmax`：safe softmax、online softmax、per-token reduce。
- `kernels/mat-transpose`：二维线程映射、访存合并、shared memory、bank conflict。
- `kernels/layer-norm`、`kernels/rms-norm`：LLM 常用的 fused reduce + elementwise kernel。
- `kernels/sgemv`、`kernels/hgemv`：矩阵向量乘。
- `kernels/sgemm`：CUDA Cores GEMM 优化路线。
- `kernels/hgemm`：Tensor Cores、WMMA、MMA PTX、CuTe、swizzle、多 stage pipeline。
- `kernels/flash-attn`：高阶融合 attention kernel。
- `kernels/openai-triton`：Triton 对照实现。
- `kernels/nvidia-nsight`：Nsight profiling 笔记。

## 第 0 阶段：环境和运行流程

目标：先跑通一个已有 kernel，理解这个仓库的基本使用方式。

主要文件：

- `kernels/elementwise/README.md`
- `kernels/elementwise/elementwise.py`
- `kernels/elementwise/elementwise.cu`

推荐命令：

```bash
cd /home/eason/LeetCUDA/kernels/elementwise
export TORCH_CUDA_ARCH_LIST=Ada
python3 elementwise.py
```

如果你的显卡不是 Ada 架构，后续需要根据 GPU 型号调整 `TORCH_CUDA_ARCH_LIST`，例如 `Ampere`、`Ada`、`Hopper`。

需要掌握：

- PyTorch 如何 JIT 编译 CUDA extension。
- `torch.utils.cpp_extension.load` 的作用。
- `.cu` 里的 CUDA kernel 如何变成 Python 可调用函数。
- benchmark 脚本如何做 warmup、计时、同步和输出。

练习任务：

- 运行 `elementwise.py`。
- 记录是否编译成功。
- 记录 GPU 型号、CUDA 版本、PyTorch 版本。

进度：

- [ ] 确认 GPU 型号。
- [ ] 确认 CUDA / PyTorch 环境。
- [ ] 跑通 `kernels/elementwise/elementwise.py`。
- [ ] 理解 Python benchmark wrapper。

## 第 1 阶段：CUDA 最小心智模型

目标：理解最小可用 CUDA kernel。

主要文件：

- `kernels/elementwise/elementwise.cu`
- `kernels/elementwise/elementwise.py`
- `kernels/relu/relu.cu`

核心概念：

- `__global__` kernel 函数。
- `blockIdx.x`、`blockDim.x`、`threadIdx.x`。
- 一维 grid 和一维 block。
- 一个线程负责一个输出元素。
- 边界检查：`if (idx < N)`。
- PyTorch tensor 如何作为指针传入 CUDA kernel。

重点 kernel：

- `elementwise_add_f32_kernel`
- `elementwise_add_f32x4_kernel`
- `elementwise_add_f16_kernel`

练习任务：

- 写一个简单的 `vector_add_f32`。
- 改成 `vector_mul_f32`。
- 改成 `relu_f32`。
- 加入 `float4` 向量化版本。
- 和 `torch.add` 或 `torch.relu` 对比性能。

进度：

- [ ] 能解释 `idx = blockIdx.x * blockDim.x + threadIdx.x`。
- [ ] 能解释为什么需要边界检查。
- [ ] 能解释 Python 如何调用 CUDA 函数。
- [ ] 实现一个新的小型 elementwise kernel。

## 第 2 阶段：访存、向量化和 half

目标：理解为什么很多简单 kernel 是 memory-bound，以及向量化为什么有用。

主要文件：

- `kernels/elementwise/elementwise.cu`
- `kernels/sigmoid/sigmoid.cu`
- `kernels/gelu/gelu.cu`
- `kernels/swish/swish.cu`

核心概念：

- global memory coalescing。
- `float4` / `int4` / 128-bit load-store。
- `half`、`half2`。
- `__hadd`、`__hadd2`。
- 通过 reinterpret cast 做 packed load/store。
- `--use_fast_math` 的影响。

重点 kernel：

- `elementwise_add_f32x4_kernel`
- `elementwise_add_f16x2_kernel`
- `elementwise_add_f16x8_kernel`
- `elementwise_add_f16x8_pack_kernel`

练习任务：

- 增加 `sub`、`mul` 或 `div` 版本。
- 实现 `f32x4`、`f16x2`、`f16x8_pack`。
- 对不同 shape 做 benchmark。
- 判断每个 kernel 是 memory-bound 还是 compute-bound。

进度：

- [ ] 理解 `FLOAT4(value)` 宏。
- [ ] 理解 `HALF2(value)` 宏。
- [ ] 能解释为什么 `f16x8_pack` 可能更快。
- [ ] 记录 benchmark 观察。

## 第 3 阶段：warp 和 block 协作

目标：学习线程如何在 warp 内、block 内协作。

主要文件：

- `kernels/reduce/block_all_reduce.cu`
- `kernels/reduce/block_all_reduce.py`
- `kernels/dot-product/dot_product.cu`

核心概念：

- warp 是 GPU 调度的重要单位。
- `WARP_SIZE = 32`。
- `__shfl_xor_sync`。
- warp reduce。
- shared memory 做跨 warp reduce。
- `__syncthreads`。
- `atomicAdd`。
- accumulate dtype 和数值误差。

重点 kernel：

- `warp_reduce_sum_f32`
- `block_all_reduce_sum_f32_f32_kernel`
- `block_all_reduce_sum_f32x4_f32_kernel`
- `block_all_reduce_sum_f16_f32_kernel`

练习任务：

- 写 `sum_f32`。
- 写 `max_f32`。
- 写 `dot_product_f32`。
- 写 `f16 input + f32 accumulation` 版本。
- 对比精度和性能。

进度：

- [ ] 能解释 warp-level reduce。
- [ ] 能解释 block-level reduce。
- [ ] 能解释为什么跨 warp 需要 shared memory。
- [ ] 能解释 fp16 accumulate 的取舍。

## 第 4 阶段：Softmax 和在线算法

目标：把 reduce 模式连接到 LLM 里非常关键的 softmax。

主要文件：

- `kernels/softmax/softmax.cu`
- `kernels/softmax/softmax.py`

核心概念：

- naive softmax。
- 数值溢出和 safe softmax。
- 先 reduce max，再 reduce sum。
- 使用 `(m, d)` 状态的 online softmax。
- per-token kernel 映射。

重点 kernel：

- `softmax_f32_per_token_kernel`
- `safe_softmax_f32_per_token_kernel`
- `safe_softmax_f32x4_per_token_kernel`
- `safe_softmax_f16_f32_per_token_kernel`
- `online_safe_softmax_f32_per_token_kernel`

练习任务：

- 实现 safe softmax f32。
- 加入 `f32x4` 版本。
- 加入 `f16 input + f32 accumulation` 版本。
- 和 `torch.softmax` 对比。
- 解释 online softmax 如何为 FlashAttention 做准备。

进度：

- [ ] 能解释 naive softmax 为什么不稳定。
- [ ] 能解释 safe softmax。
- [ ] 能解释 online softmax 的 `(m, d)` 状态。
- [ ] benchmark safe 和 online 版本。

## 第 5 阶段：二维映射和 shared memory

目标：学习二维线程映射、访存合并和 shared memory layout。

主要文件：

- `kernels/mat-transpose/mat_transpose.cu`
- `kernels/mat-transpose/mat_transpose.py`
- `kernels/nvidia-nsight/bank_conflicts.md`
- `kernels/swizzle/README.md`

核心概念：

- 二维 grid 和二维 block。
- row-major 和 column-major 访问差异。
- coalesced read 和 coalesced write。
- shared memory tile。
- bank conflict。
- padding 和 swizzle。

重点 kernel：

- `mat_transpose_f32_col2row_kernel`
- `mat_transpose_f32_row2col_kernel`
- `mat_transpose_f32x4_shared_col2row_kernel`
- `mat_transpose_f32x4_shared_bcf_col2row_kernel`

练习任务：

- 写 naive transpose。
- 写 2D block transpose。
- 加 shared memory tile。
- 加 padding 减少 bank conflict。
- 使用 Nsight Compute 观察 shared memory 行为。

进度：

- [ ] 能解释 coalesced memory access。
- [ ] 能解释 shared memory tile transpose。
- [ ] 能解释 bank conflict。
- [ ] 对比 naive 和 shared-memory transpose。

## 第 6 阶段：LLM 常用 Norm 和位置编码 kernel

目标：学习 transformer 模型中常见的 fused kernel。

主要文件：

- `kernels/layer-norm/layer_norm.cu`
- `kernels/rms-norm/rms_norm.cu`
- `kernels/rope/rope.cu`
- `kernels/embedding/embedding.cu`

核心概念：

- reduce + elementwise fusion。
- per-token / per-row 执行方式。
- fp16 input + fp32 accumulation。
- 通过融合减少 global memory traffic。
- LLM kernel 的常见 shape 模式。

练习任务：

- 实现 RMSNorm f32。
- 实现 f16 input + f32 accumulation 的 RMSNorm。
- 实现 LayerNorm。
- 加入向量化 load/store。
- 和 PyTorch baseline 对比。

进度：

- [ ] 能解释 RMSNorm 计算过程。
- [ ] 能解释 LayerNorm 计算过程。
- [ ] 能解释为什么 kernel fusion 重要。
- [ ] benchmark 自定义 norm kernel。

## 第 7 阶段：从 GEMV 到 GEMM

目标：进入矩阵乘法，学习 CUDA Cores GEMM 的逐步优化路线。

主要文件：

- `kernels/sgemv/sgemv.cu`
- `kernels/hgemv/hgemv.cu`
- `kernels/sgemm/sgemm.cu`
- `kernels/sgemm/sgemm_async.cu`
- `kernels/sgemm/README.md`

核心概念：

- GEMV 和 GEMM 的区别。
- block tile：`BM x BN x BK`。
- thread tile。
- register tile。
- shared memory A/B tile。
- double buffering。
- async copy。
- bank-conflict-free layout。

建议学习顺序：

1. `sgemm_naive_f32`
2. `sgemm_sliced_k_f32`
3. `sgemm_t_8x8_sliced_k_f32x4`
4. `sgemm_t_8x8_sliced_k_f32x4_bcf`
5. `sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf`
6. `sgemm_async.cu`

练习任务：

- 画出一个 GEMM kernel 的 tile layout。
- 解释哪些数据在 global memory、shared memory、register 中。
- benchmark 每一步优化。
- 记录 TFLOPS 和 speedup。

进度：

- [ ] 理解 naive GEMM。
- [ ] 理解 sliced-K shared memory GEMM。
- [ ] 理解 thread tile。
- [ ] 理解 double buffering。
- [ ] 理解 async copy。

## 第 8 阶段：Tensor Cores、WMMA、MMA PTX 和 CuTe

目标：从 CUDA Cores GEMM 进入 Tensor Core GEMM。

主要文件：

- `kernels/hgemm/README.md`
- `kernels/hgemm/wmma/hgemm_wmma.cu`
- `kernels/hgemm/mma/basic/hgemm_mma.cu`
- `kernels/hgemm/mma/swizzle`
- `kernels/hgemm/cutlass`
- `HGEMM`

核心概念：

- CUDA Cores vs Tensor Cores。
- WMMA API。
- MMA PTX：`m16n8k16`。
- warp-level matrix multiply。
- MMA tile layout。
- shared memory swizzle。
- multi-stage pipeline。
- CuTe / CUTLASS 抽象。

练习任务：

- 阅读 WMMA naive kernel。
- 阅读 MMA naive kernel。
- 对比 WMMA 和 MMA API 灵活性。
- 学习 stage-2 / stage-3 pipeline。
- 学习 swizzle 和 bank conflict 避免方法。

进度：

- [ ] 能解释 Tensor Cores 在计算什么。
- [ ] 能解释 WMMA fragment。
- [ ] 能解释 MMA PTX tile shape。
- [ ] 能从高层解释 shared memory swizzle。

## 第 9 阶段：FlashAttention

目标：理解仓库里的高阶 fused attention kernel。

主要文件：

- `kernels/flash-attn/README.md`
- `kernels/flash-attn/mma/basic`
- `kernels/flash-attn/mma/swizzle`
- `ffpa-attn`

核心概念：

- attention = `QK^T` + softmax + `PV`。
- online softmax 避免保存完整 score matrix。
- split-KV 和 split-Q 策略。
- shared KV / shared QKV memory。
- QK tiling 和 QKV tiling。
- MMA + softmax fusion。
- SRAM 复杂度和 occupancy。

练习任务：

- 从矩阵运算重新推导 attention。
- 解释 attention 里的 online softmax。
- 对比 split-KV 和 split-Q。
- 阅读一个 basic FlashAttention MMA kernel。
- 再进入 swizzled variants。

进度：

- [ ] 能解释 attention 计算过程。
- [ ] 能解释 FlashAttention 里的 online softmax。
- [ ] 能解释 split-Q vs split-KV。
- [ ] 完整阅读一个 basic kernel。

## 学习日志

这个区域用来记录实际学习进度和路线修正。

### 2026-06-08

- 创建 CUDA 学习大纲。
- 初始化子仓库：`HGEMM`、`ffpa-attn`、`third-party/cutlass`。
- 确定从 `kernels/elementwise` 开始。
- 将学习大纲改为中文版本，便于后续持续维护。

## 下一步

从这里开始：

```bash
cd /home/eason/LeetCUDA/kernels/elementwise
export TORCH_CUDA_ARCH_LIST=Ada
python3 elementwise.py
```

然后按这个顺序学习：

1. `elementwise.py`
2. `elementwise_add_f32_kernel`
3. `elementwise_add_f32x4_kernel`
4. `elementwise_add_f16_kernel`
5. `elementwise_add_f16x2_kernel`
6. `elementwise_add_f16x8_pack_kernel`
