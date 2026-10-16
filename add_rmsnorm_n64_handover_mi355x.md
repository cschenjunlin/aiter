# N=64 fused add + RMSNorm 交接说明（MI355X 这一轮，供 MI308X 继续）

日期：2026-10-16
代码：personal `cjl_backup/rmsnorm_perf`。代码改动到 `a16cda40e` 为止，本文档是其后单独的一笔提交。本地 `cjl/rmsnorm_perf` 也在 `a16cda40e`，但不含本文档；personal `cjl/rmsnorm_perf` 仍停在 `9c59aa0b3`。

上一份交接说明是在 MI308X 上写的 `add_rmsnorm_n64_handover.md`（没有入库）。本文只写这一轮新增的内容，背景只做简述。

## 背景

- 接口是 `aiter::add_rmsnorm` 的非量化路径：`residual_out = input + residual_in`，`out = RMSNorm(residual_out, weight, eps)`。
- BF16、contiguous、hidden=64、`eps = 1/128`。目标 shape 为 `[59392, 64]`（四块数据共 30.4 MB）和 `[32768, 64]`（16.8 MB）。
- 客户记录：H20 参考 7.744 / 4.288 µs；MI308X 上客户 Triton 13.321 / 10.881 µs，原 aiter 通用 kernel 43.76 / 22.2 µs。
- 上一轮在 MI308X 上用同一脚本测 `fb29dcb33`（每 block 8 行、先读完再算）是 22.05 / 14.54 µs。

## 这一轮的提交

| 提交 | 内容 |
|---|---|
| `9c59aa0b3` | 计时脚本 `op_tests/_bench_add_rmsnorm_n64.py` |
| `5439c34e1` | 新增 `add_rmsnorm_n64_vec_kernel` 并接入分发 |
| `a16cda40e` | 新 kernel 的两处写回改用 non-temporal store |

### 新 kernel 的做法

代码在 `csrc/kernels/rmsnorm_quant_kernels.cu`，`add_rmsnorm_n64_vec_kernel` 紧接在原 `add_rmsnorm_n64_kernel` 之后，启动宏是 `ADD_RMSNORM_N64_VEC_KERNEL_IMPL`。

- 每个 64 线程的 block 负责 8 行。每行 64 个元素分给 8 个 lane，每个 lane 一次读写 8 个元素（16 字节，`global_load_dwordx4` / `global_store_dwordx4`），8 行同时处理。网格大小和原 kernel 相同，都是 m/8 个 block。
- 平方和：每个 lane 先累加自己 8 个元素的平方，再用 `multithread_reduce(…, 8)` 在 8 个 lane 内做 3 级 DPP，不再需要 `ds_bpermute` 广播。
- `epsilon` 在启动宏里转成 float，计算 `rsqrtf(sum * (1/64) + eps)`。
- bf16/fp16 转换用 `__builtin_convertvector`，在两种架构上都是就近舍入。不要换成 opus 的 `cast` 或 `store_vector`：`OPUS_FP32_to_BF16_DEFAULT=2` 在 gfx942 上是截断。
- 尾块：每行做一次运行时越界判断（`valid`）。
- 写回：`a16cda40e` 起改用 `__builtin_nontemporal_store`，生成 `global_store_dwordx4 … nt`。
- gfx950 汇编：26 个 VGPR，3 条 `global_load_dwordx4`（weight 加两份输入），3 级 DPP，8 条 `v_cvt_pk_bf16_f32`。gfx942 能编过，结构相同，只是 bf16 转换由 `v_bfe_u32`、`v_add3_u32`、`v_perm_b32` 等指令在软件里完成。

### 分发

`ADD_RMSNORM_KERNEL_DISPATCH` 的 `n == 64` 分支：

- `m >= cu_num * 32` 且 `n64_rows_aligned`：走新 kernel。`n64_rows_aligned` 要求五个指针都 16 字节对齐、四个 stride 都是 8 的倍数。
- `m >= cu_num * 32` 但不对齐：原 kernel，每 block 8 行。
- `m < cu_num * 32`：不论是否对齐，都走原 kernel，每 block 1 行。

门槛在 MI355X 上是 8192，在 MI308X 上是 2560。两种目标 shape 在两台机器上都会走新 kernel。

## MI355X 上的测量

环境：MI355X（gfx950，256 CU，HBM3E 峰值 8 TB/s，256 MB Infinity Cache），容器镜像 `rocm/pytorch-private:ato-dsv3-optimized-9-29`（ROCm 7.2.4，torch 2.15 dev）。这个镜像缺 `psutil` 和 `pybind11`，按 `requirements.txt` 的版本另外装了。

### 计时脚本（主指标）

`run_perftest` 默认轮换 100 份参数，测的是冷 cache 时间。三个版本的 `.so` 预先编好，测量时交替拷回 `aiter/jit/` 再跑脚本，不重编。每遍前后都检查 GPU 上没有其他进程，6 轮全部干净：

| 版本 | `[59392, 64]` | `[32768, 64]` |
|---|---:|---:|
| `fb29dcb33` | 9.21–9.39，均值 9.32 µs | 6.84–6.97，均值 6.92 µs |
| `5439c34e1` | 8.54–9.10，均值 8.91 µs（−4%） | 5.24–5.99，均值 5.83 µs（−16%） |
| `a16cda40e` | 7.80–7.92，均值 7.88 µs（−15%） | 5.63–5.69，均值 5.66 µs（−18%） |

按四块数据计算，`a16cda40e` 的有效带宽约 3.9 / 3.0 TB/s，`fb29dcb33` 约 3.3 / 2.4 TB/s。

### 独立程序（辅助）

用 `rocprofv3` 计时，各变体在同一进程里交替测，3 个进程取平均。发射方式有两种：背靠背，以及两次发射之间在 host 端留 30 µs 空闲（模拟 Python 发射的间隙）。冷 cache 用 100 套缓冲区轮换，热 cache 只用一套。下面的数字是相对 `fb29dcb33` 的耗时变化，每格依次为背靠背的大 / 小 shape、留空闲的大 / 小 shape：

| 版本 | 冷 cache | 热 cache |
|---|---|---|
| 只改布局 | −9 / −8 · −4 / −17% | −11 / −11 · −25 / −22% |
| 改布局 + nt 写回 | −14 / −13 · −14 / −19% | −28 / −39 · −40 / −37% |

- **离下限还有多远：** 冷 cache 下，和搬运同样字节数的 nt copy kernel 相差在 ±3% 以内（背靠背：6.96 / 5.71 µs 对 6.81 / 5.90 µs）。本机冷 cache 下基本没有余量了。
- **对下游的影响：** 每次 kernel 后紧接一个读 `out` 的 kernel，nt 写回没有让它变慢（大 shape 3.81–3.90 µs）。不过这个测试的灵敏度有限。
- **小 `m`：** 目前 `m < cu_num * 32` 走原 kernel，所以和 `fb29dcb33` 一样快。如果改成也走新 kernel，本机冷 cache 下 `m=16` 快 23%、`m=4096` 快 15%、`m=8191` 快 28%，`m` 为 1、128、1024 时持平（±0.1 µs）。为了保持改动最小，这一项没有提交。

### 正确性

覆盖 17 种情况：两种目标 shape；`m` 为 8195、8192、1000、9、7、1；对齐的非 64 stride；奇数 stride 和不对齐指针的回退；`gemma_norm` 及其回退；fp16 及其尾块和回退。

- `residual_out` 与 CPU 参考逐位一致，`out` 与参考的差异不超过 1 ulp。
- 每种情况实际调用的 kernel 都用 `rocprofv3` 确认过。
- 规约顺序变了，和原 kernel 相比，380 万个输出里有 2–5 个元素差 1 ulp。

## 试过但没采用的

除特别说明外，都是本机冷 cache 下的结果。

| 思路 | MI355X 上的结果 | 在 MI308X 上值得重测吗 |
|---|---|---|
| 每行数据一到就先算（上一份交接的建议，原布局） | 快 1–3%，在噪声范围内 | 低：新 kernel 每个 lane 只取一组数据，不适用 |
| 每 block 16 / 32 行（上一份交接的建议，原布局） | 慢 2–10% / 慢 16–30% | 低：原布局已被替换 |
| 每 block 2 / 4 行（原布局） | 2 行慢 4–40%，4 行持平 | 低 |
| 每行 4 个 lane（每 lane 32 字节）或 16 个 lane（每 lane 8 字节） | 比 8 lane 落后 5–14 个百分点 | 低 |
| 每 wave 2 或 4 组行、每 block 2 / 4 / 8 个 wave | 都不如每 wave 1 组、每 block 1 wave，落后 2–12 个百分点 | **高**，见下一节 |
| persistent kernel + 预取下一组，每 CU 1 / 2 / 4 / 8 个 block | 落后 4–14 个百分点 | **高**，见下一节 |
| 读取也用 nt | 冷 cache 小 shape 再快约 4%；热 cache 大 shape 明显变差（相对 baseline 只有 −10 / −14%，nt 写回版本是 −36 / −38%） | 中：要同时看冷、热 cache |
| 只对 `residual_out` 用 nt | 不比两处都用 nt 更好 | 低 |
| `__restrict__`、`__launch_bounds__(64)`、两段式尾块 | 冷 cache 下 ≤1%，热 cache 个别格子 2–3%；已删除 | 低 |
| `epsilon` 改回 double | 稳定慢 2–4% | 不要改回 |

有一项没有收尾：第一轮里"每 wave 2 组、每 block 4 wave、每行运行时越界判断"的版本（读写都用 nt）冷 cache 是 −19 / −17 · −23 / −25%，比最终结构更好。第二轮改成编译期分两路处理尾块后，它退到 −9 / −12 · −11 / −17%；带运行时判断的那一版没有再复测。

## MI308X 与 MI355X 的差别

| | MI355X | MI308X |
|---|---|---|
| 架构 | gfx950 | gfx942 |
| CU 数 | 256 | 80 |
| 同时驻留的 wave（每 CU 32 个） | 8192 | 2560 |
| 两种 shape 的 7424 / 4096 个 block 要跑几轮 | 不到 1 轮 | 约 2.9 / 1.6 轮 |
| 每个 CU 分到的 HBM 带宽 | 约 31 GB/s | 约 66 GB/s |
| fp32 转 bf16 | 硬件指令 | 软件就近舍入 |

这些差别会影响下面几个结论：

- **"每 wave 1 组、每 block 1 个 wave 最好""persistent 没用"不一定成立。** 这两个结论成立的前提是整个网格一轮就能全部驻留。MI308X 要跑好几轮，每个 CU 承担的带宽也约是本机的两倍，需要更多读取同时在路上，所以每个 wave 多带几组、或者用 persistent 加预取，都可能变得有用。
- **nt 写回在 gfx942 上的效果未知。** `5439c34e1` 和 `a16cda40e` 拆成两笔提交，就是为了方便在 MI308X 上直接对比。
- **gfx942 上 VALU 指令更多**（bf16 软件转换）。这个 kernel 主要受访存限制，预计影响不大，但要看汇编和实测确认。

## MI308X 上建议的步骤

1. 取代码：`git fetch personal`，然后把工作分支对到 `personal/cjl_backup/rmsnorm_perf`（代码部分和 `a16cda40e` 相同）。
2. 跑脚本：`AITER_REBUILD=1 python op_tests/_bench_add_rmsnorm_n64.py`。用 torch profiler 确认实际执行的是 `add_rmsnorm_n64_vec_kernel<std::bfloat16_t>`。
3. 三个版本交替对比，`fb29dcb33`、`5439c34e1`、`a16cda40e` 各跑 6 遍以上：
   - 对每个提交执行 `git checkout <提交> -- csrc/kernels/rmsnorm_quant_kernels.cu`，用 `AITER_REBUILD=1` 编一次，把 `aiter/jit/module_rmsnorm_quant.so` 另存一份；
   - 之后不再重编，轮流把各版本的 `.so` 拷回原位运行脚本；
   - 测完用 `git checkout HEAD -- csrc/kernels/rmsnorm_quant_kernels.cu` 还原源码。
4. 如果 `a16cda40e` 不比 `5439c34e1` 快，可以只 revert nt 写回那一笔，或者按架构选择。
5. 在 MI308X 上重新扫结构参数：每 wave 1 / 2 / 4 组（读取全部提前发出）、每 block 1 / 4 个 wave、persistent 加预取、nt 读取，冷、热 cache 都要看。上一节没收尾的"2 组 × 4 wave + 运行时越界判断"也一并测。
6. 在 gfx942 上补跑正确性：尾块、stride 回退、fp16、`gemma_norm`。这里的 bf16 转换路径和 gfx950 不同。
7. 和客户 Triton（13.321 / 10.881 µs）比较。上一轮同一脚本测 `fb29dcb33` 是 22.05 / 14.54 µs。

## 测量注意事项

- **注意其他用户的任务。** 本机这一轮多次被别人的任务干扰（占满 8 张卡的训练和 pytest），测出过正常值 2–10 倍的异常结果。每遍前后用 `amd-smi process` 确认 GPU 上没有其他进程。
- **两种计时的绝对值不同。** 脚本用 torch profiler，从 Python 逐次发射，kernel 之间有空隙；背靠背发射的 `rocprofv3` 计时结果会更低。比较版本时要交替运行，看配对差。
- **进程间有漂移。** 同一个程序换一个进程重跑，结果可能整体漂 0.5 µs 左右，可能和缓冲区的物理地址有关。
- **编译参数。** ROCm 7.2 的 LLVM 不认识 `-amdgpu-coerce-illegal-types`，aiter JIT 会自动过滤掉；手工用 `hipcc` 编时要去掉这个参数。实际使用的编译参数可以在 `aiter/jit/build/module_rmsnorm_quant/build/build.ninja` 里查到。
- **镜像依赖。** 有些镜像缺 `psutil` 和 `pybind11`，`import aiter` 会在 JIT 编译时失败。

## MI355X 机器上的工具（不在仓库里）

都在 `/tmp/n64_verify/`，不依赖 torch，直接 include `rmsnorm_quant_kernels.cu` 编译：

- `bench.cpp`：调用 `aiter::add_rmsnorm` 的 A/B 程序，带正确性检查；
- `opt/explore.cpp`：变体扫描（原布局变体、向量化各配置、nt、persistent、copy 参照）；
- `opt/final_check.cpp`：17 种正确性情况，以及小 `m` 计时；
- `opt/slim_ab.cpp`：可去掉项的 16 种组合消融；
- `opt/build.sh`：和 aiter JIT 相同的编译参数（ROCm 7.2，gfx950）；
- `opt/parse_explore.py`、`opt/ab_idle.sh`：结果解析，以及只在 GPU 空闲时运行的脚本 A/B。

这些文件里写死了本机的路径和容器名，在其他机器上使用前要先修改。
