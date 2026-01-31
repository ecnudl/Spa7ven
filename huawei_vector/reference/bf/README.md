小提示（不改算法但通常能提速）

编译：-O3 基本必开；如果允许可试 -march=native（有些赛题禁用/无效）

dot_product 里我用 double 累加更稳；如果你只追分数/速度，可改成 float 累加

若你要继续做你之前写的 4/8/16 按 doc 分块循环展开，这两份代码的扫描循环位置就是你要替换的地方（我也可以在下一条直接给你一个带 4/8/16 分块版本）


“开 SIMD 指令”在 C++ 里通常指两件事：

1. **让编译器做自动向量化**（把你的标量循环变成 SSE/AVX/NEON 指令）
2. **你手写 SIMD**（intrinsics / asm / ISPC）

你原文档里说“无法开启编译器 SIMD 指令优化”，一般是指：**线上评测的编译命令/目标架构被固定**（例如禁止 `-march=native`，甚至用 `-fno-tree-vectorize` 之类关掉向量化）。这时就只能用“循环展开 + 代码形态诱导 SLP”来榨性能（你前面总结的思路就是这个）。

下面按“你能控制编译参数”与“你不能控制编译参数”分别说怎么做。

---

## 1) 你能控制编译参数：最直接的“开法”

### GCC（x86：SSE/AVX）

最常用的一组：

```bash
g++ -O3 -march=native -ffast-math -funroll-loops -DNDEBUG main.cpp -o main
```

* `-O3`：开启更激进的优化（很多自动向量化在高优化等级才更积极）
* `-march=native`：按你这台机器的 CPU 指令集生成（可能启用 AVX2/AVX-512/FMA 等）
* `-ffast-math`：允许更激进的浮点重排/近似，**可能改变数值**（竞赛得看容忍度）
* `-funroll-loops`：循环展开（有时对向量化/流水有帮助）

如果你**不能用 `-march=native`**（评测机不让/不稳定），只能显式指定，比如（以 AVX2+FMA 为例）：

```bash
g++ -O3 -mavx2 -mfma -ffast-math -funroll-loops main.cpp -o main
```

⚠️ 注意：`-mavx2` 在不支持 AVX2 的机器上会直接 **Illegal instruction** 崩掉；线上评测 CPU 不确定时别乱开。

### Clang/LLVM（x86）

```bash
clang++ -O3 -march=native -ffast-math -funroll-loops main.cpp -o main
```

Clang 一般默认也会开 loop vectorize + SLP vectorize（在 `-O2/-O3` 下），但你也可以显式指定：

```bash
clang++ -O3 -march=native -fvectorize -fslp-vectorize main.cpp -o main
```

### ARM（NEON）

在 ARMv8-A 上 NEON 通常是基础能力；常见做法还是：

```bash
g++ -O3 -march=native main.cpp -o main
```

---

## 2) 看看“到底有没有向量化成功”：开报告最靠谱

### GCC：向量化报告

```bash
g++ -O3 -march=native -fopt-info-vec-optimized -fopt-info-vec-missed main.cpp -o main
```

你会看到哪些循环“vectorized”、哪些“missed（失败原因）”。（现代 CPU 的向量化通常是编译器自动完成的，这类报告是定位问题的最快方式。([weedge.github.io][1])）

### Clang：向量化报告

```bash
clang++ -O3 -march=native \
  -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize \
  -Rpass=slp-vectorize  -Rpass-missed=slp-vectorize  -Rpass-analysis=slp-vectorize \
  main.cpp -o main
```

---

## 3) 代码层“让编译器更敢向量化”的关键写法

即使你开了 `-O3 -march=...`，代码写法不对也会向量化失败。常见要点：

### (1) 消除别名（aliasing）

```cpp
float dot(const float* __restrict__ a,
          const float* __restrict__ b, int L)
```

`__restrict__` 能显著提升编译器把循环向量化的概率。

### (2) 保证内存连续、步长为 1

* 用 `float*` 连续数组，而不是 `vector<vector<float>>`
* 尽量 `docs[m*L + d]` 这种线性访问

### (3) 尽量避免循环里有分支/函数调用

* 把 if/边界处理放到循环外
* dot 内联（`static inline`）

### (4) 用 pragma 明示可向量化（可选）

* OpenMP SIMD（不一定需要开多线程）：

```cpp
#pragma omp simd reduction(+:sum)
for (int i = 0; i < L; ++i) sum += a[i] * b[i];
```

* GCC/Clang 提示（效果视编译器而定）：

```cpp
#pragma GCC ivdep
for (...) ...
```

---

## 4) 如果线上“禁止开 SIMD 编译选项”，你还能怎么办？

这就是你原文档说的情况：**编译器/平台限制导致你没法通过 flags 开 SIMD**。

此时可行的是：

* **循环展开（4/8/16）**：把代码形态改得更像“可做 SLP 向量化/更好流水”的模式
* **数据布局 + 连续访问**：让编译器更容易生成宽 load / 更好预取
* **减少运算（归一化→点积 / 预计算 norm²）**：让瓶颈更纯粹、减少标量开销

这也解释了他们为什么强调“无法开 SIMD → 用展开诱导编译器尝试生成向量化访存/计算”。

---

## 5) 你想“强制 SIMD”而不靠自动向量化（手写 intrinsics）怎么办？

如果赛题不禁止，你就可以：

* x86：`#include <immintrin.h>` 用 `_mm256_loadu_ps / _mm256_fmadd_ps ...`
* ARM：`#include <arm_neon.h>` 用 `vld1q_f32 / vfmaq_f32 ...`

但很多竞速赛会：

* 限制编译参数 / 禁止特定指令集
* 或在多种 CPU 上评测，导致手写 AVX2 不可移植

所以要先看赛题的“编译命令/评测机架构说明”。

---

如果你把**赛题的编译命令**（比如评测脚本里 g++ 的参数）贴出来，我可以直接判断：

* 到底是“不能用 -march=native”，还是“直接关了向量化”，
* 以及你应该用哪一种展开/pragma 组合最稳。

[1]: https://weedge.github.io/perf-book-cn/zh/chapters/9-Optimizing-Computations/9-4_Vectorization_cn.html?utm_source=chatgpt.com "9-4.向量化· 现代CPU上的性能分析与优化"
