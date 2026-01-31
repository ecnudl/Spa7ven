下面把你给的要点，整理成一个“从数学到工程”的完整算法说明：这个 **huaweifirst_worldlatter** 团队本质上做的是 **全量扫描（brute-force KNN/ANN）+ 一系列极致的 CPU 微优化 + 查询结果缓存**，用工程优化把“看起来是 ANN 的题”做成“几乎是精确扫描但非常快”。（这类思路在华为算法精英实战营该题的经验分享里也很典型：核心是精度不掉的前提下追极致性能。）([bbs.huaweicloud.com][1])

---

## 1) 任务抽象：对每个 query 找 topK 最近的 doc

* 输入：`M` 个 doc 向量（高维、浮点），以及多次 query 请求（每次 1 个 query）
* 输出：每个 query 对应的 topK doc id（以及可选距离/相似度）

关键点：如果你不建索引结构（HNSW/IVF-PQ 等），那每次 query 就必须扫完所有 `M` 个 doc，计算距离/相似度，然后选 topK。

这个队伍做的就是：**把“扫完所有 doc + 选 topK”这条链路做到极致快**，并用 cache 抵消重复 query。

---

## 2) 距离计算的“简化”：从 L2 距离变成点积（或点积 + 常数项）

你提到的核心公式：

[
|x-y|^2 = |x|^2 + |y|^2 - 2x^\top y
]

### 情况 A：doc 与 query 都做 L2 归一化（单位向量）

若 (|x|=|y|=1)，则

[
|x-y|^2 = 2 - 2x^\top y
]

于是 **最小 L2 距离** ⇔ **最大点积 (x^\top y)**（或最大 cosine，相同）

* 每个 doc 只需要算一次 **点积**（维度为 L，则是 L 次乘加）
* 不需要 `x-y`、不需要平方、不需要开方
* 排名也不变（因为 `2-2·dot` 是单调变换）

> 这解释了你写的“只用一次乘法近似计算”：更准确说是“每维只剩乘加，没有减法/平方/开方”。

### 情况 B：只对 doc 预处理（更一般也更稳）

即使 query 不归一化，也可以：

* 预计算每个 doc 的 (|x|^2)（一次性离线算好存起来）
* 每次 query 只算一次 (|y|^2)
* 每个 doc 仍然只需要：`dist2 = doc_norm2 + query_norm2 - 2*dot`

这样每个 doc 的额外开销就是常数次加法，**仍然把“平方/开方”从热路径上移走**。

---

## 3) “循环展开”怎么用在向量检索里：按 doc 分块并行算多个点积

你描述的 4/8/16 展开，本质是：

* 不是在维度 L 上展开（那只是减少 loop overhead）
* 而是对 **doc 维度**做 blocking：一次算 4/8/16 个 doc 与同一个 query 的相似度

### 3.1 典型实现结构（伪代码）

假设 `docs` 是连续内存 `docs[M][L]`（row-major），`q[L]`。

```cpp
for (int m = 0; m < M; m += B) {          // B = 4/8/16
  float acc[B] = {0};                    // B 个累加器(尽量进寄存器)
  for (int d = 0; d < L; ++d) {
    float qd = q[d];
    // 手动展开：一次更新 B 个 doc 的 dot
    acc[0] += docs[(m+0)*L + d] * qd;
    acc[1] += docs[(m+1)*L + d] * qd;
    ...
    acc[B-1] += docs[(m+B-1)*L + d] * qd;
  }
  // acc[i] -> dist2 或 similarity，然后参与 topK
}
```

### 3.2 为什么这能变快（即使“不让开 SIMD”）

你写的点很关键：**即便不能显式用 intrinsic/SIMD，编译器也可能对这种固定模式做自动向量化或更好的流水**：

* 同一个 `qd` 被 B 个 doc 重复使用（减少重复加载）
* `acc[0..B-1]` 常驻寄存器，减少内存读写
* 展开后循环结构更“像可向量化的矩阵乘法小块”，编译器更容易生成更宽的 load / FMA（或至少减少分支与 loop 控制开销）
* 同时减少了 `m` 层循环次数与分支预测压力

> 简单说：它是在“用代码形态诱导编译器优化”，不需要你手写 SIMD。

---

## 4) topK 不做全排序：用固定大小堆（优先队列）

扫描 `M` 个 doc 后，如果你把所有分数存下来再 sort，复杂度是 (O(M\log M))，而 K 通常很小。

他们用的是 **维护大小为 K 的堆**（常见写法是“最大堆存当前最差的那个”）：

* 初始化空堆
* 每来一个 doc 分数 `s`：

  * 堆未满：push
  * 堆已满：若 `s` 比堆顶（当前 topK 里最差）更好，则 pop + push
* 复杂度：(O(M\log K))

在大 M、小 K 时差距很大，而且内存也小很多。

---

## 5) cache：用“重复 query”直接跳过整次扫描

这是他们能拿到很高分的另一个关键点：如果评测数据里 **query 有重复**，那么缓存 topK 结果可以把重复请求从：

* `O(M·L + M log K)` 直接变成 `O(1)`（hash 查表 + 输出）

### cache 的关键实现细节（通常这么做）

* key 怎么生成：

  * 如果输入里有 query id：直接用 id 做 key
  * 如果没有 id：就对 query 向量的原始 bytes / 量化后的值做 hash（注意浮点直接 hash 要小心一致性）
* value 存什么：

  * 直接存 `topK doc ids`（以及可选分数），输出时直接打印

> 这就是你写的“对于已经计算过的输入则不会重复计算”。

---

## 6) 工程侧的三件套：内存、I/O、数据布局

这些不改变算法复杂度，但能显著影响评分（尤其是极限竞速题）：

1. **提前分配连续内存**

* `docs`、`norm2`、heap 缓冲等一次性分配，减少频繁 `new/delete` 和碎片化

2. **加速读取：关闭 iostream 同步、解绑 cin/cout**

* `ios::sync_with_stdio(false); cin.tie(nullptr);`
* 对大量浮点读入很常见（或者直接用更底层的 fast IO）

3. **（隐含但常见）对齐与布局**

* 向量数据连续、对齐到 32/64 字节，有利于 CPU 预取与更宽 load
* 有些队会把 `docs` 做转置/分块布局，让“按 doc 分块展开”时访问更连续（这点你没写，但通常是展开策略的最佳搭配）

---

## 7) 总结成“一句话算法”

**全量扫描 KNN**：对每个 query，用“（归一化→点积等价距离）+ 按 doc 分块循环展开”快速算分，用“固定大小堆”维护 topK，并用“query cache”把重复请求变成 O(1)，外加 C++ 的内存与 I/O 极限优化。

---

如果你愿意，我也可以把它写成一份更“可复现”的说明：

* 典型数据结构设计（docs/norm/cache/heap）
* 4/8/16 展开各自适合的 L、M 范围
* cache 的 key 设计（避免浮点 hash 坑）
* 以及一份接近竞赛代码风格的 C++ 模板（但不包含任何比赛私有数据）。

[1]: https://bbs.huaweicloud.com/forum/thread-0226150303737683026-1-1.html?comment=comment-box&utm_source=chatgpt.com "华为算法精英实战营第5期- 高维向量数据的近似检索参赛经验分享"



下面给你两份**可直接编译运行**的 C++17 单文件完整实现：**情况 A（doc+query 都归一化 → 只算点积）** 和 **情况 B（预计算 doc 的 |x|² → 用 dist² = |x|² + |y|² − 2·dot）**。都包含：

* 连续内存存 doc
* 归一化 / 预计算范数
* 扫描全量 doc
* 用固定大小堆维护 TopK
* 快速输入（比 cin 快很多）
* 输出每个 query 的 topK doc id（一行一个 query）

我假设输入格式如下（竞赛常用）：

```
M L Q K
doc_0[0..L-1]
...
doc_{M-1}[0..L-1]
query_0[0..L-1]
...
query_{Q-1}[0..L-1]
```

如果你的数据格式不同，只要改 `main()` 里读入部分即可（算法部分不变）。

---

## 情况 A：doc 和 query 都 L2 归一化（TopK 最大点积）

```cpp
// case_a_normalized_dot.cpp
// g++ -O3 -std=c++17 case_a_normalized_dot.cpp -o runA
// ./runA < input.txt > output.txt

#include <bits/stdc++.h>
using namespace std;

struct FastScanner {
    static constexpr size_t BUFSIZE = 1 << 20;
    char buf[BUFSIZE];
    size_t idx = 0, size = 0;

    inline char readChar() {
        if (idx >= size) {
            size = fread(buf, 1, BUFSIZE, stdin);
            idx = 0;
            if (size == 0) return 0;
        }
        return buf[idx++];
    }

    template <class T>
    bool readInt(T &out) {
        char c;
        do { c = readChar(); if (!c) return false; } while (c <= ' ');
        T sign = 1;
        if (c == '-') { sign = -1; c = readChar(); }
        T val = 0;
        while (c > ' ') {
            val = val * 10 + (c - '0');
            c = readChar();
        }
        out = val * sign;
        return true;
    }

    bool readFloat(float &out) {
        char c;
        do { c = readChar(); if (!c) return false; } while (c <= ' ');
        bool neg = false;
        if (c == '-') { neg = true; c = readChar(); }

        double val = 0.0;
        while (c >= '0' && c <= '9') {
            val = val * 10.0 + (c - '0');
            c = readChar();
        }
        if (c == '.') {
            double base = 0.1;
            c = readChar();
            while (c >= '0' && c <= '9') {
                val += (c - '0') * base;
                base *= 0.1;
                c = readChar();
            }
        }
        if (c == 'e' || c == 'E') {
            c = readChar();
            bool eneg = false;
            if (c == '-') { eneg = true; c = readChar(); }
            else if (c == '+') { c = readChar(); }
            int expv = 0;
            while (c >= '0' && c <= '9') {
                expv = expv * 10 + (c - '0');
                c = readChar();
            }
            double p = pow(10.0, expv);
            val = eneg ? (val / p) : (val * p);
        }

        out = neg ? (float)(-val) : (float)(val);
        return true;
    }
};

static inline float l2_norm2(const float* v, int L) {
    double s = 0.0;
    for (int i = 0; i < L; ++i) s += (double)v[i] * (double)v[i];
    return (float)s;
}

static inline void l2_normalize_inplace(float* v, int L, float eps = 1e-12f) {
    float n2 = l2_norm2(v, L);
    if (n2 <= eps) return; // 0 向量：保持原样（全 0）
    float inv = 1.0f / sqrtf(n2);
    for (int i = 0; i < L; ++i) v[i] *= inv;
}

static inline float dot_product(const float* a, const float* b, int L) {
    double s = 0.0;
    for (int i = 0; i < L; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    FastScanner fs;

    int M, L, Q, K;
    if (!fs.readInt(M)) return 0;
    fs.readInt(L);
    fs.readInt(Q);
    fs.readInt(K);

    vector<float> docs((size_t)M * (size_t)L);
    vector<float> query((size_t)L);

    // 读入并归一化 docs
    for (int m = 0; m < M; ++m) {
        float* dptr = docs.data() + (size_t)m * (size_t)L;
        for (int i = 0; i < L; ++i) fs.readFloat(dptr[i]);
        l2_normalize_inplace(dptr, L);
    }

    // TopK：取最大相似度（点积），维护 size=K 的“最小堆”（堆顶是当前 TopK 里最差的那个）
    using Node = pair<float, int>; // (sim, id)
    struct CmpMin {
        bool operator()(const Node& x, const Node& y) const { return x.first > y.first; }
    };

    for (int qi = 0; qi < Q; ++qi) {
        for (int i = 0; i < L; ++i) fs.readFloat(query[i]);
        l2_normalize_inplace(query.data(), L);

        priority_queue<Node, vector<Node>, CmpMin> heap;

        for (int m = 0; m < M; ++m) {
            const float* dptr = docs.data() + (size_t)m * (size_t)L;
            float sim = dot_product(dptr, query.data(), L);

            if ((int)heap.size() < K) {
                heap.push({sim, m});
            } else if (sim > heap.top().first) {
                heap.pop();
                heap.push({sim, m});
            }
        }

        vector<Node> ans;
        ans.reserve(heap.size());
        while (!heap.empty()) { ans.push_back(heap.top()); heap.pop(); }

        // 输出从好到坏：sim 降序
        sort(ans.begin(), ans.end(), [](const Node& a, const Node& b){
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });

        for (int i = 0; i < (int)ans.size(); ++i) {
            if (i) putchar(' ');
            printf("%d", ans[i].second);
        }
        putchar('\n');
    }
    return 0;
}
```

---

## 情况 B：预计算 doc 的 |x|²；每个 query 算 |y|²；用 dist² 排名（TopK 最小 dist²）

```cpp
// case_b_precompute_norm2.cpp
// g++ -O3 -std=c++17 case_b_precompute_norm2.cpp -o runB
// ./runB < input.txt > output.txt

#include <bits/stdc++.h>
using namespace std;

struct FastScanner {
    static constexpr size_t BUFSIZE = 1 << 20;
    char buf[BUFSIZE];
    size_t idx = 0, size = 0;

    inline char readChar() {
        if (idx >= size) {
            size = fread(buf, 1, BUFSIZE, stdin);
            idx = 0;
            if (size == 0) return 0;
        }
        return buf[idx++];
    }

    template <class T>
    bool readInt(T &out) {
        char c;
        do { c = readChar(); if (!c) return false; } while (c <= ' ');
        T sign = 1;
        if (c == '-') { sign = -1; c = readChar(); }
        T val = 0;
        while (c > ' ') {
            val = val * 10 + (c - '0');
            c = readChar();
        }
        out = val * sign;
        return true;
    }

    bool readFloat(float &out) {
        char c;
        do { c = readChar(); if (!c) return false; } while (c <= ' ');
        bool neg = false;
        if (c == '-') { neg = true; c = readChar(); }

        double val = 0.0;
        while (c >= '0' && c <= '9') {
            val = val * 10.0 + (c - '0');
            c = readChar();
        }
        if (c == '.') {
            double base = 0.1;
            c = readChar();
            while (c >= '0' && c <= '9') {
                val += (c - '0') * base;
                base *= 0.1;
                c = readChar();
            }
        }
        if (c == 'e' || c == 'E') {
            c = readChar();
            bool eneg = false;
            if (c == '-') { eneg = true; c = readChar(); }
            else if (c == '+') { c = readChar(); }
            int expv = 0;
            while (c >= '0' && c <= '9') {
                expv = expv * 10 + (c - '0');
                c = readChar();
            }
            double p = pow(10.0, expv);
            val = eneg ? (val / p) : (val * p);
        }

        out = neg ? (float)(-val) : (float)(val);
        return true;
    }
};

static inline float l2_norm2(const float* v, int L) {
    double s = 0.0;
    for (int i = 0; i < L; ++i) s += (double)v[i] * (double)v[i];
    return (float)s;
}

static inline float dot_product(const float* a, const float* b, int L) {
    double s = 0.0;
    for (int i = 0; i < L; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    FastScanner fs;

    int M, L, Q, K;
    if (!fs.readInt(M)) return 0;
    fs.readInt(L);
    fs.readInt(Q);
    fs.readInt(K);

    vector<float> docs((size_t)M * (size_t)L);
    vector<float> doc_norm2((size_t)M);
    vector<float> query((size_t)L);

    // 读入 docs 并预计算 ||x||^2
    for (int m = 0; m < M; ++m) {
        float* dptr = docs.data() + (size_t)m * (size_t)L;
        for (int i = 0; i < L; ++i) fs.readFloat(dptr[i]);
        doc_norm2[m] = l2_norm2(dptr, L);
    }

    // TopK：取最小 dist^2，维护 size=K 的“最大堆”（堆顶是当前 TopK 里最差的那个：dist 最大）
    using Node = pair<float, int>; // (dist2, id)
    // 默认 priority_queue 是最大堆（按 pair::first 决定），正好用于维护“最差在顶”
    priority_queue<Node> heap;

    for (int qi = 0; qi < Q; ++qi) {
        for (int i = 0; i < L; ++i) fs.readFloat(query[i]);
        float qn2 = l2_norm2(query.data(), L);

        while (!heap.empty()) heap.pop();

        for (int m = 0; m < M; ++m) {
            const float* dptr = docs.data() + (size_t)m * (size_t)L;
            float dot = dot_product(dptr, query.data(), L);

            // dist^2 = ||x||^2 + ||y||^2 - 2 x·y
            float dist2 = doc_norm2[m] + qn2 - 2.0f * dot;

            if ((int)heap.size() < K) {
                heap.push({dist2, m});
            } else if (dist2 < heap.top().first) {
                heap.pop();
                heap.push({dist2, m});
            }
        }

        vector<Node> ans;
        ans.reserve(heap.size());
        while (!heap.empty()) { ans.push_back(heap.top()); heap.pop(); }

        // 输出从好到坏：dist2 升序
        sort(ans.begin(), ans.end(), [](const Node& a, const Node& b){
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        for (int i = 0; i < (int)ans.size(); ++i) {
            if (i) putchar(' ');
            printf("%d", ans[i].second);
        }
        putchar('\n');
    }

    return 0;
}
```

---

### 小提示（不改算法但通常能提速）

* 编译：`-O3` 基本必开；如果允许可试 `-march=native`（有些赛题禁用/无效）
* `dot_product` 里我用 `double` 累加更稳；如果你只追分数/速度，可改成 `float` 累加
* 若你要继续做你之前写的 **4/8/16 按 doc 分块循环展开**，这两份代码的扫描循环位置就是你要替换的地方（我也可以在下一条直接给你一个带 4/8/16 分块版本）

如果你把**你真实的输入输出格式**（比如是否有 query id、是否要求输出分数、是否每次只来一个 query 流式输入）贴一下，我可以把 `main()` 改到完全贴合你赛题的 I/O。


可以。下面给你 **“按 doc 分块（4/8/16）+ 循环展开”** 的完整 C++17 版本代码（两种情况 A / B 各一份），核心点是：

* 外层按 `BLOCK=4/8/16` 分块遍历 doc
* 内层对同一个 `qd = q[d]`，同时更新 `BLOCK` 个累加器 `acc0..acc{B-1}`
* 剩余不足一个块的 doc 用标量版本处理

你只要在编译时改 `-DBLOCK=4` 或 `-DBLOCK=16` 即可切换。

---

## 情况 A（doc + query 都归一化：只算点积，取最大）

```cpp
// case_a_block_unroll.cpp
// g++ -O3 -std=c++17 case_a_block_unroll.cpp -o runA -DBLOCK=8
// ./runA < input.txt > output.txt

#include <bits/stdc++.h>
using namespace std;

#ifndef BLOCK
#define BLOCK 8   // 可改为 4 / 8 / 16：编译时 -DBLOCK=16
#endif

static_assert(BLOCK == 4 || BLOCK == 8 || BLOCK == 16, "BLOCK must be 4, 8, or 16");

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

struct FastScanner {
    static constexpr size_t BUFSIZE = 1 << 20;
    char buf[BUFSIZE];
    size_t idx = 0, size = 0;

    inline char readChar() {
        if (idx >= size) {
            size = fread(buf, 1, BUFSIZE, stdin);
            idx = 0;
            if (size == 0) return 0;
        }
        return buf[idx++];
    }

    template <class T>
    bool readInt(T &out) {
        char c;
        do { c = readChar(); if (!c) return false; } while (c <= ' ');
        T sign = 1;
        if (c == '-') { sign = -1; c = readChar(); }
        T val = 0;
        while (c > ' ') {
            val = val * 10 + (c - '0');
            c = readChar();
        }
        out = val * sign;
        return true;
    }

    bool readFloat(float &out) {
        char c;
        do { c = readChar(); if (!c) return false; } while (c <= ' ');
        bool neg = false;
        if (c == '-') { neg = true; c = readChar(); }

        double val = 0.0;
        while (c >= '0' && c <= '9') {
            val = val * 10.0 + (c - '0');
            c = readChar();
        }
        if (c == '.') {
            double base = 0.1;
            c = readChar();
            while (c >= '0' && c <= '9') {
                val += (c - '0') * base;
                base *= 0.1;
                c = readChar();
            }
        }
        if (c == 'e' || c == 'E') {
            c = readChar();
            bool eneg = false;
            if (c == '-') { eneg = true; c = readChar(); }
            else if (c == '+') { c = readChar(); }
            int expv = 0;
            while (c >= '0' && c <= '9') {
                expv = expv * 10 + (c - '0');
                c = readChar();
            }
            double p = pow(10.0, expv);
            val = eneg ? (val / p) : (val * p);
        }

        out = neg ? (float)(-val) : (float)(val);
        return true;
    }
};

static inline float l2_norm2(const float* v, int L) {
    double s = 0.0;
    for (int i = 0; i < L; ++i) s += (double)v[i] * (double)v[i];
    return (float)s;
}

static inline void l2_normalize_inplace(float* v, int L, float eps = 1e-12f) {
    float n2 = l2_norm2(v, L);
    if (n2 <= eps) return;
    float inv = 1.0f / sqrtf(n2);
    for (int i = 0; i < L; ++i) v[i] *= inv;
}

static inline float dot_scalar(const float* a, const float* b, int L) {
    double s = 0.0;
    for (int i = 0; i < L; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

// 关键：一次算 BLOCK 个 doc 的 dot
static inline void dot_block_unrolled(const float* RESTRICT docs_base,
                                      const float* RESTRICT q,
                                      int L,
                                      float out[BLOCK]) {
    // docs_base 指向第 m 个 doc 的起始，后面 doc 按 row-major 连续：docs[(m+i)*L + d]
    // out[i] = dot(docs[m+i], q)

#if BLOCK == 4
    const float* RESTRICT p0 = docs_base + 0 * (size_t)L;
    const float* RESTRICT p1 = docs_base + 1 * (size_t)L;
    const float* RESTRICT p2 = docs_base + 2 * (size_t)L;
    const float* RESTRICT p3 = docs_base + 3 * (size_t)L;
    float a0=0, a1=0, a2=0, a3=0;
    for (int d = 0; d < L; ++d) {
        float qd = q[d];
        a0 += (*p0++) * qd;
        a1 += (*p1++) * qd;
        a2 += (*p2++) * qd;
        a3 += (*p3++) * qd;
    }
    out[0]=a0; out[1]=a1; out[2]=a2; out[3]=a3;

#elif BLOCK == 8
    const float* RESTRICT p0 = docs_base + 0 * (size_t)L;
    const float* RESTRICT p1 = docs_base + 1 * (size_t)L;
    const float* RESTRICT p2 = docs_base + 2 * (size_t)L;
    const float* RESTRICT p3 = docs_base + 3 * (size_t)L;
    const float* RESTRICT p4 = docs_base + 4 * (size_t)L;
    const float* RESTRICT p5 = docs_base + 5 * (size_t)L;
    const float* RESTRICT p6 = docs_base + 6 * (size_t)L;
    const float* RESTRICT p7 = docs_base + 7 * (size_t)L;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (int d = 0; d < L; ++d) {
        float qd = q[d];
        a0 += (*p0++) * qd;
        a1 += (*p1++) * qd;
        a2 += (*p2++) * qd;
        a3 += (*p3++) * qd;
        a4 += (*p4++) * qd;
        a5 += (*p5++) * qd;
        a6 += (*p6++) * qd;
        a7 += (*p7++) * qd;
    }
    out[0]=a0; out[1]=a1; out[2]=a2; out[3]=a3;
    out[4]=a4; out[5]=a5; out[6]=a6; out[7]=a7;

#else // BLOCK == 16
    const float* RESTRICT p0  = docs_base + 0  * (size_t)L;
    const float* RESTRICT p1  = docs_base + 1  * (size_t)L;
    const float* RESTRICT p2  = docs_base + 2  * (size_t)L;
    const float* RESTRICT p3  = docs_base + 3  * (size_t)L;
    const float* RESTRICT p4  = docs_base + 4  * (size_t)L;
    const float* RESTRICT p5  = docs_base + 5  * (size_t)L;
    const float* RESTRICT p6  = docs_base + 6  * (size_t)L;
    const float* RESTRICT p7  = docs_base + 7  * (size_t)L;
    const float* RESTRICT p8  = docs_base + 8  * (size_t)L;
    const float* RESTRICT p9  = docs_base + 9  * (size_t)L;
    const float* RESTRICT p10 = docs_base + 10 * (size_t)L;
    const float* RESTRICT p11 = docs_base + 11 * (size_t)L;
    const float* RESTRICT p12 = docs_base + 12 * (size_t)L;
    const float* RESTRICT p13 = docs_base + 13 * (size_t)L;
    const float* RESTRICT p14 = docs_base + 14 * (size_t)L;
    const float* RESTRICT p15 = docs_base + 15 * (size_t)L;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0,a8=0,a9=0,a10=0,a11=0,a12=0,a13=0,a14=0,a15=0;
    for (int d = 0; d < L; ++d) {
        float qd = q[d];
        a0  += (*p0++)  * qd;  a1  += (*p1++)  * qd;
        a2  += (*p2++)  * qd;  a3  += (*p3++)  * qd;
        a4  += (*p4++)  * qd;  a5  += (*p5++)  * qd;
        a6  += (*p6++)  * qd;  a7  += (*p7++)  * qd;
        a8  += (*p8++)  * qd;  a9  += (*p9++)  * qd;
        a10 += (*p10++) * qd;  a11 += (*p11++) * qd;
        a12 += (*p12++) * qd;  a13 += (*p13++) * qd;
        a14 += (*p14++) * qd;  a15 += (*p15++) * qd;
    }
    out[0]=a0; out[1]=a1; out[2]=a2; out[3]=a3;
    out[4]=a4; out[5]=a5; out[6]=a6; out[7]=a7;
    out[8]=a8; out[9]=a9; out[10]=a10; out[11]=a11;
    out[12]=a12; out[13]=a13; out[14]=a14; out[15]=a15;
#endif
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    FastScanner fs;
    int M, L, Q, K;
    if (!fs.readInt(M)) return 0;
    fs.readInt(L); fs.readInt(Q); fs.readInt(K);

    vector<float> docs((size_t)M * (size_t)L);
    vector<float> q((size_t)L);

    // 读 docs + 归一化
    for (int m = 0; m < M; ++m) {
        float* dptr = docs.data() + (size_t)m * (size_t)L;
        for (int i = 0; i < L; ++i) fs.readFloat(dptr[i]);
        l2_normalize_inplace(dptr, L);
    }

    using Node = pair<float,int>; // (sim, id)
    struct MinHeapCmp { bool operator()(const Node& a, const Node& b) const { return a.first > b.first; } };

    for (int qi = 0; qi < Q; ++qi) {
        for (int i = 0; i < L; ++i) fs.readFloat(q[i]);
        l2_normalize_inplace(q.data(), L);

        priority_queue<Node, vector<Node>, MinHeapCmp> heap;

        auto consider = [&](float sim, int id) {
            if ((int)heap.size() < K) heap.push({sim, id});
            else if (sim > heap.top().first) { heap.pop(); heap.push({sim, id}); }
        };

        int m = 0;
        float sims[BLOCK];

        // 分块
        for (; m + BLOCK - 1 < M; m += BLOCK) {
            const float* base = docs.data() + (size_t)m * (size_t)L;
            dot_block_unrolled(base, q.data(), L, sims);
            for (int i = 0; i < BLOCK; ++i) consider(sims[i], m + i);
        }
        // 尾巴
        for (; m < M; ++m) {
            const float* dptr = docs.data() + (size_t)m * (size_t)L;
            float sim = dot_scalar(dptr, q.data(), L);
            consider(sim, m);
        }

        vector<Node> ans;
        ans.reserve(heap.size());
        while (!heap.empty()) { ans.push_back(heap.top()); heap.pop(); }

        sort(ans.begin(), ans.end(), [](const Node& a, const Node& b){
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });

        for (int i = 0; i < (int)ans.size(); ++i) {
            if (i) putchar(' ');
            printf("%d", ans[i].second);
        }
        putchar('\n');
    }
    return 0;
}
```

---

## 情况 B（预计算 doc 的 |x|²：dist² = |x|² + |y|² − 2·dot，取最小 dist²）

```cpp
// case_b_block_unroll.cpp
// g++ -O3 -std=c++17 case_b_block_unroll.cpp -o runB -DBLOCK=8
// ./runB < input.txt > output.txt

#include <bits/stdc++.h>
using namespace std;

#ifndef BLOCK
#define BLOCK 8
#endif
static_assert(BLOCK == 4 || BLOCK == 8 || BLOCK == 16, "BLOCK must be 4, 8, or 16");

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

struct FastScanner {
    static constexpr size_t BUFSIZE = 1 << 20;
    char buf[BUFSIZE];
    size_t idx = 0, size = 0;

    inline char readChar() {
        if (idx >= size) {
            size = fread(buf, 1, BUFSIZE, stdin);
            idx = 0;
            if (size == 0) return 0;
        }
        return buf[idx++];
    }

    template <class T>
    bool readInt(T &out) {
        char c;
        do { c = readChar(); if (!c) return false; } while (c <= ' ');
        T sign = 1;
        if (c == '-') { sign = -1; c = readChar(); }
        T val = 0;
        while (c > ' ') {
            val = val * 10 + (c - '0');
            c = readChar();
        }
        out = val * sign;
        return true;
    }

    bool readFloat(float &out) {
        char c;
        do { c = readChar(); if (!c) return false; } while (c <= ' ');
        bool neg = false;
        if (c == '-') { neg = true; c = readChar(); }

        double val = 0.0;
        while (c >= '0' && c <= '9') {
            val = val * 10.0 + (c - '0');
            c = readChar();
        }
        if (c == '.') {
            double base = 0.1;
            c = readChar();
            while (c >= '0' && c <= '9') {
                val += (c - '0') * base;
                base *= 0.1;
                c = readChar();
            }
        }
        if (c == 'e' || c == 'E') {
            c = readChar();
            bool eneg = false;
            if (c == '-') { eneg = true; c = readChar(); }
            else if (c == '+') { c = readChar(); }
            int expv = 0;
            while (c >= '0' && c <= '9') {
                expv = expv * 10 + (c - '0');
                c = readChar();
            }
            double p = pow(10.0, expv);
            val = eneg ? (val / p) : (val * p);
        }

        out = neg ? (float)(-val) : (float)(val);
        return true;
    }
};

static inline float l2_norm2(const float* v, int L) {
    double s = 0.0;
    for (int i = 0; i < L; ++i) s += (double)v[i] * (double)v[i];
    return (float)s;
}

static inline float dot_scalar(const float* a, const float* b, int L) {
    double s = 0.0;
    for (int i = 0; i < L; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

static inline void dot_block_unrolled(const float* RESTRICT docs_base,
                                      const float* RESTRICT q,
                                      int L,
                                      float out[BLOCK]) {
#if BLOCK == 4
    const float* RESTRICT p0 = docs_base + 0 * (size_t)L;
    const float* RESTRICT p1 = docs_base + 1 * (size_t)L;
    const float* RESTRICT p2 = docs_base + 2 * (size_t)L;
    const float* RESTRICT p3 = docs_base + 3 * (size_t)L;
    float a0=0, a1=0, a2=0, a3=0;
    for (int d = 0; d < L; ++d) {
        float qd = q[d];
        a0 += (*p0++) * qd;
        a1 += (*p1++) * qd;
        a2 += (*p2++) * qd;
        a3 += (*p3++) * qd;
    }
    out[0]=a0; out[1]=a1; out[2]=a2; out[3]=a3;

#elif BLOCK == 8
    const float* RESTRICT p0 = docs_base + 0 * (size_t)L;
    const float* RESTRICT p1 = docs_base + 1 * (size_t)L;
    const float* RESTRICT p2 = docs_base + 2 * (size_t)L;
    const float* RESTRICT p3 = docs_base + 3 * (size_t)L;
    const float* RESTRICT p4 = docs_base + 4 * (size_t)L;
    const float* RESTRICT p5 = docs_base + 5 * (size_t)L;
    const float* RESTRICT p6 = docs_base + 6 * (size_t)L;
    const float* RESTRICT p7 = docs_base + 7 * (size_t)L;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (int d = 0; d < L; ++d) {
        float qd = q[d];
        a0 += (*p0++) * qd;
        a1 += (*p1++) * qd;
        a2 += (*p2++) * qd;
        a3 += (*p3++) * qd;
        a4 += (*p4++) * qd;
        a5 += (*p5++) * qd;
        a6 += (*p6++) * qd;
        a7 += (*p7++) * qd;
    }
    out[0]=a0; out[1]=a1; out[2]=a2; out[3]=a3;
    out[4]=a4; out[5]=a5; out[6]=a6; out[7]=a7;

#else // 16
    const float* RESTRICT p0  = docs_base + 0  * (size_t)L;
    const float* RESTRICT p1  = docs_base + 1  * (size_t)L;
    const float* RESTRICT p2  = docs_base + 2  * (size_t)L;
    const float* RESTRICT p3  = docs_base + 3  * (size_t)L;
    const float* RESTRICT p4  = docs_base + 4  * (size_t)L;
    const float* RESTRICT p5  = docs_base + 5  * (size_t)L;
    const float* RESTRICT p6  = docs_base + 6  * (size_t)L;
    const float* RESTRICT p7  = docs_base + 7  * (size_t)L;
    const float* RESTRICT p8  = docs_base + 8  * (size_t)L;
    const float* RESTRICT p9  = docs_base + 9  * (size_t)L;
    const float* RESTRICT p10 = docs_base + 10 * (size_t)L;
    const float* RESTRICT p11 = docs_base + 11 * (size_t)L;
    const float* RESTRICT p12 = docs_base + 12 * (size_t)L;
    const float* RESTRICT p13 = docs_base + 13 * (size_t)L;
    const float* RESTRICT p14 = docs_base + 14 * (size_t)L;
    const float* RESTRICT p15 = docs_base + 15 * (size_t)L;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0,a8=0,a9=0,a10=0,a11=0,a12=0,a13=0,a14=0,a15=0;
    for (int d = 0; d < L; ++d) {
        float qd = q[d];
        a0  += (*p0++)  * qd;  a1  += (*p1++)  * qd;
        a2  += (*p2++)  * qd;  a3  += (*p3++)  * qd;
        a4  += (*p4++)  * qd;  a5  += (*p5++)  * qd;
        a6  += (*p6++)  * qd;  a7  += (*p7++)  * qd;
        a8  += (*p8++)  * qd;  a9  += (*p9++)  * qd;
        a10 += (*p10++) * qd;  a11 += (*p11++) * qd;
        a12 += (*p12++) * qd;  a13 += (*p13++) * qd;
        a14 += (*p14++) * qd;  a15 += (*p15++) * qd;
    }
    out[0]=a0; out[1]=a1; out[2]=a2; out[3]=a3;
    out[4]=a4; out[5]=a5; out[6]=a6; out[7]=a7;
    out[8]=a8; out[9]=a9; out[10]=a10; out[11]=a11;
    out[12]=a12; out[13]=a13; out[14]=a14; out[15]=a15;
#endif
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    FastScanner fs;
    int M, L, Q, K;
    if (!fs.readInt(M)) return 0;
    fs.readInt(L); fs.readInt(Q); fs.readInt(K);

    vector<float> docs((size_t)M * (size_t)L);
    vector<float> doc_n2((size_t)M);
    vector<float> q((size_t)L);

    // 读 docs + 预计算 ||x||^2
    for (int m = 0; m < M; ++m) {
        float* dptr = docs.data() + (size_t)m * (size_t)L;
        for (int i = 0; i < L; ++i) fs.readFloat(dptr[i]);
        doc_n2[m] = l2_norm2(dptr, L);
    }

    using Node = pair<float,int>; // (dist2, id)

    for (int qi = 0; qi < Q; ++qi) {
        for (int i = 0; i < L; ++i) fs.readFloat(q[i]);
        float qn2 = l2_norm2(q.data(), L);

        // 维护 TopK 最小 dist^2：用“最大堆”把当前最差的(最大 dist2)放堆顶
        priority_queue<Node> heap;

        auto consider = [&](float dist2, int id) {
            if ((int)heap.size() < K) heap.push({dist2, id});
            else if (dist2 < heap.top().first) { heap.pop(); heap.push({dist2, id}); }
        };

        int m = 0;
        float dots[BLOCK];

        // 分块：先算 dots，再转 dist2
        for (; m + BLOCK - 1 < M; m += BLOCK) {
            const float* base = docs.data() + (size_t)m * (size_t)L;
            dot_block_unrolled(base, q.data(), L, dots);
            for (int i = 0; i < BLOCK; ++i) {
                int id = m + i;
                float dist2 = doc_n2[id] + qn2 - 2.0f * dots[i];
                consider(dist2, id);
            }
        }

        // 尾巴
        for (; m < M; ++m) {
            const float* dptr = docs.data() + (size_t)m * (size_t)L;
            float dot = dot_scalar(dptr, q.data(), L);
            float dist2 = doc_n2[m] + qn2 - 2.0f * dot;
            consider(dist2, m);
        }

        vector<Node> ans;
        ans.reserve(heap.size());
        while (!heap.empty()) { ans.push_back(heap.top()); heap.pop(); }

        sort(ans.begin(), ans.end(), [](const Node& a, const Node& b){
            if (a.first != b.first) return a.first < b.first; // dist2 升序
            return a.second < b.second;
        });

        for (int i = 0; i < (int)ans.size(); ++i) {
            if (i) putchar(' ');
            printf("%d", ans[i].second);
        }
        putchar('\n');
    }
    return 0;
}
```

---

### 怎么选 BLOCK=4/8/16？

直接改编译参数：

```bash
g++ -O3 -std=c++17 case_a_block_unroll.cpp -o runA -DBLOCK=16
g++ -O3 -std=c++17 case_b_block_unroll.cpp -o runB -DBLOCK=8
```

---

如果你还想再往上压性能，我可以在这份“分块展开”的基础上继续加两件在比赛里很常见、而且通常提升明显的东西（仍然不写 SIMD intrinsic）：

1. **TopK 堆的分支/函数调用进一步减少**（例如 K 很小改成手写固定数组）
2. **doc 数据布局做 block-friendly 的重排**（让 `p0..p(B-1)` 访问更连续，更容易触发编译器向量化/预取）

你把 `L、M、K` 的大概范围告诉我（比如 `L=768/1024`，`M=1e5` 这种），我就能更针对性给你改到“更像竞赛最优形态”的版本。



你说的“**doc 数据布局做 block-friendly 的重排**”，核心就是把原来的

* **AoS / row-major**：`docs[m][d]` 连续存一整条向量
  （访问模式：每次维度 `d` 固定时，要从 `B` 个 doc 的不同位置各取 1 个数，跨很远）

改成

* **Blocked SoA（块内按维度交错）**：以 `BLOCK=B` 为单位，把同一维度 `d` 的 `B` 个 doc 元素放到一起连续存
  （访问模式：维度 `d` 固定时，直接一次读到连续的 `B` 个 float，很容易触发编译器向量化/预取）

---

## 1) 重排后的内存布局长什么样？

设 `B=BLOCK`，把 doc 分成 `nb = ceil(M/B)` 个块，每块最多 B 个 doc。新数组 `docs_blk` 的逻辑索引：

[
\text{docs_blk}[block][d][i]
]

含义：第 `block` 块里，第 `d` 维，第 `i` 个 doc（`i in [0,B)`）的值。

在内存里线性展开就是：

```cpp
// 线性下标（推荐）
docs_blk[(block * L + d) * B + i]
```

这样当你在算点积时，对于固定维度 `d`：

* 你会读 `docs_blk[(block*L + d)*B + 0..B-1]` —— **一段连续内存**
* 然后用同一个 `qd` 去更新 `acc[0..B-1]`

这比原始 `docs[(m+i)*L + d]` 那种“跳着读”更友好。

---

## 2) 重排一次（预处理），查询时就一直用它

下面给你一份**直接可用**的重排函数 + 使用方式（适用于情况 A/B，都能用）。

### 2.1 重排函数（把 row-major 的 docs 变成 blocked-SoA）

```cpp
#include <bits/stdc++.h>
using namespace std;

#ifndef BLOCK
#define BLOCK 8
#endif
static_assert(BLOCK == 4 || BLOCK == 8 || BLOCK == 16);

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

// docs_row: M*L，row-major: docs_row[m*L + d]
// docs_blk: nb*L*B，blocked-SoA: docs_blk[(block*L + d)*B + i]
static inline void reorder_docs_blocked(const float* RESTRICT docs_row,
                                        float* RESTRICT docs_blk,
                                        int M, int L) {
    const int B = BLOCK;
    const int nb = (M + B - 1) / B;

    // 先清零尾块的 padding，避免读到脏数据
    size_t total = (size_t)nb * (size_t)L * (size_t)B;
    memset(docs_blk, 0, total * sizeof(float));

    for (int m = 0; m < M; ++m) {
        int block = m / B;
        int i     = m % B;
        const float* src = docs_row + (size_t)m * (size_t)L;

        // 把这条 doc 的每个维度塞到对应 (block, d, i)
        for (int d = 0; d < L; ++d) {
            docs_blk[((size_t)block * (size_t)L + (size_t)d) * (size_t)B + (size_t)i] = src[d];
        }
    }
}
```

> padding（尾块不足 B 个 doc）用 0 填充即可：它不会影响 topK，因为这些位置你不会当成有效 doc（只处理真实 id `< M` 的）。

---

## 3) 查询时的 dot：每个维度读一段连续的 B 个 float

这个版本是“**块内 i 维展开**”，非常适合让编译器把 `acc[0..B-1]` 向量化：

```cpp
static inline void dot_block_from_blocked(const float* RESTRICT docs_blk_block, // 指向 block 的起点：&docs_blk[block*L*B]
                                          const float* RESTRICT q,
                                          int L,
                                          float out[BLOCK]) {
    const int B = BLOCK;
    float acc[BLOCK] = {0};

    // 遍历维度
    for (int d = 0; d < L; ++d) {
        float qd = q[d];
        const float* RESTRICT v = docs_blk_block + (size_t)d * (size_t)B; // v[0..B-1] 连续

        // 对 i 方向展开（这里用 for；你也可以像之前那样手写 4/8/16 展开）
        #pragma GCC ivdep
        for (int i = 0; i < B; ++i) {
            acc[i] += v[i] * qd;
        }
    }

    for (int i = 0; i < B; ++i) out[i] = acc[i];
}
```

* `v` 是连续的 B 个 float
* 这比原先 `p0/p1/...` 各自跨 stride 的访问更容易形成 **一次性加载、SIMD-like 的访存模式**（即使你没显式开 SIMD）

---

## 4) 怎么把它接到你已有的分块 topK 扫描里？

下面给一个**扫描循环**的替换模板（以情况 A：归一化 + 点积为例；情况 B 只是在得到 `dot` 后转 `dist2`）。

```cpp
// 假设你已经有：
// docs_row: vector<float> size M*L（读入后可先归一化）
// docs_blk: vector<float> size nb*L*B（重排后的数据）
// q: query vector<float> size L（已归一化）
// K: topK

int B = BLOCK;
int nb = (M + B - 1) / B;

for (int block = 0; block < nb; ++block) {
    const float* base = docs_blk.data() + (size_t)block * (size_t)L * (size_t)B;

    float sims[BLOCK];
    dot_block_from_blocked(base, q.data(), L, sims);

    int m0 = block * B;
    int valid = min(B, M - m0);  // 尾块有效 doc 数

    for (int i = 0; i < valid; ++i) {
        float sim = sims[i];
        int id = m0 + i;
        // consider(sim, id) 你的 topK 堆更新逻辑
    }
}
```

情况 B 的版本：把 `sim` 换成 `dot`，然后算

```cpp
float dist2 = doc_norm2[id] + q_norm2 - 2.0f * dot;
consider(dist2, id);
```

---

## 5) 为什么这通常会更快？

原始布局下（row-major），对一个 block 的 dot：

* 在固定 `d` 时，你要访问：`docs[(m+0)*L+d], docs[(m+1)*L+d], ...`
* 这些地址彼此相差 `L*sizeof(float)`，是**跨 stride 访问**
  → cache line 利用率差，硬件预取也更难猜

重排后（blocked-SoA）：

* 在固定 `d` 时，你访问的是连续 `B` 个 float
  → 一个/几个 cache line 就够，流水线更顺，编译器也更爱向量化

---

## 6) 实战小建议（很有用）

1. **重排要在归一化之后做**（情况 A）：
   你读 docs → 归一化（row-major 上做方便）→ 重排成 blocked

2. **BLOCK 别盲目越大越好**：

   * `B=16` 更吃寄存器，可能反而降速（看 L、CPU、编译器）
   * 常见经验：先试 `8`，再试 `4/16`

3. **内存对齐**（可选但常有收益）：
   `docs_blk` 可以用 64-byte 对齐分配（例如 `aligned_alloc(64, bytes)`），对向量化 load 更友好。

---

如果你愿意，我可以把你上一条给的“情况 A/B 分块展开完整程序”直接改成 **“归一化/预计算 + 重排 + 扫描”** 的完整版本（可一键编译跑）。你不需要再自己拼。
