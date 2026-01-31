下面按“**向量怎么压缩**”和“**怎么检索（ANN）**”两条线，把常用的高性能方案梳理一下，并给出典型组合与选型建议。

---

## 1) 向量压缩（Vector Compression）主流方法

### A. 标量量化（Scalar Quantization, SQ）

* 思路：把每个维度的 float32 量化成 int8 / int4（甚至更低），存储与带宽大幅下降。
* 优点：实现简单、解码快，适合 CPU/GPU SIMD。
* 缺点：精度损失依数据分布而定，通常需要校准/归一化；更低 bit 需要更精细的工程。
  （在工业检索里经常和“候选召回后 rerank 用原向量/高精度向量”搭配。）

### B. 乘积量化（Product Quantization, PQ）

* 思路：把 d 维向量切成 M 个子向量，每段用一个小码本（k-means）编码；距离用 **ADC（Asymmetric Distance Computation）** 快速近似。
* 优点：**压缩率高**（典型 8–16 bytes/向量），距离计算可用查表+SIMD 加速。
* 关键增强：

  * **OPQ**：对向量做学习到的旋转/变换，让 PQ 的误差更小（常见于 IVFOPQ / OPQ-PQ）。([微软][1])
  * 更快的 ADC/SIMD 实现（如 AVX512 优化思路）。([arXiv][2])

### C. 残差/加性量化（Residual / Additive Quantization 家族）

* 思路：分多级编码残差（RQ、AQ、LSQ 等），通常比单层 PQ 更准但构建更复杂、计算更重。
* 优点：在相同码长下可能更高 recall。
* 缺点：训练/编码更复杂，工程上需要权衡。

### D. 二值哈希（Binary Hashing, 1-bit/维 或若干bit）

* 思路：把向量映射到二进制码，用 Hamming 距离。
* 优点：极致省内存、极快。
* 缺点：对高召回/高精度通常不如 PQ/SQ（但在粗召回/过滤中仍很有用）。

---

## 2) 高性能向量检索（ANN Search / Indexing）主流方法

### A. 图索引：HNSW（内存型高召回“王者”之一）

* 核心：分层小世界图，搜索近似复杂度接近对数；调参主要是 M、efConstruction、efSearch。
* 优点：高 recall、低延迟，适合中大型（百万~千万）内存场景。
* 缺点：索引内存占用较高，构建时间偏长；更新/删除需要实现细节。([arXiv][3])

### B. 倒排聚类：IVF（Inverted File）家族

* 核心：先用粗聚类（k-means）把库向量分桶；查询时只扫 Top-nprobe 个桶。
* 优点：非常适合和 PQ/SQ 结合做压缩（经典 **IVF-PQ / IVF-OPQ-PQ**）。
* 缺点：需要训练聚类；nlist/nprobe 调参影响大。
* 代表实现：FAISS（支持 IVF、PQ、OPQ、GPU 等）。([Khoury College of Computer Sciences][4])

### C. 分区 + 量化的工业化路线：ScaNN

* 核心：分区（partition）+（可选）向量压缩/打分优化，强调在高吞吐下的召回-延迟权衡。
* 优点：工程上对吞吐/延迟很强，常用于大规模检索管线。([谷歌研究][5])

### D. 磁盘/SSD 向量检索：DiskANN（超大规模、低内存）

* 核心：图索引结合 SSD I/O，把大库放 SSD，内存只留必要结构/压缩表示。
* 优点：在有限 RAM 下做十亿级检索是经典选择。([微软][6])
* 也有大量后续工作围绕“过滤条件 + SSD 图检索”扩展（如 Filtered-DiskANN）。([ACM Digital Library][7])

---

## 3) 典型“压缩 + 检索”黄金组合

1. **IVF + PQ/OPQ（IVF-PQ / IVF-OPQ-PQ）**
   适合：超大库、需要强压缩、可接受训练与调参；常见于 FAISS 体系。([微软][1])

2. **HNSW +（原向量 / SQ / PQ）**
   适合：低延迟高召回的内存检索；压缩更多是为了省内存与带宽，但要注意图本身的额外开销。([arXiv][3])

3. **DiskANN（SSD）+ 内存压缩表示（SQ/PQ）**
   适合：数据量极大、RAM 成本敏感、SSD 读延迟可控的场景。([微软][6])

4. **ScaNN 风格：分区 +（压缩/打分优化）**
   适合：工业检索服务，追求吞吐与端到端成本。([谷歌研究][5])

---

## 4) 选型速记（按你的约束选）

* **你最在意 recall/延迟（内存够）**：优先 HNSW；如果量大再考虑 IVF/HNSW 的混合或分片策略。([arXiv][3])
* **你最在意内存成本（仍要较高 recall）**：IVF-OPQ-PQ 是经典答案；码长（bytes/vec）+ nprobe 是关键旋钮。([微软][1])
* **你库巨大到必须上 SSD**：DiskANN/Filtered-DiskANN 路线更合适。([微软][6])
* **你要高吞吐线上服务**：看 ScaNN/FAISS 的工程栈与是否易于部署到你的 CPU/GPU 环境。([谷歌研究][5])

---

## 5) 一个通用落地流程（很少踩坑的版本）

1. **明确距离度量**：余弦相似通常先 L2-normalize，然后用 inner product / L2 等价形式。
2. **先用原向量做“上限”**：跑 exact 或高召回配置，得到 recall@k/latency 的基准。
3. **再引入压缩**：先试 SQ（实现简单）→ 再试 PQ/OPQ（压缩更强）。([微软][1])
4. **最后定索引**：HNSW（低延迟）vs IVF（大规模+压缩）vs DiskANN（SSD）。([arXiv][3])

---

如果你告诉我三件事：**向量维度 d、数据规模 N、目标指标（P99 延迟 / QPS / recall@k / 内存预算）**，我可以直接给你一套“可落地的参数起步表”（比如 IVF 的 nlist/nprobe、PQ 的 M/nbits，HNSW 的 M/efSearch/efConstruction，是否需要 OPQ 等），以及对应的测评方法。

[1]: https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf?utm_source=chatgpt.com "Optimized Product Quantization"
[2]: https://arxiv.org/pdf/1812.09162?utm_source=chatgpt.com "Unlocking the Hidden Potential of Product Quantization ..."
[3]: https://arxiv.org/abs/1603.09320?utm_source=chatgpt.com "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
[4]: https://khoury.northeastern.edu/home/pandey/courses/cs7270/fall25/papers/vectordb/FAISS.pdf?utm_source=chatgpt.com "The FAISS Library"
[5]: https://research.google/blog/announcing-scann-efficient-vector-similarity-search/?utm_source=chatgpt.com "Announcing ScaNN: Efficient Vector Similarity Search"
[6]: https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/?utm_source=chatgpt.com "DiskANN: Fast Accurate Billion-point Nearest Neighbor ..."
[7]: https://dl.acm.org/doi/10.1145/3543507.3583552?utm_source=chatgpt.com "Filtered-DiskANN: Graph Algorithms for Approximate ..."


如果把“高性能向量压缩与检索”做成 **3 小时竞赛题**，最可能不是传统“AC/WA + 固定时间限制”的风格，而是 **带评分的性能题（challenge/optimization）**：你既要“够准”，还要“够快/省内存”。下面给你一套很贴近真实会出的题目设计（含输入输出、约束、评分、出题人常加的坑点）。

---

## 题目形态 1：单机离线评分（最常见、3 小时也能开干）

### 题意

给定一批 **文档向量** `N x D` 和一批 **查询向量** `Q x D`，要求对每个 query 输出 topK 个最相似的 doc id（余弦或内积/L2 之一）。
允许近似（不要求完全正确），系统会用 **召回率/精度** 来打分，同时计入运行时间/内存。

### 输入（典型）

* 第一行：`N D Q K`
* 接下来 N 行：doc 向量（float32 或 int8）
* 接下来 Q 行：query 向量
* （可选）多组 test case，或一个大 case

### 输出

* Q 行，每行 K 个 doc id（从高到低）

### 约束（出题人会这样卡）

* `D = 96/128/256`（让你能做 loop unroll / 查表）
* `N = 2e5 ~ 2e6`，`Q = 1e3 ~ 5e4`
* 内存限制例如 2GB（逼你压缩）
* 时间限制可能不严格“超时就 0”，而是“越快越高分”

### 评分（很像工业/高校性能赛）

* 先算 **Recall@K** 或 **NDCG@K**（基于隐藏的 ground truth）
* 再结合耗时：
  `Score = A * quality - B * log(runtime_ms)`
  或者分段：质量过阈值才计速度分
* 也可能加入内存项：`- C * memory_MB`

> 这类设计的好处：3 小时内你可以从 baseline（暴力/部分剪枝）逐步升级到“量化 + 候选筛选 + cache”。

---

## 题目形态 2：两阶段接口题（更“工程化”，但仍适合 3 小时）

### 题意

你要实现两个阶段：

1. `build_index(docs)`：预处理/训练/压缩
2. `search(queries)`：回答所有 query

评测系统会分别计时 build 和 search（或只计 search），并会多次调用 search（鼓励 cache/复用）。

### 常见“暗示解法”的规则

* 文档向量可能 **提前归一化**（鼓励用内积近似距离）
* 编译参数禁止 `-march=native` / 禁止 intrinsics（逼你用 **循环展开**、数据布局、cache）
* Q 里会有 **重复 query** 或 **相近 query**（鼓励你做 query cache / centroid cache）

---

## 出题人会加的“关键坑点”（让题目更像你之前描述那种）

1. **数据分布有结构**

   * 向量已 L2 normalize ⇒ 余弦=内积
   * 或者 doc 分簇明显 ⇒ 适合 IVF / coarse clustering
   * 或者维度不大但 N 巨大 ⇒ 适合 PQ/SQ + 查表

2. **评测会混合多档 case**

   * 小 N：考“正确性/边界处理”
   * 大 N：考“性能/压缩/召回权衡”
   * D 不同：考“泛化/不能写死某个 D”

3. **会放“对抗性边界”**

   * `K` 变大（比如 1/10/100）
   * query 的范数/分布和 docs 不完全一致（防止过拟合）

4. **输入输出量巨大**

   * 让你必须用 fast I/O、二进制读取（或者给 mmap 文件）
   * 输出也很大，逼你减少格式化开销

---

## 3 小时内“最可能的官方预期路线”（因为可实现、收益大）

出题人会希望你走这种“简单但很能提分”的工程路线：

* **候选集缩小（coarse filter）**
  简易 k-means/随机采样聚类（IVF-ish），每个 query 只扫少量桶
* **向量压缩**
  int8 标量量化（SQ）或简化 PQ（不用写完整训练也能跑）
* **距离计算优化**
  loop unroll、AoS→SoA、预取、查表（LUT）、批处理 query
* **cache**
  重复 query / 重复桶 / 重复 LUT 复用

这套在 3 小时内从 0 写到“能跑+能提分”是现实的。

---

## 如果要把题目设计得“更竞赛、但仍 3 小时可做”

我会建议题面明确提供一个“可落地的方向”，避免变成纯玄学调参：

* 明确允许近似，并给一个 **quality 指标定义**（Recall@K）
* 给一个 **基准实现**（暴力或简单 IVF）
* 给若干公开样例数据用于本地测分
* 限制一些太强外挂（比如禁 GPU / 禁 AVX512 intrinsics）以保证公平

---

如果你愿意，我可以把上面这套直接“写成完整题面”（像 Codeforces/ICPC 那种格式），并给一份 **baseline C++**（fast I/O + 简单量化 + 简单候选筛选），保证你 3 小时内能从 baseline 快速往上加分。你想用 **余弦/内积/L2** 哪一种做主指标？K 通常设 10 还是 100？


下面给你一份**完整、可直接发布的 3 小时竞赛题面**（Codeforces/ICPC 风格，但带性能评分），以及一份**可跑的 baseline C++**（实现：L2 归一化 + per-vector int8 量化 + 256 桶粗召回 + 候选精算 topK）。

---

# 题目：VecRace — 高性能向量压缩与近似检索

## 题目背景

你需要在单机上对大规模向量库进行近似最近邻检索（ANN）。评测不仅看检索质量，也看运行时间和内存占用。你可以自由使用任意近似方法（量化、分桶、倒排、图等），只需输出每个查询的 topK 文档编号。

---

## 相似度定义

给定文档向量 (x_i \in \mathbb{R}^D) 与查询向量 (q\in\mathbb{R}^D)。

* 评测使用 **余弦相似度**：
  [
  \text{sim}(q, x_i)=\frac{q}{|q|}\cdot\frac{x_i}{|x_i|}
  ]
* 你可以在程序中自行对 doc/query 做 L2 归一化（推荐）。

---

## 输入格式

从标准输入读取：

* 第一行：`N D Q K`

  * `N`：文档数量
  * `D`：向量维度
  * `Q`：查询数量
  * `K`：需要输出的 topK

* 接下来 `N` 行：每行 `D` 个浮点数，表示一个文档向量 `x_i`

* 接下来 `Q` 行：每行 `D` 个浮点数，表示一个查询向量 `q_j`

> 文档编号为 `0..N-1`，查询编号为 `0..Q-1`。

---

## 输出格式

输出 `Q` 行，每行 `K` 个整数，表示该查询的 topK 文档编号（按相似度从高到低排序）：

* 若相似度相同，编号小的排在前面。
* 允许输出近似结果（不要求与精确 topK 完全一致）。

---

## 数据范围（示例设计）

评测包含多档数据（隐藏）：

* `D ∈ {96, 128, 192, 256}`
* `N` 最大可到 `2,000,000`
* `Q` 最大可到 `50,000`
* `K ∈ {10, 20, 100}`

资源限制：

* 内存限制：2 GB
* 运行时间：计入评分（不是硬性 TLE）

---

## 评分规则（典型）

对每个测试点，评测器会预先计算精确 topK 作为 ground truth，定义：

* `Recall@K`：对每个 query，输出集合与真值集合交集大小 / K，最后对所有 query 平均。

记：

* `R = Recall@K ∈ [0,1]`
* `T = search 阶段耗时（毫秒）`（只统计从读完输入到输出结束的整体时间，或只统计 search，取决于赛制；此处按整体计时更常见）
* `M = 峰值内存（MB）`

则该测试点分数（示例）：
[
\text{Score} = 10^6 \cdot \frac{R}{1 + 0.02\cdot \log(1+T)} - 50\cdot \log(1+M)
]
最终总分为各测试点分数之和（或加权和）。

> 出题人常用这种形式：**质量先决定上限**，速度/内存是惩罚项；也可能设置质量阈值（如 `R < 0.6` 则该点得分为 0）。

---

## 备注与提示

* 你可以做：

  * int8/int4 标量量化、PQ/OPQ、倒排（IVF）、HNSW、DiskANN 等
* 也可以做工程优化：

  * 数据布局（SoA）、循环展开、批处理、多线程（若允许）、cache、预取等
* I/O 很可能是瓶颈：建议使用 fast I/O。

---

## 样例

### 输入

```
5 4 2 2
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
1 1 0 0
1 0 0 0
0 1 0 0
```

### 输出（示例之一）

```
0 4
1 4
```

---

# Baseline C++（可跑、可得分）

这个 baseline 做了：

1. doc/query L2 归一化（余弦=内积）
2. per-vector int8 量化（每个 doc 一个 scale）
3. 256 桶粗召回：用 8 个维度的符号位做签名（0/1）组成 8-bit bucket
4. 查询时扫描本桶 + 8 个单比特翻转邻桶（共 9 桶），对候选做近似内积并取 topK

> 这是“3 小时内能写完且能跑”的典型 baseline：速度比全库暴力好很多，召回也不会太差（取决于数据分布）。

```cpp
#include <bits/stdc++.h>
using namespace std;

// -------- Fast Input (stdin) --------
struct FastInput {
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
        do {
            c = readChar();
            if (!c) return false;
        } while (c <= ' ');

        bool neg = false;
        if (c == '-') { neg = true; c = readChar(); }

        T val = 0;
        while (c > ' ') {
            val = val * 10 + (c - '0');
            c = readChar();
        }
        out = neg ? -val : val;
        return true;
    }

    bool readFloat(float &out) {
        char c;
        do {
            c = readChar();
            if (!c) return false;
        } while (c <= ' ');

        // parse as double for robustness
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
            if (c == '+' || c == '-') { eneg = (c == '-'); c = readChar(); }
            int expv = 0;
            while (c >= '0' && c <= '9') {
                expv = expv * 10 + (c - '0');
                c = readChar();
            }
            double pow10 = pow(10.0, expv);
            val = eneg ? (val / pow10) : (val * pow10);
        }

        out = (float)(neg ? -val : val);
        return true;
    }
};

// -------- Utility --------
static inline float l2norm(const float* v, int D) {
    double s = 0.0;
    for (int i = 0; i < D; i++) s += (double)v[i] * (double)v[i];
    return (float)sqrt(s);
}

static inline int8_t clamp_int8(int x) {
    if (x > 127) return 127;
    if (x < -127) return -127; // keep symmetric
    return (int8_t)x;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    FastInput in;
    int N, D, Q, K;
    if (!in.readInt(N)) return 0;
    in.readInt(D);
    in.readInt(Q);
    in.readInt(K);

    // Choose 8 dimensions to build a signature (spread out).
    // For D < 128, modulo will wrap; still works.
    int sigDims[8] = {0, 16, 32, 48, 64, 80, 96, 112};
    for (int i = 0; i < 8; i++) sigDims[i] %= max(1, D);

    // Storage: int8 codes for docs, plus per-doc scale.
    vector<int8_t> codes((size_t)N * (size_t)D);
    vector<float> scales(N);

    // Buckets: 256 inverted lists
    vector<vector<int>> buckets(256);
    buckets.reserve(256);

    // Temp buffer for reading one vector
    vector<float> tmp(D);

    // ---- Read & preprocess docs ----
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) in.readFloat(tmp[j]);

        // L2 normalize
        float nrm = l2norm(tmp.data(), D);
        if (nrm > 0) {
            float inv = 1.0f / nrm;
            for (int j = 0; j < D; j++) tmp[j] *= inv;
        }

        // signature: 8-bit sign pattern
        unsigned sig = 0;
        for (int b = 0; b < 8; b++) {
            float v = tmp[sigDims[b]];
            if (v >= 0) sig |= (1u << b);
        }
        buckets[sig].push_back(i);

        // per-vector scale for int8 quantization
        float maxabs = 0.0f;
        for (int j = 0; j < D; j++) maxabs = max(maxabs, fabsf(tmp[j]));
        if (maxabs < 1e-12f) maxabs = 1e-12f;

        float scale = maxabs / 127.0f;
        scales[i] = scale;

        float invs = 1.0f / scale;
        int8_t* dst = &codes[(size_t)i * (size_t)D];
        for (int j = 0; j < D; j++) {
            int qv = (int)lrintf(tmp[j] * invs);
            dst[j] = clamp_int8(qv);
        }
    }

    // ---- Search queries ----
    string out;
    out.reserve((size_t)Q * (size_t)K * 6);

    vector<float> qv(D);

    // Helper lambda: compute approximate dot(q, doc) using int8 codes & scale
    auto approx_dot = [&](const float* q, int docId) -> float {
        const int8_t* dc = &codes[(size_t)docId * (size_t)D];
        // dot ~= scale * sum(int8 * q)
        // Use float accumulation; can be optimized to SIMD/unroll later.
        float s = 0.0f;
        for (int j = 0; j < D; j++) s += (float)dc[j] * q[j];
        return s * scales[docId];
    };

    // Candidate bucket list (query bucket + 1-bit neighbors)
    int neigh[9];
    for (int qi = 0; qi < Q; qi++) {
        for (int j = 0; j < D; j++) in.readFloat(qv[j]);

        // L2 normalize
        float nrm = l2norm(qv.data(), D);
        if (nrm > 0) {
            float inv = 1.0f / nrm;
            for (int j = 0; j < D; j++) qv[j] *= inv;
        }

        unsigned sig = 0;
        for (int b = 0; b < 8; b++) {
            float v = qv[sigDims[b]];
            if (v >= 0) sig |= (1u << b);
        }

        neigh[0] = (int)sig;
        for (int b = 0; b < 8; b++) neigh[1 + b] = (int)(sig ^ (1u << b));

        // Min-heap for topK: (score, -id) so that lower score pops first,
        // tie-break: smaller id preferred => in heap we treat larger id as worse.
        using Node = pair<float,int>;
        auto worse = [](const Node& a, const Node& b){
            if (a.first != b.first) return a.first > b.first; // min-heap by score
            return a.second > b.second; // min-heap by id (larger id worse)
        };
        std::priority_queue<Node, vector<Node>, decltype(worse)> heap(worse);

        size_t candCount = 0;
        for (int t = 0; t < 9; t++) candCount += buckets[neigh[t]].size();

        // Fallback: if too few candidates, scan all buckets (still approximate)
        bool fallback_all = (candCount < (size_t)K);

        if (!fallback_all) {
            for (int t = 0; t < 9; t++) {
                const auto& lst = buckets[neigh[t]];
                for (int id : lst) {
                    float sc = approx_dot(qv.data(), id);
                    if ((int)heap.size() < K) {
                        heap.push({sc, id});
                    } else {
                        const Node& top = heap.top();
                        // replace if better: higher score or same score smaller id
                        if (sc > top.first || (sc == top.first && id < top.second)) {
                            heap.pop();
                            heap.push({sc, id});
                        }
                    }
                }
            }
        } else {
            for (int b = 0; b < 256; b++) {
                const auto& lst = buckets[b];
                for (int id : lst) {
                    float sc = approx_dot(qv.data(), id);
                    if ((int)heap.size() < K) {
                        heap.push({sc, id});
                    } else {
                        const Node& top = heap.top();
                        if (sc > top.first || (sc == top.first && id < top.second)) {
                            heap.pop();
                            heap.push({sc, id});
                        }
                    }
                }
            }
        }

        // Extract and sort descending score, tie id ascending
        vector<Node> ans;
        ans.reserve(heap.size());
        while (!heap.empty()) { ans.push_back(heap.top()); heap.pop(); }
        sort(ans.begin(), ans.end(), [](const Node& a, const Node& b){
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });

        // If less than K found (possible when N < K)
        while ((int)ans.size() < K) ans.push_back({-1e30f, 0});

        for (int i = 0; i < K; i++) {
            if (i) out.push_back(' ');
            out += to_string(ans[i].second);
        }
        out.push_back('\n');
    }

    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
```

---

# 你可以用这份题面“自然引导”的升级方向（3 小时内常见提分点）

如果你要把题目设计成“能分层拉开差距”，常见的 next steps 会是：

* **更好的粗召回**：把 256 桶升级成 IVF（k-means nlist=1k/4k），query 扫 nprobe 个桶
* **更强压缩**：从 per-vector int8 → per-dim int8 校准，或 PQ（M=16, nbits=8）
* **更快打分**：查表 ADC、循环展开（4/8/16）、SoA、批量 query
* **更聪明的 cache**：重复 query cache；或 centroid 距离 cache

---

如果你希望我把这道题再“做得更像真实线上性能赛”，我可以再给你两个版本（都能直接发布）：

1. **Build/Search 两阶段接口版**（更工程、更适合缓存与索引）
2. **二进制输入版**（IO 不再成为瓶颈，更能比拼算法与 SIMD）



可以的。你提到的这套 baseline 其实就是三块东西拼起来：**（1）per-vector int8 量化** + **（2）用符号位做 256 桶的粗召回（很像 LSH）** + **（3）在候选上用近似内积打分取 topK**。下面把每一块“具体怎么做”讲清楚。

---

## 0) 先统一：余弦相似度怎么变成内积

题面说余弦相似度：
[
\text{sim}(q,x)=\frac{q}{|q|}\cdot\frac{x}{|x|}
]
所以我们在读入 doc/query 后都做 **L2 归一化**：
[
\tilde{x}=x/|x|,\quad \tilde{q}=q/|q|
]
之后：
[
\text{sim}(q,x)=\tilde{q}\cdot \tilde{x}
]
也就是 **内积**，方便后面用量化去近似计算。

---

## 1) Per-vector int8 量化（每个 doc 一个 scale）怎么做

### 1.1 存什么？

对每个 doc 向量 (\tilde{x}\in\mathbb{R}^D)，我们存两样：

1. `codes[i][j]`：每个维度一个 int8（范围大概 ([-127,127])）
2. `scale[i]`：一个 float（这个 doc 的缩放因子）

所以 doc 的近似重构是：
[
\tilde{x}_j \approx \text{scale}[i] \cdot \text{codes}[i][j]
]

### 1.2 scale 怎么选？

baseline 用的是 “**该向量的最大绝对值**” 来决定 scale：

[
\text{maxabs}=\max_j|\tilde{x}_j|,\quad \text{scale}=\text{maxabs}/127
]

这样保证最大的维度量化后大致落在 127 附近，不会溢出。

### 1.3 codes 怎么算？

对每个维度：
[
\text{codes}[j]=\text{round}(\tilde{x}_j/\text{scale})
]
并 clamp 到 ([-127,127])。

> 直觉：`scale` 越小，codes 越大；用 maxabs 让所有维度都能在 int8 范围内表达。

### 1.4 查询时怎么用它算相似度？

我们要算 (\tilde{q}\cdot \tilde{x})。用近似：
[
\tilde{q}\cdot \tilde{x} \approx \sum_j \tilde{q}_j \cdot (\text{scale}\cdot \text{codes}_j)
= \text{scale}\cdot \sum_j \text{codes}_j \tilde{q}_j
]

也就是 baseline 里的：

* 先算 `s = sum(codes[j] * q[j])`
* 再乘 `scale`

**重点：** `codes` 是 int8，但 `q[j]` 是 float，所以这是“int8 × float”的点积（可被循环展开/向量化优化）。

---

## 2) 256 桶粗召回：8 个维度的符号位签名怎么做

这一步的目的：别对全体 N 个 doc 都算点积，先用很便宜的方式筛一小部分候选。

### 2.1 选 8 个维度

baseline 选了 8 个固定维度（比如 0,16,32,...,112），记为 `sigDims[0..7]`。

### 2.2 计算 8-bit 签名（bucket id）

对每个 doc（归一化后的 (\tilde{x})）：

* 看这 8 个维度的符号（>=0 记 1，否则记 0）
* 拼成 8 位二进制数：

[
\text{sig}=\sum_{b=0}^{7} \mathbf{1}[\tilde{x}_{\text{sigDims}[b]}\ge 0]\cdot 2^b
]

sig ∈ [0,255]，就是 bucket 编号。

### 2.3 建倒排表（inverted lists）

维护 `buckets[256]`，每个桶里放 doc id 列表：

* `buckets[sig].push_back(docId)`

> 直觉：这相当于把空间按“8 个坐标的正负（正交象限）”粗略切分。
> 这也可以看成一种非常简化的 LSH（更一般的做法是 SimHash：用随机超平面投影取符号）。

---

## 3) 查询时：本桶 + 8 个单比特翻转邻桶（共 9 桶）怎么做

### 3.1 query 也算一个 sig

同样用 (\tilde{q}) 的 8 个维度符号得到 `sig_q`。

### 3.2 为什么要扫邻桶？

只扫本桶可能召回太低：真实最近邻可能只有某个维度符号不同（尤其该维度接近 0 时，符号很容易翻转）。

所以 baseline 扫：

* 本桶：`sig_q`
* 以及把 8 个 bit 分别翻转一次得到 8 个桶：
  [
  \text{sig}_q \oplus (1\ll b),; b=0..7
  ]
  总共 9 个桶。

这就是代码里的：

* `neigh[0] = sig`
* `neigh[1+b] = sig ^ (1<<b)`

### 3.3 候选集合怎么来？

把这 9 个桶的 doc id 列表合并（不去重也行，但最好去重；baseline没去重也能跑）。

候选数大概是：
[
C = \sum_{t=1}^{9} |bucket_t|
]
（通常远小于 N）

### 3.4 对候选做近似打分并取 topK

对每个候选 docId：

* 计算近似分数 `approx_dot(q, docId)`（第 1 节的式子）
* 用一个大小为 K 的最小堆维护 topK（更快）

最后把堆里元素取出来按分数降序排序输出。

---

## 4) 这套方法“本质上是什么”？它为什么能工作？

* **符号桶**：是一个超轻量的粗检索（粗召回），把不太可能相似的向量大量过滤掉。
* **int8 量化点积**：是一个轻量的近似打分（精排），比 float32 点积快、内存更省。

它的效果取决于数据是否“方向上有结构”（比如向量归一化后确实按方向聚类）。

---

## 5) 你可以马上升级的两个关键点（仍然很像“具体怎么做”）

### A) 桶签名从“看 8 个坐标符号”升级成 SimHash（更稳）

不用固定维度，而是取 8/16 个随机超平面 (r_b)，签名 bit 为：
[
\mathbf{1}[\tilde{x}\cdot r_b \ge 0]
]
这比只看某些坐标的符号更抗“维度选择不佳”。

### B) 量化从 per-vector scale 升级成 per-dimension scale（更准）

当前做法每个向量一个 scale，会对“分布差异大的维度”不友好。
可以改成每个维度一个 scale（全局统计 max/均方等），然后统一量化：
[
\text{codes}[j]=\text{round}(\tilde{x}_j/\text{scaleDim}[j])
]
查询点积也更好向量化（不用每个 doc 再乘一个 scale）。

---

如果你想更贴近比赛“能显著提分”的版本，我可以在不引入太复杂训练的前提下，把这套 baseline 直接改成：**SimHash(16-bit) 粗召回 + int8 点积 + 去重 + 候选上限控制**，并给你一版更快的循环展开点积（4/8/16）。


下面给你一版**可直接替换 baseline 的“3 小时可写完、但明显更强”的版本**：

* **粗召回：SimHash 16-bit**（用 *稀疏随机超平面*，每个 bit 只看 8 个维度，极快）
* **倒排结构：CSR（单数组 ids + offsets）**（65536 个桶不会被 `vector<vector<int>>` 的开销拖死）
* **精排：int8 量化 + 点积（8-way loop unroll）**
* **自适应 probe**：先扫 Hamming 半径 0/1，候选不足再加半径 2（最多 137 个桶）

---

## 这套具体怎么做（很短但抓重点）

### A) 建索引（docs）

1. 读 doc 向量，做 L2 normalize。
2. **算 16-bit SimHash 签名**：每个 bit b 有 8 个随机维度和 ±1 符号
   [
   s_b = \sum_{t=1}^{8} \text{sign}*{b,t}\cdot x*{\text{dim}_{b,t}},\quad \text{bit}_b = [s_b \ge 0]
   ]
   拼成 `sig∈[0,65535]`。
3. 把 doc 放进 `sig` 对应桶：用 `counts[sig]++` 统计大小（第一遍）。
4. 同时做 **per-vector int8 量化**（每个 doc 一个 scale）：

   * `scale = maxabs/127`
   * `code[j] = round(x[j]/scale)` clamp 到 int8
5. docs 读完后把 `counts` 前缀和成 `offsets`，再用 `sig_doc[i]` 把 doc id 写进 `ids`（CSR）。

### B) 查询（queries）

1. 读 query，L2 normalize，算 `sig_q`。
2. 生成要 probe 的桶：

   * 半径 0：`sig_q`
   * 半径 1：`sig_q ^ (1<<b)`，b=0..15（共 16 个）
   * 如果候选不够，再加半径 2（120 个两位翻转），最多 137 桶
3. 对所有候选 doc：计算近似内积
   [
   \tilde{q}\cdot \tilde{x}\approx scale_x\cdot \sum_j code_x[j]\cdot q[j]
   ]
   用 8-way unroll 加速，并维护 topK 小根堆。

---

## 参考实现（C++17，一份文件可编译运行）

> 默认：`B=16`, `SPARSE=8`, 候选目标 `TARGET_CANDS=20000`，候选上限 `MAX_CANDS=200000`
> 你可以在代码里改这几个常量。

```cpp
#include <bits/stdc++.h>
using namespace std;

// ================= Fast Input =================
struct FastInput {
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
        do {
            c = readChar();
            if (!c) return false;
        } while (c <= ' ');

        bool neg = false;
        if (c == '-') { neg = true; c = readChar(); }

        T val = 0;
        while (c > ' ') {
            val = val * 10 + (c - '0');
            c = readChar();
        }
        out = neg ? -val : val;
        return true;
    }

    bool readFloat(float &out) {
        char c;
        do {
            c = readChar();
            if (!c) return false;
        } while (c <= ' ');

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
            if (c == '+' || c == '-') { eneg = (c == '-'); c = readChar(); }
            int expv = 0;
            while (c >= '0' && c <= '9') {
                expv = expv * 10 + (c - '0');
                c = readChar();
            }
            double pow10 = pow(10.0, expv);
            val = eneg ? (val / pow10) : (val * pow10);
        }

        out = (float)(neg ? -val : val);
        return true;
    }
};

// ================= Utils =================
static inline float l2norm(const float* v, int D) {
    double s = 0.0;
    for (int i = 0; i < D; i++) s += (double)v[i] * (double)v[i];
    return (float)sqrt(s);
}
static inline int8_t clamp_int8(int x) {
    if (x > 127) return 127;
    if (x < -127) return -127;
    return (int8_t)x;
}

// ================= Main =================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    FastInput in;
    int N, D, Q, K;
    if (!in.readInt(N)) return 0;
    in.readInt(D);
    in.readInt(Q);
    in.readInt(K);

    // ---- Tunables ----
    constexpr int B = 16;              // SimHash bits => 2^16 buckets
    constexpr int SPARSE = 8;          // each bit uses 8 dims with +/-1
    constexpr size_t TARGET_CANDS = 20000;   // aim for about this many candidates
    constexpr size_t MAX_CANDS    = 200000;  // hard cap, avoid worst-case

    const int NB = 1 << B;

    // Build sparse random hyperplanes: for each bit, store dims and signs
    // Use fixed seed for determinism
    std::mt19937 rng(1234567);
    std::uniform_int_distribution<int> dimDist(0, max(1, D) - 1);
    std::uniform_int_distribution<int> sgnDist(0, 1);

    array<array<int, SPARSE>, B> planeDim;
    array<array<int8_t, SPARSE>, B> planeSgn;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < SPARSE; t++) {
            planeDim[b][t] = dimDist(rng);
            planeSgn[b][t] = (sgnDist(rng) ? (int8_t)+1 : (int8_t)-1);
        }
    }

    // Storage
    vector<int8_t> codes((size_t)N * (size_t)D);  // int8 codes
    vector<float> scales(N);                      // per-doc scale
    vector<uint16_t> sig_doc(N);                  // bucket signature per doc

    // First pass counts
    vector<uint32_t> counts(NB, 0);

    vector<float> tmp(D);

    // ---- Read & preprocess docs ----
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) in.readFloat(tmp[j]);

        // L2 normalize
        float nrm = l2norm(tmp.data(), D);
        if (nrm > 0) {
            float inv = 1.0f / nrm;
            for (int j = 0; j < D; j++) tmp[j] *= inv;
        }

        // SimHash signature (sparse)
        uint16_t sig = 0;
        for (int b = 0; b < B; b++) {
            float s = 0.0f;
            // sum +/- x[dim]
            // SPARSE is small -> loop is tiny
            for (int t = 0; t < SPARSE; t++) {
                s += (float)planeSgn[b][t] * tmp[planeDim[b][t]];
            }
            if (s >= 0) sig |= (uint16_t)(1u << b);
        }
        sig_doc[i] = sig;
        counts[sig]++;

        // per-vector int8 quantization
        float maxabs = 0.0f;
        for (int j = 0; j < D; j++) maxabs = max(maxabs, fabsf(tmp[j]));
        if (maxabs < 1e-12f) maxabs = 1e-12f;

        float scale = maxabs / 127.0f;
        scales[i] = scale;
        float invs = 1.0f / scale;

        int8_t* dst = &codes[(size_t)i * (size_t)D];
        for (int j = 0; j < D; j++) {
            int qv = (int)lrintf(tmp[j] * invs);
            dst[j] = clamp_int8(qv);
        }
    }

    // ---- Build CSR buckets: offsets + ids ----
    vector<uint32_t> offsets(NB + 1, 0);
    for (int b = 0; b < NB; b++) offsets[b + 1] = offsets[b] + counts[b];

    vector<uint32_t> cursor = offsets;          // write pointers
    vector<int> ids(N);                         // all doc ids stored contiguously

    for (int i = 0; i < N; i++) {
        uint16_t sig = sig_doc[i];
        uint32_t pos = cursor[sig]++;
        ids[pos] = i;
    }

    // ---- Dot product: float q with int8 doc, unroll 8 ----
    auto approx_dot_unroll8 = [&](const float* q, int docId) -> float {
        const int8_t* dc = &codes[(size_t)docId * (size_t)D];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

        int j = 0;
        int limit = D & ~7;
        for (; j < limit; j += 8) {
            a0 += (float)dc[j + 0] * q[j + 0];
            a1 += (float)dc[j + 1] * q[j + 1];
            a2 += (float)dc[j + 2] * q[j + 2];
            a3 += (float)dc[j + 3] * q[j + 3];
            a0 += (float)dc[j + 4] * q[j + 4];
            a1 += (float)dc[j + 5] * q[j + 5];
            a2 += (float)dc[j + 6] * q[j + 6];
            a3 += (float)dc[j + 7] * q[j + 7];
        }
        float s = (a0 + a1) + (a2 + a3);
        for (; j < D; j++) s += (float)dc[j] * q[j];
        return s * scales[docId];
    };

    // ---- Probe generator (adaptive): radius 0 -> 1 -> 2 ----
    auto append_radius1 = [&](uint16_t sig, vector<uint16_t>& probes) {
        for (int b = 0; b < B; b++) probes.push_back((uint16_t)(sig ^ (1u << b)));
    };
    auto append_radius2 = [&](uint16_t sig, vector<uint16_t>& probes) {
        for (int i = 0; i < B; i++) {
            for (int j = i + 1; j < B; j++) {
                probes.push_back((uint16_t)(sig ^ (1u << i) ^ (1u << j)));
            }
        }
    };

    // ---- Query loop ----
    string out;
    out.reserve((size_t)Q * (size_t)K * 6);

    vector<float> qv(D);

    for (int qi = 0; qi < Q; qi++) {
        for (int j = 0; j < D; j++) in.readFloat(qv[j]);

        // L2 normalize
        float nrm = l2norm(qv.data(), D);
        if (nrm > 0) {
            float inv = 1.0f / nrm;
            for (int j = 0; j < D; j++) qv[j] *= inv;
        }

        // SimHash signature
        uint16_t sigq = 0;
        for (int b = 0; b < B; b++) {
            float s = 0.0f;
            for (int t = 0; t < SPARSE; t++) s += (float)planeSgn[b][t] * qv[planeDim[b][t]];
            if (s >= 0) sigq |= (uint16_t)(1u << b);
        }

        // Build probe list adaptively
        vector<uint16_t> probes;
        probes.reserve(1 + B + (B * (B - 1)) / 2);
        probes.push_back(sigq);
        append_radius1(sigq, probes);

        // estimate candidate count for current probes
        auto count_cands = [&](const vector<uint16_t>& ps) -> size_t {
            size_t c = 0;
            for (uint16_t s : ps) c += (offsets[s + 1] - offsets[s]);
            return c;
        };

        size_t cand = count_cands(probes);
        if (cand < (size_t)K || cand < TARGET_CANDS) {
            // try radius-2
            vector<uint16_t> probes2 = probes;
            probes2.reserve(probes.size() + (B * (B - 1)) / 2);
            append_radius2(sigq, probes2);

            size_t cand2 = count_cands(probes2);
            if (cand2 <= MAX_CANDS) {
                probes.swap(probes2);
                cand = cand2;
            } else {
                // too many; keep radius-1 only
            }
        }

        // topK min-heap: (score, id) with worst on top
        using Node = pair<float,int>;
        auto worse = [](const Node& a, const Node& b){
            if (a.first != b.first) return a.first > b.first; // min-heap by score
            return a.second > b.second; // larger id worse
        };
        priority_queue<Node, vector<Node>, decltype(worse)> heap(worse);

        // Scan candidates bucket by bucket
        for (uint16_t s : probes) {
            uint32_t l = offsets[s], r = offsets[s + 1];
            for (uint32_t p = l; p < r; p++) {
                int id = ids[p];
                float sc = approx_dot_unroll8(qv.data(), id);
                if ((int)heap.size() < K) {
                    heap.push({sc, id});
                } else {
                    const Node& top = heap.top();
                    if (sc > top.first || (sc == top.first && id < top.second)) {
                        heap.pop();
                        heap.push({sc, id});
                    }
                }
            }
        }

        // Extract + sort
        vector<Node> ans;
        ans.reserve(heap.size());
        while (!heap.empty()) { ans.push_back(heap.top()); heap.pop(); }
        sort(ans.begin(), ans.end(), [](const Node& a, const Node& b){
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });

        while ((int)ans.size() < K) ans.push_back({-1e30f, 0});

        for (int i = 0; i < K; i++) {
            if (i) out.push_back(' ');
            out += to_string(ans[i].second);
        }
        out.push_back('\n');
    }

    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
```

---

## 你接下来最容易“再提一档”的两个改动（很推荐）

1. **把 `SPARSE=8` 调到 12 或 16**：签名更稳，召回↑，建索引稍慢但通常值。
2. **候选策略更聪明**：

   * `TARGET_CANDS` 设成跟 K、N 成比例（比如 `max(5000, 200*K)`）
   * 或者先扫 radius-1，如果 bucket 太大就只取前一部分（采样）避免极端。

你如果把你比赛数据的大概规模（N/D/Q/K）和时间/内存限制贴一下，我可以直接给你推荐一组更贴合的默认参数（B、SPARSE、TARGET_CANDS、MAX_CANDS、是否扫 radius-2）。


你问的这一步其实是在把 “`buckets[sig].push_back(docId)` 这种很多小 vector” 换成一种**更省内存、更连续、更快遍历**的数据结构：**CSR / Compressed Sparse Row（压缩行存储）**。在向量检索里也常叫“倒排表的压缩存储”。

下面把它拆开讲清楚。

---

## 目标：用两个数组表示 65536 个桶的倒排表

我们想表达的东西是：

* 每个桶 `sig` 里有一串 docId：`bucket[sig] = [id1, id2, ...]`

CSR 用两个数组就能表示全部桶：

1. `offsets[sig]`：桶 `sig` 在大数组 `ids` 里的起始位置
2. `ids[]`：把所有桶的 docId **按桶顺序**连在一起的大数组

于是桶 `sig` 的 doc 列表就是：

* `ids[ offsets[sig] ... offsets[sig+1]-1 ]`

---

## Step 1：`counts` 是每个桶的大小

第一遍读 docs 时，我们已经算出了每个 doc 的 `sig_doc[i]`（它属于哪个桶），于是可以统计：

* `counts[sig]++`

最终 `counts[sig]` 表示：桶 `sig` 里有多少个 doc。

---

## Step 2：把 `counts` 做前缀和得到 `offsets`

我们要给每个桶在 `ids` 里分一段连续空间，长度就是 `counts[sig]`。

所以定义：

* `offsets[0] = 0`
* `offsets[sig+1] = offsets[sig] + counts[sig]`

这就是“前缀和”。

这样一来：

* 桶 `sig` 的空间就是 `[offsets[sig], offsets[sig+1])`
* 总长度 `offsets[NB] = N`（所有 doc 总数）

---

## Step 3：用 `cursor` 把 docId 填进 `ids`

现在每个桶的区间已经分配好了，但我们还没把具体 docId 写进去。

做法是：

* `cursor = offsets` 的拷贝，表示“每个桶当前写到哪里了”
* 遍历所有 doc i：

  * `sig = sig_doc[i]`
  * `pos = cursor[sig]`（该桶下一个可写位置）
  * `ids[pos] = i`
  * `cursor[sig]++`

这样每个 doc 就被写进了它所属桶的连续区间里。

---

## 一个很小的例子（看完就懂）

假设只有 4 个桶（0..3），N=6 个 doc，它们的桶是：

* doc0→桶2
* doc1→桶0
* doc2→桶2
* doc3→桶1
* doc4→桶2
* doc5→桶0

### 1) 统计 counts

* counts[0]=2（doc1, doc5）
* counts[1]=1（doc3）
* counts[2]=3（doc0, doc2, doc4）
* counts[3]=0

### 2) 前缀和 offsets

* offsets[0]=0
* offsets[1]=0+2=2
* offsets[2]=2+1=3
* offsets[3]=3+3=6
* offsets[4]=6+0=6

所以桶区间：

* 桶0：ids[0..1]
* 桶1：ids[2..2]
* 桶2：ids[3..5]
* 桶3：空

### 3) 填 ids（用 cursor）

初始 cursor = offsets = [0,2,3,6]

按 doc 顺序写：

* doc0(sig2): pos=3 → ids[3]=0, cursor2=4
* doc1(sig0): pos=0 → ids[0]=1, cursor0=1
* doc2(sig2): pos=4 → ids[4]=2, cursor2=5
* doc3(sig1): pos=2 → ids[2]=3, cursor1=3
* doc4(sig2): pos=5 → ids[5]=4, cursor2=6
* doc5(sig0): pos=1 → ids[1]=5, cursor0=2

最终 ids 大数组可能是：

* ids = [1, 5, 3, 0, 2, 4]

于是：

* 桶0 = ids[offsets0..offsets1-1] = ids[0..1] = [1,5]
* 桶1 = ids[2..2] = [3]
* 桶2 = ids[3..5] = [0,2,4]

完事。

---

## 为什么要这么做（比 vector<vector<int>> 好在哪）

* **内存更省**：不会有 65536 个 `vector` 的对象开销和碎片
* **更快**：每个桶的 docId 在内存里连续，遍历时 cache 友好
* **构建也稳定**：两遍线性流程，几乎没有小块动态分配

---

如果你愿意，我也可以把这段 CSR 构建抽成一个独立函数（更好复用/更不容易写错），或者再讲讲**如何在 probes 很多时做去重**（避免同一 doc 被重复打分）。
