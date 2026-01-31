# ANN 向量检索库 - 赛时使用手册

## 目录

1. [快速上手](#1-快速上手)
2. [架构总览](#2-架构总览)
3. [三条流水线详解](#3-三条流水线详解)
4. [核心组件与接口](#4-核心组件与接口)
5. [参数调优实战](#5-参数调优实战)
6. [常见问题诊断](#6-常见问题诊断)
7. [赛时策略建议](#7-赛时策略建议)
8. [性能优化技巧](#8-性能优化技巧)

---

## 1. 快速上手

### 1.1 最小可运行示例

```cpp
#include "ann/pipeline_factory.hpp"
using namespace ann;

int main() {
    // 1. 准备数据（实际比赛中从题目接口获取）
    std::vector<std::vector<float>> base;   // N 个 D 维向量
    std::vector<std::vector<float>> queries; // Q 个查询向量

    // 2. 选择流水线（三选一）
    auto pipeline = make_pipeline(PipelineType::HNSW_SQ_RERANK);

    // 3. 配置参数
    SearchParams params;
    params.k = 10;
    params.metric = Metric::L2;
    params.threads = 4;

    // 4. 构建索引（离线阶段）
    pipeline->build(base, params);

    // 5. 执行查询（在线阶段）
    auto results = pipeline->batch_search(queries, params);

    // 6. 提取结果
    for (size_t i = 0; i < results.size(); ++i) {
        for (const auto& r : results[i]) {
            // r.id 是底库向量编号，r.score 是距离
        }
    }
    return 0;
}
```

### 1.2 编译命令

```bash
# 基础编译
g++ -std=c++17 -O3 -march=native -I include your_code.cpp -o solution

# 启用 OpenMP 并行（强烈推荐）
g++ -std=c++17 -O3 -march=native -fopenmp -DANN_USE_OPENMP -I include your_code.cpp -o solution
```

### 1.3 三条流水线速查表

| 流水线 | 构建速度 | 查询速度 | 召回率 | 内存占用 | 推荐场景 |
|--------|----------|----------|--------|----------|----------|
| `HNSW_RAW` | 慢 | 中 | 高 | 高 | 追求召回、数据量中等 |
| `HNSW_SQ_RERANK` | 慢 | 中 | 最高 | 中 | 平衡场景（**首选**） |
| `LSH_SQ_RERANK` | **快** | 快 | 中 | 低 | 超大数据、构建时间紧 |

---

## 2. 架构总览

### 2.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │Preprocess│ -> │  Index   │ -> │  Scorer  │ -> │ Reranker │  │
│  │(归一化)   │    │(召回候选) │    │(近似打分) │    │(精确重排) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 文件与职责

| 文件 | 职责 | 关键类/函数 |
|------|------|-------------|
| `common.hpp` | 类型定义、接口声明 | `SearchResult`, `SearchParams`, `Pipeline` |
| `metric.hpp` | 距离计算 | `l2_distance_squared()`, `inner_product()` |
| `preprocess.hpp` | 向量预处理 | `NormalizePreprocess` |
| `topk.hpp` | TopK 选择算法 | `SmallKTopK`, `HeapTopK`, `select_topk()` |
| `sq.hpp` | 标量量化 | `SQCompressor` |
| `index_hnsw.hpp` | HNSW 图索引 | `HNSWIndex` |
| `index_lsh.hpp` | LSH 哈希索引 | `LSHIndex` |
| `scorer_sq.hpp` | SQ 近似打分 | `SQScorer` |
| `reranker_raw.hpp` | 原始向量精排 | `RawReranker` |
| `pipeline.hpp` | 三条流水线实现 | `HNSWRawPipeline`, `HNSWSQRerankPipeline`, `LSHSQRerankPipeline` |
| `pipeline_factory.hpp` | 工厂函数 | `make_pipeline()` |

---

## 3. 三条流水线详解

### 3.1 Pipeline 1: HNSW_RAW

**原理：**
```
Query -> HNSW 图搜索 -> 直接返回 TopK
```

**工作流程：**
1. 从入口点开始，在高层快速定位到目标区域
2. 逐层下降，在每层贪心搜索最近邻
3. 在第 0 层扩展搜索 `efSearch` 个候选
4. 返回最近的 `k` 个结果

**参数影响：**
```
M (连接数)
├── 增大 -> 图更稠密 -> 召回↑ 内存↑ 构建慢
└── 减小 -> 图更稀疏 -> 召回↓ 内存↓ 构建快

efConstruction (构建搜索宽度)
├── 增大 -> 找到更好的邻居 -> 图质量↑ 构建慢
└── 减小 -> 邻居质量下降 -> 图质量↓ 构建快

efSearch (查询搜索宽度)
├── 增大 -> 探索更多候选 -> 召回↑ 查询慢
└── 减小 -> 探索更少候选 -> 召回↓ 查询快
```

**代码位置：** `include/ann/index_hnsw.hpp`

### 3.2 Pipeline 2: HNSW_SQ_RERANK

**原理：**
```
Query -> HNSW 召回 rerankK 个候选 -> SQ 近似打分 -> Raw 精排 -> TopK
```

**工作流程：**
1. HNSW 召回 `rerankK` 个候选（粗筛）
2. SQ 量化向量快速打分（可选，当前实现直接精排）
3. 用原始 float32 向量精确计算距离
4. 返回最近的 `k` 个结果

**为什么召回率更高？**
- HNSW 可能漏掉一些真正的近邻
- 但召回 `rerankK`（如 200）个候选后精排
- 精排保证这 200 个中选出的 top-k 是最准确的

**参数影响：**
```
rerankK (精排候选数)
├── 增大 -> 候选池更大 -> 召回↑ 精排慢
└── 减小 -> 候选池更小 -> 召回↓ 精排快

经验公式：rerankK = 10 * k ~ 20 * k
```

**代码位置：** `include/ann/pipeline.hpp` -> `HNSWSQRerankPipeline`

### 3.3 Pipeline 3: LSH_SQ_RERANK

**原理：**
```
Query -> 多表 LSH 哈希 -> 收集候选 -> SQ 打分 -> Raw 精排 -> TopK
```

**工作流程：**
1. 对 query 计算每个表的哈希值
2. 查找哈希桶 + multi-probe 邻近桶
3. 合并所有候选并去重
4. 精排返回 TopK

**LSH 哈希原理：**
```
每个表有 numBits 个随机超平面
hash = 0
for each 超平面 h[i]:
    if dot(query, h[i]) >= 0:
        hash |= (1 << i)
相似向量大概率落入同一个桶
```

**参数影响：**
```
numTables (哈希表数)
├── 增大 -> 更多独立哈希 -> 召回↑ 内存↑
└── 减小 -> 更少哈希表 -> 召回↓ 内存↓

numBits (每表位数)
├── 增大 -> 桶更细 (2^numBits 个桶) -> 每桶向量少 -> 召回↓
└── 减小 -> 桶更粗 -> 每桶向量多 -> 召回↑ 但候选多

probes (多探测数)
├── 增大 -> 探测更多邻近桶 -> 召回↑ 查询慢
└── 减小 -> 只探测原桶 -> 召回↓ 查询快

关键公式：
  桶数 = 2^numBits
  理想每桶向量数 = 10~50
  推荐 numBits ≈ log2(N / 20)

  N=10000  -> numBits=9
  N=100000 -> numBits=12
  N=1000000 -> numBits=16
```

**代码位置：** `include/ann/index_lsh.hpp`

---

## 4. 核心组件与接口

### 4.1 SearchParams 完整说明

```cpp
struct SearchParams {
    // ===== 通用参数 =====
    int k = 10;                    // 返回的近邻数量
    int threads = 1;               // batch_search 的并行线程数
    Metric metric = Metric::L2;    // 距离度量类型
    bool normalize = false;        // 是否归一化（COS 自动启用）

    // ===== HNSW 参数 =====
    int M = 16;                    // 每层每节点的最大连接数
    int efConstruction = 200;      // 构建时的搜索宽度
    int efSearch = 64;             // 查询时的搜索宽度

    // ===== 两阶段参数 =====
    int rerankK = 200;             // 精排候选数量

    // ===== LSH 参数 =====
    int numTables = 4;             // 哈希表数量
    int numBits = 14;              // 每表的哈希位数
    int probes = 2;                // multi-probe 探测数
};
```

### 4.2 Metric 度量类型

```cpp
enum class Metric { L2, IP, COS };
```

| 度量 | 公式 | 分数含义 | 适用场景 |
|------|------|----------|----------|
| `L2` | `Σ(a[i]-b[i])²` | 越小越近 | 欧氏空间、图像特征 |
| `IP` | `-Σ(a[i]*b[i])` | 越小越相似 | 已归一化向量、推荐系统 |
| `COS` | `-cos(a,b)` | 越小越相似 | 文本嵌入、语义相似度 |

**注意：** 所有分数都是**越小越好**，结果按升序排列。

### 4.3 SearchResult 结构

```cpp
struct SearchResult {
    int id;      // 底库向量的索引 (0-based)
    float score; // 距离分数（越小越好）
};
```

### 4.4 Pipeline 接口

```cpp
class Pipeline {
public:
    // 构建索引（离线阶段调用一次）
    virtual void build(const std::vector<std::vector<float>>& base,
                       const SearchParams& p) = 0;

    // 单条查询
    virtual std::vector<SearchResult> search(
        const std::vector<float>& query,
        const SearchParams& p) const = 0;

    // 批量查询（自动并行）
    virtual std::vector<std::vector<SearchResult>> batch_search(
        const std::vector<std::vector<float>>& queries,
        const SearchParams& p) const = 0;
};
```

### 4.5 工厂函数

```cpp
enum class PipelineType {
    HNSW_RAW,        // 单阶段 HNSW
    HNSW_SQ_RERANK,  // HNSW + 精排
    LSH_SQ_RERANK    // LSH + 精排
};

std::unique_ptr<Pipeline> make_pipeline(PipelineType t);
```

---

## 5. 参数调优实战

### 5.1 HNSW 调参流程

```
Step 1: 固定 efConstruction=200, efSearch=64, 调 M
        M=8  -> 召回太低？ -> M=16
        M=16 -> 还不够？   -> M=32
        M=32 -> 内存爆了？ -> 回退 M=24

Step 2: 固定 M，调 efSearch
        efSearch=32  -> 召回 0.85
        efSearch=64  -> 召回 0.92
        efSearch=128 -> 召回 0.96（但慢了）
        -> 选择满足召回要求的最小 efSearch

Step 3: 如果构建太慢，降低 efConstruction
        efConstruction=100 通常够用
```

**参数速查表：**

| 数据规模 | M | efConstruction | efSearch | 预期召回 |
|----------|---|----------------|----------|----------|
| 1万 | 16 | 100 | 64 | 0.90+ |
| 10万 | 16 | 200 | 100 | 0.90+ |
| 100万 | 24 | 200 | 128 | 0.85+ |
| 1000万 | 32 | 200 | 200 | 0.80+ |

### 5.2 LSH 调参流程

```
Step 1: 根据数据量确定 numBits
        numBits = floor(log2(N / 20))
        N=10000  -> numBits=9
        N=100000 -> numBits=12

Step 2: 设置足够的 numTables
        numTables=8  -> 召回不够？
        numTables=16 -> 召回提升
        numTables=32 -> 内存翻倍，谨慎

Step 3: 调整 probes
        probes=2 -> 基础
        probes=4 -> 召回提升明显
        probes=8 -> 查询变慢，收益递减
```

**参数速查表：**

| 数据规模 | numTables | numBits | probes | 预期召回 |
|----------|-----------|---------|--------|----------|
| 1万 | 16 | 8 | 4 | 0.85+ |
| 10万 | 16 | 10 | 4 | 0.80+ |
| 100万 | 24 | 12 | 6 | 0.75+ |

### 5.3 rerankK 调参

```
基本原则：rerankK = α * k

α=10  -> 基础召回
α=20  -> 较好召回
α=50  -> 高召回（精排开销大）

示例：
  k=10, rerankK=100  -> 召回 ~0.90
  k=10, rerankK=200  -> 召回 ~0.95
  k=10, rerankK=500  -> 召回 ~0.98（慢）
```

### 5.4 线程数调优

```cpp
params.threads = std::thread::hardware_concurrency(); // 使用所有核心

// 或者保守一点
params.threads = std::thread::hardware_concurrency() - 1;
```

---

## 6. 常见问题诊断

### 6.1 召回率低

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| HNSW 召回 < 0.8 | efSearch 太小 | 增大 efSearch 到 128+ |
| HNSW 召回 < 0.7 | M 太小 | 增大 M 到 24-32 |
| LSH 召回 < 0.5 | numBits 太大，桶太稀疏 | 减小 numBits |
| LSH 召回 < 0.3 | numTables 太少 | 增加 numTables 到 16+ |
| LSH 召回 < 0.2 | probes 太小 | 增加 probes 到 4+ |
| 两阶段召回低 | rerankK 太小 | 增大 rerankK |

### 6.2 构建太慢

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| HNSW 构建 > 10分钟 | efConstruction 太大 | 降到 100-150 |
| HNSW 构建 > 10分钟 | M 太大 | 降到 16-24 |
| LSH 构建慢 | numTables 太多 | 减少到 8-16 |

### 6.3 查询太慢

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 单查询 > 10ms | efSearch 太大 | 降低 efSearch |
| 单查询 > 10ms | rerankK 太大 | 降低 rerankK |
| 批量查询慢 | 未启用并行 | 设置 threads > 1 |
| LSH 查询慢 | probes 太大 | 减小 probes |
| LSH 查询慢 | 候选太多 | 增大 numBits |

### 6.4 内存不足

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| HNSW OOM | M 太大 | 减小 M |
| LSH OOM | numTables 太多 | 减少 numTables |
| 通用 OOM | 数据量太大 | 换用 LSH（内存更省） |

### 6.5 结果异常

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 结果全是 0 | 数据未正确加载 | 检查 base 向量 |
| 分数全是 inf | 向量包含 NaN | 预处理清洗数据 |
| COS 结果差 | 未归一化 | 确认 metric=COS 会自动归一化 |
| IP 结果反了 | 分数方向理解错误 | 分数越小越相似 |

---

## 7. 赛时策略建议

### 7.1 拿到题目后的检查清单

```
□ 数据规模：N=? D=? Q=?
□ 度量类型：L2 / IP / COS ?
□ 召回要求：Recall@K >= ?
□ 延迟要求：QPS >= ? 或 单查询 < ?ms
□ 内存限制：< ?GB
□ 构建时间限制：< ?分钟
```

### 7.2 流水线选择决策树

```
                    ┌─ 构建时间紧（<1分钟）？
                    │       ├─ 是 -> LSH_SQ_RERANK
                    │       └─ 否 ─┐
                    │              │
数据规模 N ─────────┤              ▼
                    │       ┌─ 召回要求高（>0.95）？
                    │       │       ├─ 是 -> HNSW_SQ_RERANK
                    │       │       └─ 否 -> HNSW_RAW
                    │       │
                    └─ N > 100万？
                            ├─ 是 -> LSH_SQ_RERANK（内存友好）
                            └─ 否 -> HNSW_SQ_RERANK
```

### 7.3 快速调参模板

**场景 A：追求最高召回**
```cpp
SearchParams params;
params.metric = Metric::L2;  // 根据题目调整
params.k = 10;

// HNSW_SQ_RERANK
params.M = 32;
params.efConstruction = 300;
params.efSearch = 200;
params.rerankK = 500;
params.threads = 8;
```

**场景 B：平衡召回与速度**
```cpp
SearchParams params;
params.metric = Metric::L2;
params.k = 10;

// HNSW_SQ_RERANK
params.M = 16;
params.efConstruction = 200;
params.efSearch = 100;
params.rerankK = 200;
params.threads = 8;
```

**场景 C：超大数据量**
```cpp
SearchParams params;
params.metric = Metric::L2;
params.k = 10;

// LSH_SQ_RERANK
params.numTables = 20;
params.numBits = 12;  // 根据 N 调整
params.probes = 5;
params.rerankK = 300;
params.threads = 8;
```

### 7.4 调参优先级

```
1. 先跑通：使用默认参数确保代码正确
2. 调召回：efSearch / rerankK / numTables
3. 调速度：降低上述参数直到满足延迟要求
4. 调内存：M / numTables
5. 微调：在召回-速度边界上二分搜索最优点
```

---

## 8. 性能优化技巧

### 8.1 编译优化

```bash
# 必须使用的编译选项
-O3              # 最高优化级别
-march=native    # 针对当前 CPU 优化
-fopenmp         # 启用 OpenMP 并行
-DANN_USE_OPENMP # 启用库内 OpenMP 支持

# 完整命令
g++ -std=c++17 -O3 -march=native -fopenmp -DANN_USE_OPENMP \
    -I include your_code.cpp -o solution
```

### 8.2 数据预处理

```cpp
// 如果数据有 NaN 或 Inf，先清洗
for (auto& vec : base) {
    for (auto& v : vec) {
        if (std::isnan(v) || std::isinf(v)) v = 0.0f;
    }
}

// 如果使用 COS 度量，可以预先归一化（虽然库会自动做）
// 这样可以避免重复归一化
```

### 8.3 批量查询优化

```cpp
// 总是使用 batch_search 而不是循环调用 search
// 差的写法
for (const auto& q : queries) {
    auto r = pipeline->search(q, params);  // 无法并行
}

// 好的写法
params.threads = 8;
auto results = pipeline->batch_search(queries, params);  // 自动并行
```

### 8.4 内存预分配

```cpp
// 如果知道数据规模，预分配内存
base.reserve(N);
for (int i = 0; i < N; ++i) {
    base.emplace_back(D);
    // 读取数据...
}
```

---

## 附录 A：完整代码模板

```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include "ann/pipeline_factory.hpp"

using namespace ann;

int main() {
    // ========== 1. 读取数据 ==========
    int N, D, Q, K;
    // 从题目接口读取 N, D, Q, K

    std::vector<std::vector<float>> base(N, std::vector<float>(D));
    std::vector<std::vector<float>> queries(Q, std::vector<float>(D));
    // 从题目接口读取向量数据

    // ========== 2. 选择流水线 ==========
    auto pipeline = make_pipeline(PipelineType::HNSW_SQ_RERANK);

    // ========== 3. 配置参数 ==========
    SearchParams params;
    params.k = K;
    params.metric = Metric::L2;  // 根据题目要求
    params.threads = 8;

    // HNSW 参数
    params.M = 16;
    params.efConstruction = 200;
    params.efSearch = 100;
    params.rerankK = K * 20;

    // ========== 4. 构建索引 ==========
    auto t1 = std::chrono::high_resolution_clock::now();
    pipeline->build(base, params);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cerr << "Build time: " << build_ms << " ms" << std::endl;

    // ========== 5. 执行查询 ==========
    auto t3 = std::chrono::high_resolution_clock::now();
    auto results = pipeline->batch_search(queries, params);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    std::cerr << "Search time: " << search_ms << " ms" << std::endl;

    // ========== 6. 输出结果 ==========
    for (size_t i = 0; i < results.size(); ++i) {
        for (const auto& r : results[i]) {
            std::cout << r.id << " ";  // 根据题目要求的格式输出
        }
        std::cout << "\n";
    }

    return 0;
}
```

---

## 附录 B：调试技巧

### 打印中间结果

```cpp
// 检查候选数量
std::vector<int> candidates;
index.search_candidates(query, rerankK, candidates, params);
std::cerr << "Candidates: " << candidates.size() << std::endl;

// 检查分数分布
for (int i = 0; i < std::min(10, (int)results.size()); ++i) {
    std::cerr << "id=" << results[i].id << " score=" << results[i].score << std::endl;
}
```

### 计算真实召回率

```cpp
// 暴力计算 ground truth
std::vector<SearchResult> gt;
for (int i = 0; i < N; ++i) {
    float dist = compute_distance(query, base[i], params.metric);
    gt.push_back({i, dist});
}
std::sort(gt.begin(), gt.end());
gt.resize(K);

// 计算召回
std::unordered_set<int> gt_set;
for (const auto& r : gt) gt_set.insert(r.id);

int hits = 0;
for (const auto& r : results) {
    if (gt_set.count(r.id)) hits++;
}
float recall = (float)hits / K;
std::cerr << "Recall@" << K << " = " << recall << std::endl;
```

---

**祝比赛顺利！**
