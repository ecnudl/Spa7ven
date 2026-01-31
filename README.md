# ANN 向量检索库

一个纯头文件的 C++17 近似最近邻 (ANN) 搜索库，无外部依赖（仅 STL，OpenMP 可选）。

## 特性

- **无 I/O**：所有数据通过内存数组/向量传递
- **无训练**：仅轻量统计（SQ 量化的 min/max）
- **三条可切换流水线**：统一接口，组件可插拔
- **性能优化**：连续内存布局、高效 TopK 选择、批量并行化

## 编译

```bash
mkdir build && cd build
cmake ..
make
./sanity_test
```

启用 OpenMP（推荐用于批量搜索）：
```bash
cmake -DCMAKE_CXX_FLAGS="-fopenmp" ..
```

## 三条流水线

### Pipeline 1: HNSW_RAW（单阶段）

直接使用 HNSW 搜索返回原始距离。

**适用场景：**
- 需要最高召回率
- 内存不是瓶颈
- 可接受较高查询延迟

**关键参数：**
- `M`：每个节点的连接数（默认：16）。越大召回越好，内存越多
- `efConstruction`：构建时搜索宽度（默认：200）。越大图质量越好
- `efSearch`：查询时搜索宽度（默认：64）。越大召回越好，速度越慢

### Pipeline 2: HNSW_SQ_RERANK（两阶段）

HNSW 召回 + SQ 近似打分 + 原始向量精排。

**适用场景：**
- 需要速度与精度的平衡
- 大数据集全精度计算代价高
- 内存效率重要

**关键参数：**
- HNSW 参数同 Pipeline 1
- `rerankK`：精排候选数量（默认：200）。越大召回越好

### Pipeline 3: LSH_SQ_RERANK（两阶段）

LSH 多表哈希 + SQ 打分 + 原始向量精排。

**适用场景：**
- 超大规模数据集
- 可接受近似结果
- 需要快速构建

**关键参数：**
- `numTables`：哈希表数量（默认：4）。越多召回越好
- `numBits`：每表哈希位数（默认：14）。位数越多桶越细
- `probes`：多探测扩展（默认：2）。越大候选越多

## 参数调优指南

### HNSW 参数

| 参数 | 范围 | 影响 |
|------|------|------|
| M | 8-64 | 内存与召回的权衡 |
| efConstruction | 100-500 | 构建质量（一次性开销） |
| efSearch | 32-256 | 查询召回与延迟的权衡 |

**建议：**
- 起始值：M=16, efConstruction=200, efSearch=64
- 提高召回：增大 efSearch
- 高维数据：增大 M

### 两阶段参数

| 参数 | 范围 | 影响 |
|------|------|------|
| rerankK | 50-500 | 候选池大小 |

**建议：**
- rerankK = k 的 10~20 倍可获得较好召回
- rerankK 越大召回越高，但延迟增加

### LSH 参数

| 参数 | 范围 | 影响 |
|------|------|------|
| numTables | 4-16 | 召回提升 |
| numBits | 6-12 | 桶粒度 |
| probes | 1-5 | 多探测扩展 |

**建议：**
- 起始值：numTables=16, numBits=8, probes=4
- 提高召回：增加 numTables 或 probes
- numBits 推荐值 ≈ log2(n / 20)

## 分数方向

**重要**：所有分数采用"越小越好"的约定。

| 度量 | 分数含义 |
|------|----------|
| L2 | 欧氏距离平方（越小越近） |
| IP | 负内积（越小相似度越高） |
| COS | 负余弦相似度（越小相似度越高） |

结果按分数升序排列。

## 使用示例

```cpp
#include "ann/pipeline_factory.hpp"

using namespace ann;

int main() {
    // 准备数据
    std::vector<std::vector<float>> base = /* 底库向量 */;
    std::vector<std::vector<float>> queries = /* 查询向量 */;

    // 创建流水线
    auto pipeline = make_pipeline(PipelineType::HNSW_SQ_RERANK);

    // 配置参数
    SearchParams params;
    params.k = 10;
    params.metric = Metric::L2;
    params.M = 16;
    params.efConstruction = 200;
    params.efSearch = 64;
    params.rerankK = 100;
    params.threads = 4;

    // 构建索引
    pipeline->build(base, params);

    // 单条查询
    auto results = pipeline->search(queries[0], params);

    // 批量查询（并行化）
    auto batch_results = pipeline->batch_search(queries, params);

    return 0;
}
```

## API 参考

### SearchResult
```cpp
struct SearchResult {
    int id;      // 底库中的向量 ID
    float score; // 距离/相似度分数
};
```

### SearchParams
```cpp
struct SearchParams {
    int k = 10;                    // 返回结果数
    int threads = 1;               // batch_search 并行度
    Metric metric = Metric::L2;    // L2, IP 或 COS

    // HNSW
    int M = 16;
    int efConstruction = 200;
    int efSearch = 64;

    // 两阶段
    int rerankK = 200;

    // LSH
    int numTables = 4;
    int numBits = 14;
    int probes = 2;

    bool normalize = false;        // COS 度量时自动启用
};
```

### Pipeline 接口
```cpp
class Pipeline {
    void build(const std::vector<std::vector<float>>& base, const SearchParams& p);
    std::vector<SearchResult> search(const std::vector<float>& query, const SearchParams& p) const;
    std::vector<std::vector<SearchResult>> batch_search(
        const std::vector<std::vector<float>>& queries, const SearchParams& p) const;
};
```

### 工厂函数
```cpp
enum class PipelineType { HNSW_RAW, HNSW_SQ_RERANK, LSH_SQ_RERANK };
std::unique_ptr<Pipeline> make_pipeline(PipelineType t);
```

## 文件结构

```
include/ann/
  common.hpp          - 类型、枚举、接口定义
  metric.hpp          - 距离函数（L2, IP）
  preprocess.hpp      - 归一化预处理
  topk.hpp            - TopK 选择（SmallK, Heap）
  sq.hpp              - 标量量化
  index_hnsw.hpp      - HNSW 索引
  index_lsh.hpp       - LSH 索引
  scorer_sq.hpp       - SQ 打分器
  reranker_raw.hpp    - 原始向量精排器
  pipeline.hpp        - 流水线实现
  pipeline_factory.hpp - 工厂函数
tests/
  sanity_test.cpp     - 验证测试
```

## 许可证

MIT
