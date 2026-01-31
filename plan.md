========================
总体约束
========================
1) 不实现任何 I/O（不读文件、不写文件、不打印大日志）
2) 不实现任何训练层（不允许 kmeans / PQ 训练 / OPQ 训练）
   - 允许 build 时做轻量统计：例如 SQ 的 per-dim min/max 或 mean/std（扫描 base 一遍）
3) C++17
4) 性能友好：
   - 尽量连续内存
   - TopK 对小 k 使用固定数组插入
   - 支持 batch_search 的 query 级并行（OpenMP 优先；否则 std::thread）
5) 需要 tests/sanity_test.cpp：随机数据自测 3 条 pipeline 的 build/search 是否可用；对比 Flat 精确结果计算 Recall@K（在测试里可用暴力算 gt）

========================
必须实现的三条 Pipeline
========================
Pipeline 1: HNSW(raw) 一阶段
- 用 HNSWIndex 直接返回 topK（raw 距离）
- 参数：M, efConstruction, efSearch

Pipeline 2: HNSW 召回 + SQ 近似打分 + Raw 精排（二阶段）
- Index：HNSWIndex（召回候选 candidateK = rerankK）
- Scorer：SQScorer（对候选快速打分，可选不必严格准确）
- Reranker：RawReranker（对候选 top rerankK 做精确距离重排，返回最终 topK）
- 参数：M, efConstruction, efSearch, rerankK（例如 50/100/200/400）

Pipeline 3: LSH/随机投影分桶 + SQ 打分 + Raw 精排（二阶段）
- Index：LSHIndex（多表随机超平面哈希；multi-probe 扩桶）
- Scorer：SQScorer
- Reranker：RawReranker
- 参数：numTables, numBits, probes, rerankK

========================
统一接口（必须严格实现）
========================
数据结构：
struct SearchResult { int id; float score; };

enum class Metric { L2, IP, COS };

struct SearchParams {
  int k = 10;
  int threads = 1;
  Metric metric = Metric::L2;

  // HNSW
  int M = 16;
  int efConstruction = 200;
  int efSearch = 64;

  // Two-stage
  int rerankK = 200;

  // LSH
  int numTables = 4;
  int numBits = 14;
  int probes = 2;

  // preprocessing
  bool normalize = false; // if metric==COS, should enable normalize automatically
};

组件接口：
struct IPreprocess {
  virtual void fit(const std::vector<std::vector<float>>& base) = 0; // 允许空实现
  virtual void transform_inplace(std::vector<float>& v) const = 0;   // normalize 等
  virtual ~IPreprocess() = default;
};

struct IIndex {
  virtual void build(const std::vector<std::vector<float>>& base, const SearchParams& p) = 0;
  virtual void search_candidates(const std::vector<float>& q,
                                 int candidateK,
                                 std::vector<int>& out_ids,
                                 const SearchParams& p) const = 0;
  virtual ~IIndex() = default;
};

struct IScorer {
  virtual void score_candidates(const std::vector<float>& q,
                                const std::vector<int>& ids,
                                std::vector<float>& out_scores,
                                const SearchParams& p) const = 0;
  virtual ~IScorer() = default;
};

struct IReranker {
  virtual void rerank_topk(const std::vector<int>& ids,
                           const std::vector<float>& scores,
                           int k,
                           std::vector<SearchResult>& out) const = 0;
  virtual ~IReranker() = default;
};

Pipeline 接口：
class Pipeline {
public:
  virtual void build(const std::vector<std::vector<float>>& base, const SearchParams& p) = 0;
  virtual std::vector<SearchResult> search(const std::vector<float>& query, const SearchParams& p) const = 0;
  virtual std::vector<std::vector<SearchResult>> batch_search(const std::vector<std::vector<float>>& queries,
                                                              const SearchParams& p) const = 0;
  virtual ~Pipeline() = default;
};

提供工厂：
enum class PipelineType { HNSW_RAW, HNSW_SQ_RERANK, LSH_SQ_RERANK };
std::unique_ptr<Pipeline> make_pipeline(PipelineType t);

========================
实现细节要求（关键）
========================
1) Metric:
- L2: 越小越近（score 用负距离或直接距离，需文档说明）
- IP: 越大越近
- COS: normalize + IP（pipeline 内部自动 normalize）

2) TopK:
- 实现 SmallKTopK（固定数组插入，适用 k<=64）
- 实现 HeapTopK（适用更大 k 或候选很多）
- 提供 select_topk(ids, scores, k) 工具函数

3) SQ:
- 不允许训练，但允许 build 统计
- SQCompressor:
  - build_stats(base): per-dim min/max 或 mean/std（二选一即可，推荐 min/max）
  - encode_to_int8(x): 输出 int8 或 uint8
  - fast_score(q, code): 近似打分（IP/L2 至少支持一种；可用解量化近似）
- 注意数值稳定、处理 min==max

4) HNSW:
- 实现一个“可用的简化 HNSW”
- 支持多层（level）和 efSearch
- 构建与搜索都尽量清晰可调参
- 允许先做 L2/IP 两种，COS 通过 normalize 实现

5) LSH:
- 多表随机超平面哈希（随机种子固定、可复现）
- 每表 numBits 个随机向量；hash = sign(dot(q, r_i))
- 桶结构：unordered_map<uint64_t, vector<int>>
- multi-probe：生成与原 hash 汉明距离 <= probes 的若干邻居 key（可简化为翻转前 probes 位或生成少量邻居即可；但必须可用）
- 候选去重：用 visited bitmap 或 unordered_set（按性能选）

6) Two-stage:
- 候选 = rerankK（或 >=rerankK）
- SQScorer 输出近似分数
- RawReranker 用 base 原向量精确距离重排取最终 topK

7) 并行：
- batch_search 按 query 并行（OpenMP 如果可用）
- 无 OpenMP：用 threads 均分 queries

========================
文件结构（必须输出完整文件内容）
========================
repo/
  CMakeLists.txt
  README.md
  include/ann/
    common.hpp            // types, SearchParams, enums
    metric.hpp
    preprocess.hpp        // NormalizePreprocess
    topk.hpp
    sq.hpp
    index_hnsw.hpp
    index_lsh.hpp
    scorer_sq.hpp
    reranker_raw.hpp
    pipeline.hpp
    pipeline_factory.hpp
  src/
    (可选：若你愿意放 cpp 实现；否则全部 header-only 也可以)
  tests/
    sanity_test.cpp

README 需要说明：
- 三条 pipeline 的用途与何时使用
- 如何调参（M/efSearch/rerankK，tables/bits/probes）
- score 的排序方向（L2 是越小越好还是用负号转成越大越好）

========================
输出要求
========================
请直接输出所有文件的完整内容，按文件路径分块展示，例如：
--- CMakeLists.txt ---
...内容...
--- include/ann/common.hpp ---
...内容...
不要省略，不要用占位符。生成的代码必须能在 Linux + g++(C++17) 下编译通过并运行 tests/sanity_test.cpp。

现在开始生成。
