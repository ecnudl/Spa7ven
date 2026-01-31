#pragma once

#include <vector>
#include <memory>
#include <cstdint>

namespace ann {

struct SearchResult {
    int id;
    float score;

    bool operator<(const SearchResult& o) const { return score < o.score; }
    bool operator>(const SearchResult& o) const { return score > o.score; }
};

enum class Metric { L2, IP, COS };

enum class PipelineType { HNSW_RAW, HNSW_SQ_RERANK, LSH_SQ_RERANK };

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
    bool normalize = false;
};

struct IPreprocess {
    virtual void fit(const std::vector<std::vector<float>>& base) = 0;
    virtual void transform_inplace(std::vector<float>& v) const = 0;
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

class Pipeline {
public:
    virtual void build(const std::vector<std::vector<float>>& base, const SearchParams& p) = 0;
    virtual std::vector<SearchResult> search(const std::vector<float>& query, const SearchParams& p) const = 0;
    virtual std::vector<std::vector<SearchResult>> batch_search(const std::vector<std::vector<float>>& queries,
                                                                const SearchParams& p) const = 0;
    virtual ~Pipeline() = default;
};

} // namespace ann
