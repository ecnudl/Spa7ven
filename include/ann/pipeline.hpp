#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <algorithm>
#include "common.hpp"
#include "metric.hpp"
#include "preprocess.hpp"
#include "topk.hpp"
#include "sq.hpp"
#include "index_hnsw.hpp"
#include "index_lsh.hpp"
#include "scorer_sq.hpp"
#include "reranker_raw.hpp"

#ifdef ANN_USE_OPENMP
#include <omp.h>
#endif

namespace ann {

class HNSWRawPipeline : public Pipeline {
public:
    void build(const std::vector<std::vector<float>>& base, const SearchParams& p) override {
        metric_ = p.metric;
        normalize_ = (p.metric == Metric::COS) || p.normalize;

        if (normalize_) {
            normalized_base_.resize(base.size());
            for (size_t i = 0; i < base.size(); ++i) {
                normalized_base_[i] = base[i];
                NormalizePreprocess::normalize(normalized_base_[i]);
            }
            base_ptr_ = &normalized_base_;
        } else {
            base_ptr_ = &base;
            original_base_ = &base;
        }

        SearchParams build_params = p;
        if (normalize_) {
            build_params.metric = Metric::IP;
        }
        index_.build(*base_ptr_, build_params);
    }

    std::vector<SearchResult> search(const std::vector<float>& query, const SearchParams& p) const override {
        std::vector<float> q = query;
        if (normalize_) {
            NormalizePreprocess::normalize(q);
        }

        int ef = p.efSearch;
        return index_.search_topk(q, p.k, ef);
    }

    std::vector<std::vector<SearchResult>> batch_search(const std::vector<std::vector<float>>& queries,
                                                        const SearchParams& p) const override {
        std::vector<std::vector<SearchResult>> results(queries.size());

#ifdef ANN_USE_OPENMP
        #pragma omp parallel for num_threads(p.threads) schedule(dynamic)
        for (size_t i = 0; i < queries.size(); ++i) {
            results[i] = search(queries[i], p);
        }
#else
        if (p.threads <= 1) {
            for (size_t i = 0; i < queries.size(); ++i) {
                results[i] = search(queries[i], p);
            }
        } else {
            std::vector<std::thread> threads;
            size_t chunk = (queries.size() + p.threads - 1) / p.threads;

            for (int t = 0; t < p.threads; ++t) {
                size_t start = t * chunk;
                size_t end = std::min(start + chunk, queries.size());
                if (start >= end) break;

                threads.emplace_back([this, &queries, &results, &p, start, end]() {
                    for (size_t i = start; i < end; ++i) {
                        results[i] = search(queries[i], p);
                    }
                });
            }

            for (auto& th : threads) {
                th.join();
            }
        }
#endif
        return results;
    }

private:
    HNSWIndex index_;
    Metric metric_ = Metric::L2;
    bool normalize_ = false;
    std::vector<std::vector<float>> normalized_base_;
    const std::vector<std::vector<float>>* base_ptr_ = nullptr;
    const std::vector<std::vector<float>>* original_base_ = nullptr;
};

class HNSWSQRerankPipeline : public Pipeline {
public:
    void build(const std::vector<std::vector<float>>& base, const SearchParams& p) override {
        metric_ = p.metric;
        normalize_ = (p.metric == Metric::COS) || p.normalize;

        if (normalize_) {
            normalized_base_.resize(base.size());
            for (size_t i = 0; i < base.size(); ++i) {
                normalized_base_[i] = base[i];
                NormalizePreprocess::normalize(normalized_base_[i]);
            }
            base_ptr_ = &normalized_base_;
        } else {
            base_ptr_ = &base;
        }

        SearchParams build_params = p;
        Metric effective_metric = normalize_ ? Metric::IP : p.metric;
        build_params.metric = effective_metric;

        index_.build(*base_ptr_, build_params);

        compressor_.build_stats(*base_ptr_);
        scorer_.set_compressor(&compressor_);

        reranker_.set_base(base_ptr_, effective_metric);
    }

    std::vector<SearchResult> search(const std::vector<float>& query, const SearchParams& p) const override {
        std::vector<float> q = query;
        if (normalize_) {
            NormalizePreprocess::normalize(q);
        }

        Metric effective_metric = normalize_ ? Metric::IP : p.metric;
        SearchParams search_params = p;
        search_params.metric = effective_metric;

        std::vector<int> candidate_ids;
        index_.search_candidates(q, p.rerankK, candidate_ids, search_params);

        if (candidate_ids.empty()) {
            return {};
        }

        std::vector<float> sq_scores;
        scorer_.score_candidates(q, candidate_ids, sq_scores, search_params);

        std::vector<SearchResult> results;
        reranker_.rerank_with_query(q, candidate_ids, p.k, results);

        return results;
    }

    std::vector<std::vector<SearchResult>> batch_search(const std::vector<std::vector<float>>& queries,
                                                        const SearchParams& p) const override {
        std::vector<std::vector<SearchResult>> results(queries.size());

#ifdef ANN_USE_OPENMP
        #pragma omp parallel for num_threads(p.threads) schedule(dynamic)
        for (size_t i = 0; i < queries.size(); ++i) {
            results[i] = search(queries[i], p);
        }
#else
        if (p.threads <= 1) {
            for (size_t i = 0; i < queries.size(); ++i) {
                results[i] = search(queries[i], p);
            }
        } else {
            std::vector<std::thread> threads;
            size_t chunk = (queries.size() + p.threads - 1) / p.threads;

            for (int t = 0; t < p.threads; ++t) {
                size_t start = t * chunk;
                size_t end = std::min(start + chunk, queries.size());
                if (start >= end) break;

                threads.emplace_back([this, &queries, &results, &p, start, end]() {
                    for (size_t i = start; i < end; ++i) {
                        results[i] = search(queries[i], p);
                    }
                });
            }

            for (auto& th : threads) {
                th.join();
            }
        }
#endif
        return results;
    }

private:
    HNSWIndex index_;
    SQCompressor compressor_;
    SQScorer scorer_;
    RawReranker reranker_;
    Metric metric_ = Metric::L2;
    bool normalize_ = false;
    std::vector<std::vector<float>> normalized_base_;
    const std::vector<std::vector<float>>* base_ptr_ = nullptr;
};

class LSHSQRerankPipeline : public Pipeline {
public:
    void build(const std::vector<std::vector<float>>& base, const SearchParams& p) override {
        metric_ = p.metric;
        normalize_ = (p.metric == Metric::COS) || p.normalize;

        if (normalize_) {
            normalized_base_.resize(base.size());
            for (size_t i = 0; i < base.size(); ++i) {
                normalized_base_[i] = base[i];
                NormalizePreprocess::normalize(normalized_base_[i]);
            }
            base_ptr_ = &normalized_base_;
        } else {
            base_ptr_ = &base;
        }

        SearchParams build_params = p;
        Metric effective_metric = normalize_ ? Metric::IP : p.metric;
        build_params.metric = effective_metric;

        index_.build(*base_ptr_, build_params);

        compressor_.build_stats(*base_ptr_);
        scorer_.set_compressor(&compressor_);

        reranker_.set_base(base_ptr_, effective_metric);
    }

    std::vector<SearchResult> search(const std::vector<float>& query, const SearchParams& p) const override {
        std::vector<float> q = query;
        if (normalize_) {
            NormalizePreprocess::normalize(q);
        }

        Metric effective_metric = normalize_ ? Metric::IP : p.metric;
        SearchParams search_params = p;
        search_params.metric = effective_metric;

        std::vector<int> candidate_ids;
        index_.search_candidates(q, p.rerankK, candidate_ids, search_params);

        if (candidate_ids.empty()) {
            return {};
        }

        std::vector<float> sq_scores;
        scorer_.score_candidates(q, candidate_ids, sq_scores, search_params);

        std::vector<SearchResult> results;
        reranker_.rerank_with_query(q, candidate_ids, p.k, results);

        return results;
    }

    std::vector<std::vector<SearchResult>> batch_search(const std::vector<std::vector<float>>& queries,
                                                        const SearchParams& p) const override {
        std::vector<std::vector<SearchResult>> results(queries.size());

#ifdef ANN_USE_OPENMP
        #pragma omp parallel for num_threads(p.threads) schedule(dynamic)
        for (size_t i = 0; i < queries.size(); ++i) {
            results[i] = search(queries[i], p);
        }
#else
        if (p.threads <= 1) {
            for (size_t i = 0; i < queries.size(); ++i) {
                results[i] = search(queries[i], p);
            }
        } else {
            std::vector<std::thread> threads;
            size_t chunk = (queries.size() + p.threads - 1) / p.threads;

            for (int t = 0; t < p.threads; ++t) {
                size_t start = t * chunk;
                size_t end = std::min(start + chunk, queries.size());
                if (start >= end) break;

                threads.emplace_back([this, &queries, &results, &p, start, end]() {
                    for (size_t i = start; i < end; ++i) {
                        results[i] = search(queries[i], p);
                    }
                });
            }

            for (auto& th : threads) {
                th.join();
            }
        }
#endif
        return results;
    }

private:
    LSHIndex index_;
    SQCompressor compressor_;
    SQScorer scorer_;
    RawReranker reranker_;
    Metric metric_ = Metric::L2;
    bool normalize_ = false;
    std::vector<std::vector<float>> normalized_base_;
    const std::vector<std::vector<float>>* base_ptr_ = nullptr;
};

} // namespace ann
