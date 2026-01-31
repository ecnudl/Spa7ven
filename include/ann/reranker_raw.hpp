#pragma once

#include <vector>
#include <algorithm>
#include "common.hpp"
#include "metric.hpp"
#include "topk.hpp"

namespace ann {

class RawReranker : public IReranker {
public:
    RawReranker() = default;

    void set_base(const std::vector<std::vector<float>>* base, Metric metric) {
        base_ = base;
        metric_ = metric;
    }

    void rerank_topk(const std::vector<int>& ids,
                     const std::vector<float>& /*scores*/,
                     int k,
                     std::vector<SearchResult>& out) const override {
        out.clear();
        if (ids.empty() || base_ == nullptr || query_ == nullptr) return;

        std::vector<SearchResult> results;
        results.reserve(ids.size());

        for (int id : ids) {
            float dist = compute_distance(*query_, (*base_)[id], metric_);
            results.push_back({id, dist});
        }

        select_topk_from_results(results, k);
        out = std::move(results);
    }

    void rerank_with_query(const std::vector<float>& query,
                           const std::vector<int>& ids,
                           int k,
                           std::vector<SearchResult>& out) const {
        out.clear();
        if (ids.empty() || base_ == nullptr) return;

        std::vector<SearchResult> results;
        results.reserve(ids.size());

        for (int id : ids) {
            float dist = compute_distance(query, (*base_)[id], metric_);
            results.push_back({id, dist});
        }

        select_topk_from_results(results, k);
        out = std::move(results);
    }

    void set_query(const std::vector<float>* query) {
        query_ = query;
    }

private:
    const std::vector<std::vector<float>>* base_ = nullptr;
    const std::vector<float>* query_ = nullptr;
    Metric metric_ = Metric::L2;
};

} // namespace ann
