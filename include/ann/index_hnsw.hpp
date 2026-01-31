#pragma once

#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <limits>
#include "common.hpp"
#include "metric.hpp"
#include "topk.hpp"

namespace ann {

class HNSWIndex : public IIndex {
public:
    HNSWIndex() = default;

    void build(const std::vector<std::vector<float>>& base, const SearchParams& p) override {
        if (base.empty()) return;

        base_ = &base;
        n_ = static_cast<int>(base.size());
        dim_ = static_cast<int>(base[0].size());
        M_ = p.M;
        M_max0_ = M_ * 2;
        ef_construction_ = p.efConstruction;
        metric_ = p.metric;

        ml_ = 1.0 / std::log(static_cast<double>(M_));

        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        levels_.resize(n_);
        neighbors_.resize(n_);

        int max_level = 0;
        for (int i = 0; i < n_; ++i) {
            int level = static_cast<int>(-std::log(dist(rng)) * ml_);
            levels_[i] = level;
            max_level = std::max(max_level, level);
            neighbors_[i].resize(level + 1);
        }

        entry_point_ = 0;
        max_level_ = levels_[0];

        for (int i = 1; i < n_; ++i) {
            insert(i);
            if (levels_[i] > max_level_) {
                max_level_ = levels_[i];
                entry_point_ = i;
            }
        }
    }

    void search_candidates(const std::vector<float>& q,
                           int candidateK,
                           std::vector<int>& out_ids,
                           const SearchParams& p) const override {
        out_ids.clear();
        if (n_ == 0) return;

        int ef = std::max(candidateK, p.efSearch);
        auto results = search_layer0(q, ef);

        out_ids.reserve(std::min(candidateK, static_cast<int>(results.size())));
        for (int i = 0; i < std::min(candidateK, static_cast<int>(results.size())); ++i) {
            out_ids.push_back(results[i].id);
        }
    }

    std::vector<SearchResult> search_topk(const std::vector<float>& q, int k, int ef) const {
        if (n_ == 0) return {};

        ef = std::max(k, ef);
        auto results = search_layer0(q, ef);

        if (static_cast<int>(results.size()) > k) {
            results.resize(k);
        }
        return results;
    }

    const std::vector<std::vector<float>>* base_ptr() const { return base_; }
    Metric metric() const { return metric_; }

private:
    struct Candidate {
        int id;
        float dist;
        bool operator<(const Candidate& o) const { return dist < o.dist; }
        bool operator>(const Candidate& o) const { return dist > o.dist; }
    };

    float distance(const std::vector<float>& a, const std::vector<float>& b) const {
        return compute_distance(a, b, metric_);
    }

    float distance(const std::vector<float>& a, int id) const {
        return compute_distance(a, (*base_)[id], metric_);
    }

    void insert(int id) {
        const auto& vec = (*base_)[id];
        int level = levels_[id];

        int ep = entry_point_;
        float ep_dist = distance(vec, ep);

        for (int lc = max_level_; lc > level; --lc) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int neighbor : neighbors_[ep][lc]) {
                    float d = distance(vec, neighbor);
                    if (d < ep_dist) {
                        ep = neighbor;
                        ep_dist = d;
                        changed = true;
                    }
                }
            }
        }

        for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
            auto candidates = search_layer(vec, ep, ef_construction_, lc);

            int M_cur = (lc == 0) ? M_max0_ : M_;
            auto selected = select_neighbors(vec, candidates, M_cur);

            neighbors_[id][lc] = selected;

            for (int neighbor : selected) {
                neighbors_[neighbor][lc].push_back(id);
                if (static_cast<int>(neighbors_[neighbor][lc].size()) > M_cur) {
                    prune_neighbors(neighbor, lc, M_cur);
                }
            }

            if (!candidates.empty()) {
                ep = candidates[0].id;
            }
        }
    }

    std::vector<Candidate> search_layer(const std::vector<float>& q, int ep, int ef, int level) const {
        std::unordered_set<int> visited;
        visited.insert(ep);

        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
        std::priority_queue<Candidate> results;

        float d = distance(q, ep);
        candidates.push({ep, d});
        results.push({ep, d});

        while (!candidates.empty()) {
            auto curr = candidates.top();
            candidates.pop();

            if (curr.dist > results.top().dist && static_cast<int>(results.size()) >= ef) {
                break;
            }

            for (int neighbor : neighbors_[curr.id][level]) {
                if (visited.count(neighbor)) continue;
                visited.insert(neighbor);

                float nd = distance(q, neighbor);

                if (static_cast<int>(results.size()) < ef || nd < results.top().dist) {
                    candidates.push({neighbor, nd});
                    results.push({neighbor, nd});
                    if (static_cast<int>(results.size()) > ef) {
                        results.pop();
                    }
                }
            }
        }

        std::vector<Candidate> result_vec;
        result_vec.reserve(results.size());
        while (!results.empty()) {
            result_vec.push_back(results.top());
            results.pop();
        }
        std::reverse(result_vec.begin(), result_vec.end());
        return result_vec;
    }

    std::vector<SearchResult> search_layer0(const std::vector<float>& q, int ef) const {
        int ep = entry_point_;
        float ep_dist = distance(q, ep);

        for (int lc = max_level_; lc > 0; --lc) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int neighbor : neighbors_[ep][lc]) {
                    float d = distance(q, neighbor);
                    if (d < ep_dist) {
                        ep = neighbor;
                        ep_dist = d;
                        changed = true;
                    }
                }
            }
        }

        auto candidates = search_layer(q, ep, ef, 0);

        std::vector<SearchResult> results;
        results.reserve(candidates.size());
        for (const auto& c : candidates) {
            results.push_back({c.id, c.dist});
        }
        return results;
    }

    std::vector<int> select_neighbors(const std::vector<float>& /*q*/,
                                      const std::vector<Candidate>& candidates,
                                      int M) const {
        std::vector<int> result;
        result.reserve(M);

        for (const auto& c : candidates) {
            if (static_cast<int>(result.size()) >= M) break;
            result.push_back(c.id);
        }
        return result;
    }

    void prune_neighbors(int id, int level, int M) {
        auto& nbrs = neighbors_[id][level];
        if (static_cast<int>(nbrs.size()) <= M) return;

        const auto& vec = (*base_)[id];
        std::vector<Candidate> candidates;
        candidates.reserve(nbrs.size());
        for (int n : nbrs) {
            candidates.push_back({n, distance(vec, n)});
        }
        std::sort(candidates.begin(), candidates.end());

        nbrs.clear();
        nbrs.reserve(M);
        for (int i = 0; i < M && i < static_cast<int>(candidates.size()); ++i) {
            nbrs.push_back(candidates[i].id);
        }
    }

    const std::vector<std::vector<float>>* base_ = nullptr;
    int n_ = 0;
    int dim_ = 0;
    int M_ = 16;
    int M_max0_ = 32;
    int ef_construction_ = 200;
    double ml_ = 0.0;
    int entry_point_ = 0;
    int max_level_ = 0;
    Metric metric_ = Metric::L2;

    std::vector<int> levels_;
    std::vector<std::vector<std::vector<int>>> neighbors_;
};

} // namespace ann
