#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "common.hpp"
#include "metric.hpp"

namespace ann {

class LSHIndex : public IIndex {
public:
    LSHIndex() = default;

    void build(const std::vector<std::vector<float>>& base, const SearchParams& p) override {
        if (base.empty()) return;

        base_ = &base;
        n_ = static_cast<int>(base.size());
        dim_ = static_cast<int>(base[0].size());
        num_tables_ = p.numTables;
        num_bits_ = p.numBits;
        metric_ = p.metric;

        std::mt19937 rng(12345);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        hyperplanes_.resize(num_tables_);
        tables_.resize(num_tables_);

        for (int t = 0; t < num_tables_; ++t) {
            hyperplanes_[t].resize(num_bits_);
            for (int b = 0; b < num_bits_; ++b) {
                hyperplanes_[t][b].resize(dim_);
                for (int d = 0; d < dim_; ++d) {
                    hyperplanes_[t][b][d] = normal(rng);
                }
                float norm = 0.0f;
                for (int d = 0; d < dim_; ++d) {
                    norm += hyperplanes_[t][b][d] * hyperplanes_[t][b][d];
                }
                norm = std::sqrt(norm);
                if (norm > 1e-9f) {
                    for (int d = 0; d < dim_; ++d) {
                        hyperplanes_[t][b][d] /= norm;
                    }
                }
            }
        }

        for (int i = 0; i < n_; ++i) {
            for (int t = 0; t < num_tables_; ++t) {
                uint64_t hash = compute_hash(base[i], t);
                tables_[t][hash].push_back(i);
            }
        }
    }

    void search_candidates(const std::vector<float>& q,
                           int candidateK,
                           std::vector<int>& out_ids,
                           const SearchParams& p) const override {
        out_ids.clear();
        if (n_ == 0) return;

        std::unordered_set<int> visited;
        int probes = p.probes;

        for (int t = 0; t < num_tables_; ++t) {
            uint64_t base_hash = compute_hash(q, t);

            auto probe_hashes = generate_probe_hashes(base_hash, probes);

            for (uint64_t h : probe_hashes) {
                auto it = tables_[t].find(h);
                if (it != tables_[t].end()) {
                    for (int id : it->second) {
                        visited.insert(id);
                    }
                }
            }
        }

        out_ids.reserve(visited.size());
        for (int id : visited) {
            out_ids.push_back(id);
        }

        if (static_cast<int>(out_ids.size()) > candidateK * 2) {
            std::vector<std::pair<float, int>> scored;
            scored.reserve(out_ids.size());
            for (int id : out_ids) {
                float dist = compute_distance(q, (*base_)[id], metric_);
                scored.push_back({dist, id});
            }
            std::partial_sort(scored.begin(),
                              scored.begin() + std::min(candidateK, static_cast<int>(scored.size())),
                              scored.end());
            out_ids.clear();
            for (int i = 0; i < std::min(candidateK, static_cast<int>(scored.size())); ++i) {
                out_ids.push_back(scored[i].second);
            }
        }
    }

    const std::vector<std::vector<float>>* base_ptr() const { return base_; }
    Metric metric() const { return metric_; }

private:
    uint64_t compute_hash(const std::vector<float>& vec, int table) const {
        uint64_t hash = 0;
        for (int b = 0; b < num_bits_; ++b) {
            float dot = 0.0f;
            for (int d = 0; d < dim_; ++d) {
                dot += vec[d] * hyperplanes_[table][b][d];
            }
            if (dot >= 0.0f) {
                hash |= (1ULL << b);
            }
        }
        return hash;
    }

    std::vector<uint64_t> generate_probe_hashes(uint64_t base_hash, int probes) const {
        std::vector<uint64_t> result;
        result.push_back(base_hash);

        if (probes <= 0) return result;

        for (int b = 0; b < num_bits_ && static_cast<int>(result.size()) < (1 + probes * num_bits_); ++b) {
            uint64_t flipped = base_hash ^ (1ULL << b);
            result.push_back(flipped);
        }

        if (probes >= 2) {
            int max_pairs = std::min(probes * 2, num_bits_ * (num_bits_ - 1) / 2);
            int count = 0;
            for (int b1 = 0; b1 < num_bits_ && count < max_pairs; ++b1) {
                for (int b2 = b1 + 1; b2 < num_bits_ && count < max_pairs; ++b2) {
                    uint64_t flipped = base_hash ^ (1ULL << b1) ^ (1ULL << b2);
                    result.push_back(flipped);
                    ++count;
                }
            }
        }

        return result;
    }

    const std::vector<std::vector<float>>* base_ = nullptr;
    int n_ = 0;
    int dim_ = 0;
    int num_tables_ = 4;
    int num_bits_ = 14;
    Metric metric_ = Metric::L2;

    std::vector<std::vector<std::vector<float>>> hyperplanes_;
    std::vector<std::unordered_map<uint64_t, std::vector<int>>> tables_;
};

} // namespace ann
