#pragma once

#include <vector>
#include <algorithm>
#include <queue>
#include <cstddef>
#include "common.hpp"

namespace ann {

template<int MaxK = 64>
class SmallKTopK {
public:
    explicit SmallKTopK(int k) : k_(k), size_(0) {
        static_assert(MaxK <= 256, "MaxK too large for SmallKTopK");
    }

    void push(int id, float score) {
        if (size_ < k_) {
            int pos = size_;
            while (pos > 0 && scores_[pos - 1] < score) {
                ids_[pos] = ids_[pos - 1];
                scores_[pos] = scores_[pos - 1];
                --pos;
            }
            ids_[pos] = id;
            scores_[pos] = score;
            ++size_;
        } else if (score < scores_[0]) {
            int pos = 0;
            while (pos + 1 < k_ && scores_[pos + 1] > score) {
                ids_[pos] = ids_[pos + 1];
                scores_[pos] = scores_[pos + 1];
                ++pos;
            }
            ids_[pos] = id;
            scores_[pos] = score;
        }
    }

    float threshold() const {
        return (size_ < k_) ? std::numeric_limits<float>::max() : scores_[0];
    }

    void finalize(std::vector<SearchResult>& out) const {
        out.resize(size_);
        for (int i = 0; i < size_; ++i) {
            out[size_ - 1 - i] = {ids_[i], scores_[i]};
        }
    }

    int size() const { return size_; }

private:
    int k_;
    int size_;
    int ids_[MaxK];
    float scores_[MaxK];
};

class HeapTopK {
public:
    explicit HeapTopK(int k) : k_(k) {}

    void push(int id, float score) {
        if (static_cast<int>(heap_.size()) < k_) {
            heap_.push_back({id, score});
            std::push_heap(heap_.begin(), heap_.end(), cmp);
        } else if (score < heap_[0].score) {
            std::pop_heap(heap_.begin(), heap_.end(), cmp);
            heap_.back() = {id, score};
            std::push_heap(heap_.begin(), heap_.end(), cmp);
        }
    }

    float threshold() const {
        return (static_cast<int>(heap_.size()) < k_)
            ? std::numeric_limits<float>::max()
            : heap_[0].score;
    }

    void finalize(std::vector<SearchResult>& out) {
        std::sort_heap(heap_.begin(), heap_.end(), cmp);
        out = std::move(heap_);
    }

    int size() const { return static_cast<int>(heap_.size()); }

private:
    static bool cmp(const SearchResult& a, const SearchResult& b) {
        return a.score < b.score;
    }

    int k_;
    std::vector<SearchResult> heap_;
};

inline void select_topk(const std::vector<int>& ids,
                        const std::vector<float>& scores,
                        int k,
                        std::vector<SearchResult>& out) {
    int n = static_cast<int>(ids.size());
    k = std::min(k, n);

    if (k <= 0) {
        out.clear();
        return;
    }

    if (k <= 64) {
        SmallKTopK<64> topk(k);
        for (int i = 0; i < n; ++i) {
            topk.push(ids[i], scores[i]);
        }
        topk.finalize(out);
    } else {
        HeapTopK topk(k);
        for (int i = 0; i < n; ++i) {
            topk.push(ids[i], scores[i]);
        }
        topk.finalize(out);
    }
}

inline void select_topk_from_results(std::vector<SearchResult>& results, int k) {
    if (static_cast<int>(results.size()) <= k) {
        std::sort(results.begin(), results.end());
        return;
    }

    std::partial_sort(results.begin(), results.begin() + k, results.end());
    results.resize(k);
}

} // namespace ann
