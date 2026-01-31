#pragma once

#include <vector>
#include "common.hpp"
#include "sq.hpp"

namespace ann {

class SQScorer : public IScorer {
public:
    SQScorer() = default;

    void set_compressor(const SQCompressor* compressor) {
        compressor_ = compressor;
    }

    void score_candidates(const std::vector<float>& q,
                          const std::vector<int>& ids,
                          std::vector<float>& out_scores,
                          const SearchParams& p) const override {
        out_scores.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            out_scores[i] = compressor_->fast_score(q, ids[i], p.metric);
        }
    }

private:
    const SQCompressor* compressor_ = nullptr;
};

} // namespace ann
