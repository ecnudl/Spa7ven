#pragma once

#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>
#include <algorithm>
#include "common.hpp"
#include "metric.hpp"

namespace ann {

class SQCompressor {
public:
    SQCompressor() = default;

    void build_stats(const std::vector<std::vector<float>>& base) {
        if (base.empty()) return;

        dim_ = static_cast<int>(base[0].size());
        min_vals_.resize(dim_);
        max_vals_.resize(dim_);
        scale_.resize(dim_);
        inv_scale_.resize(dim_);

        for (int d = 0; d < dim_; ++d) {
            min_vals_[d] = std::numeric_limits<float>::max();
            max_vals_[d] = std::numeric_limits<float>::lowest();
        }

        for (const auto& vec : base) {
            for (int d = 0; d < dim_; ++d) {
                min_vals_[d] = std::min(min_vals_[d], vec[d]);
                max_vals_[d] = std::max(max_vals_[d], vec[d]);
            }
        }

        for (int d = 0; d < dim_; ++d) {
            float range = max_vals_[d] - min_vals_[d];
            if (range < 1e-9f) {
                range = 1.0f;
            }
            scale_[d] = 255.0f / range;
            inv_scale_[d] = range / 255.0f;
        }

        codes_.resize(base.size());
        for (size_t i = 0; i < base.size(); ++i) {
            encode_to_uint8(base[i], codes_[i]);
        }
    }

    void encode_to_uint8(const std::vector<float>& x, std::vector<uint8_t>& out) const {
        out.resize(dim_);
        for (int d = 0; d < dim_; ++d) {
            float val = (x[d] - min_vals_[d]) * scale_[d];
            val = std::max(0.0f, std::min(255.0f, val));
            out[d] = static_cast<uint8_t>(val + 0.5f);
        }
    }

    void decode_from_uint8(const std::vector<uint8_t>& code, std::vector<float>& out) const {
        out.resize(dim_);
        for (int d = 0; d < dim_; ++d) {
            out[d] = static_cast<float>(code[d]) * inv_scale_[d] + min_vals_[d];
        }
    }

    float fast_score_l2(const std::vector<float>& q, int id) const {
        const auto& code = codes_[id];
        float sum = 0.0f;
        for (int d = 0; d < dim_; ++d) {
            float decoded = static_cast<float>(code[d]) * inv_scale_[d] + min_vals_[d];
            float diff = q[d] - decoded;
            sum += diff * diff;
        }
        return sum;
    }

    float fast_score_ip(const std::vector<float>& q, int id) const {
        const auto& code = codes_[id];
        float sum = 0.0f;
        for (int d = 0; d < dim_; ++d) {
            float decoded = static_cast<float>(code[d]) * inv_scale_[d] + min_vals_[d];
            sum += q[d] * decoded;
        }
        return -sum;
    }

    float fast_score(const std::vector<float>& q, int id, Metric metric) const {
        switch (metric) {
            case Metric::L2:
                return fast_score_l2(q, id);
            case Metric::IP:
            case Metric::COS:
                return fast_score_ip(q, id);
            default:
                return fast_score_l2(q, id);
        }
    }

    int dim() const { return dim_; }
    size_t num_vectors() const { return codes_.size(); }
    const std::vector<uint8_t>& get_code(int id) const { return codes_[id]; }

private:
    int dim_ = 0;
    std::vector<float> min_vals_;
    std::vector<float> max_vals_;
    std::vector<float> scale_;
    std::vector<float> inv_scale_;
    std::vector<std::vector<uint8_t>> codes_;
};

} // namespace ann
