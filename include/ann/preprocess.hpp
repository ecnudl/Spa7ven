#pragma once

#include <vector>
#include <cmath>
#include "common.hpp"

namespace ann {

class NormalizePreprocess : public IPreprocess {
public:
    void fit(const std::vector<std::vector<float>>& /*base*/) override {
        // No fitting needed for normalization
    }

    void transform_inplace(std::vector<float>& v) const override {
        float norm = 0.0f;
        for (float x : v) {
            norm += x * x;
        }
        if (norm > 1e-12f) {
            norm = 1.0f / std::sqrt(norm);
            for (float& x : v) {
                x *= norm;
            }
        }
    }

    static void normalize(std::vector<float>& v) {
        float norm = 0.0f;
        for (float x : v) {
            norm += x * x;
        }
        if (norm > 1e-12f) {
            norm = 1.0f / std::sqrt(norm);
            for (float& x : v) {
                x *= norm;
            }
        }
    }

    static std::vector<float> normalized_copy(const std::vector<float>& v) {
        std::vector<float> result = v;
        normalize(result);
        return result;
    }
};

class IdentityPreprocess : public IPreprocess {
public:
    void fit(const std::vector<std::vector<float>>& /*base*/) override {}
    void transform_inplace(std::vector<float>& /*v*/) const override {}
};

} // namespace ann
