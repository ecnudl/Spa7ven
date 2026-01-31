#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include "common.hpp"

namespace ann {

inline float l2_distance_squared(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float d0 = a[i] - b[i];
        float d1 = a[i+1] - b[i+1];
        float d2 = a[i+2] - b[i+2];
        float d3 = a[i+3] - b[i+3];
        sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline float inner_product(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3];
    }
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float compute_distance(const float* a, const float* b, size_t dim, Metric metric) {
    switch (metric) {
        case Metric::L2:
            return l2_distance_squared(a, b, dim);
        case Metric::IP:
            return -inner_product(a, b, dim);
        case Metric::COS:
            return -inner_product(a, b, dim);
        default:
            return l2_distance_squared(a, b, dim);
    }
}

inline float compute_distance(const std::vector<float>& a, const std::vector<float>& b, Metric metric) {
    return compute_distance(a.data(), b.data(), a.size(), metric);
}

inline bool is_score_better(float a, float b, Metric metric) {
    (void)metric;
    return a < b;
}

inline float worst_score(Metric metric) {
    (void)metric;
    return std::numeric_limits<float>::max();
}

} // namespace ann
