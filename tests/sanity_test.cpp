#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <cassert>

#include "ann/common.hpp"
#include "ann/metric.hpp"
#include "ann/pipeline_factory.hpp"

using namespace ann;

std::vector<std::vector<float>> generate_random_data(int n, int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> data(n);
    for (int i = 0; i < n; ++i) {
        data[i].resize(dim);
        for (int d = 0; d < dim; ++d) {
            data[i][d] = dist(rng);
        }
    }
    return data;
}

std::vector<SearchResult> brute_force_search(const std::vector<std::vector<float>>& base,
                                              const std::vector<float>& query,
                                              int k,
                                              Metric metric) {
    std::vector<SearchResult> results;
    results.reserve(base.size());

    for (size_t i = 0; i < base.size(); ++i) {
        float dist = compute_distance(query, base[i], metric);
        results.push_back({static_cast<int>(i), dist});
    }

    std::partial_sort(results.begin(), results.begin() + std::min(k, static_cast<int>(results.size())), results.end());
    results.resize(std::min(k, static_cast<int>(results.size())));
    return results;
}

float compute_recall(const std::vector<SearchResult>& pred,
                     const std::vector<SearchResult>& gt,
                     int k) {
    std::unordered_set<int> gt_ids;
    for (int i = 0; i < std::min(k, static_cast<int>(gt.size())); ++i) {
        gt_ids.insert(gt[i].id);
    }

    int hits = 0;
    for (int i = 0; i < std::min(k, static_cast<int>(pred.size())); ++i) {
        if (gt_ids.count(pred[i].id)) {
            ++hits;
        }
    }

    return static_cast<float>(hits) / static_cast<float>(std::min(k, static_cast<int>(gt.size())));
}

void test_pipeline(PipelineType type, const std::string& name,
                   const std::vector<std::vector<float>>& base,
                   const std::vector<std::vector<float>>& queries,
                   Metric metric, int k) {
    std::cout << "\n=== Testing " << name << " ===" << std::endl;

    auto pipeline = make_pipeline(type);

    SearchParams params;
    params.k = k;
    params.metric = metric;
    params.threads = 1;

    params.M = 16;
    params.efConstruction = 100;
    params.efSearch = 64;
    params.rerankK = 100;

    params.numTables = 16;
    params.numBits = 8;
    params.probes = 4;

    auto build_start = std::chrono::high_resolution_clock::now();
    pipeline->build(base, params);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "Build time: " << build_ms << " ms" << std::endl;

    float total_recall = 0.0f;
    auto search_start = std::chrono::high_resolution_clock::now();

    for (const auto& query : queries) {
        auto results = pipeline->search(query, params);
        auto gt = brute_force_search(base, query, k, metric);
        float recall = compute_recall(results, gt, k);
        total_recall += recall;
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    auto search_us = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count();

    float avg_recall = total_recall / queries.size();
    float avg_latency = static_cast<float>(search_us) / queries.size();

    std::cout << "Avg Recall@" << k << ": " << avg_recall << std::endl;
    std::cout << "Avg search latency: " << avg_latency << " us" << std::endl;

    auto batch_start = std::chrono::high_resolution_clock::now();
    params.threads = 2;
    auto batch_results = pipeline->batch_search(queries, params);
    auto batch_end = std::chrono::high_resolution_clock::now();
    auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();

    std::cout << "Batch search (" << queries.size() << " queries, " << params.threads << " threads): " << batch_ms << " ms" << std::endl;

    assert(batch_results.size() == queries.size());
    std::cout << name << " PASSED" << std::endl;
}

void test_metric(Metric metric, const std::string& metric_name,
                 const std::vector<std::vector<float>>& base,
                 const std::vector<std::vector<float>>& queries,
                 int k) {
    std::cout << "\n\n========== Testing with " << metric_name << " metric ==========" << std::endl;

    test_pipeline(PipelineType::HNSW_RAW, "HNSW_RAW", base, queries, metric, k);
    test_pipeline(PipelineType::HNSW_SQ_RERANK, "HNSW_SQ_RERANK", base, queries, metric, k);
    test_pipeline(PipelineType::LSH_SQ_RERANK, "LSH_SQ_RERANK", base, queries, metric, k);
}

int main() {
    std::cout << "ANN Library Sanity Test" << std::endl;
    std::cout << "=======================" << std::endl;

    const int n_base = 5000;
    const int n_queries = 100;
    const int dim = 128;
    const int k = 10;

    std::cout << "\nGenerating random data..." << std::endl;
    std::cout << "Base vectors: " << n_base << " x " << dim << std::endl;
    std::cout << "Query vectors: " << n_queries << " x " << dim << std::endl;

    auto base = generate_random_data(n_base, dim, 42);
    auto queries = generate_random_data(n_queries, dim, 123);

    test_metric(Metric::L2, "L2", base, queries, k);
    test_metric(Metric::IP, "IP", base, queries, k);
    test_metric(Metric::COS, "COS", base, queries, k);

    std::cout << "\n\n========================================" << std::endl;
    std::cout << "All sanity tests PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
