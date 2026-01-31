#pragma once

#include <memory>
#include "common.hpp"
#include "pipeline.hpp"

namespace ann {

inline std::unique_ptr<Pipeline> make_pipeline(PipelineType t) {
    switch (t) {
        case PipelineType::HNSW_RAW:
            return std::make_unique<HNSWRawPipeline>();
        case PipelineType::HNSW_SQ_RERANK:
            return std::make_unique<HNSWSQRerankPipeline>();
        case PipelineType::LSH_SQ_RERANK:
            return std::make_unique<LSHSQRerankPipeline>();
        default:
            return std::make_unique<HNSWRawPipeline>();
    }
}

} // namespace ann
