#ifndef ANDROID_ML_NN_BUILDER_GNA_H
#define ANDROID_ML_NN_BUILDER_GNA_H

#pragma once

#include "ie_builders.hpp"
#include "ie_network.hpp"

namespace IEBuilder = InferenceEngine::Builder;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace IRBuilder {

class BuilderNetwork {
    public:
        InferenceEngine::Builder::Network* getBuilder() {
            return mBuilder;
        }

        BuilderNetwork(std::string network_name) {
            InferenceEngine::Context ctx;
            mBuilder = new IEBuilder::Network(ctx, "graph-builder");
        }

        int memory_layer_cnt = 0;

        std::vector<InferenceEngine::idx_t> mConnections;
        InferenceEngine::idx_t finalMemLayerId;
        std::vector<InferenceEngine::idx_t> mOutputIds;
    
    private:
        IEBuilder::Network* mBuilder;
};

}
}
}
}
}

#endif