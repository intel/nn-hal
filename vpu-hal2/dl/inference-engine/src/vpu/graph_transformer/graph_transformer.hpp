//
// INTEL CONFIDENTIAL
// Copyright 2017-2018 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <ie_icnn_network.hpp>
#include <vpu_logger.h>

namespace VPU {

struct BlobMetaData {
    std::string name;
    std::string exec_type;
    std::string layer_type;
    InferenceEngine::InferenceEngineProfileInfo::LayerStatus status;
};

struct BlobConfig {
    uint16_t firstShave, lastShave;
    bool memoryOptimization, hwOptimization, useCmxBuffers;
    bool copyOptimization, reshapeOptimization;
    uint32_t cmxBufferStart, cmxBufferSize;
    float inputScale, inputBias;
    std::vector<std::string> NoneLayers;
    std::vector<std::string> hwWhiteList;
    std::vector<std::string> hwBlackList;
    bool ignoreUnknownLayers;
};

class IGraphTransformer {
public:
    virtual ~IGraphTransformer() = default;

    virtual void generate(InferenceEngine::ICNNNetwork& network,
                          std::vector<char>& blob,
                          std::vector<BlobMetaData>& metadata,
                          size_t& numStages) = 0;
};

std::shared_ptr<IGraphTransformer> createGraphTransformer(const BlobConfig& blobConfig,
                                                          const Common::LoggerPtr& log);

}  // namespace VPU
