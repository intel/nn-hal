//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
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

#include <ie_common.h>
#include <mkldnn_node.h>
#include <caseless.hpp>
#include <string>
#include <memory>

namespace MKLDNNPlugin {

class MKLDNNActivationNode : public MKLDNNNode {
public:
    MKLDNNActivationNode(Type type, const std::string &name);
    explicit MKLDNNActivationNode(InferenceEngine::CNNLayerPtr layer);
    virtual ~MKLDNNActivationNode() {}

    void createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) override;
    void createPrimitive() override;
    bool created() override;

    mkldnn::algorithm getAlgorithm() {
        if (!initialized)
            initValues();
        return algorithm;
    }

    float getAlpha() {
        if (!initialized)
            initValues();
        return alpha;
    }

    float getBeta() {
        if (!initialized)
            initValues();
        return beta;
    }

private:
    void initValues();
    static Register<MKLDNNActivationNode> reg;
    bool initialized = false;
    float alpha = 0.0f;
    float beta = 0.0f;
    static caseless_map<std::string,
            std::function<void(InferenceEngine::GenericLayer*, mkldnn::algorithm&, float&, float&)>> initializers;
    mkldnn::algorithm algorithm;
};

}  // namespace MKLDNNPlugin

