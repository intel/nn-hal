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
#include "mkldnn_power_node.h"
#include "ie_layers.h"
#include <string>
#include <cmath>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPowerNode::MKLDNNPowerNode(Type type, const std::string &name) : MKLDNNNode(type, name),
                                                                       scale(1.0f), shift(1.0f), power(1.0f) {}
MKLDNNPowerNode::MKLDNNPowerNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer),
                                                                       scale(1.0f), shift(1.0f), power(1.0f) {}

void MKLDNNPowerNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    auto * powerLayer = dynamic_cast<PowerLayer*>(getCnnLayer().get());

    if (powerLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert power layer.";
    scale = powerLayer->scale;
    power = powerLayer->power;
    shift = powerLayer->offset;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
}

void MKLDNNPowerNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{autoBlockingDims(getParentEdgeAt(0)->getDims(), format), getInputDataType(), format}},
                                                 {{autoBlockingDims(getChildEdgeAt(0)->getDims(), format), getOutputDataType(), format}},
                                                 impl_desc_type::unknown});
    }
}

void MKLDNNPowerNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNPowerNode::execute(mkldnn::stream strm) {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();
    auto& dstMemory = getChildEdgeAt(0)->getMemory();
    const size_t data_size = srcMemory.GetSize() / sizeof(float);

    const auto *src_ptr = reinterpret_cast<const float*>(srcMemory.GetData()) +
            srcMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_ptr = reinterpret_cast<float*>(dstMemory.GetData()) +
            dstMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;

    if (power == 1.0f) {
        #pragma omp parallel for
        for (int i = 0; i < data_size; i++)
            dst_ptr[i] = src_ptr[i] * scale + shift;
    } else {
        #pragma omp parallel for
        for (int i = 0; i < data_size; i++)
            dst_ptr[i] = static_cast<float>(pow(src_ptr[i] * scale + shift, power));
    }
}

bool MKLDNNPowerNode::created() {
    return getType() == Power;
}
