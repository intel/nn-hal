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
#include "mkldnn_clamp_node.h"
#include "ie_layers.h"
#include <string>
#include <cmath>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNClampNode::MKLDNNClampNode(Type type, const std::string &name) : MKLDNNNode(type, name),
                                                                       min_value(0.0f), max_value(1.0f) {}
MKLDNNClampNode::MKLDNNClampNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer),
                                                                       min_value(0.0f), max_value(1.0f) {}

void MKLDNNClampNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    ClampLayer* clampLayer = dynamic_cast<ClampLayer*>(getCnnLayer().get());

    if (clampLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert clamp layer.";
    min_value = clampLayer->min_value;
    max_value = clampLayer->max_value;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
}

void MKLDNNClampNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    supportedPrimitiveDescriptors.push_back({engine,
                                             {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::format::any}},
                                             {{getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::any}},
                                             impl_desc_type::unknown});
}

void MKLDNNClampNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory wasn't allocated.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory wasn't allocated.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor wasn't set.";
}

void MKLDNNClampNode::execute(mkldnn::stream strm) {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();
    const size_t data_size = srcMemory.GetSize() / sizeof(float);

    const float *src_ptr = reinterpret_cast<const float*>(srcMemory.GetData()) +
            srcMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_ptr = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    #pragma omp parallel for
    for (int i = 0; i < data_size; i++) {
        float src_data = src_ptr[i];
        if (src_data > max_value)
            src_data = max_value;
        if (src_data < min_value)
            src_data = min_value;

        dst_ptr[i] = src_data;
    }
}

bool MKLDNNClampNode::created() {
    return getType() == Clamp;
}
