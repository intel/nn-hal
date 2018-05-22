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
#include "mkldnn_roi_pooling_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNROIPoolingNode::MKLDNNROIPoolingNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNROIPoolingNode::MKLDNNROIPoolingNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNROIPoolingNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (descs.size())
        return;
    GenericLayer* genericLayer = getCnnLayer().get();

    if (genericLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert ROIPooling layer.";

    if (!getParentEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    pooled_h = genericLayer->GetParamAsInt("pooled_h");
    pooled_w = genericLayer->GetParamAsInt("pooled_w");
    spatial_scale = genericLayer->GetParamAsFloat("spatial_scale");

    auto parentDims = getParentEdgeAt(0)->getDims();
    for (auto format : getAvailableFormatsForDims(parentDims)) {
        std::vector<memory::desc> srcs;
        srcs.push_back({autoBlockingDims(getParentEdgeAt(0)->getDims(), format), inputDataType, format});
        srcs.push_back({getParentEdgeAt(1)->getDims(), inputDataType, memory::nc});


        memory::desc out_candidate{autoBlockingDims(getChildEdgeAt(0)->getDims(), format), outputDataType, format};

        MKLDNNDescriptor desc(std::shared_ptr<roi_pooling_forward::desc>(
                new roi_pooling_forward::desc(prop_kind::forward_scoring, srcs, out_candidate, pooled_h, pooled_w,
                                              spatial_scale)));
        descs.push_back(desc);
    }
}

void MKLDNNROIPoolingNode::createPrimitive() {
    if (prim)
        return;

    std::vector<memory::desc> srcs;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        srcs.push_back(getParentEdgeAt(i)->getMemory().GetDescriptor());
    }

    memory::desc out_candidate = getChildEdgeAt(0)->getMemory().GetDescriptor();

    MKLDNNDescriptor desc(std::shared_ptr<roi_pooling_forward::desc>(
            new roi_pooling_forward::desc(prop_kind::forward_scoring, srcs, out_candidate, pooled_h, pooled_w,
                                          spatial_scale)));

    descs[0] = desc;
    std::shared_ptr<roi_pooling_forward::desc> selected_desc_ptr = descs[0];

    const MKLDNNPrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set for node " << getName() << ".";

    auto prim_desc = roi_pooling_forward::primitive_desc(*selected_desc_ptr, selected_pd->getEngine());
    primitive_desc_iterator itpd = descs[0].createPrimitiveDescriptorIterator(selected_pd->getEngine());

    std::vector<primitive::at> src_p;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        src_p.push_back(getParentEdgeAt(i)->getMemoryPtr()->GetPrimitive());
    }
    prim.reset(new roi_pooling_forward(prim_desc, src_p, getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNROIPoolingNode::created() {
    return getType() == ROIPooling;
}
