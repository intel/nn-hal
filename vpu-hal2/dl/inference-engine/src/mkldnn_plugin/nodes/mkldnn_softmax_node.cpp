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
#include "mkldnn_softmax_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNSoftMaxNode::MKLDNNSoftMaxNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNSoftMaxNode::MKLDNNSoftMaxNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNSoftMaxNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (descs.size())
        return;

    SoftMaxLayer* smLayer = dynamic_cast<SoftMaxLayer*>(getCnnLayer().get());
    if (smLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert softmax layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    axis = smLayer->axis;

    if (axis >= getParentEdgeAt(0)->getDims().ndims()) {
        THROW_IE_EXCEPTION << "Incorrect axis!";
    }

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        MKLDNNDims dims = getParentEdgeAt(0)->getDims();
        if (dims != autoBlockingDims(dims, format)) continue;

        memory::desc in_candidate{dims, inputDataType, format};

        MKLDNNDescriptor desc(std::shared_ptr<softmax_forward::desc>(
                new softmax_forward::desc(prop_kind::forward_scoring, in_candidate, axis)));
        descs.push_back(desc);
    }
}

void MKLDNNSoftMaxNode::createPrimitive() {
    if (prim)
        return;

    memory::desc in_candidate = getParentEdgeAt(0)->getMemory().GetDescriptor();
    MKLDNNDescriptor desc(std::shared_ptr<softmax_forward::desc>(
            new softmax_forward::desc(prop_kind::forward_scoring, in_candidate, axis)));
    descs[0] = desc;
    std::shared_ptr<softmax_forward::desc> selected_desc_ptr = descs[0];

    const MKLDNNPrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set for node " << getName() << ".";

    auto prim_desc = softmax_forward::primitive_desc(*selected_desc_ptr, selected_pd->getEngine());
    primitive_desc_iterator itpd = descs[0].createPrimitiveDescriptorIterator(selected_pd->getEngine());

    prim.reset(new softmax_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNSoftMaxNode::created() {
    return getType() == SoftMax;
}
