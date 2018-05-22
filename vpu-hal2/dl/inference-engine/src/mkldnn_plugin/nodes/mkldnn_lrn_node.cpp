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
#include "mkldnn_lrn_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNLrnNode::MKLDNNLrnNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNLrnNode::MKLDNNLrnNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNLrnNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (descs.size())
        return;
    NormLayer* lrnLayer = dynamic_cast<NormLayer*>(getCnnLayer().get());

    if (lrnLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert lrn layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";


    algorithm alg = (lrnLayer->_isAcrossMaps) ? lrn_across_channels : lrn_within_channel;

    auto parentDims = getParentEdgeAt(0)->getDims();

    for (auto format : getAvailableFormatsForDims(parentDims)) {
        memory::desc in_candidate{autoBlockingDims(parentDims, format), inputDataType, format};
        MKLDNNDescriptor desc(std::shared_ptr<lrn_forward::desc>(
                new lrn_forward::desc(prop_kind::forward_scoring, alg, in_candidate,
                                      static_cast<int>(lrnLayer->_size),
                                      lrnLayer->_alpha, lrnLayer->_beta, lrnLayer->_k)));
        descs.push_back(desc);
    }
}

void MKLDNNLrnNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<lrn_forward::primitive_desc, lrn_forward::desc>();

    prim.reset(new lrn_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                               getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNLrnNode::created() {
    return getType() == Lrn;
}
