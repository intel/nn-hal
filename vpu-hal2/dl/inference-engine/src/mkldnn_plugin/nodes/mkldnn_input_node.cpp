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
#include "mkldnn_input_node.h"
#include "../mkldnn_extension_utils.h"
#include <string>

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNInputNode::MKLDNNInputNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNInputNode::MKLDNNInputNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNInputNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (getType() == Input) {
        // User-specified input data precision is being put into outData for some reason
        if (this->getCnnLayer() != nullptr && this->getCnnLayer()->outData.size() > 0) {
            InferenceEngine::Precision cnnInputPrecision = this->getCnnLayer()->outData[0]->getPrecision();

            // As MKLDNN doesn't support U16, we prepare to FP32 in that case
            if (cnnInputPrecision == InferenceEngine::Precision::U16) cnnInputPrecision = InferenceEngine::Precision::FP32;


            if (outputDataType == mkldnn::memory::data_undef) {
                // If the output type is undefined, we set it to CNN input data type
                this->outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(cnnInputPrecision);
            }
        }

        if (!getParentEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of input edges.";
        if (getChildEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges.";
    } else if (getType() == Output) {
        if (getParentEdges().size() != 1)
            THROW_IE_EXCEPTION << "Incorrect number of input edges.";
        if (!getChildEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges.";
    }

    auto layer = getCnnLayer();
    if (layer && CaselessEq<std::string>()(layer->type, "const")) {
        if (layer->blobs.size() != 1 || getType() != Input || !layer->blobs.begin()->second)
            THROW_IE_EXCEPTION << "Incorrect const input " << getName();
        constBlob = layer->blobs.begin()->second;
    }
}

void MKLDNNInputNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    if (getType() == Input || getType() == MemoryInput)
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {},
                                                 {{getChildEdgeAt(0)->getDims(), getOutputDataType(),
                                                     MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims())}},
                                                 impl_desc_type::unknown});
    else if (getType() == Output)
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(),
                                                     MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims())}},
                                                 {},
                                                 impl_desc_type::unknown});
}

void MKLDNNInputNode::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    const MKLDNNPrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set for node " << getName() << ".";
}

bool MKLDNNInputNode::created() {
    return getType() == Input || getType() == Output;
}

bool MKLDNNInputNode::isConstant(bool fromCache) {
    if (!getCnnLayer() || getCnnLayer()->type != "Const")
        return MKLDNNNode::isConstant(fromCache);

    return true;
}

void MKLDNNInputNode::execute(mkldnn::stream strm) {
    if (!constBlob)
        return;
    auto dstBlob = getChildEdgeAt(0)->getMemory().GetBlob();
    const float *srcData = constBlob->cbuffer().as<float *>();
    float *dstData = dstBlob->buffer();
    if (constBlob->size() != dstBlob->size()) {
        THROW_IE_EXCEPTION << "Incorrect blob sizes for node " << getName();
    }
    for (size_t i = 0; i < constBlob->size(); i++) {
        // srcData without offset() because constBlob should be planar
        dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];
    }
}
