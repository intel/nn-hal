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
#include "mkldnn_split_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <map>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNSplitNode::MKLDNNSplitNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNSplitNode::MKLDNNSplitNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNSplitNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    SplitLayer* splitLayer = dynamic_cast<SplitLayer*>(getCnnLayer().get());

    if (splitLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert split layer.";

    axis = splitLayer->_axis;

    if (axis != 1)
        THROW_IE_EXCEPTION << "Split support only axis 1.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input nodes.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output nodes.";
}

void MKLDNNSplitNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto inDims = getParentEdgeAt(0)->getDims();
    auto outFirst = getChildEdgeAt(0)->getDims();

    bool allCanBeOptimized = outFirst[0] == 1 && getChildEdgeAt(0)->getChild()->getType() != Concatenation;
    supported_nChw8c = inDims.ndims() == 4 && inDims[1] % 8 == 0 && outFirst.ndims() == 4 && outFirst[1] % 8 == 0;
    supported_nChw16c = inDims.ndims() == 4 && inDims[1] % 16 == 0 && outFirst.ndims() == 4 && outFirst[1] % 16 == 0;
    std::vector<MKLDNNMemoryDesc> dstFormats;
    dstFormats.push_back({outFirst, getOutputDataType(), MKLDNNMemory::GetPlainFormat(outFirst)});

    if (inDims.ndims() < 2)
        THROW_IE_EXCEPTION << "Split " << getName() << " isn't supported 1d blobs";

    size_t num_chanels = outFirst[1];

    for (size_t i = 1; i < getChildEdges().size(); i++) {
        auto outDims = getChildEdgeAt(i)->getDims();
        if (outFirst.ndims() != outDims.ndims()) {
            THROW_IE_EXCEPTION << "Split " << getName() << " supports only output blob with equal number of dimensions";
        }
        if (allCanBeOptimized && getChildEdgeAt(i)->getChild()->getType() == Concatenation)
            allCanBeOptimized = false;
        if (supported_nChw8c && outDims.ndims() == 4 && outDims[1] % 8) {
            supported_nChw8c = false;
        }
        if (supported_nChw16c && outDims.ndims() == 4 && outDims[1] % 16) {
            supported_nChw16c = false;
        }
        dstFormats.push_back({outDims, getOutputDataType(), MKLDNNMemory::GetPlainFormat(outFirst)});
        num_chanels += outDims[1];
        for (size_t j = 0; j < outFirst.ndims(); j++) {
            if (j == axis)
                continue;
            if (outDims[j] != outFirst[j])
                THROW_IE_EXCEPTION << "Split " << getName() << "has incorrect output dimensions";
        }
    }
    outFirst[1] = num_chanels;
    if (outFirst.size() != inDims.size())
        THROW_IE_EXCEPTION << "The sizes of input blob and sum of output blobs are not equal.";
    supportedPrimitiveDescriptors.push_back({engine,
                                             {{getParentEdgeAt(0)->getDims(), getInputDataType(), MKLDNNMemory::GetPlainFormat(inDims)}},
                                             {dstFormats}, impl_desc_type::unknown});

    if (supported_nChw8c) {
        dstFormats.clear();
        for (size_t i = 0; i < getChildEdges().size(); i++) {
            auto outDims = getChildEdgeAt(i)->getDims();
            dstFormats.push_back({outDims, getOutputDataType(), memory::nChw8c});
        }
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::nChw8c}},
                                                 dstFormats,
                                                 impl_desc_type::unknown});
    }

    if (supported_nChw16c) {
        dstFormats.clear();
        for (size_t i = 0; i < getChildEdges().size(); i++) {
            auto outDims = getChildEdgeAt(i)->getDims();
            dstFormats.push_back({outDims, getOutputDataType(), memory::nChw16c});
        }
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::nChw16c}},
                                                 dstFormats,
                                                 impl_desc_type::unknown});
    }

    if (allCanBeOptimized) {
        canOptimize = true;
    }
}

void MKLDNNSplitNode::createPrimitive() {
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        if (!getChildEdgeAt(i)->getMemoryPtr() || !getChildEdgeAt(i)->getMemory().GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNSplitNode::execute(mkldnn::stream strm) {
    if (canOptimize)
        return;

    MKLDNNDims par_dims = getParentEdgeAt(0)->getDims();
    int MB = batchToProcess(par_dims[0]);
    int par_slice_size = par_dims.size(1);
    const float *parData = static_cast<const float *>(getParentEdgeAt(0)->getMemory().GetData()) +
            getParentEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    for (size_t i = 0; i < getChildEdges().size(); i++) {
        MKLDNNEdgePtr childEdge = getChildEdgeAt(i);
        float *chldData = static_cast<float *>(childEdge->getMemory().GetData()) +
                childEdge->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
        unsigned chld_slice_size = (unsigned)childEdge->getDims().size(1);

        for (unsigned b = 0; b < MB; b++) {
            memcpy(chldData, parData + b * par_slice_size, chld_slice_size* sizeof(float));
            chldData += chld_slice_size;
        }
        parData += chld_slice_size;
    }
}

bool MKLDNNSplitNode::created() {
    return getType() == Split;
}

void MKLDNNSplitNode::initEdges() {
    if (canOptimize) {
        for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
            auto childEdge = getChildEdgeAt(cIdx);
            if (!childEdge)
                THROW_IE_EXCEPTION << "Node " << getName() << " contains empty child edge";
            childEdge->sharedMemFrom(getParentEdgeAt(0));
        }
        for (size_t pIdx = 0; pIdx < getParentEdges().size(); pIdx++) {
            auto parentEdge = getParentEdgeAt(pIdx);
            if (!parentEdge)
                THROW_IE_EXCEPTION << "Node " << getName() << " contains empty parent edge";
            parentEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        }
    } else {
        for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
            auto childEdge = getChildEdgeAt(cIdx);
            if (!childEdge)
                THROW_IE_EXCEPTION << "Node " << getName() << " contains empty child edge";
            childEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        }
        for (size_t pIdx = 0; pIdx < getParentEdges().size(); pIdx++) {
            auto parentEdge = getParentEdgeAt(pIdx);
            if (!parentEdge)
                THROW_IE_EXCEPTION << "Node " << getName() << " contains empty parent edge";
            parentEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        }
    }
}

void MKLDNNSplitNode::selectOptimalPrimitiveDescriptor() {
    std::map<mkldnn::memory::format, size_t> formatFrequency;
    memory::format convertTo = memory::format::format_undef;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto parent = parentEdge->getParent();

        if (parent->getSelectedPrimitiveDescriptor() == nullptr) {
            if (parent->getType() == Concatenation)
                continue;
            parent->selectOptimalPrimitiveDescriptor();
        }

        auto outDesc = parent->getSelectedPrimitiveDescriptor()->getOutputDescs()[0];
        if (!outDesc)
            continue;
        if (formatFrequency.find(outDesc.getFormat()) != formatFrequency.end())
            formatFrequency[outDesc.getFormat()] += 1;
        else
            formatFrequency[outDesc.getFormat()] = 1;
    }
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto child = childEdge->getChild();
        if (child->getSelectedPrimitiveDescriptor() == nullptr) {
            if (child->getType() == Concatenation)
                continue;
            child->selectOptimalPrimitiveDescriptor();
        }
        int inputIndex = childEdge->getOutputNum();
        if (inputIndex < 0)
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        if (inputIndex >= child->getSelectedPrimitiveDescriptor()->getInputDescs().size())
            inputIndex = 0;
        auto outDesc = child->getSelectedPrimitiveDescriptor()->getInputDescs()[inputIndex];
        if (!outDesc)
            continue;
        if (formatFrequency.find(outDesc.getFormat()) != formatFrequency.end())
            formatFrequency[outDesc.getFormat()] += 1;
        else
            formatFrequency[outDesc.getFormat()] = 1;
    }

    size_t maxCount = 0;
    for (auto &it : formatFrequency) {
        if (it.second > maxCount) {
            maxCount = it.second;
            convertTo = it.first;
        }
    }


    if (!canOptimize) {
        if (convertTo == memory::nChw16c && supported_nChw16c) {
            if (supported_nChw8c) {
                MKLDNNNode::selectPrimitiveDescriptorByIndex(2);
            } else {
                MKLDNNNode::selectPrimitiveDescriptorByIndex(1);
            }
        } else if (convertTo == memory::nChw8c && supported_nChw8c) {
            MKLDNNNode::selectPrimitiveDescriptorByIndex(1);
        } else {
            MKLDNNNode::selectPrimitiveDescriptorByIndex(0);
        }
        return;
    }

    if (!supported_nChw8c && convertTo == memory::nChw8c) {
        convertTo = memory::nchw;
    }

    if (!supported_nChw16c && convertTo == memory::nChw16c) {
        convertTo = memory::nchw;
    }

    std::vector<MKLDNNMemoryDesc> dstFormats;
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto outDims = getChildEdgeAt(i)->getDims();
        dstFormats.push_back({outDims, getOutputDataType(), convertTo});
    }

    supportedPrimitiveDescriptors.push_back({supportedPrimitiveDescriptors[0].getEngine(),
                                             {{getParentEdgeAt(0)->getDims(), getInputDataType(), convertTo}},
                                             dstFormats,
                                             impl_desc_type::unknown});
    selectPrimitiveDescriptorByIndex(static_cast<int>(supportedPrimitiveDescriptors.size()) - 1);
}

void MKLDNNSplitNode::resolveNotAllocatedEdges() {
    if (canOptimize) {
        int offset = 0;
        for (size_t i = 0; i < getChildEdges().size(); i++) {
            auto childEdge = getChildEdgeAt(i);
            offset += childEdge->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding * sizeof(float);
            char* memPtr = reinterpret_cast<char*>(childEdge->getMemory().GetData());
            memory::format splitFormat = childEdge->getMemory().GetFormat();
            childEdge->getMemoryPtr().reset(new MKLDNNMemory(getSelectedPrimitiveDescriptor()->getEngine()));
            auto child = childEdge->getChild();
            childEdge->getMemoryPtr()->Create(childEdge->getDims(), outputDataType, splitFormat, memPtr + offset);
            offset += childEdge->getDims().size() * sizeof(float);

            childEdge->changeStatus(MKLDNNEdge::Status::Allocated);
            child->resolveNotAllocatedEdges();
        }
    }
    MKLDNNNode::resolveNotAllocatedEdges();
}
