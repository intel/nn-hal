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

#include "mkldnn_concat_node.h"

#include <map>
#include <utility>
#include <vector>

#include "details/ie_exception.hpp"
#include "ie_layers.h"
#include "mkldnn.hpp"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_dims.h"
#include "mkldnn_edge.h"
#include "mkldnn_memory.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConcatNode::MKLDNNConcatNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNConcatNode::MKLDNNConcatNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNConcatNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    auto * conLayer = dynamic_cast<ConcatLayer*>(getCnnLayer().get());

    if (conLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert concat layer.";

    axis = conLayer->_axis;

    if (!getParentEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
    auto& firstParentDims = getParentEdgeAt(0)->getDims();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        auto& dims = getParentEdgeAt(i)->getDims();
        bool incorrectDims = false;
        for (size_t j = 0; j < firstParentDims.ndims(); j++) {
            if (j == axis)
                continue;
            if (dims.ndims() != firstParentDims.ndims() || firstParentDims[j] != dims[j]) {
                incorrectDims = true;
                break;
            }
        }
        if (incorrectDims || firstParentDims.ndims() == 0) {
            THROW_IE_EXCEPTION << "Incorrect input dimensions for concat node " << getName();
        }
    }
}

void MKLDNNConcatNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    bool canBeOptimized = axis == 1 && getChildEdgeAt(0)->getDims()[0] == 1;
    std::vector<MKLDNNMemoryDesc> srcFormats;
    int numOfDim = -1;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (numOfDim < 0) {
            numOfDim = parentEdge->getDims().ndims();
        }
        if (canBeOptimized && parentEdge->getDims().ndims() != numOfDim)
            canBeOptimized = false;
        srcFormats.push_back({parentEdge->getDims(), getInputDataType(), memory::format::any});
    }

    auto dims = getChildEdgeAt(0)->getDims();

    MKLDNNMemoryDesc dstFormat = MKLDNNMemoryDesc(dims, getOutputDataType(), MKLDNNMemory::GetPlainFormat(dims));
    supportedPrimitiveDescriptors.push_back({engine, srcFormats, {dstFormat}, impl_desc_type::ref});
    if (dims.ndims() == 4) {
        if (dims[1] % 8 == 0) {
            dstFormat = MKLDNNMemoryDesc(dims, getOutputDataType(), mkldnn::memory::nChw8c);
            supportedPrimitiveDescriptors.push_back({engine, srcFormats, {dstFormat}, impl_desc_type::ref});

            if (dims[1] % 16 == 0) {
                dstFormat = MKLDNNMemoryDesc(dims, getOutputDataType(), mkldnn::memory::nChw16c);
                supportedPrimitiveDescriptors.push_back({engine, srcFormats, {dstFormat}, impl_desc_type::ref});
                if (canBeOptimized) {
                    srcFormats.clear();
                    dstFormat = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::nChw16c);
                    for (size_t i = 0; i < getParentEdges().size(); i++) {
                        srcFormats.emplace_back(
                                autoBlockingDims(getParentEdgeAt(i)->getDims(), memory::format::nChw16c),
                                getInputDataType(), memory::format::nChw16c);
                    }

                    supportedPrimitiveDescriptors.push_back({engine, srcFormats, {dstFormat},
                                                             impl_desc_type::unknown});
                }
            }
            if (canBeOptimized) {
                srcFormats.clear();
                dstFormat = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::nChw8c);
                for (size_t i = 0; i < getParentEdges().size(); i++) {
                    srcFormats.emplace_back(autoBlockingDims(getParentEdgeAt(i)->getDims(), memory::format::nChw8c),
                                            getInputDataType(), memory::format::nChw8c);
                }

                supportedPrimitiveDescriptors.push_back({engine, srcFormats, {dstFormat},
                                                         impl_desc_type::unknown});
            }
        }
    }

    if (canBeOptimized) {
        if (numOfDim == 2) {
            srcFormats.clear();
            dstFormat = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::nc);
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                srcFormats.push_back({getParentEdgeAt(i)->getDims(), getInputDataType(), memory::format::nc});
            }
            supportedPrimitiveDescriptors.push_back({engine, srcFormats, {dstFormat},
                                                     impl_desc_type::unknown});
        }
        if (numOfDim == 3) {
            srcFormats.clear();
            dstFormat = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::blocked);
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                srcFormats.push_back({getParentEdgeAt(i)->getDims(), getInputDataType(), memory::format::blocked});
            }

            supportedPrimitiveDescriptors.push_back({engine, srcFormats, {dstFormat},
                impl_desc_type::unknown});
        }
        if (numOfDim == 4) {
            srcFormats.clear();
            dstFormat = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::nchw);
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                srcFormats.push_back({getParentEdgeAt(i)->getDims(), getInputDataType(), memory::format::nchw});
            }

            supportedPrimitiveDescriptors.push_back({engine, srcFormats, {dstFormat},
                                                     impl_desc_type::unknown});
        }
    }
}

void MKLDNNConcatNode::selectOptimalPrimitiveDescriptor() {
    bool hasAny = true;
    bool hasUnknown = false;
    std::vector<size_t> canSelectPrimitive;
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        auto &primDescInfo = supportedPrimitiveDescriptors[i];
        if (primDescInfo.getImplementationType() != impl_desc_type::unknown)
            continue;
        hasUnknown = true;
        for (auto iInfo : primDescInfo.getInputDescs()) {
            if (iInfo) {
                hasAny = false;
                break;
            }
        }

        if (hasAny) {
            for (auto oInfo : primDescInfo.getOutputDescs()) {
                if (oInfo) {
                    hasAny = false;
                    break;
                }
            }
        }

        if (!hasAny) {
            canSelectPrimitive.push_back(i);
        }
    }

    if (hasUnknown) {
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto parentEdge = getParentEdgeAt(i);
            auto parent = parentEdge->getParent();
            if (parent->getType() == Concatenation || parent->getType() == Convolution_Sum ||
                    parent->getType() == Convolution_Sum_Activation) {
                // FIXME: (hot fix for pc detect) set memory for convolution from concat
                hasUnknown = false;
                break;
            }
            for (size_t j = 0; j < parent->getChildEdges().size(); j++) {
                auto child = parent->getChildEdgeAt(j)->getChild();
                auto *concatNode = dynamic_cast<MKLDNNConcatNode *>(child.get());
                if (concatNode != nullptr && concatNode != this && concatNode->isOptimized()) {
                    hasUnknown = false;
                    break;
                }
            }
            if (!hasUnknown)
                break;
        }
    }

    if (hasUnknown) {
        canOptimize = true;

        if (canSelectPrimitive.size() == 1) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive[0]));
            return;
        }
    }

    std::map<mkldnn::memory::format, size_t> formatFrequency;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parent = getParentEdgeAt(i)->getParent();

        if (parent->getSelectedPrimitiveDescriptor() == nullptr) {
            if (!canOptimize)
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
            if (!canOptimize)
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
    mkldnn::memory::format convertTo = MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims());
    for (auto &it : formatFrequency) {
        if (it.second > maxCount) {
            maxCount = it.second;
            convertTo = it.first;
        }
    }

    if (canOptimize && autoBlockingDims(getChildEdgeAt(0)->getDims(), convertTo) != getChildEdgeAt(0)->getDims())
        canOptimize = false;
    for (size_t i = 0; canOptimize && i < getParentEdges().size(); i++) {
        if (autoBlockingDims(getParentEdgeAt(i)->getDims(), convertTo) != getParentEdgeAt(i)->getDims())
            canOptimize = false;
    }

    if (!canOptimize) {
        for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
            auto &primDescInfo = supportedPrimitiveDescriptors[i];
            if (primDescInfo.getImplementationType() == impl_desc_type::unknown)
                continue;
            if (convertTo == supportedPrimitiveDescriptors[i].getOutputDescs()[0].getFormat()) {
                size_t num = 0;
                for (num = 0; num < getParentEdges().size(); num++) {
                    if (autoBlockingDims(getParentEdgeAt(num)->getDims(), convertTo) != getParentEdgeAt(num)->getDims())
                        break;
                }
                if (num == getParentEdges().size()) {
                    selectPrimitiveDescriptorByIndex(i);
                    return;
                }
            }
        }
        selectPrimitiveDescriptorByIndex(0);
        return;
    }

    for (auto supportedPdIndex : canSelectPrimitive) {
        if (supportedPrimitiveDescriptors[supportedPdIndex].getInputDescs()[0].getFormat() == convertTo) {
            selectPrimitiveDescriptorByIndex(supportedPdIndex);
            return;
        }
    }

    std::vector<MKLDNNMemoryDesc> srcFormats;
    MKLDNNMemoryDesc dstFormat = MKLDNNMemoryDesc(autoBlockingDims(getChildEdgeAt(0)->getDims(), convertTo), getOutputDataType(), convertTo);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        srcFormats.emplace_back(autoBlockingDims(getParentEdgeAt(i)->getDims(), convertTo), getInputDataType(), convertTo);
    }

    supportedPrimitiveDescriptors.push_back({supportedPrimitiveDescriptors[0].getEngine(), srcFormats, {dstFormat},
                                             impl_desc_type::unknown});
    selectPrimitiveDescriptorByIndex(static_cast<int>(supportedPrimitiveDescriptors.size()) - 1);
}

void MKLDNNConcatNode::initEdges() {
    const MKLDNNPrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        return;

    if (!canOptimize) {
        MKLDNNNode::initEdges();
        return;
    }

    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        auto childEdge = getChildEdgeAt(cIdx);
        if (cIdx) {
            childEdge->sharedMemFrom(getChildEdgeAt(0));
        } else {
            childEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        }
    }
    for (size_t pIdx = 0; pIdx < getParentEdges().size(); pIdx++) {
        auto parentEdge = getParentEdgeAt(pIdx);
        if (parentEdge->getStatus() == MKLDNNEdge::Status::NotAllocated) {
            parentEdge = parentEdge->getSharedEdge();
            if (parentEdge.get() != getChildEdgeAt(0).get() ||
                    parentEdge->getStatus() == MKLDNNEdge::Status::NotAllocated) {
                canOptimize = false;
                break;
            }
        }
    }
    if (canOptimize) {
        for (size_t pIdx = 0; pIdx < getParentEdges().size(); pIdx++) {
            auto parentEdge = getParentEdgeAt(pIdx);
            if (parentEdge->getStatus() == MKLDNNEdge::Status::NotAllocated) {
                parentEdge = parentEdge->getSharedEdge();
            }
            parentEdge->sharedMemFrom(getChildEdgeAt(0));
        }
    } else {
        MKLDNNNode::initEdges();
        return;
    }
}

void MKLDNNConcatNode::resolveNotAllocatedEdges() {
    if (canOptimize) {
        int offset = 0;
        auto firstChildEdge = getChildEdgeAt(0);
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto parentEdge = getParentEdgeAt(i);

            if (parentEdge->getSharedEdge().get() != getChildEdgeAt(0).get()) {
                parentEdge = parentEdge->getSharedEdge();
            }

            auto * memPtr = reinterpret_cast<char*>(parentEdge->getMemory().GetData());
            memory::format concatFormat = parentEdge->getMemory().GetFormat();
            parentEdge->getMemoryPtr().reset(new MKLDNNMemory(getSelectedPrimitiveDescriptor()->getEngine()));
            auto parent = parentEdge->getParent();
            parentEdge->getMemoryPtr()->Create(parentEdge->getDims(), inputDataType, concatFormat, memPtr + offset);
            offset += parentEdge->getDims().size() * sizeof(float);

            parentEdge->changeStatus(MKLDNNEdge::Status::Allocated);
        }
    }
    MKLDNNNode::resolveNotAllocatedEdges();
}

bool MKLDNNConcatNode::created() {
    return getType() == Concatenation;
}

bool MKLDNNConcatNode::isOptimized() {
    return canOptimize;
}

void MKLDNNConcatNode::createPrimitive() {
    if (prim || canOptimize)
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<primitive::at> srcs_p;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr()) {
            auto parent = getParentEdgeAt(i)->getParent();
            THROW_IE_EXCEPTION << "Source memory from " << parent->getName() << " didn't allocate for node "
                               << getName() << ".";
        }

        auto desc = srcMemPtr->GetDescriptor();
        auto dims = getParentEdgeAt(i)->getDims();
        for (size_t i = 0; i < dims.ndims(); i++) {
            desc.data.dims[i] = dims[i];
        }

        srcs_pd.push_back(memory::primitive_desc(desc, srcMemPtr->GetPrimitiveDescriptor().get_engine()));
        srcs_p.push_back(srcMemPtr->GetPrimitive());
    }

    auto primitive_desc = concat::primitive_desc(getChildEdgeAt(0)->getMemory().GetDescriptor(), axis, srcs_pd);

    prim.reset(new concat(primitive_desc, srcs_p, getChildEdgeAt(0)->getMemory().GetPrimitive()));
}
