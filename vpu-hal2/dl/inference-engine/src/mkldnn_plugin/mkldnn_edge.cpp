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
#include "mkldnn_edge.h"
#include "mkldnn_node.h"
#include "mkldnn_extension_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNPlugin::MKLDNNEdge::MKLDNNEdge(const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> &parent,
                                     const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> &child) {
    this->parent = parent;
    this->child = child;
}

const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> MKLDNNPlugin::MKLDNNEdge::getParent() const {
    auto parentPtr = parent.lock();
    if (!parentPtr)
        THROW_IE_EXCEPTION << "Edge contains empty parent node";
    return parentPtr;
}

const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> MKLDNNPlugin::MKLDNNEdge::getChild() const {
    auto childPtr = child.lock();
    if (!childPtr)
        THROW_IE_EXCEPTION << "Edge contains empty child node";
    return childPtr;
}

bool MKLDNNPlugin::MKLDNNEdge::isDropped() {
    return getInputNum() == -1 && getOutputNum() == -1;
}

bool MKLDNNPlugin::MKLDNNEdge::needReorder() {
    return getInputDesc() != getOutputDesc();
}

MKLDNNPlugin::MKLDNNMemoryDesc MKLDNNPlugin::MKLDNNEdge::getInputDesc() {
    if (!inputDesc) {
        memory::format fmt = getSpecifiedInputFormat({});
        if (getDims().ndims() == 2 && fmt == memory::nchw) fmt = memory::nc;

        memory::data_type dataType = getParent()->getOutputDataType();
        inputDesc = MKLDNNMemoryDesc(autoBlockingDims(getDims(), fmt), dataType, fmt);
    }
    return inputDesc;
}

MKLDNNPlugin::MKLDNNMemoryDesc MKLDNNPlugin::MKLDNNEdge::getOutputDesc() {
    if (!outputDesc) {
        memory::format fmt = getSpecifiedOutputFormat({});
        if (getDims().ndims() == 2 && fmt == memory::nchw) fmt = memory::nc;


        memory::data_type dataType = getChild()->getInputDataType();
        outputDesc = MKLDNNMemoryDesc(autoBlockingDims(getDims(), fmt), dataType, fmt);
    }
    return outputDesc;
}

int MKLDNNPlugin::MKLDNNEdge::getInputNum() {
    auto parentPtr = parent.lock();
    if (!parentPtr)
        return -1;
    for (size_t i = 0; i < parentPtr->getChildEdges().size(); i++) {
        auto childEdge = parentPtr->getChildEdges()[i].lock();
        if (childEdge && childEdge.get() == this) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int MKLDNNPlugin::MKLDNNEdge::getOutputNum() {
    auto childPtr = child.lock();
    if (!childPtr)
        return -1;
    for (size_t i = 0; i < childPtr->getParentEdges().size(); i++) {
        auto parentEdge = childPtr->getParentEdges()[i].lock();
        if (parentEdge && parentEdge.get() == this) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MKLDNNPlugin::MKLDNNEdge::allocate() {
    if (status != Status::NeedAllocation)
        return;

    if (memoryPtr)
        THROW_IE_EXCEPTION << "Unexpected behaviour: status == NeedAllocation but memory is already allocated.";

    auto inputDesc = getInputDesc();
    auto outputDesc = getOutputDesc();
    if (outputDesc != inputDesc)
        THROW_IE_EXCEPTION << "Cannot allocate memory. Nodes have primitive descriptors with different formats.";
    if (!inputDesc)
        THROW_IE_EXCEPTION << "Cannot get input descriptor!";

    auto parentPtr = getParent();
    memoryPtr.reset(new MKLDNNMemory(parentPtr->getSelectedPrimitiveDescriptor()->getEngine()));
    memoryPtr->Create(inputDesc);
    status = Status::Allocated;
}

void MKLDNNPlugin::MKLDNNEdge::changeStatus(MKLDNNPlugin::MKLDNNEdge::Status state) {
    if (state == Status::NotAllocated) {
        THROW_IE_EXCEPTION << "Incorrect behaviour! Use method sharedMemFrom()";
    }
    if (state == Status::Validated) {
        THROW_IE_EXCEPTION << "Incorrect behaviour! Use method validate()";
    }
    if (status != Status::Uninitialized && state == Status::NeedAllocation)
        return;
    if (status == Status::NotAllocated)
        memoryFromEdge.reset();
    status = state;
}

MKLDNNPlugin::MKLDNNDims &MKLDNNPlugin::MKLDNNEdge::getDims() {
    if (!dims.ndims()) {
        MKLDNNDims outDims;
        MKLDNNDims inDims;
        auto childPtr = getChild();
        auto parentPtr = getParent();

        int inNum = getOutputNum();
        if (inNum < 0) {
            THROW_IE_EXCEPTION << "Error cannot find input data for " << child.lock()->getName()
                               << " from " << parent.lock()->getName();
        }
        if (inNum < childPtr->inDims.size()) {
            outDims = childPtr->inDims[inNum];
        }

        int outNum = getInputNum();
        if (outNum < 0) {
            THROW_IE_EXCEPTION << "Error cannot find output data for " << parent.lock()->getName()
                               << " to " << child.lock()->getName();
        }
        if (outNum >= parentPtr->outDims.size())
            outNum = 0;
        if (outNum < parentPtr->outDims.size()) {
            inDims = parentPtr->outDims[outNum];
        }

        if (inDims.ndims() && outDims.ndims() && inDims.ndims() != outDims.ndims() && inDims.size() != outDims.size())
            THROW_IE_EXCEPTION << "Nodes " << getParent()->getName() << " and " << getChild()->getName()
                               << " have incompatible dimensions!";

        dims = outDims.ndims() ? outDims : inDims;

        if (!dims.ndims())
            THROW_IE_EXCEPTION << "Cannot detect right dims for nodes " << getParent()->getName()
                               << " and " << getChild()->getName();
    }
    return dims;
}

void MKLDNNPlugin::MKLDNNEdge::setDims(MKLDNNPlugin::MKLDNNDims &dims) {
    this->dims = dims;
}

bool MKLDNNPlugin::MKLDNNEdge::nodeCanChangeDesc(const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> &node) const {
    MKLDNNPrimitiveDescInfo * selectedPd = node->getSelectedPrimitiveDescriptor();
    if (selectedPd == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << node->getName() << " is not selected.";

    for (auto &inputDesc : selectedPd->getInputDescs()) {
        if (inputDesc) {
            return true;
        }
    }

    for (auto &outDesc : selectedPd->getOutputDescs()) {
        if (outDesc) {
            return true;
        }
    }

    MKLDNNDims inputDims;
    for (size_t i = 0; i < node->getParentEdges().size(); i++) {
        if (inputDims.size() == 1 && inputDims.ndims() == 0) {
            inputDims = node->getParentEdgeAt(i)->getDims();
            continue;
        }

        if (inputDims.ndims() != node->getParentEdgeAt(i)->getDims().ndims()) {
            return true;
        }
    }
    for (size_t i = 0; i < node->getChildEdges().size(); i++) {
        if (inputDims.size() == 1 && inputDims.ndims() == 0) {
            inputDims = node->getChildEdgeAt(i)->getDims();
            continue;
        }

        if (inputDims.ndims() != node->getChildEdgeAt(i)->getDims().ndims()) {
            return true;
        }
    }

    return false;
}

/// In we have {any, any, any} -> {any} or {any} -> {any, any, any} or {any} -> {any} it means that
/// layer doesn't change memory format
/// We don't support {any, any, nchw} -> {any}
mkldnn::memory::format MKLDNNPlugin::MKLDNNEdge::getSpecifiedInputFormat(std::map<mkldnn::memory::format, size_t> formats) {
    memory::format inFormat;
    static int enterCount = 0;
    enterCount++;

    if (inputDesc) {
        --enterCount;
        return inputDesc.getFormat();
    }

    auto parentPtr = getParent();
    if (parentPtr->getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << parentPtr->getName() << " is not selected.";

    int inputIdx = getInputNum();
    if (inputIdx < 0)
        THROW_IE_EXCEPTION << "Edge cannot be found for node" << parentPtr->getName() << ".";

    if (inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs().size())
        inFormat = parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[inputIdx].getFormat();
    else
        inFormat = parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[0].getFormat();

    if (inFormat != memory::any && inFormat != memory::format_undef) {
        --enterCount;
        return inFormat;
    }

    bool isFormatChanging = nodeCanChangeDesc(parentPtr);

    if (!isFormatChanging && inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getInputDescs().size() &&
            parentPtr->getSelectedPrimitiveDescriptor()->getInputDescs()[inputIdx]) {
        inFormat = parentPtr->getSelectedPrimitiveDescriptor()->getInputDescs()[inputIdx].getFormat();
        parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[inputIdx] =
                MKLDNNMemoryDesc(autoBlockingDims(getDims(), inFormat), parentPtr->getOutputDataType(), inFormat);
        --enterCount;
        return inFormat;
    }

    parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[inputIdx] = MKLDNNMemoryDesc();

    for (size_t i = 0; i < parentPtr->getChildEdges().size(); i++) {
        auto childEdge = parentPtr->getChildEdgeAt(i);
        auto child = childEdge->getChild();
        int childIdx = childEdge->getOutputNum();
        if (!child->getSelectedPrimitiveDescriptor() || childIdx < 0 ||
                childEdge->getDims().ndims() != getDims().ndims()) {
            continue;
        }
        if (child->getSelectedPrimitiveDescriptor()->getInputDescs().size() <= childIdx) {
            childIdx = 0;
        }
        memory::format childInDesc = child->getSelectedPrimitiveDescriptor()->getInputDescs()[childIdx].getFormat();
        if (childInDesc != memory::any && childInDesc != memory::format_undef) {
            if (formats.find(childInDesc) == formats.end())
                formats[childInDesc] = 1;
            else
                formats[childInDesc] += 1;
            continue;
        }
        if (nodeCanChangeDesc(child))
            continue;

        if (enterCount < 2) {
            childInDesc = childEdge->getSpecifiedOutputFormat(formats);
            if (childInDesc != memory::any && childInDesc != memory::format_undef) {
                if (formats.find(childInDesc) == formats.end())
                    formats[childInDesc] = 1;
                else
                    formats[childInDesc] += 1;
            }
        }
    }

    if (!isFormatChanging) {
        for (size_t i = 0; i < parentPtr->getParentEdges().size(); i++) {
            auto parentEdge = parentPtr->getParentEdgeAt(i);
            auto parent = parentEdge->getParent();
            int parentIdx = parentEdge->getInputNum();
            if (!parent->getSelectedPrimitiveDescriptor() || parentIdx < 0 ||
                    parentEdge->getDims().ndims() != getDims().ndims()) {
                continue;
            }
            if (parent->getSelectedPrimitiveDescriptor()->getOutputDescs().size() <= parentIdx) {
                parentIdx = 0;
            }
            memory::format parentOutDesc = parent->getSelectedPrimitiveDescriptor()->getOutputDescs()[parentIdx].getFormat();
            if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
                if (formats.find(parentOutDesc) == formats.end())
                    formats[parentOutDesc] = 1;
                else
                    formats[parentOutDesc] += 1;
                continue;
            }
            if (nodeCanChangeDesc(parent))
                continue;

            if (enterCount < 2) {
                parentOutDesc = parentEdge->getSpecifiedInputFormat(formats);
                if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
                    if (formats.find(parentOutDesc) == formats.end())
                        formats[parentOutDesc] = 1;
                    else
                        formats[parentOutDesc] += 1;
                }
            }
        }
    }

    size_t maxFormatCount = 0;
    memory::format desc =  MKLDNNMemory::GetPlainFormat(getDims());
    for (auto &it : formats) {
        if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getDims(), it.first)) {
            maxFormatCount = it.second;
            desc = it.first;
        }
    }

    parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[inputIdx] =
            MKLDNNMemoryDesc(autoBlockingDims(getDims(), desc), parentPtr->getOutputDataType(), desc);
    if (!isFormatChanging && inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getInputDescs().size() &&
            !parentPtr->getSelectedPrimitiveDescriptor()->getInputDescs()[inputIdx]) {
        parentPtr->getSelectedPrimitiveDescriptor()->getInputDescs()[inputIdx] =
                MKLDNNMemoryDesc(autoBlockingDims(getDims(), desc), parentPtr->getOutputDataType(), desc);
    }

    --enterCount;
    return desc;
}

mkldnn::memory::format MKLDNNPlugin::MKLDNNEdge::getSpecifiedOutputFormat(std::map<mkldnn::memory::format, size_t> formats) {
    static int enterCount = 0;
    enterCount++;
    memory::format outFormat;

    if (outputDesc) {
        enterCount--;
        return outputDesc.getFormat();
    }

    auto childPtr = getChild();
    auto parentPtr = getParent();

    if (childPtr->getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << childPtr->getName() << " is not selected.";

    int outputIdx = getOutputNum();
    int inputIdx = getInputNum();
    if (outputIdx < 0) {
        THROW_IE_EXCEPTION << "Edge cannot be found for node" << childPtr->getName() << ".";
    }
    if (outputIdx >= childPtr->getSelectedPrimitiveDescriptor()->getInputDescs().size())
        outputIdx = 0;
    outFormat = childPtr->getSelectedPrimitiveDescriptor()->getInputDescs()[outputIdx].getFormat();

    if (outFormat != memory::any && outFormat != memory::format_undef) {
        enterCount--;
        return outFormat;
    }

    if (inputIdx >= parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs().size())
        inputIdx = 0;

    bool isFormatChanging = nodeCanChangeDesc(childPtr);

    if ((!isFormatChanging && outputIdx < childPtr->getSelectedPrimitiveDescriptor()->getOutputDescs().size() &&
            childPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[outputIdx]) ||
            (isFormatChanging && inputIdx >= 0 && parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[inputIdx])) {
        if (!isFormatChanging)
            outFormat = childPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[outputIdx].getFormat();
        else
            outFormat = parentPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[inputIdx].getFormat();
        childPtr->getSelectedPrimitiveDescriptor()->getInputDescs()[outputIdx] =
                MKLDNNMemoryDesc(autoBlockingDims(getDims(), outFormat), childPtr->getInputDataType(), outFormat);
        enterCount--;
        return outFormat;
    }

    childPtr->getSelectedPrimitiveDescriptor()->getInputDescs()[outputIdx] = MKLDNNMemoryDesc();

    for (size_t i = 0; i < childPtr->getParentEdges().size(); i++) {
        auto parentEdge = childPtr->getParentEdgeAt(i);
        auto parent = parentEdge->getParent();
        int parentIdx = parentEdge->getInputNum();
        if (!parent->getSelectedPrimitiveDescriptor() || parentIdx < 0 ||
                parentEdge->getDims().ndims() != getDims().ndims()) {
            continue;
        }
        if (parent->getSelectedPrimitiveDescriptor()->getOutputDescs().size() <= parentIdx) {
            parentIdx = 0;
        }
        memory::format parentOutDesc = parent->getSelectedPrimitiveDescriptor()->getOutputDescs()[parentIdx].getFormat();
        if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
            if (formats.find(parentOutDesc) == formats.end())
                formats[parentOutDesc] = 1;
            else
                formats[parentOutDesc] += 1;
            continue;
        }
        if (nodeCanChangeDesc(parent))
            continue;

        if (enterCount < 2) {
            parentOutDesc = parentEdge->getSpecifiedInputFormat(formats);
            if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
                if (formats.find(parentOutDesc) == formats.end())
                    formats[parentOutDesc] = 1;
                else
                    formats[parentOutDesc] += 1;
            }
        }
    }

    if (!isFormatChanging) {
        for (size_t i = 0; i < childPtr->getChildEdges().size(); i++) {
            auto childEdge = childPtr->getChildEdgeAt(i);
            auto child = childEdge->getChild();
            int childIdx = childEdge->getOutputNum();
            if (!child->getSelectedPrimitiveDescriptor() || childIdx < 0 ||
                    childEdge->getDims().ndims() != getDims().ndims()) {
                continue;
            }
            if (child->getSelectedPrimitiveDescriptor()->getInputDescs().size() <= childIdx) {
                childIdx = 0;
            }
            memory::format childInDesc = child->getSelectedPrimitiveDescriptor()->getInputDescs()[childIdx].getFormat();
            if (childInDesc != memory::any && childInDesc != memory::format_undef) {
                if (formats.find(childInDesc) == formats.end())
                    formats[childInDesc] = 1;
                else
                    formats[childInDesc] += 1;
                continue;
            }
            if (nodeCanChangeDesc(child))
                continue;

            if (enterCount < 2) {
                childInDesc = childEdge->getSpecifiedOutputFormat(formats);
                if (childInDesc != memory::any && childInDesc != memory::format_undef) {
                    if (formats.find(childInDesc) == formats.end())
                        formats[childInDesc] = 1;
                    else
                        formats[childInDesc] += 1;
                }
            }
        }
    }

    size_t maxFormatCount = 0;
    memory::format format =  MKLDNNMemory::GetPlainFormat(getDims());
    for (auto &it : formats) {
        if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getDims(), it.first)) {
            maxFormatCount = it.second;
            format = it.first;
        }
    }

    childPtr->getSelectedPrimitiveDescriptor()->getInputDescs()[outputIdx] =
            MKLDNNMemoryDesc(autoBlockingDims(getDims(), format), childPtr->getInputDataType(), format);
    if (!isFormatChanging && outputIdx < childPtr->getSelectedPrimitiveDescriptor()->getOutputDescs().size() &&
            !childPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[outputIdx]) {
        childPtr->getSelectedPrimitiveDescriptor()->getOutputDescs()[outputIdx] =
                MKLDNNMemoryDesc(autoBlockingDims(getDims(), format), childPtr->getInputDataType(), format);
    }

    enterCount--;
    return format;
}

const MKLDNNPlugin::MKLDNNMemory &MKLDNNPlugin::MKLDNNEdge::getMemory() {
    if (status == Status::NotAllocated) {
        memoryPtr = getSharedEdge()->getMemoryPtr();
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return *memoryPtr;
}

MKLDNNPlugin::MKLDNNMemoryPtr &MKLDNNPlugin::MKLDNNEdge::getMemoryPtr() {
    if (status == Status::NotAllocated) {
        memoryPtr = getSharedEdge()->getMemoryPtr();
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return memoryPtr;
}

void MKLDNNPlugin::MKLDNNEdge::sharedMemFrom(const MKLDNNPlugin::MKLDNNEdgePtr &edge) {
    memoryFromEdge = edge;
    status = Status::NotAllocated;
}

void MKLDNNPlugin::MKLDNNEdge::validate() {
    if (status == Status::Validated)
        return;
    getMemory();
    getParent();
    getChild();
    getDims();
    if (status != Status::Allocated) {
        THROW_IE_EXCEPTION << "Error memory is not allocated!";
    }
    status = Status::Validated;
}

MKLDNNPlugin::MKLDNNEdgePtr MKLDNNPlugin::MKLDNNEdge::getSharedEdge() const {
    auto memoryFromEdgePtr = memoryFromEdge.lock();
    if (!memoryFromEdgePtr) {
        THROW_IE_EXCEPTION << "Cannot get memory ptr for edge(" << getParent()->getName() << "->"
                           << getChild()->getName() << "). The pointer on the edge with memory is empty!";
    }
    return memoryFromEdgePtr;
}
