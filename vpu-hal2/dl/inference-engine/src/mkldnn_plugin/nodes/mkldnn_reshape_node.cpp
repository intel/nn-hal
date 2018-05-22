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
#include "mkldnn_reshape_node.h"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNReshapeNode::MKLDNNReshapeNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNReshapeNode::MKLDNNReshapeNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNReshapeNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
}

void MKLDNNReshapeNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto& inDims = getParentEdgeAt(0)->getDims();
    auto& outDims = getChildEdgeAt(0)->getDims();
    if ((inDims.ndims() != 2 && inDims.ndims() != 3 && inDims.ndims() != 4) ||
            (outDims.ndims() != 2 && outDims.ndims() != 3 && outDims.ndims() != 4))
        THROW_IE_EXCEPTION << "Reshape layer: Currently unsupported reshape mode.";

    if (inDims.size() != outDims.size()) {
        THROW_IE_EXCEPTION << "Reshape layer: Sizes of input and output blobs should equal.";
    }

    auto outFormat = MKLDNNMemoryDesc(outDims, getOutputDataType(), MKLDNNMemory::GetPlainFormat(outDims));

    if (inDims.ndims() == 4 && inDims[1] % 8 == 0 && outDims.ndims() == 4 &&outDims[1] % 8 == 0) {
        outFormat = MKLDNNMemoryDesc(outDims, getOutputDataType(), memory::format::any);
    }

    supportedPrimitiveDescriptors.push_back({engine,
                                             {{inDims, getInputDataType(), memory::format::any}},
                                             {outFormat}, impl_desc_type::unknown});
}

void MKLDNNReshapeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    if (srcMem && dstMem) {
        if (srcMemPtr->GetSize() == srcMem->GetSize()) {
            srcPrim.reset(new mkldnn::reorder(srcMemPtr->GetPrimitive(), srcMem->GetPrimitive()));
        } else {
            // Autoblocking mode
            memory::dims dims = srcMem->GetDims();  // contains logical dims

            memory::desc src_d = srcMemPtr->GetPrimitive().get_primitive_desc().desc();
            void *src_data_hdl = srcMemPtr->GetPrimitive().get_data_handle();

            for (int i = 0; i < dims.size(); i++)
                src_d.data.dims[i] =  dims[i];

            memory::primitive_desc tmp_src_pd(src_d, supportedPrimitiveDescriptors[0].getEngine());
            src_blocked.reset(new memory(tmp_src_pd, src_data_hdl));

            srcPrim.reset(new mkldnn::reorder(*src_blocked, srcMem->GetPrimitive()));
        }

        if (dstMemPtr->GetSize() == dstMem->GetSize()) {
            dstPrim.reset(new mkldnn::reorder(dstMem->GetPrimitive(), dstMemPtr->GetPrimitive()));
        } else {
            // Autoblocking mode
            memory::dims dims = srcMem->GetDims();

            memory::desc dst_d = dstMemPtr->GetPrimitive().get_primitive_desc().desc();
            void *dst_data_hdl = dstMemPtr->GetPrimitive().get_data_handle();

            for (int i = 0; i < dims.size(); i++)
                dst_d.data.dims[i] =  dims[i];

            memory::primitive_desc tmp_dst_pd(dst_d, supportedPrimitiveDescriptors[0].getEngine());
            dst_blocked.reset(new memory(tmp_dst_pd, dst_data_hdl));

            dstPrim.reset(new mkldnn::reorder(*dst_blocked, dstMemPtr->GetPrimitive()));
        }
    }
}

void MKLDNNReshapeNode::execute(mkldnn::stream strm) {
    if (srcPrim && dstPrim) {
        if (src_blocked)
            src_blocked->set_data_handle(getParentEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
        if (dst_blocked)
            dst_blocked->set_data_handle(getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
        strm.submit({*srcPrim, *dstPrim});
    }
}

void MKLDNNReshapeNode::initEdges() {
    const MKLDNNPrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        return;

    memory::format sourceFormat;
    if (!getParentEdgeAt(0)->getParent()->getSelectedPrimitiveDescriptor())
        getParentEdgeAt(0)->getParent()->selectOptimalPrimitiveDescriptor();
    memory::format outFormat = getParentEdgeAt(0)->getParent()->getSelectedPrimitiveDescriptor()->getOutputDescs()[0].getFormat();
    memory::format inFormat = selected_pd->getInputDescs()[0].getFormat();

    if (!MKLDNNMemory::formatEquals(outFormat, inFormat) && inFormat != memory::any &&
            inFormat != memory::format_undef && outFormat != memory::any && outFormat != memory::format_undef) {
        THROW_IE_EXCEPTION << "Nodes have primitive descriptors with different formats";
    } else {
        sourceFormat = (inFormat != memory::any && inFormat != memory::format_undef) ? inFormat : outFormat;
    }

    if (!MKLDNNMemory::IsPlainFormat(sourceFormat)) {
        inPlace = false;
    } else {
        for (size_t cIdx = 0 ; cIdx < getChildEdges().size(); cIdx++) {
            if (getChildEdgeAt(cIdx)->getChild()->getType() == Concatenation) {
                inPlace = false;
                break;
            }
        }
    }

    if (!inPlace) {
        MKLDNNNode::initEdges();
    } else {
        getParentEdgeAt(0)->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        for (size_t cIdx = 0 ; cIdx < getChildEdges().size(); cIdx++) {
            getChildEdgeAt(cIdx)->sharedMemFrom(cIdx ? getChildEdgeAt(0) : getParentEdgeAt(0));
        }
    }
}

void MKLDNNReshapeNode::resolveNotAllocatedEdges() {
    if (!inPlace) {
        MKLDNNNode::resolveNotAllocatedEdges();

        auto dims = getParentEdgeAt(0)->getDims();

        srcMem.reset(new MKLDNNMemory(getSelectedPrimitiveDescriptor()->getEngine()));
        srcMem->Create(dims, inputDataType, MKLDNNMemory::GetPlainFormat(dims));

        dstMem.reset(new MKLDNNMemory(getSelectedPrimitiveDescriptor()->getEngine()));
        dstMem->Create(getChildEdgeAt(0)->getDims(), outputDataType,
                       MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims()), srcMem->GetData());
    } else {
        for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
            auto child = getChildEdgeAt(cIdx);
            if (child->getStatus() != MKLDNNEdge::Status::NotAllocated) {
                THROW_IE_EXCEPTION << "Reshape unexpected in-place behaviour!";
            }
            if (cIdx) {
                child->getMemoryPtr() = getChildEdgeAt(0)->getMemoryPtr();
            } else {
                void *data = (static_cast<float *>(child->getMemory().GetData()) +
                        child->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding);
                child->getMemoryPtr().reset(new MKLDNNMemory(getSelectedPrimitiveDescriptor()->getEngine()));
                child->getMemoryPtr()->Create(child->getDims(), outputDataType,
                                              MKLDNNMemory::GetPlainFormat(child->getDims()), data);
                child->changeStatus(MKLDNNEdge::Status::Allocated);
            }
        }
    }
}

bool MKLDNNReshapeNode::created() {
    return getType() == Reshape || getType() == Flatten;
}
