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
#include "mkldnn_reorder_node.h"
#include <string>
#include <algorithm>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNReorderNode::MKLDNNReorderNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNReorderNode::MKLDNNReorderNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNReorderNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
}

void MKLDNNReorderNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto parent = getParentEdgeAt(0)->getParent();
    auto child = getChildEdgeAt(0)->getChild();

    if (input && output) {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {input},
                                                 {output},
                                                 impl_desc_type::reorder});
    } else if (parent->getSelectedPrimitiveDescriptor() != nullptr &&
               child->getSelectedPrimitiveDescriptor() != nullptr) {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {parent->getSelectedPrimitiveDescriptor()->getOutputDescs()[0]},
                                                 {child->getSelectedPrimitiveDescriptor()->getInputDescs()[0]},
                                                 impl_desc_type::reorder});
    } else {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), inputDataType, memory::format::any}},
                                                 {{getChildEdgeAt(0)->getDims(), outputDataType, memory::format::any}},
                                                 impl_desc_type::reorder});
    }
}

void MKLDNNReorderNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    if (srcMemPtr->GetSize() == dstMemPtr->GetSize()) {
        // No autoblocking. Reorder can be applied as is
        prim.reset(new mkldnn::reorder(srcMemPtr->GetPrimitive(), dstMemPtr->GetPrimitive()));
    } else {
        // Autoblocking case. nchw<=>nChw8c are only supported, but memory descriptor
        // should be with strides. Prepare it from enlarged blob
        memory::dims dims = srcMemPtr->GetDims();
        memory::dims dims_dst = dstMemPtr->GetDims();

        for (int i = 0; i < dims.size(); i++)  // min dims is a logical dims
            dims[i] = std::min(dims[i], dims_dst[i]);

        memory::desc src_d = srcMemPtr->GetPrimitive().get_primitive_desc().desc();
        void *src_data_hdl = srcMemPtr->GetPrimitive().get_data_handle();

        memory::desc dst_d = dstMemPtr->GetPrimitive().get_primitive_desc().desc();
        void *dst_data_hdl = dstMemPtr->GetPrimitive().get_data_handle();

        for (int i = 0; i < dims.size(); i++)
            src_d.data.dims[i] = dst_d.data.dims[i] = dims[i];

        memory::primitive_desc tmp_src_pd(src_d, supportedPrimitiveDescriptors[0].getEngine());
        src_blocked.reset(new memory(tmp_src_pd, src_data_hdl));

        memory::primitive_desc tmp_dst_pd(dst_d, supportedPrimitiveDescriptors[0].getEngine());
        dst_blocked.reset(new memory(tmp_dst_pd, dst_data_hdl));

        // output blob should be zeroed. NaN value can occur in untouched place.
        dstMemPtr->FillZero();

        prim.reset(new mkldnn::reorder(*src_blocked, *dst_blocked));
    }
}

void MKLDNNReorderNode::selectOptimalPrimitiveDescriptor() {
    if (getSupportedPrimitiveDescriptors().size()) {
        selectPrimitiveDescriptorByIndex(0);
    }
}

bool MKLDNNReorderNode::created() {
    return getType() == Reorder;
}

void MKLDNNReorderNode::execute(mkldnn::stream strm) {
    if (src_blocked)
        src_blocked->set_data_handle(getParentEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
    if (dst_blocked)
        dst_blocked->set_data_handle(getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
    MKLDNNNode::execute(strm);
}
