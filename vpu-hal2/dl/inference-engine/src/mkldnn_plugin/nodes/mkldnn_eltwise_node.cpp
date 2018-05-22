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
#include "mkldnn_eltwise_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNEltwiseNode::MKLDNNEltwiseNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNEltwiseNode::MKLDNNEltwiseNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

bool MKLDNNEltwiseNode::isSum() {
    EltwiseLayer* eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());
    return eltwiseLayer->_operation == EltwiseLayer::Sum;
}

bool MKLDNNEltwiseNode::isUnitScales() {
    EltwiseLayer* eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());

    if (eltwiseLayer->coeff.size() == 0)
        return true;

    for (auto scale : eltwiseLayer->coeff) {
        if (scale != 1.0f)
            return false;
    }

    return true;
}

void MKLDNNEltwiseNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    EltwiseLayer* eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());

    if (eltwiseLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert eltwise layer.";
    op = eltwiseLayer->_operation;

    if (!getParentEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    auto outDims = getParentEdgeAt(0)->getDims();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        auto oDims = getParentEdgeAt(i)->getDims();
        if (outDims.size() != oDims.size() || outDims.ndims() != oDims.ndims())
            THROW_IE_EXCEPTION << "Dimensions of input layers don't equal";
    }

    bool with_coeffs = eltwiseLayer->coeff.size() != 0;
    if (op != EltwiseLayer::Sum && with_coeffs)
        THROW_IE_EXCEPTION << "Only sum operation supports operands coefficients";

    if (with_coeffs && eltwiseLayer->coeff.size() != getParentEdges().size())
        THROW_IE_EXCEPTION << "Number of provided coefficients is not equal to number of operands";

    sum_scales.clear();
    for (int i = 0; i < getParentEdges().size(); i++)
        sum_scales.push_back(with_coeffs ? eltwiseLayer->coeff[i] : 1.0f);
}

void MKLDNNEltwiseNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto same = [&] (memory::format fmt) -> MKLDNNPrimitiveDescInfo {
        MKLDNNMemoryDesc dstFormat {autoBlockingDims(getChildEdgeAt(0)->getDims(), fmt), getOutputDataType(), fmt};
        std::vector<MKLDNNMemoryDesc> srcFormats;
        for (size_t i = 0; i < getParentEdges().size(); i++)
            srcFormats.push_back({autoBlockingDims(getParentEdgeAt(i)->getDims(), fmt), getInputDataType(), fmt});
        return {engine, srcFormats, {dstFormat}, impl_desc_type::ref};
    };

    MKLDNNDims in_dims = getChildEdgeAt(0)->getDims();

    if (in_dims.ndims() == 4) {
        supportedPrimitiveDescriptors.push_back(same(memory::format::nchw));
        supportedPrimitiveDescriptors.push_back(same(memory::format::nChw8c));
        supportedPrimitiveDescriptors.push_back(same(memory::format::nChw16c));
    } else {
        memory::format pln_fmt = MKLDNNMemory::GetPlainFormat(in_dims);
        pln_fmt = memory::format::any;
        supportedPrimitiveDescriptors.push_back(same(pln_fmt));
    }
}

void MKLDNNEltwiseNode::createPrimitive() {
    if (prim)
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
            THROW_IE_EXCEPTION << "Source memory from " << parent->getName() << " didn't allocate.";
        }

        if (op == EltwiseLayer::Sum) {
            srcs_pd.push_back(srcMemPtr->GetPrimitiveDescriptor());
            srcs_p.push_back(srcMemPtr->GetPrimitive());
        }
    }
    if (op == EltwiseLayer::Sum) {
        auto primitive_desc = sum::primitive_desc(dstMemPtr->GetDescriptor(), sum_scales, srcs_pd);
        prim = std::shared_ptr<sum>(new sum(primitive_desc, srcs_p, dstMemPtr->GetPrimitive()));
    }
}

void MKLDNNEltwiseNode::execute(mkldnn::stream strm) {
    if (op == EltwiseLayer::Sum) {
        strm.submit({*prim});
    } else {
        IE_ASSERT(getParentEdges().size() > 1);

        auto& srcMemory0 = getParentEdgeAt(0)->getMemory();
        auto& srcMemory1 = getParentEdgeAt(1)->getMemory();
        const float *src0_ptr = reinterpret_cast<const float*>(srcMemory0.GetData()) +
                srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
        const float *src1_ptr = reinterpret_cast<const float*>(srcMemory1.GetData()) +
                srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
        float *dst_ptr = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
                getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
        const size_t data_size = srcMemory0.GetSize() / sizeof(float);

        if (op == EltwiseLayer::Prod) {
            #pragma omp parallel for
            for (int i = 0; i < data_size; i++)
                dst_ptr[i] = src0_ptr[i] * src1_ptr[i];

            for (int j = 2; j < getParentEdges().size(); j++) {
                const float *src_ptr = reinterpret_cast<const float *>(getParentEdgeAt(j)->getMemory().GetData()) +
                        getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

                #pragma omp parallel for
                for (int i = 0; i < data_size; i++)
                    dst_ptr[i] = dst_ptr[i] * src_ptr[i];
            }
        } else if (op == EltwiseLayer::Max)  {
            #pragma omp parallel for
            for (int i = 0; i < data_size; i++)
                dst_ptr[i] = std::max(src0_ptr[i], src1_ptr[i]);

            for (int j = 2; j < getParentEdges().size(); j++) {
                const float *src_ptr = reinterpret_cast<const float*>(getParentEdgeAt(j)->getMemory().GetData()) +
                        getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

                #pragma omp parallel for
                for (int i = 0; i < data_size; i++)
                    dst_ptr[i] = std::max(dst_ptr[i], src_ptr[i]);
            }
        }
    }
}

bool MKLDNNEltwiseNode::created() {
    return getType() == Eltwise;
}

bool MKLDNNEltwiseNode::initAsInPlace() {
    size_t inPlaceWithParent = getParentEdges().size();
    for (size_t i = 0; i < inPlaceWithParent; i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge->getParent()->getChildEdges().size() == 1 &&
                (parentEdge->getStatus() == MKLDNNEdge::Status::NeedAllocation ||
                        parentEdge->getStatus() == MKLDNNEdge::Status::Uninitialized)) {
            inPlaceWithParent = i;
            break;
        }
    }
    // This is WA for MKLDNN implementation
    if (inPlaceWithParent != 0)
        return false;
    MKLDNNDims dims = getParentEdgeAt(0)->getDims();
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        if (getChildEdgeAt(cIdx)->getDims() != dims) {
            return false;
        }
    }


    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        auto childEdge = getChildEdgeAt(cIdx);
        if (childEdge->getStatus() != MKLDNNEdge::Status::NeedAllocation &&
            childEdge->getStatus() != MKLDNNEdge::Status::Uninitialized)
            return false;
    }
    for (size_t pIdx = 0; pIdx < getParentEdges().size(); pIdx++) {
        auto parentEdge = getParentEdgeAt(pIdx);
        parentEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
    }
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        auto childEdge = getChildEdgeAt(cIdx);
        childEdge->sharedMemFrom(getParentEdgeAt(inPlaceWithParent));
    }

    return true;
}
