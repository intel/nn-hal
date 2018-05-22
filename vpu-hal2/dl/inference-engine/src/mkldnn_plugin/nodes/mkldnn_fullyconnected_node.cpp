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
#include "mkldnn_fullyconnected_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <vector>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNFullyConnectedNode::MKLDNNFullyConnectedNode(Type type, const std::string &name) : MKLDNNNode(type, name) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (internalBlobs.size() <= 1)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}
MKLDNNFullyConnectedNode::MKLDNNFullyConnectedNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (internalBlobs.size() <= 1)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}

void MKLDNNFullyConnectedNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (descs.size())
        return;
    FullyConnectedLayer* fcLayer = dynamic_cast<FullyConnectedLayer*>(getCnnLayer().get());
    if (fcLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert fully connected layer.";
    if (fcLayer->_weights == nullptr) {
        THROW_IE_EXCEPTION << "Weights are empty for layer: " << fcLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use ReadWeights and SetWeights methods of InferenceEngine::CNNNetReader"
                           << " to load them from .bin part of the IR";
    }

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getParentEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    MKLDNNDims inDims(fcLayer->input()->getDims());

    SizeVector weightsDims;

    if (inDims.ndims() == 2) {
        weightsDims = {fcLayer->_out_num, static_cast<size_t>(inDims.size(1))};
    } else if (inDims.ndims() == 4) {
        weightsDims = {fcLayer->_out_num, static_cast<size_t>(inDims[1]), static_cast<size_t>(inDims[2]),
                       static_cast<size_t>(inDims[3])};
    } else {
        THROW_IE_EXCEPTION << "Unsupported source format for FC layer. Expected 4 or 2, got: "
                           << inDims.ndims() << " dims.";
    }

    internalBlobs.push_back(createInternalBlob(weightsDims, true));

    SizeVector biasesDims;
    bool withBiases = (fcLayer->_biases != nullptr && fcLayer->_biases->size() != 0);
    if (withBiases) {
        biasesDims.push_back(static_cast<int>(fcLayer->_out_num));
        internalBlobs.push_back(createInternalBlob(biasesDims, false));
    }

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        MKLDNNDims blk_in_dims = autoBlockingDims(inDims, format);

        memory::desc in_candidate{blk_in_dims, inputDataType, format};
        memory::desc out_candidate{getChildEdgeAt(0)->getDims(), outputDataType, memory::any};

        memory::format weights_fmt = weightsFormatForSrcFormat(format);

        memory::desc wgh_candidate{autoBlockingDims(MKLDNNDims(weightsDims), weights_fmt), inputDataType, weights_fmt};
        memory::desc bias_candidate{MKLDNNDims(biasesDims), inputDataType, memory::any};

        if (withBiases) {
            MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                    new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                    bias_candidate, out_candidate)));
            descs.push_back(desc);
        } else {
            MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                    new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                    out_candidate)));
            descs.push_back(desc);
        }
    }
}

void MKLDNNFullyConnectedNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<inner_product_forward::primitive_desc, inner_product_forward::desc>();

    if (internalBlobs.size() > 1) {
        prim.reset(new inner_product_forward(prim_desc,
                                             getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                             internalBlobMemory[0]->GetPrimitive(),
                                             internalBlobMemory[1]->GetPrimitive(),
                                             getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new inner_product_forward(prim_desc,
                                             getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                             internalBlobMemory[0]->GetPrimitive(),
                                             getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

bool MKLDNNFullyConnectedNode::created() {
    return getType() == FullyConnected;
}

void MKLDNNFullyConnectedNode::selectOptimalPrimitiveDescriptor() {
    static const std::vector<impl_desc_type> fcPriority = {
            impl_desc_type::unknown,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm,
            impl_desc_type::ref,
    };
    MKLDNNNode::selectPreferPrimitiveDescriptor(fcPriority);
}

memory::format MKLDNNFullyConnectedNode::weightsFormatForSrcFormat(memory::format sourceFormat) {
    switch (sourceFormat) {
        case memory::format::x:
            return memory::format::x;
        case memory::format::nc:
            return memory::format::oi;
        case memory::format::nchw:
            return memory::format::oihw;
        case memory::format::nChw8c:
            return memory::format::oIhw8i;
        case memory::format::nChw16c:
            return memory::format::oIhw16i;
        default:
            THROW_IE_EXCEPTION << "Unsupported source format for node " << getName();
    }
}
