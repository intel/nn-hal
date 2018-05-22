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
#include "mkldnn_tile_node.h"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNTileNode::MKLDNNTileNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNTileNode::MKLDNNTileNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNTileNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    auto * tileLayer = dynamic_cast<TileLayer*>(getCnnLayer().get());

    if (tileLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert tile layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    axis = tileLayer->axis;
    tiles = tileLayer->tiles;
}

void MKLDNNTileNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto& inDims = getParentEdgeAt(0)->getDims();
    if (inDims.ndims() == 2) {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::format::nc}},
                                                 {{getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::nc}},
                                                 impl_desc_type::unknown});
    } else if (inDims.ndims() == 4) {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::format::nchw}},
                                                 {{getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::nchw}},
                                                 impl_desc_type::unknown});
    } else {
        THROW_IE_EXCEPTION << "Tile " << getName() << " supports only 2d and 4d dimensions!";
    }
}

void MKLDNNTileNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
}

void MKLDNNTileNode::execute(mkldnn::stream strm) {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    const float *src_ptr = reinterpret_cast<const float*>(srcMemory.GetData()) +
            srcMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_ptr = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    int m_inner_dim = 1;
    int m_outer_dim = 1;
    memory::dims inDims = srcMemory.GetDims();
    for (int i=0; i < axis; i++ ) m_outer_dim *= inDims[i];
    for (int i=axis; i < inDims.size(); i++ ) m_inner_dim *= inDims[i];

    if (m_inner_dim == 1 && inDims.size() == 4 && m_outer_dim%8 == 0 && srcMemory.GetFormat() == memory::nChw8c) {
        /*
         * We may enable tile processing directly to appropriate output format (nChw8c)
         */
        m_inner_dim *= 8;
        m_outer_dim /= 8;
    } else if (m_inner_dim == 1 && inDims.size() == 4 && m_outer_dim%16 == 0
               && srcMemory.GetFormat() == memory::nChw16c) {
        /*
         * We may enable tile processing directly to appropriate output format (nChw16c)
         */
        m_inner_dim *= 16;
        m_outer_dim /= 16;
    }

    for (int i = 0; i < m_outer_dim; ++i) {
        for (int t = 0; t < tiles; ++t) {
            memcpy(dst_ptr, src_ptr, m_inner_dim* sizeof(float));
            dst_ptr += m_inner_dim;
        }
        src_ptr += m_inner_dim;
    }
}

bool MKLDNNTileNode::created() {
    return getType() == Tile;
}
