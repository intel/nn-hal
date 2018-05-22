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
#include "mkldnn_crop_node.h"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNCropNode::MKLDNNCropNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNCropNode::MKLDNNCropNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNCropNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    CropLayer* cropLayer = dynamic_cast<CropLayer*>(getCnnLayer().get());

    if (cropLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert crop layer.";

    channelAxis = 1;
    if (getParentEdges().size() != 1) {
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    }

    MKLDNNDims childDims = getChildEdgeAt(0)->getDims();

    offsets.resize(static_cast<size_t>(childDims.ndims()));  // plus one dim for batch
    dims.resize(static_cast<size_t>(childDims.ndims()));  // plus one dim for batch
    for (int i = 0; i < childDims.ndims(); i++)
        dims[i] = childDims[i];

    for (int i = 0; i < cropLayer->axis.size(); i++) {
        offsets[cropLayer->axis[i]] = cropLayer->offset[i];
        dims[cropLayer->axis[i]] = cropLayer->dim[i];
    }

    if (cropLayer->axis.size() == dims.size()) {
        for (size_t i = 0; i < cropLayer->axis.size(); i++) {
            if (cropLayer->axis[i] == 1) {
                channelAxis = static_cast<int>(i);
                break;
            }
        }
    }

    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
}

void MKLDNNCropNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto& inDims = getParentEdgeAt(0)->getDims();
    if (inDims.ndims() != 4) {
        THROW_IE_EXCEPTION << "Crop supports only 4d blobs.";
    }

    if (channelAxis >= 0 && dims[channelAxis] % 8) {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::format::nchw}},
                                                 {{getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::nchw}},
                                                 impl_desc_type::unknown});
    } else {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::format::any}},
                                                 {{getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::format::any}},
                                                 impl_desc_type::unknown});
    }
}

void MKLDNNCropNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNCropNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();

    int m_block_size = 1;
    if (!MKLDNNMemory::IsPlainFormat(parentMem.GetFormat())) {
        m_block_size = parentMem.GetDescriptor().data.layout_desc.blocking.block_dims[1];
    }
    int m_inner_dim = dims[dims.size() - 1] * m_block_size;

    const memory &dst_d = getChildEdgeAt(0)->getMemory().GetPrimitive();

    int dst_ndims = dst_d.get_primitive_desc().desc().data.ndims;

    // TODO: Rewrite it in general case. For every tensor
    // and rank, without using letter N,C,H,W
    int OFFSET_N = (dst_ndims > 0) ? offsets[0] : 0;
    int OFFSET_C = (dst_ndims > 1) ? offsets[1] : 0;
    int OFFSET_H = (dst_ndims > 2) ? offsets[2] : 0;
    int OFFSET_W = (dst_ndims > 3) ? offsets[3] : 0;

    // TODO: Check applicability of dyn_batch_lim in early steps.
    //       crop of batch dimension doesn't support dyn batch.
    const int ON = (dst_ndims  > 0) ? batchToProcess(dims[0]) : 1;
    const int OC = (dst_ndims  > 1) ? dims[1] : 1;
    const int OH = (dst_ndims  > 2) ? dims[2] : 1;
    const int OW = (dst_ndims  > 3) ? dims[3] : 1;

    memory::dims src_dims = parentMem.GetDims();
    int src_ndims = static_cast<int>(src_dims.size());

    const int IC = (src_ndims  > 1) ? src_dims[1] : 1;
    const int IH = (src_ndims  > 2) ? src_dims[2] : 1;
    const int IW = (src_ndims  > 3) ? src_dims[3] : 1;

    const auto *src_data = reinterpret_cast<const float*>(parentMem.GetData()) +
            parentMem.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_data = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

#   pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < ON; ++n) {
        for (int c = 0; c < OC; c += m_block_size) {
            for (int h = 0; h < OH; ++h) {
                int dst_ind =
                        n*OC*OH*OW + c*OH*OW +
                        h*OW*m_block_size;

                int src_ind =
                        (n+OFFSET_N)*IC*IH*IW +
                        (c+OFFSET_C)*IH*IW +
                        (h+OFFSET_H)*IW*m_block_size +
                        OFFSET_W*m_block_size;

                memcpy(dst_data + dst_ind, src_data + src_ind, m_inner_dim * sizeof(float));
            }
        }
    }
}

bool MKLDNNCropNode::created() {
    return getType() == Crop;
}
