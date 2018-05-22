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
#include "mkldnn_permute_node.h"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPermuteNode::MKLDNNPermuteNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNPermuteNode::MKLDNNPermuteNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNPermuteNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    auto& layer = getCnnLayer();
    if (!layer) {
        THROW_IE_EXCEPTION << "Cannot get CNNLayer.";
    }

    order.clear();
    std::vector<int> layerOrder = layer->GetParamAsInts("order");
    for (auto ord : layerOrder)
        order.push_back(static_cast<size_t>(ord));
}

void MKLDNNPermuteNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    if (getParentEdgeAt(0)->getDims().ndims() == 4) {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::nchw}},
                                                 {{getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::nchw}},
                                                 impl_desc_type::unknown});

        auto srcDims = getParentEdgeAt(0)->getDims();

        if (srcDims[1] % 8 == 0) {
            supportedPrimitiveDescriptors.push_back({engine,
                                                     {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::nChw8c}},
                                                     {{getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::nchw}},
                                                     impl_desc_type::unknown});
        }

        if (srcDims[1] % 16 == 0) {
            supportedPrimitiveDescriptors.push_back({engine,
                                                     {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::nChw16c}},
                                                     {{getChildEdgeAt(0)->getDims(), getOutputDataType(), memory::nchw}},
                                                     impl_desc_type::unknown});
        }
    } else {
        supportedPrimitiveDescriptors.push_back({engine,
                                                 {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::any}},
                                                 {{getChildEdgeAt(0)->getDims(), getOutputDataType(),
                                                     MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims())}},
                                                 impl_desc_type::unknown});
    }
}

void MKLDNNPermuteNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNPermuteNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    float *src_data = reinterpret_cast<float *>(srcMemPtr->GetData()) +
                      srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_data = reinterpret_cast<float *>(dstMemPtr->GetData()) +
                      dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    if (order[0] == 0 && order[1] == 2 && order[2] == 3 && order[3] == 1) {
        // Supports only NCHW to NHWC
        int block_size = 1;
        if (!MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat())) {
            block_size = srcMemPtr->GetDescriptor().data.layout_desc.blocking.block_dims[1];
        }

        const int MB = batchToProcess(srcMemPtr->GetDims()[0]);
        const int C = srcMemPtr->GetDims()[1];
        const int H = srcMemPtr->GetDims()[2];
        const int W = srcMemPtr->GetDims()[3];

        // NHWC
        const int src_stride = H * W * block_size;

        int src_off = 0;
        int dst_off = 0;

        for (int n = 0; n < MB; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    src_off = n * C * H * W + (h * W + w) * block_size;

                    for (int c = 0; c < C; c += block_size) {
                        for (int b = 0; b < block_size; b++) {
                            dst_data[dst_off] = src_data[src_off + b];
                            dst_off++;
                        }

                        src_off += src_stride;
                    }
                }
            }
        }
    } else {
        auto srcBlob = getParentEdgeAt(0)->getMemory().GetBlob();
        TensorDesc srcDesc = srcBlob->getTensorDesc();

        SizeVector& dims = srcDesc.getDims();
        InferenceEngine::SizeVector orderedDims;
        for (auto ord : order) {
            orderedDims.push_back(dims[ord]);
        }
        TensorDesc dstDesc(InferenceEngine::Precision::FP32, dims, {orderedDims, order});

        size_t dataSize = srcBlob->size();
#pragma omp parallel for
        for (size_t i = 0; i < dataSize; i++) {
            dst_data[dstDesc.offset(i)] = src_data[srcDesc.offset(i)];
        }
    }
}

bool MKLDNNPermuteNode::created() {
    return getType() == Permute;
}
