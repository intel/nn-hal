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
#include "mkldnn_pooling_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPoolingNode::MKLDNNPoolingNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNPoolingNode::MKLDNNPoolingNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNPoolingNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (descs.size())
        return;

    PoolingLayer* cnnLayer = dynamic_cast<PoolingLayer*>(getCnnLayer().get());
    if (cnnLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert pooling layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    auto type = cnnLayer->_type;

    algorithm alg;
    if (type == PoolingLayer::PoolType::AVG) {
        if (!cnnLayer->_exclude_pad && (cnnLayer->_padding_y != 0 || cnnLayer->_padding_x != 0))
            alg = pooling_avg_include_padding;
        else
            alg = pooling_avg_exclude_padding;
    } else if (type == PoolingLayer::PoolType::MAX) {
        alg = pooling_max;
    } else {
        // TODO: Handle rest of the possible: STOCH, ROI, SPACIAL_PYRAMID
        THROW_IE_EXCEPTION << "Unsupported pooling type";
    }

    std::vector<int> stride =
            {static_cast<int>(cnnLayer->_stride_y), static_cast<int>(cnnLayer->_stride_x)};
    std::vector<int> paddingL =
            {static_cast<int>(cnnLayer->_padding_y), static_cast<int>(cnnLayer->_padding_x)};
    std::vector<int> kernel =
            {static_cast<int>(cnnLayer->_kernel_y), static_cast<int>(cnnLayer->_kernel_x)};

    auto parentDims = getParentEdgeAt(0)->getDims();
    auto childDims = getChildEdgeAt(0)->getDims();
    if (parentDims.ndims() != 4)
        THROW_IE_EXCEPTION << "Pooling layer. Unsupported mode, only 4D blob as input.";

    int calculated_y = (getParentEdgeAt(0)->getDims()[2] - cnnLayer->_kernel_y
                        + cnnLayer->_padding_y*2) / cnnLayer->_stride_y + 1;
    int calculated_x = (getParentEdgeAt(0)->getDims()[3]  - cnnLayer->_kernel_x
                        + cnnLayer->_padding_x*2) / cnnLayer->_stride_x + 1;
    int shift_pad_y = (getChildEdgeAt(0)->getDims()[2] - calculated_y) * cnnLayer->_stride_y;
    int shift_pad_x = (getChildEdgeAt(0)->getDims()[3] - calculated_x) * cnnLayer->_stride_x;


    std::vector<int> paddingR = {static_cast<int>(cnnLayer->_padding_y) + shift_pad_y,
                                 static_cast<int>(cnnLayer->_padding_x) + shift_pad_x};

    // It doesn't support any format
    for (auto format : getAvailableFormatsForDims(parentDims)) {
        MKLDNNDims blk_in_dims = autoBlockingDims(parentDims, format);
        MKLDNNDims blk_out_dims = autoBlockingDims(childDims, format);

        memory::desc in_candidate{blk_in_dims, inputDataType, format};
        memory::desc out_candidate{blk_out_dims, outputDataType, format};

        std::shared_ptr<pooling_forward::desc> desc_ptr(
                new pooling_forward::desc(prop_kind::forward_scoring, alg,
                        in_candidate, out_candidate,
                        stride, kernel, paddingL, paddingR,
                        mkldnn::padding_kind::zero));

        if (alg == pooling_avg_include_padding && (shift_pad_y || shift_pad_y)) {
            // In case of AVG including padings the norm coeff should be calculated
            // with tacking into account original pads. So we need to restore
            // original values (R_padding = L_padding).
            //
            // WA. Because mkldnn uses different formula to calculate AVG norm coeff
            //     in compare with Caffe. In mkldnn coeff is always 1/(KH*KW)
            for (int i = 0; i < paddingL.size(); i++) desc_ptr->data.padding[1][i] = paddingL[i];
        }

        descs.push_back(MKLDNNDescriptor(desc_ptr));
    }
}

void MKLDNNPoolingNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<pooling_forward::primitive_desc, pooling_forward::desc>();

    prim.reset(new pooling_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                   getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNPoolingNode::created() {
    return getType() == Pooling;
}
