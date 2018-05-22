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
#include "mkldnn_batchnorm_node.h"
#include "mkldnn_scaleshift_node.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNBatchNormalizationNode::MKLDNNBatchNormalizationNode(Type type, const std::string &name):
        MKLDNNNode(type, name) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return GetVarianceDesc(primitive_desc_it.fetch());
    });
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return GetMeanDesc(primitive_desc_it.fetch());
    });
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!fusedWithScale())
            return MKLDNNMemoryDesc();
        return GetScaleShiftWeightsDesc(primitive_desc_it.fetch());
    });
}
MKLDNNBatchNormalizationNode::MKLDNNBatchNormalizationNode(InferenceEngine::CNNLayerPtr layer): MKLDNNNode(layer) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return GetVarianceDesc(primitive_desc_it.fetch());
    });
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return GetMeanDesc(primitive_desc_it.fetch());
    });

    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!fusedWithScale())
            return MKLDNNMemoryDesc();
        return GetScaleShiftWeightsDesc(primitive_desc_it.fetch());
    });
}

void MKLDNNBatchNormalizationNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (!descs.empty())
        return;
    auto * bnLayer = dynamic_cast<BatchNormalizationLayer*>(getCnnLayer().get());
    if (bnLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert batch normalization layer.";
    if (bnLayer->_weights == nullptr || bnLayer->_biases == nullptr) {
        THROW_IE_EXCEPTION << "Weights/biases are empty for layer: " << bnLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use ReadWeights and SetWeights methods of InferenceEngine::CNNNetReader"
                           << " to load them from .bin part of the IR";
    }

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    float eps = bnLayer->epsilon;

    size_t variancesSize = MKLDNNDims(bnLayer->_weights->dims()).size();
    size_t meansSize = MKLDNNDims(bnLayer->_biases->dims()).size();

    if (variancesSize != meansSize && variancesSize != 1)
        THROW_IE_EXCEPTION << "Incorrect weights and biases sizes!";

    internalBlobs.push_back(createInternalBlob(bnLayer->_weights->dims(), true));
    internalBlobs.push_back(createInternalBlob(bnLayer->_biases->dims(), false));

    auto parentOutDims = getParentEdgeAt(0)->getDims();

    if (fusedWith.size() > 1)
        THROW_IE_EXCEPTION << "BatchNorm fusion is possible with only one layer!";

    for (const auto &node : fusedWith) {
        auto * scshLayer = dynamic_cast<ScaleShiftLayer*>(node->getCnnLayer().get());
        if (scshLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast to the ScaleShift layer to fuse with BatchNorm.";

        size_t C = static_cast<size_t>(getChildEdgeAt(0)->getDims()[1]);
        SizeVector mkldnn_weights = {2, C};
        TensorDesc desc(scshLayer->_weights->precision(), mkldnn_weights, InferenceEngine::NC);
        InferenceEngine::TBlob<float>::Ptr internalBlob = InferenceEngine::make_shared_blob<float>(desc);
        internalBlob->allocate();
        float * data = internalBlob->buffer();

        InferenceEngine::Blob::Ptr blb = scshLayer->_weights;
        if (blb == nullptr)
            THROW_IE_EXCEPTION << "Cannot get weights blob for node " << getName() << ".";

        size_t weightsByteSize = blb->byteSize();
        memcpy(data, blb->buffer(), weightsByteSize);
        data += blb->size();
        blb = scshLayer->_biases;

        if (blb == nullptr) {
            memset(data, 0, weightsByteSize);
        } else {
            if (weightsByteSize != blb->byteSize())
                THROW_IE_EXCEPTION << "ScaleShift has incorrect weights!";
            memcpy(data, blb->buffer(), weightsByteSize);
        }
        internalBlobs.push_back(internalBlob);
    }

    for (auto format : getAvailableFormatsForDims(parentOutDims)) {
        memory::desc in_candidate{autoBlockingDims(parentOutDims, format), inputDataType, format};

        if (parentOutDims.ndims() == 2) {
            // Make it 4D
            MKLDNNDims dims = parentOutDims;
            dims.push_back(1);  // H
            dims.push_back(1);  // W
            format = memory::nchw;
            in_candidate = {dims, inputDataType, format};
        }

        unsigned flag = mkldnn_use_global_stats;
        if (fusedWithScale())
            flag |= mkldnn_use_scaleshift;
        MKLDNNDescriptor desc(std::shared_ptr<batch_normalization_forward::desc>(
                new batch_normalization_forward::desc(prop_kind::forward_scoring, in_candidate, eps,
                                                      flag)));
        descs.push_back(desc);
    }
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetVarianceDesc(const memory::primitive_desc &primitive_desc) const {
    memory::primitive_desc aprimitive_desc;
    mkldnn_primitive_desc_t bndesc;
    mkldnn_batch_normalization_desc_t *p;
    error::wrap_c_api(mkldnn_primitive_desc_query(
            primitive_desc.get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                      "could not get a batch-normalization descriptor");
    const_mkldnn_primitive_desc_t const_bndesc =
            (p->flags & use_global_stats) ?
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                                  mkldnn::convert_to_c(src_pd), 2) :
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                                  mkldnn::convert_to_c(dst_pd), 2);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                                                         const_bndesc),
                      "could not clone a variance primitive descriptor");
    aprimitive_desc.reset(bndesc);
    return MKLDNNMemoryDesc(aprimitive_desc.desc());
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetMeanDesc(const memory::primitive_desc &primitive_desc) const {
    memory::primitive_desc aprimitive_desc;
    mkldnn_primitive_desc_t bndesc;
    mkldnn_batch_normalization_desc_t *p;
    error::wrap_c_api(mkldnn_primitive_desc_query(
            primitive_desc.get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                      "could not get a batch-normalization descriptor");
    const_mkldnn_primitive_desc_t const_bndesc =
            (p->flags & use_global_stats) ?
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                                  mkldnn::convert_to_c(src_pd), 1) :
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                                  mkldnn::convert_to_c(dst_pd), 1);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                                                         const_bndesc),
                      "could not clone a mean primitive descriptor");
    aprimitive_desc.reset(bndesc);
    return MKLDNNMemoryDesc(aprimitive_desc.desc());
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetScaleShiftWeightsDesc(const memory::primitive_desc &primitive_desc) const {
    memory::primitive_desc adesc;
    mkldnn_primitive_desc_t bndesc;
    const_mkldnn_primitive_desc_t const_bndesc =
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                           mkldnn::convert_to_c(weights_pd), 0);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                                                  const_bndesc),
                      "could not clone a weights primitive descriptor");
    adesc.reset(bndesc);
    return MKLDNNMemoryDesc(adesc.desc());
}

bool MKLDNNBatchNormalizationNode::created() {
    return getType() == BatchNormalization;
}

void MKLDNNBatchNormalizationNode::createPrimitive() {
    if (prim)
        return;

    if (fusedWithScale()) {
        const size_t channel = getChildEdgeAt(0)->getDims()[1];
        size_t blk_channel = getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.block_dims[1];
        if (channel % blk_channel) {
            // Weights mem for the ScaleShift do not fit blocking layout and needs padding
            blk_channel = ((size_t) ((channel + blk_channel - 1) / blk_channel)) * blk_channel;
            const SizeVector mkldnn_weights = {blk_channel, 2};
            InferenceEngine::TBlob<float>::Ptr internalBlob =
                    InferenceEngine::make_shared_blob<float>(internalBlobs[2]->precision(), InferenceEngine::NC,
                                                             mkldnn_weights);
            internalBlob->allocate();
            float *data = internalBlob->buffer();
            memset(data, 0, internalBlob->byteSize());
            const float *realData = internalBlobs[2]->buffer();
            memcpy(data, realData, channel * sizeof(float));
            memcpy(data + blk_channel, realData + channel, channel * sizeof(float));

            // Replace the original weights blob for ScaleShift with the padded
            internalBlobs[2] = internalBlob;
        }
        auto prim_desc = createPrimitiveDescriptor<batch_normalization_forward::primitive_desc,
                batch_normalization_forward::desc>();
        prim.reset(new batch_normalization_forward(prim_desc,
                                                   getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[1]->GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[0]->GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[2]->GetPrimitive(),
                                                   getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }  else {
        auto prim_desc = createPrimitiveDescriptor<batch_normalization_forward::primitive_desc,
                batch_normalization_forward::desc>();
        prim.reset(new batch_normalization_forward(prim_desc,
                                                   getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[1]->GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[0]->GetPrimitive(),
                                                   getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}
