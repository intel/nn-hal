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
#include "mkldnn_conv_node.h"
#include "mkldnn_reorder_node.h"
#include "mkldnn_input_node.h"
#include "mkldnn_activation_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConvolutionNode::MKLDNNConvolutionNode(Type type, const std::string &name) : MKLDNNNode(type, name),
                                                                                   withBiases(false) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!withBiases)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}
MKLDNNConvolutionNode::MKLDNNConvolutionNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer),
                                                                                   withBiases(false) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!withBiases)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}

void MKLDNNConvolutionNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (!descs.empty())
        return;

    auto * convLayer = dynamic_cast<ConvolutionLayer*>(getCnnLayer().get());
    if (convLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert convolution layer.";

    if (getParentEdges().size() != 1 &&
        ((getType() != Convolution_Sum && getType() != Convolution_Sum_Activation) || getParentEdges().size() != 2))
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    if (getParentEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << "Convolution layer. Unsupported mode, only 4D blob as input.";
    }

    bool isMerged = (!getMergeWith().empty());  // grouped convolution was constructed from split->concat subgraph
    bool isGrouped = convLayer->_group != 1;    // group info available from IR
    if (isMerged && isGrouped)
        THROW_IE_EXCEPTION << "Convolution initialization. Group splitted mode are used together with direct group specification.";

    // default values. Can be replaced in next steps
    size_t groupNum = convLayer->_group;
    size_t groupIC = convLayer->input()->getDims()[1];
    size_t groupOC = convLayer->_out_depth;

    bool isDW = groupNum == groupOC && groupNum == groupIC;

    if (isMerged) {
        groupNum = getMergeWith().size() + 1;
    }
    if (isGrouped) {
        groupIC /= groupNum;
        groupOC /= groupNum;
    }

    SizeVector weightDims = { groupOC, groupIC, convLayer->_kernel_y, convLayer->_kernel_x};
    SizeVector biasesDims = { groupOC * groupNum};

    if (isGrouped || isMerged) weightDims.insert(weightDims.begin(), groupNum);

    withBiases = (convLayer->_biases != nullptr && convLayer->_biases->size() != 0);

    internalBlobs.push_back(createInternalBlob(weightDims, true));
    if (withBiases) {
        internalBlobs.push_back(createInternalBlob(biasesDims, false));
    }

    std::vector<int> stride =
            {static_cast<int>(convLayer->_stride_y), static_cast<int>(convLayer->_stride_x)};
    std::vector<int> dilation =
            {static_cast<int>(convLayer->_dilation_y) - 1, static_cast<int>(convLayer->_dilation_x) - 1};
    std::vector<int> paddingL =
            {static_cast<int>(convLayer->_padding_y), static_cast<int>(convLayer->_padding_x)};
    std::vector<int> paddingR = {0, 0};

    MKLDNNDims weightsDims = MKLDNNDims(weightDims);

    for (int i = 0; i < 2; i++) {
        int with_group = (isGrouped || isMerged) ? 1 : 0;
        int krn = weightsDims[with_group + 2 + i];
        int src = getParentEdgeAt(0)->getDims()[2 + i];
        int dst = getChildEdgeAt(0)->getDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }

    withActivation = getType() == Convolution_Activation || getType() == Convolution_Sum_Activation;
    withSum = getType() == Convolution_Sum || getType() == Convolution_Sum_Activation;

    int O_IND = (isGrouped || isMerged) ? 1 : 0;
    int I_IND = (isGrouped || isMerged) ? 2 : 1;

    auto try_add_pd = [&] (memory::format in_fmt, memory::format out_fmt) {
        MKLDNNDims i_dims = getParentEdgeAt(0)->getDims();
        MKLDNNDims o_dims = getChildEdgeAt(0)->getDims();

        // grouping and autoblicking is not compatible
        if (((isGrouped && !isDW) || isMerged) &&
            ((autoBlockingDims(i_dims, in_fmt) != i_dims) ||
             (autoBlockingDims(o_dims, in_fmt) != o_dims)))
            return;

        memory::desc in_candidate{autoBlockingDims(getParentEdgeAt(0)->getDims(), in_fmt),
                inputDataType, in_fmt};
        memory::desc out_candidate{autoBlockingDims(getChildEdgeAt(0)->getDims(), out_fmt),
                outputDataType, out_fmt};

        MKLDNNDims blocked_weightDims(weightDims);
        MKLDNNDims blocked_biasesDims(biasesDims);

        if (!isGrouped && !isMerged) {
            if (in_fmt == memory::nChw16c) {
                blocked_weightDims[I_IND] = rnd_up(blocked_weightDims[I_IND], 16);
            } else if (in_fmt == memory::nChw8c) {
                blocked_weightDims[I_IND] = rnd_up(blocked_weightDims[I_IND], 8);
            }

            if (out_fmt == memory::nChw16c) {
                blocked_weightDims[O_IND] = rnd_up(blocked_weightDims[O_IND], 16);
                blocked_biasesDims[0] = rnd_up(blocked_biasesDims[0], 16);
            } else if (out_fmt == memory::nChw8c) {
                blocked_weightDims[O_IND] = rnd_up(blocked_weightDims[O_IND], 8);
                blocked_biasesDims[0] = rnd_up(blocked_biasesDims[0], 8);
            }
        } else if (isDW) {
            if (out_fmt != in_fmt)
                return;

            if (in_fmt == memory::nChw16c) {
                blocked_weightDims[0] = rnd_up(blocked_weightDims[0], 16);
                blocked_biasesDims[0] = rnd_up(blocked_biasesDims[0], 16);
            } else if (in_fmt == memory::nChw8c) {
                blocked_weightDims[0] = rnd_up(blocked_weightDims[0], 8);
                blocked_biasesDims[0] = rnd_up(blocked_biasesDims[0], 8);
            }
        }

        memory::desc wgh_candidate{blocked_weightDims, inputDataType, memory::any};

        std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
        if (withBiases) {
            memory::desc bias_candidate{blocked_biasesDims, inputDataType, memory::any};

            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, algorithm::convolution_direct,
                                                          in_candidate, wgh_candidate, bias_candidate, out_candidate,
                                                          stride, dilation, paddingL, paddingR, padding_kind::zero));
        } else {
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, algorithm::convolution_direct,
                                                          in_candidate, wgh_candidate, out_candidate, stride, dilation,
                                                          paddingL, paddingR, padding_kind::zero));
        }
        descs.push_back(MKLDNNDescriptor(conv_desc));
    };

    try_add_pd(memory::nchw, memory::nchw);
    if (groupIC == 3) {
        // reorder + nchw->nChwXc are faster with channel equal 3 than nChwXc->nChwXc
        try_add_pd(memory::nchw, memory::nChw16c);
        try_add_pd(memory::nchw, memory::nChw8c);
    } else {
        try_add_pd(memory::nChw16c, memory::nChw16c);
        try_add_pd(memory::nChw8c, memory::nChw8c);
    }
}


void MKLDNNConvolutionNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    mkldnn::post_ops ops;
    if (withSum) ops.append_sum(1.0);
    if (withActivation) {
        for (auto &node : fusedWith) {
            auto * activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
            if (!activationNode)
                continue;
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(),
                               activationNode->getBeta());
        }
    }

    mkldnn::primitive_attr attr;
    attr.set_post_ops(ops);

    for (auto& desc : descs) {
        try {
            primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, attr);
            do {
                std::vector<MKLDNNMemoryDesc> srcDescs;
                for (size_t i = 0; i < desc.inputNumbers() && srcMemDesc; i++)
                    srcDescs.push_back(srcMemDesc(itpd, i));

                std::vector<MKLDNNMemoryDesc> intDescs;
                for (size_t i = 0; i < internalBlobDesc.size(); i++)
                    intDescs.push_back(internalBlobDesc[i](itpd, 0));

                std::vector<MKLDNNMemoryDesc> dstDescs;
                for (size_t i = 0; i < desc.outputNumbers() && dstMemDesc; i++)
                    dstDescs.push_back(dstMemDesc(itpd, i));
                impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str().c_str());

                supportedPrimitiveDescriptors.push_back({engine, srcDescs, dstDescs, intDescs, impl_type});
            } while (itpd.next());
        } catch (std::exception e) {
            // it throw exception in case of no implementation found
            continue;
        }
    }
}


void MKLDNNConvolutionNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::post_ops ops;
    if (withSum) ops.append_sum(1.0);
    if (withActivation) {
        for (auto &node : fusedWith) {
            auto * activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
            if (!activationNode)
                continue;

            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(),
                               activationNode->getBeta());
        }
    }

    mkldnn::primitive_attr attr;
    attr.set_post_ops(ops);

    auto prim_desc = createPrimitiveDescriptor<convolution_forward::primitive_desc,
            convolution_forward::desc>(attr);

    if (internalBlobMemory.size() > 1) {
        prim.reset(new convolution_forward(prim_desc,
                                           getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                           internalBlobMemory[0]->GetPrimitive(),
                                           internalBlobMemory[1]->GetPrimitive(),
                                           getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new convolution_forward(prim_desc,
                                           getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                           internalBlobMemory[0]->GetPrimitive(),
                                           getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

void MKLDNNConvolutionNode::initEdges() {
    if (getType() != Convolution_Sum_Activation && getType() != Convolution_Sum) {
        MKLDNNNode::initEdges();
        return;
    }

    for (size_t pIdx = 0 ; pIdx < getParentEdges().size(); pIdx++) {
        getParentEdgeAt(pIdx)->changeStatus(MKLDNNEdge::Status::NeedAllocation);
    }
    for (size_t cIdx = 0 ; cIdx < getChildEdges().size(); cIdx++) {
        getChildEdgeAt(cIdx)->sharedMemFrom(getParentEdgeAt(1));
    }
}

bool MKLDNNConvolutionNode::created() {
    return getType() == Convolution || getType() == Convolution_Sum_Activation ||
           getType() == Convolution_Activation || getType() == Convolution_Sum;
}
