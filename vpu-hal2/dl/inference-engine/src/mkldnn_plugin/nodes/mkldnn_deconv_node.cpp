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
#include "mkldnn_deconv_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(Type type, const std::string &name) : MKLDNNNode(type, name) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!withBiases)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
    srcMemDesc = [&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.diff_src_primitive_desc(idx).desc());
    };
    dstMemDesc = [&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.diff_dst_primitive_desc(idx).desc());
    };
}
MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    srcMemDesc = [&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.diff_src_primitive_desc(idx).desc());
    };
    dstMemDesc = [&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.diff_dst_primitive_desc(idx).desc());
    };
}

void MKLDNNDeconvolutionNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (!descs_fwd.empty() && !descs_bwd.empty())
        return;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    auto * deconvLayer = dynamic_cast<DeconvolutionLayer*>(getCnnLayer().get());
    if (deconvLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert deconvolution layer.";
    if (deconvLayer->_weights == nullptr) {
        THROW_IE_EXCEPTION << "Weights are empty for layer: " << deconvLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use ReadWeights and SetWeights methods of InferenceEngine::CNNNetReader"
                           << " to load them from .bin part of the IR";
    }
    bool withGroups = (deconvLayer->_group > 1);
    bool isDW = withGroups && deconvLayer->_group == deconvLayer->_out_depth &&
            deconvLayer->_group == deconvLayer->input()->getDims()[1];
    withBiases = (deconvLayer->_biases != nullptr && deconvLayer->_biases->size() != 0);
    if (withBiases)
        biases = deconvLayer->_biases;

    /* Original layout format for deconv weights is iohw (from Caffe).
     * We specify oihw, but mean iohw, because there are no more
     * suitable format in MKLDNN.
     */
    SizeVector weightDims;
    if (withGroups) {
        weightDims = {
                deconvLayer->_group,
                deconvLayer->_out_depth / deconvLayer->_group,
                deconvLayer->input()->getDims()[1] / deconvLayer->_group,
                deconvLayer->_kernel_y,
                deconvLayer->_kernel_x
        };
    } else {
        weightDims = {
                deconvLayer->input()->getDims()[1],
                deconvLayer->_out_depth,
                deconvLayer->_kernel_y,
                deconvLayer->_kernel_x
        };
    }

    internalBlobs.push_back(createInternalBlob(weightDims, true));

    std::vector<int> stride =
            {static_cast<int>(deconvLayer->_stride_y), static_cast<int>(deconvLayer->_stride_x)};
    std::vector<int> paddingL =
            {static_cast<int>(deconvLayer->_padding_y), static_cast<int>(deconvLayer->_padding_x)};
    std::vector<int> dilation =
            {static_cast<int>(deconvLayer->_dilation_y) - 1, static_cast<int>(deconvLayer->_dilation_x) - 1};
    std::vector<int> paddingR = {0, 0};

    MKLDNNDims weightsDims = MKLDNNDims(weightDims);

    for (int i = 0; i < 2; i++) {
        int with_group = (withGroups) ? 1 : 0;
        int krn = weightsDims[with_group + 2 + i];
        int src = getChildEdgeAt(0)->getDims()[2 + i];
        int dst = getParentEdgeAt(0)->getDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }

    int O_IND = withGroups ? 1 : 0;
    int I_IND = withGroups ? 2 : 1;

    auto try_add_pd = [&] (memory::format in_fmt, memory::format out_fmt) {
        MKLDNNDims i_dims = getParentEdgeAt(0)->getDims();
        MKLDNNDims o_dims = getChildEdgeAt(0)->getDims();

        // grouping and autoblicking is not compatible
        if ((withGroups && !isDW) &&
            ((autoBlockingDims(i_dims, in_fmt) != i_dims) ||
             (autoBlockingDims(o_dims, in_fmt) != o_dims)))
            return;

        memory::desc in_candidate{autoBlockingDims(getParentEdgeAt(0)->getDims(), in_fmt),
                                  inputDataType, in_fmt};
        memory::desc out_candidate{autoBlockingDims(getChildEdgeAt(0)->getDims(), out_fmt),
                                   outputDataType, out_fmt};

        MKLDNNDims blocked_weightDims(weightDims);

        if (!withGroups) {
            if (in_fmt == memory::nChw16c) {
                blocked_weightDims[I_IND] = rnd_up(blocked_weightDims[I_IND], 16);
            } else if (in_fmt == memory::nChw8c) {
                blocked_weightDims[I_IND] = rnd_up(blocked_weightDims[I_IND], 8);
            }

            if (out_fmt == memory::nChw16c) {
                blocked_weightDims[O_IND] = rnd_up(blocked_weightDims[O_IND], 16);
            } else if (out_fmt == memory::nChw8c) {
                blocked_weightDims[O_IND] = rnd_up(blocked_weightDims[O_IND], 8);
            }
        } else if (isDW) {
            if (out_fmt != in_fmt)
                return;

            if (in_fmt == memory::nChw16c) {
                blocked_weightDims[0] = rnd_up(blocked_weightDims[0], 16);
            } else if (in_fmt == memory::nChw8c) {
                blocked_weightDims[0] = rnd_up(blocked_weightDims[0], 8);
            }
        }

        memory::desc wgh_candidate{blocked_weightDims, inputDataType, memory::any};

        std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
        conv_desc.reset(new convolution_forward::desc(prop_kind::forward_inference, algorithm::convolution_direct,
                                                      out_candidate, wgh_candidate, in_candidate, stride, dilation,
                                                 paddingL, paddingR, padding_kind::zero));

        std::shared_ptr<mkldnn::convolution_backward_data::desc> deconv_desc;
        deconv_desc.reset(new convolution_backward_data::desc(algorithm::convolution_direct, out_candidate, wgh_candidate,
                                                       in_candidate, stride, dilation, paddingL, paddingR,
                                                       padding_kind::zero));
        descs_fwd.push_back(conv_desc);
        descs_bwd.push_back(deconv_desc);
    };

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        try_add_pd(format, format);
    }
}

void MKLDNNDeconvolutionNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!descs.empty())
        return;

    for (size_t i = 0; i < descs_bwd.size(); i++) {
        descs.push_back({descs_bwd[i],
                    std::shared_ptr<convolution_forward::primitive_desc>(
                                 new convolution_forward::primitive_desc(*descs_fwd[i], engine))});
    }
    MKLDNNNode::initSupportedPrimitiveDescriptors(engine);
}

void MKLDNNDeconvolutionNode::execute(mkldnn::stream strm) {
    if (prim) {
        strm.submit({*prim});
    }
    if (withBiases) {
        const auto *bias = biases->buffer().as<const float*>();

        auto& dst = getChildEdgeAt(0)->getMemory();

        float *output = reinterpret_cast<float*>(dst.GetData()) +
                        dst.GetDescriptor().data.layout_desc.blocking.offset_padding;

        const auto &dims = dst.GetDims();

        const int N = dims[0];
        const int C = dims[1];
        const int H = dims[2];
        const int W = dims[3];

        memory::format fmt = getSelectedPrimitiveDescriptor()->getInputDescs().data()->getFormat();

        const int blksize = fmt == memory::nChw16c ? 16 :
                            fmt == memory::nChw8c ? 8 : 1;

#   pragma omp parallel for collapse(4) schedule(static)
        for (int n = 0; n < N; ++n)
        for (int c = 0; c < C / blksize; ++c)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            const int off =
                    n * C * H * W + c * H * W * blksize +
                    h * W * blksize + w * blksize;
            auto o = &output[off];
            for (int bc = 0; bc < blksize; ++bc) {
                o[bc] += bias[c*blksize + bc];
            }
        }
    }
}

bool MKLDNNDeconvolutionNode::created() {
    return getType() == Deconvolution;
}

void MKLDNNDeconvolutionNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<convolution_backward_data::primitive_desc,
            convolution_backward_data::desc, convolution_forward::primitive_desc>();

    prim.reset(new convolution_backward_data(prim_desc,
            getParentEdgeAt(0)->getMemory().GetPrimitive(),
            internalBlobMemory[0]->GetPrimitive(),
            getChildEdgeAt(0)->getMemory().GetPrimitive()));
}
