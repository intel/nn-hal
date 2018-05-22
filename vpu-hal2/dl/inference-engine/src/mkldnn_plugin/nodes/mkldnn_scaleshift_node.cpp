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
#include "mkldnn_scaleshift_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNScaleShiftNode::MKLDNNScaleShiftNode(Type type, const std::string &name) : MKLDNNNode(type, name) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
}
MKLDNNScaleShiftNode::MKLDNNScaleShiftNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {
    internalBlobDesc.push_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
}

void MKLDNNScaleShiftNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    auto *scshLayer = dynamic_cast<ScaleShiftLayer*>(getCnnLayer().get());
    if (scshLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert ScaleShift layer.";
    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
    if (getParentEdgeAt(0)->getDims() != getChildEdgeAt(0)->getDims())
        THROW_IE_EXCEPTION << "Incorrect output shapes. Different from input.";

    with_add = scshLayer->_biases != nullptr;
    with_mul = scshLayer->_weights != nullptr;
    weights = scshLayer->_weights;
    biases = scshLayer->_biases;
    broad_cast = static_cast<bool>(scshLayer->_broadcast);
}

void MKLDNNScaleShiftNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNScaleShiftNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    MKLDNNDims dims = getParentEdgeAt(0)->getDims();

    auto pd_same = [&] (memory::format fmt) {
        MKLDNNDims blk_dims = autoBlockingDims(dims, fmt);
        MKLDNNDims scale = autoBlockingDims(dims, memory::nc);

        supportedPrimitiveDescriptors.push_back({
            engine,
            {{blk_dims, getInputDataType(), fmt}},
            {{blk_dims, getOutputDataType(), fmt}},
            impl_desc_type::unknown
        });
    };
    std::vector<memory::format> fmts = getAvailableFormatsForDims(dims);
    for (auto fmt : fmts) pd_same(fmt);
}

template <bool MUL, bool ADD, memory::format fmt>
void MKLDNNScaleShiftNode::execute_impl() {
    auto& src = getParentEdgeAt(0)->getMemory();
    auto& dst = getChildEdgeAt(0)->getMemory();

    const auto &dims = src.GetDims();

    const float *input = reinterpret_cast<const float*>(src.GetData()) +
            src.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *output = reinterpret_cast<float*>(dst.GetData()) +
            dst.GetDescriptor().data.layout_desc.blocking.offset_padding;

    const float *scales, *shifts;
    scales = MUL ? weights->buffer().as<float*>() : nullptr;
    shifts = ADD ? biases->buffer().as<float*>()  : nullptr;

    constexpr int blksize = fmt == memory::nChw16c ? 16 :
                            fmt == memory::nChw8c ? 8 : 1;

    auto ker = [&](const float *i, float *o, int C) {
#       pragma omp simd
        for (int c = 0; c < blksize; ++c) {
            if (MUL && ADD)
                o[c] = i[c]*scales[C*blksize+c] + shifts[C*blksize+c];
            else if (MUL)
                o[c] = i[c]*scales[C*blksize+c];
            else if (ADD)
                o[c] = i[c] + shifts[C*blksize+c];
        }
    };

    const int N = dims.size() > 0 ? dims[0] : 1;
    const int C = dims.size() > 1 ? dims[1] : 1;
    const int H = dims.size() > 2 ? dims[2] : 1;
    const int W = dims.size() > 3 ? dims[3] : 1;

#   pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C / blksize; ++c)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w) {
        const int off =
            n*C*H*W + c*H*W*blksize +
            h*W*blksize + w*blksize;
        auto i = &input[off];
        auto o = &output[off];
        ker(i, o, c);
    }
}

void MKLDNNScaleShiftNode::execute(mkldnn::stream strm) {
    memory::format spd_fmt = getChildEdgeAt(0)->getMemory().GetFormat();

#define ADD_SCSH_IMPL(_mul, _add, _fmt) if (spd_fmt == (_fmt)) execute_impl<_mul, _add, _fmt>()

    if (with_mul && with_add) {
        ADD_SCSH_IMPL(true, true, memory::nChw16c);
        ADD_SCSH_IMPL(true, true, memory::nChw8c);
        ADD_SCSH_IMPL(true, true, memory::nchw);
        ADD_SCSH_IMPL(true, true, memory::nc);
    } else if (with_mul) {
        ADD_SCSH_IMPL(true, false, memory::nChw16c);
        ADD_SCSH_IMPL(true, false, memory::nChw8c);
        ADD_SCSH_IMPL(true, false, memory::nchw);
        ADD_SCSH_IMPL(true, false, memory::nc);
    } else if (with_add) {
        ADD_SCSH_IMPL(false, true, memory::nChw16c);
        ADD_SCSH_IMPL(false, true, memory::nChw8c);
        ADD_SCSH_IMPL(false, true, memory::nchw);
        ADD_SCSH_IMPL(false, true, memory::nc);
    }
#undef ADD_SCSH_IMPL
}

bool MKLDNNScaleShiftNode::created() {
    return getType() == ScaleShift;
}
