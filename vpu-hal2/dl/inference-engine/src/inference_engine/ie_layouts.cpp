/*
 * INTEL CONFIDENTIAL
 * Copyright 2016 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */
#include <map>

#include "ie_layouts.h"
#include <algorithm>

using namespace InferenceEngine;

static const std::map<Layout, SizeVector> DIM_POSITIONS = {
    { NCHW, { I_W, I_H, I_C, I_N } },
    { NHWC, { I_C, I_W, I_H, I_N } }
};

LayoutOffsetCounter::LayoutOffsetCounter(Layout layout, SizeVector dims) : _layout(layout), _dims(dims), _dims_count(dims.size()), _muls(dims.size(), -1) {
    size_t mul = 1;
    for (size_t i = 0; i < _dims_count; i++) {
        size_t index = DIM_POSITIONS.at(layout)[i];
        _muls[index] = mul;
        mul *= dims[index];
    }
}

/**
 * @brief Calculates offset for specified layout
 * @param pos Tensor position array (reverse NCHW order as in the IR: w,h,c,n)
 */
size_t LayoutOffsetCounter::Offset(SizeVector pos) {
    size_t res = 0;
    for (size_t i = 0; i < _dims_count; i++) {
        res += pos[i] * _muls[i];
    }

    return res;
}

TensorDesc::TensorDesc(const Precision& precision, SizeVector dims, Layout layout): blockingDesc(dims, layout),
                                                                                    precision(precision) {
                                                                                            
    this->dims = dims;
    this->layout = layout;
}

TensorDesc::TensorDesc(const Precision& precision, Layout layout): blockingDesc(), precision(precision) {
    this->layout = layout;
}

TensorDesc::TensorDesc(const Precision &precision, SizeVector dims, const BlockingDesc &blockDesc)
        : blockingDesc(blockDesc), precision(precision)  {
    this->dims = dims;
    this->layout = Layout::BLOCKED;
}
TensorDesc::TensorDesc() {}

void TensorDesc::setDims(const SizeVector &dims) {
    this->dims = dims;
    if (layout == Layout::BLOCKED) {
        // TODO: Prepare new blocked dims
        blockingDesc = BlockingDesc(blockingDesc.getBlockDims(), blockingDesc.getOrder());
    } else {
        blockingDesc = BlockingDesc(dims, layout);
    }
}


bool TensorDesc::operator==(const TensorDesc &rhs) const {
    return blockingDesc == rhs.blockingDesc &&
            layout == rhs.layout &&
            dims == rhs.dims;
}

bool TensorDesc::operator!=(const TensorDesc &rhs) const {
    return !(*this == rhs);
}

Layout TensorDesc::getLayoutByDims(SizeVector dims) {
    switch (dims.size()) {
        case 1:
            return Layout::C;
        case 2:
            return Layout::NC;
        case 3:
            return Layout::CHW;
        case 4:
            return Layout::NCHW;
        default:
            return Layout::BLOCKED;
    }
}

size_t TensorDesc::offset(SizeVector v) const {
    if (layout == Layout::ANY)
        THROW_IE_EXCEPTION << "Cannot calculate offset for any format!";

    const SizeVector& blockedDims = blockingDesc.getBlockDims();
    const SizeVector& strides = blockingDesc.getStrides();
    const SizeVector& order = blockingDesc.getOrder();

    size_t n_blocked_dims = order.size();
    if (blockedDims.size() != n_blocked_dims || strides.size() != n_blocked_dims) {
        THROW_IE_EXCEPTION << "Cannot calculate offset. Incorrect primitive descriptor!";
    }
    SizeVector blockedShift(n_blocked_dims);
    for (size_t i = 1; i <= n_blocked_dims; i++) {
        blockedShift[n_blocked_dims - i] = v[order[n_blocked_dims - i]] % blockedDims[n_blocked_dims - i];
        v[order[n_blocked_dims - i]] /= blockedDims[n_blocked_dims - i];
    }
    size_t offset = blockingDesc.getOffsetPadding();
    for (int d = 0; d < n_blocked_dims; ++d) {
        const size_t p = blockedShift[d] + blockingDesc.getOffsetPaddingToData()[d];
        offset += p * strides[d];
    }
    return offset;
}

size_t TensorDesc::offset(size_t l) const {
    size_t n_dims = dims.size();
    SizeVector pos(n_dims);
    for (int rd = 0; rd < n_dims; ++rd) {
        const size_t d = n_dims - 1 - rd;
        const size_t cur_dim = dims[d];
        pos[d] = l % cur_dim;
        l /= cur_dim;
    }
    return offset(pos);
}

void TensorDesc::reshape(const SizeVector &dims, Layout layout) {
    for (auto &padd : blockingDesc.getOffsetPaddingToData()) {
        if (padd)
            THROW_IE_EXCEPTION << "Cannot reshape a non-packaged blob!";
    }
    if (layout != Layout::ANY) {
        blockingDesc = BlockingDesc(dims, layout);
        this->layout = layout;
    } else {
        blockingDesc = BlockingDesc(dims, this->layout);
    }
    this->dims = dims;
}

void TensorDesc::reshape(const SizeVector &dims, const BlockingDesc &blockDesc) {
    blockingDesc = blockDesc;
    this->dims = dims;
    this->layout = Layout::BLOCKED;
}

BlockingDesc::BlockingDesc(const SizeVector& block_dims, const SizeVector & order): offsetPadding(0) {
    this->order = order;
    fillDesc(block_dims, order);
}

BlockingDesc::BlockingDesc(): BlockingDesc({}, Layout::ANY) {}

BlockingDesc::BlockingDesc(const SizeVector &blocked_dims, const SizeVector &order,
                           size_t offset): BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
}

BlockingDesc::BlockingDesc(const SizeVector &blocked_dims, const SizeVector &order, size_t offset,
                           SizeVector dimOffsets): BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
    if (blocked_dims.size() != dimOffsets.size())
        THROW_IE_EXCEPTION << "Offsets are not initialized for all dimensions.";
    this->offsetPaddingToData = dimOffsets;
}

BlockingDesc::BlockingDesc(const SizeVector &blocked_dims, const SizeVector &order, size_t offset,
                           SizeVector dimOffsets, SizeVector strides): BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
    if (blocked_dims.size() != strides.size())
        THROW_IE_EXCEPTION << "Strides are not initialized for all dimensions.";
    this->strides = strides;
    if (blocked_dims.size() != dimOffsets.size())
        THROW_IE_EXCEPTION << "Offsets are not initialized for all dimensions.";
    this->offsetPaddingToData = dimOffsets;
}

BlockingDesc::BlockingDesc(const SizeVector& dims, Layout layout): offsetPadding(0) {
    if (dims.empty())
        return;

    offsetPadding = 0;
    switch (layout) {
        case Layout::ANY:break;
        case Layout::C:
            if (dims.size() != 1)
                THROW_IE_EXCEPTION << "Dims and format are inconsistent.";
            order = {0};
            fillDesc(dims, order);
            break;
        case Layout::OIHW:
        case Layout::NCHW:
            if (dims.size() != 4)
                THROW_IE_EXCEPTION << "Dims and format are inconsistent.";
            order = {0, 1, 2, 3};
            fillDesc(dims, order);
            break;
        case Layout::NHWC:
            if (dims.size() != 4)
                THROW_IE_EXCEPTION << "Dims and format are inconsistent.";
            order = {0, 2, 3, 1};
            fillDesc(dims, order);
            break;
        case Layout::CHW:
            if (dims.size() != 3)
                THROW_IE_EXCEPTION << "Dims and format are inconsistent.";
            order = {0, 1, 2};
            fillDesc(dims, order);
            break;
        case Layout::CN:
            if (dims.size() != 2)
                THROW_IE_EXCEPTION << "Dims and format are inconsistent.";
            order = {1, 0};
            fillDesc(dims, order);
            break;
        case Layout::NC:
        case Layout::HW:
            if (dims.size() != 2)
                THROW_IE_EXCEPTION << "Dims size " << dims.size() <<"and format NC are inconsistent.";
            order = {0, 1};
            fillDesc(dims, order);
            break;
        case Layout::BLOCKED:
            order.clear();
            for (size_t i = 0; i < dims.size(); i++)
                order.push_back(i);
            fillDesc(dims, order);
    }
}

void BlockingDesc::fillDesc(const SizeVector& blocked_dims, const SizeVector &order) {
    if (order.size() != blocked_dims.size())
        THROW_IE_EXCEPTION << "Cannot fill descriptor. Incorrect arguments.";
    this->order = order;
    this->blockedDims = blocked_dims;
    offsetPadding = 0;
    offsetPaddingToData.resize(order.size());
    strides.resize(order.size());
    strides[strides.size() - 1] = 1;
    offsetPaddingToData[offsetPaddingToData.size() - 1] = 0;
    for (size_t i = 2; i <= order.size(); i++) {
        offsetPaddingToData[offsetPaddingToData.size() - i] = 0;
        strides[strides.size() - i] = strides[strides.size() - (i - 1)] * blocked_dims[blocked_dims.size() - (i - 1)];
    }

    offsetPadding = 0;
}

bool BlockingDesc::operator==(const BlockingDesc &rhs) const {
    return blockedDims == rhs.blockedDims &&
           strides == rhs.strides &&
           offsetPaddingToData == rhs.offsetPaddingToData &&
           order == rhs.order &&
           offsetPadding == rhs.offsetPadding;
}

bool BlockingDesc::operator!=(const BlockingDesc &rhs) const {
    return !(*this == rhs);
}
