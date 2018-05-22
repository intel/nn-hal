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
#pragma once

#include "perf_count.h"
#include <vector>
#include <utility>
#include <mkldnn_types.h>
#include <mkldnn.hpp>

namespace MKLDNNPlugin {

class MKLDNNDims {
public:
    MKLDNNDims() {
    }

    explicit MKLDNNDims(const InferenceEngine::SizeVector& size) {
        dims = std::vector<int>(size.begin(), size.end());
    }

    explicit MKLDNNDims(const std::vector<int>& dim) {
        dims = dim;
    }

    MKLDNNDims(const mkldnn_dims_t dnn_dims, int dnn_ndims) {
        dims = std::vector<int>(dnn_dims, dnn_dims + dnn_ndims);
    }

    explicit MKLDNNDims(std::initializer_list<int> ilist) : dims(ilist) {}

    InferenceEngine::SizeVector ToSizeVector() {
        InferenceEngine::SizeVector size;
        for (auto i : dims) {
            size.push_back(i);
        }

        return size;
    }

    int ndims() const {
        return dims.size();
    }

    int size() const {
        return size(0);
    }

    int size(int start) const {
        int size = 1;

        for (int i = start; i < dims.size(); i++) {
            size *= dims[i];
        }

        return size;
    }

    void insert(int at, int val) {
        dims.insert(dims.begin() + at, val);
    }

    void push_back(int val) {
        dims.push_back(val);
    }

    void swap(int from, int to) {
        int tmp = dims[from];
        dims[from] = dims[to];
        dims[to] = tmp;
    }

    operator mkldnn::memory::dims() const {
        return dims;
    }

    bool operator == (const MKLDNNDims& rhs) {
        if (dims.size() != rhs.dims.size()) {
            return false;
        }

        return std::equal(rhs.dims.begin(), rhs.dims.end(), dims.begin());
    }

    bool operator != (const MKLDNNDims& rhs) {
        return !(*this == rhs);
    }

    int& operator[](int idx) {
        return dims[idx];
    }

    int operator[](int idx) const {
        return dims[idx];
    }

private:
    std::vector<int> dims;
};

}  // namespace MKLDNNPlugin
