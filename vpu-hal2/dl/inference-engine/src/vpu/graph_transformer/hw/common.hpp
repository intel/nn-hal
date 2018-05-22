//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation.
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

#include <tuple>
#include <vector>
#include "graph_transformer_impl.hpp"

HwPaddingInfo getPadding(const VpuDims& inDims, const VpuDims& outDims,
                         uint32_t kernelDimX, uint32_t kernelDimY,
                         uint32_t kernelStrideX, uint32_t kernelStrideY);

using TileSoH = std::tuple<int, int, int, int, int, int, int, int>;

std::vector<TileSoH> heightSolution(int inputSize, int kernelSize, int stride,
                                    const std::tuple<int, int>& pad,
                                    int maxOutputLines);

std::vector<TileSoH> heightSolutionWithPooling(int inputSize, int kernelSize, int stride,
                                               int pad,
                                               int maxOutputLines);

// General form of hardware Parametrized ReLU (PRelu):
//                     / a0*x, if x < t0
// f(x; t0, a0, a1) = |
//                     \ a1*x, if x >= t0
bool isReluPostOp(const VpuStageHandle& postOp);

class HwWeightsWriter : public DataWriter {
public:
    HwWeightsWriter(const Blob::Ptr& blob,
                    const VpuDims& origWeightsDims,
                    const VpuDims& hwWeightsDims,
                    int numInputTiles = 1,
                    int inputTileInd = 0,
                    float scale = 1.0f)
        : _blob(blob),
          _origWeightsDims(origWeightsDims),
          _hwWeightsDims(hwWeightsDims),
          _numInputTiles(numInputTiles),
          _inputTileInd(inputTileInd),
          _scale(scale) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override;

    void write(void* dst) const override;

private:
    Blob::Ptr _blob;
    VpuDims _origWeightsDims;
    VpuDims _hwWeightsDims;
    int _numInputTiles;
    int _inputTileInd;
    float _scale;
};

class ScaledBiasesWriter : public DataWriter {
public:
    ScaledBiasesWriter(const Blob::Ptr& blob, float scale) : _blob(blob), _scale(scale) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override;

    void write(void* dst) const override;

private:
    Blob::Ptr _blob;
    float _scale;
};

uint32_t estimateHwBufferSize(const VpuDims& dims);
