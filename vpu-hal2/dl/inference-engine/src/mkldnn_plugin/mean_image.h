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

#include "inference_engine.hpp"
#include "mkldnn_dims.h"
#include <vector>
#include <limits>

namespace MKLDNNPlugin {

class MeanImage {
public:
    MeanImage();

public:
    void Load(const MKLDNNDims& inputDims, InferenceEngine::InputInfo::Ptr inputInfo);
    void Subtract(const MKLDNNDims &inputDims, float *input);

    template<typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
    void Subtract(const MKLDNNDims &inputDims, T *input) {
        IE_ASSERT(input != nullptr);

        if (inputDims.ndims() != 4) {
            THROW_IE_EXCEPTION << "Expecting input as 4 dimension blob with format NxCxHxW.";
        }

        int MB = inputDims[0];
        int srcSize = inputDims.size() / MB;

        if (meanBuffer && meanBuffer->size()) {
            const float * meanBufferValues = meanBuffer->readOnly();
        #   pragma omp parallel for collapse(2) schedule(static)
            for (int mb = 0; mb < MB; mb++) {
                for (int i = 0; i < srcSize; i++) {
                    int buf = input[srcSize * mb + i];
                    buf -= meanBufferValues[i];
                    if (buf < std::numeric_limits<T>::min()) buf = std::numeric_limits<T>::min();
                    if (buf > std::numeric_limits<T>::max()) buf = std::numeric_limits<T>::max();
                    input[srcSize * mb + i] = buf;
                }
            }
        } else if (!meanValues.empty()) {
            int C = inputDims[1];
            srcSize /= inputDims[1];

        #   pragma omp parallel for collapse(3) schedule(static)
            for (int mb = 0; mb < MB; mb++) {
                for (int c = 0; c < C; c++) {
                    for (int i = 0; i < srcSize; i++) {
                        int buf = input[srcSize * mb * C + c * srcSize + i];
                        buf -= meanValues[c];
                        if (buf < std::numeric_limits<T>::min()) buf = std::numeric_limits<T>::min();
                        if (buf > std::numeric_limits<T>::max()) buf = std::numeric_limits<T>::max();
                        input[srcSize * mb * C + c * srcSize + i] = buf;
                    }
                }
            }
        }
    }

private:
    std::vector<float> meanValues;

    InferenceEngine::TBlob<float>::Ptr meanBuffer;
};

}  // namespace MKLDNNPlugin
