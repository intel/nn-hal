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

#include "mean_image.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MeanImage::MeanImage() : meanBuffer(nullptr) {
}

void MeanImage::Load(const MKLDNNDims& inputDims, InputInfo::Ptr inputInfo) {
    PreProcessInfo &pp = inputInfo->getPreProcess();
    size_t inChannels = pp.getNumberOfChannels();
    if (inChannels == 0) {
        meanBuffer = nullptr;
        return;
    }

    if (inChannels != inputDims[1]) {
        THROW_IE_EXCEPTION << "channels mismatch between mean and input";
    }

    ResponseDesc resp;

    switch (pp.getMeanVariant()) {
        case MEAN_VALUE: {
            // mean image common value per channel (1x1xC)
            meanValues.resize(inChannels);

            for (unsigned channel = 0; channel < inChannels; channel++) {
                meanValues[channel] = pp[channel]->meanValue;
            }
        }
        break;
        case MEAN_IMAGE: {
            // since MKLDNN expects all channels in the same buffer - we copy it here as it comes from different channels...
            auto meanWidth = pp[0]->meanData->dims()[0];
            auto meanHeight = pp[0]->meanData->dims()[1];


            meanBuffer = make_shared_blob<float>(Precision::FP32, CHW, { meanWidth, meanHeight, inChannels });

            meanBuffer->allocate();

            for (unsigned channel = 0; channel < inChannels; channel++) {
                Blob::Ptr meanBlob = pp[channel]->meanData;
                if (!meanBlob || meanBlob->precision() != Precision::FP32)
                    THROW_IE_EXCEPTION << "mean image not provided or not in Float 32";
                if (meanBlob->size() != meanHeight*meanWidth) {
                    THROW_IE_EXCEPTION << "mean image size does not match expected network input, expecting " << meanWidth << " x " << meanHeight;
                }
                // todo: cast to TBlob and make sure it is floats
                memcpy(meanBuffer->data() + channel*meanBlob->size(), meanBlob->buffer(), meanBlob->byteSize());
            }
        }
            break;

        case NONE: {
            // there is no mean image. So disable mean image step
            meanBuffer = nullptr;
        }
            break;

        default: {
            THROW_IE_EXCEPTION << "Unsupported mean variant: " << pp.getMeanVariant();
        }
    }
}

void MeanImage::Subtract(const MKLDNNDims &inputDims, float *input) {
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
                input[srcSize * mb + i] -= meanBufferValues[i];
            }
        }
    } else if (!meanValues.empty()) {
        int C = inputDims[1];
        srcSize /= inputDims[1];

#   pragma omp parallel for collapse(3) schedule(static)
        for (int mb = 0; mb < MB; mb++) {
            for (int c = 0; c < C; c++) {
                for (int i = 0; i < srcSize; i++) {
                    input[srcSize * mb * C + c * srcSize + i] -= meanValues[c];
                }
            }
        }
    }
}
