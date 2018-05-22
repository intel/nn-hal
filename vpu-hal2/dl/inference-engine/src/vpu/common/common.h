//
// INTEL CONFIDENTIAL
// Copyright 2017 Intel Corporation.
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

#include <vector>
#include <memory>
#include <string>
#include <map>

#include "environment.h"
#include "GraphInfo.h"
#include "ie_common.h"
#include "ie_layouts.h"

// TODO: move implementation to a separate cpp?
namespace VPU {
namespace Common {

void GetPerformanceCounts(const std::vector<BlobMetaData> &blobMetaData,
                          const std::shared_ptr<Common::GraphInfo<float>> &graphInfo,
                          std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap,
                          const bool printReceiveTensorTime = false) {
    perfMap.clear();
    int graphElementsCount = graphInfo->numElements();

    graphElementsCount -= 1;  // Last element for fathom thread execution time

    if (!printReceiveTensorTime) {
        graphElementsCount -= 1;  // Do not use element which contain InputReceive time
    }

    unsigned timeIndex = 0;
    unsigned execIndex = 1;
    for (auto currentMetaData = blobMetaData.begin();
         currentMetaData != blobMetaData.end() && timeIndex < graphElementsCount;
         currentMetaData++) {
        InferenceEngine::InferenceEngineProfileInfo &pc = perfMap[currentMetaData->name];
        float timeMS = 0;
        if (currentMetaData->status != InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT) {
            timeMS = graphInfo->info()[timeIndex];
            timeIndex++;
        }
        pc.cpu_uSec = pc.realTime_uSec = static_cast<long long int>(timeMS * 1000);
        currentMetaData->exec_type.copy(pc.exec_type, sizeof(pc.exec_type) / sizeof(pc.exec_type[0]), 0);
        currentMetaData->layer_type.copy(pc.layer_type, sizeof(pc.layer_type) / sizeof(pc.layer_type[0]), 0);
        pc.status = currentMetaData->status;
        if (currentMetaData->name.compare("Receive-Tensor") == 0) {
            pc.execution_index = 0;
        } else if (currentMetaData->name.compare("LoadInput") == 0) {
            pc.execution_index = 0;
        } else if (currentMetaData->name.compare("GetOutput") == 0) {
            pc.execution_index = graphElementsCount - 2;
        } else if (currentMetaData->status != InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT) {
            pc.execution_index = execIndex;
            execIndex++;
        }
    }

    if (timeIndex != graphElementsCount) {
        THROW_IE_EXCEPTION << "Inconsistent profile info per layers: number of times (" << graphElementsCount
                           << ") != number of non-optimized out layers (" << timeIndex << ")";
    }
}

template<typename T>
void ConvertBlobToLayout(InferenceEngine::Layout layout, InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::Blob::Ptr convertedBlobPtr =
            InferenceEngine::make_shared_blob<T>(blob->precision(), layout, blob->dims());
    convertedBlobPtr->allocate();

    {
        auto srcPtr = blob->cbuffer().as<const T*>();
        auto dstPtr = convertedBlobPtr->buffer().as<T*>();
        const auto& newDims = blob->getTensorDesc().getDims();
        auto C = newDims[1];
        auto H = newDims[2];
        auto W = newDims[3];
        if (layout == InferenceEngine::NHWC) {
            if (blob->layout() != InferenceEngine::Layout::NCHW || newDims.size() != 4 || newDims[0] != 1) {
                // ConvertLayout expects dimensions in reversed order,
                // so we use deperecated Blob::dims() method.
                InferenceEngine::ConvertLayout<T>(blob->layout(), InferenceEngine::NHWC, srcPtr, dstPtr, blob->dims());
            } else {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            auto dstInd = c + w * C + h * C * W;
                            dstPtr[dstInd] = *srcPtr++;
                        }
                    }
                }
            }
        } else if (layout == InferenceEngine::NCHW) {
            if (blob->layout() != InferenceEngine::Layout::NHWC || newDims.size() != 4 || newDims[0] != 1) {
                InferenceEngine::ConvertLayout<T>(blob->layout(), InferenceEngine::NCHW, srcPtr, dstPtr, blob->dims());
            } else {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
                        int offs = h * C * W + c;
                        for (int w = 0; w < W; ++w) {
                            auto srcInd = w * C + offs;
                            *dstPtr = srcPtr[srcInd];
                            ++dstPtr;
                        }
                    }
                }
            }
        }
    }
    blob.swap(convertedBlobPtr);
}

}  // namespace Common
}  // namespace VPU
