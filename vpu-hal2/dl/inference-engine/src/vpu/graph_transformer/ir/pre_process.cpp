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

#include "graph_transformer_impl.hpp"
#include <vector>
#include <memory>
#ifdef NNLOG
#include <android/log.h>
#include <cutils/log.h>
#endif

namespace {

class MeanImageWeightsWriter : public DataWriter {
public:
    MeanImageWeightsWriter(const PreProcessInfo& info, const VpuDims& dims, t_MvTensorStorageOrder inputOrder)
        : _info(info), _dims(dims), _inputOrder(inputOrder) {
    }

    size_t byteSize() const override {
        return _dims.totalSize() * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto numOfChannel = _info.getNumberOfChannels();

        auto imagePixels = _dims[Dim::X] * _dims[Dim::Y];
        auto countElem = _dims[Dim::X] * _dims[Dim::Y] * _dims[Dim::Z];

        std::vector<ie_fp16> meanDataFp16(countElem);
        for (size_t i = 0; i < numOfChannel; i++) {
            auto meanDataBlobPtr = _info[i]->meanData;
            PrecisionUtils::f32tof16Arrays(meanDataFp16.data() + i * imagePixels,
                                           meanDataBlobPtr->buffer(),
                                           imagePixels,
                                           -1.f);
        }
        if (_inputOrder == orderYXZ) {
            kchw_to_hwck(meanDataFp16.data(), static_cast<ie_fp16*>(dst), _dims);
        } else if (_inputOrder == orderZYX) {
            std::copy_n(meanDataFp16.data(), meanDataFp16.size(), static_cast<ie_fp16 *>(dst));
        }
    }

private:
    PreProcessInfo _info;
    VpuDims _dims;
    t_MvTensorStorageOrder _inputOrder;
};

class MeanValueBiasesWriter : public DataWriter {
public:
    explicit MeanValueBiasesWriter(const PreProcessInfo& info) : _info(info) {
    }

    size_t byteSize() const override {
        return _info.getNumberOfChannels() * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto numOfChannel = _info.getNumberOfChannels();

        auto dstData = static_cast<ie_fp16*>(dst);
        for (size_t i = 0; i < numOfChannel; i++) {
            dstData[i] = PrecisionUtils::f32tof16(-_info[i]->meanValue);
        }
    }

private:
    PreProcessInfo _info;
};

}  // namespace

void GraphTransformerImpl::addPreProcessStages() {
    for (const auto& inputInfo : _networkInputs) {
        auto netInput = inputInfo.second;
        assert(netInput != nullptr);

        const auto& preProcess = netInput->getPreProcess();

        if (preProcess.getMeanVariant() != NONE) {
            auto input = getVpuDataFP16(netInput->getInputData());
            assert(input != nullptr);

            uint32_t numOfChannel = preProcess.getNumberOfChannels();

            LOG_DEBUG("[VPU] GraphTransformer : add pre-processing for input %s", input->name.c_str());
#ifdef NNLOG
            ALOGI("[VPU] GraphTransformer : add pre-processing for input %s", input->name.c_str());
#endif

            if (preProcess.getMeanVariant() == MEAN_IMAGE) {
                auto weights = addNewData(
                    newDataId(),
                    [input, preProcess, this](VpuData* data) {
                        data->name = input->name + "@meanImageWeights";
                        data->index = IndexBlob;
                        data->type = VpuDataType::FP16;
                        data->dims = input->dims;
                        data->order = input->order;
                        data->strides = calcStrides(data->dims, data->type, data->order);
                        data->writer = std::make_shared<MeanImageWeightsWriter>(preProcess, data->dims, data->order);
                    });

                addNewStage<VpuEltwiseStage>(
                    weights->name,
                    kSum,
                    nullptr,
                    [](VpuEltwiseStage* stage) {
                        stage->requiredInputOrder[0] = orderYXZ;
                        stage->requiredInputOrder[1] = orderYXZ;
                        stage->requiredOutputOrder[0] = orderYXZ;
                    },
                    {input, weights},
                    {input});
            } else {  // if (preProcess.getMeanVariant() == MEAN_VALUE)
                auto biases = addNewData(
                    newDataId(),
                    [input, preProcess, numOfChannel](VpuData* data) {
                        data->name = input->name + "@meanValueBiases";
                        data->index = IndexBlob;
                        data->type = VpuDataType::FP16;
                        data->dims = VpuDims({numOfChannel, 1, 1});
                        data->strides = calcStrides(data->dims, data->type, data->order);
                        data->writer = std::make_shared<MeanValueBiasesWriter>(preProcess);
                    });

                addNewStage<VpuBiasStage>(
                    biases->name,
                    input->order == orderYXZ ? kBias : kCHWBias,
                    nullptr,
                    [input](VpuBiasStage* stage) {
                        stage->requiredInputOrder[0] = input->order;
                        stage->requiredOutputOrder[0] = input->order;
                    },
                    {input, biases},
                    {input});
            }

            auto doStdScale = preProcess[0]->stdScale != 1.0f;
            if (doStdScale) {
                for (size_t i = 1; i < numOfChannel; i++) {
                    if (preProcess[i - 1]->stdScale != preProcess[i]->stdScale) {
                        doStdScale = false;
                        break;
                    }
                }
                if (!doStdScale) {
                    THROW_IE_EXCEPTION << "[VPU] Different values of stdScale are not supported";
                }

                auto weights = addNewData(
                    newDataId(),
                    [input, preProcess, this](VpuData* data) {
                        data->name = input->name + "@stdScale";
                        data->index = IndexBlob;
                        data->type = VpuDataType::FP16;
                        data->order = orderXYZ;
                        data->dims = VpuDims({1, 1, input->dims[Dim::Z]});
                        data->strides = calcStrides(data->dims, data->type, data->order);
                        data->writer = std::make_shared<ScaleWeightsWriter>(preProcess[0]->stdScale, data->dims[Dim::Z]);
                    });

                addNewStage<VpuScaleStage>(
                    weights->name,
                    input->order == orderYXZ ? kScale : kCHWScale,
                    nullptr,
                    [input](VpuScaleStage* stage) {
                        stage->requiredInputOrder[0] = input->order;
                        stage->requiredOutputOrder[0] = input->order;
                    },
                    {input, weights},
                    {input});
            }
        }
    }
}
