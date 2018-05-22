//
// INTEL CONFIDENTIAL
// Copyright 2017-2018 Intel Corporation.
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

void GraphTransformerImpl::parseConcat(const CNNLayerPtr& _layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs) {
    assert(!inputs.empty());
    assert(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ConcatLayer>(_layer);
    assert(layer != nullptr);

    assert(layer->outData.size() == 1);
    assert(layer->outData[0] != nullptr);

    VpuDims outDataOffset(3);

    int revAxis = -1;
    switch (layer->outData[0]->dims.size()) {
    case 2:
    {
        switch (layer->_axis) {
        case 0:
            revAxis = 1;
            break;
        case 1:
            revAxis = 2;
            break;
        }
        break;
    }
    case 3:
    {
        switch (layer->_axis) {
        case 0:
            revAxis = 2;
            break;
        case 1:
            revAxis = 1;
            break;
        case 2:
            revAxis = 0;
            break;
        }
        break;
    }
    case 4:
    {
        switch (layer->_axis) {
            case 1:
                revAxis = 2;
                break;
            case 2:
                revAxis = 1;
                break;
            case 3:
                revAxis = 0;
                break;
        }
        break;
    }
    }
    if (revAxis == -1) {
        THROW_IE_EXCEPTION << "[VPU] Unsupported combination of concat axis=" << layer->_axis
                << " and the number of output dims=" << layer->outData[0]->dims.size();
    }

    auto output = outputs[0];

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input = inputs[i];
        assert(input != nullptr);

        auto subOutput = addNewData(
            newDataId(),
            [layer, input, output, outDataOffset, i](VpuData* data) {
                data->name = layer->name + "@" + std::to_string(i);
                data->index = output->index;
                data->type = VpuDataType::FP16;
                data->dims = input->dims;
                data->strides = output->strides;
                data->offsetFromParent = outDataOffset;
            },
            output);

        auto curOffset = calcDataOffset(revAxis, input->dims);
        outDataOffset[Dim::X] += curOffset[Dim::X];
        outDataOffset[Dim::Y] += curOffset[Dim::Y];
        outDataOffset[Dim::Z] += curOffset[Dim::Z];

        auto actualInput = input;
        auto actualOutput = subOutput;

        if (revAxis == 0) {
            actualInput = addNewData(
                newDataId(),
                [input](VpuData* data) {
                    data->name = input->name + "@reshaped";
                    data->index = input->index;
                    data->type = input->type;
                    data->dims = input->dims;
                    data->dims[Dim::Z] *= input->dims[Dim::X];
                    data->dims[Dim::X] = 1;
                    data->strides = calcStrides(data->dims, data->type, data->order);
                },
                input);

            actualOutput = addNewData(
                newDataId(),
                [subOutput, actualInput, output](VpuData* data) {
                    data->name = subOutput->name + "@reshaped";
                    data->index = subOutput->index;
                    data->type = subOutput->type;
                    data->dims = actualInput->dims;
                    data->strides.resize(3);
                    data->strides[Dim::X] = sizeof(int16_t) * output->dims[Dim::Z] * output->dims[Dim::X];
                    data->strides[Dim::Y] = sizeof(int16_t) * output->dims[Dim::Z] * output->dims[Dim::X];
                    data->strides[Dim::Z] = sizeof(int16_t);
                },
                subOutput);
        }

        addCopyStage(subOutput->name, layer, actualInput, actualOutput);
    }
}
