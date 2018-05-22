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

void GraphTransformerImpl::parseSplit(const CNNLayerPtr& _layer,
                                      const std::vector<VpuDataHandle>& inputs,
                                      const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(!outputs.empty());

    auto layer = std::dynamic_pointer_cast<SplitLayer>(_layer);
    assert(layer != nullptr);

    auto inDims = inputs[0]->dims;

    // Check whether it is split(copy) or slice Caffe layer
    // and we do not trust to layer type value
    bool isSplit = true;
    VpuDims sumDims(3);
    for (const auto& output : outputs) {
        sumDims[Dim::X] += output->dims[Dim::X];
        sumDims[Dim::Y] += output->dims[Dim::Y];
        sumDims[Dim::Z] += output->dims[Dim::Z];
        if (inDims[Dim::X] != output->dims[Dim::X] ||
            inDims[Dim::Y] != output->dims[Dim::Y] ||
            inDims[Dim::Z] != output->dims[Dim::Z]) {
            isSplit = false;
        }
    }

    int axis = -1;
    if (!isSplit) {
        // this is slicing. it is necessary to define target axis for slicing
        VpuDims mulDims(3);
        mulDims[Dim::X] = inDims[Dim::X] * outputs.size();
        mulDims[Dim::Y] = inDims[Dim::Y] * outputs.size();
        mulDims[Dim::Z] = inDims[Dim::Z] * outputs.size();
        if (inDims[Dim::X] == sumDims[Dim::X] && mulDims[Dim::Y] == sumDims[Dim::Y] && mulDims[Dim::Z] == sumDims[Dim::Z]) {
            axis = 0;
        } else if (inDims[Dim::Y] == sumDims[Dim::Y] && mulDims[Dim::X] == sumDims[Dim::X] && mulDims[Dim::Z] == sumDims[Dim::Z]) {
            axis = 1;
        } else if (inDims[Dim::Z] == sumDims[Dim::Z] && mulDims[Dim::X] == sumDims[Dim::X] && mulDims[Dim::Y] == sumDims[Dim::Y]) {
            axis = 2;
        }

        if (axis == -1) {
            THROW_IE_EXCEPTION << "[VPU] Incorrect output data dimensions for Split layer with name: " << layer->name;
        }

        assert(layer->insData.size() == 1);
        auto layerInput = layer->insData[0].lock();
        assert(layerInput != nullptr);
        if (axis > layerInput->dims.size()) {
            THROW_IE_EXCEPTION << "Calculated splitting axis value is not correct for Split layer with name: " << layer->name;
        }
    }

    auto input = inputs[0];

    VpuDims dataOffset(3);
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];

        auto curInput = input;
        if (!isSplit) {
            curInput = addNewData(
                newDataId(),
                [layer, input, output, dataOffset, i](VpuData* data) {
                    data->name = layer->name + "@sub" + std::to_string(i);
                    data->index = input->index;
                    data->type = VpuDataType::FP16;
                    data->dims = output->dims;
                    data->strides = input->strides;
                    data->offsetFromParent = dataOffset;
                },
                input);

            auto curOffset = calcDataOffset(axis, output->dims);
            dataOffset[Dim::X] += curOffset[Dim::X];
            dataOffset[Dim::Y] += curOffset[Dim::Y];
            dataOffset[Dim::Z] += curOffset[Dim::Z];
        }

        auto actualInput = curInput;
        auto actualOutput = output;

        if (axis == 0) {
            actualOutput = addNewData(
                newDataId(),
                [output](VpuData* data) {
                    data->name = output->name + "@reshaped";
                    data->index = output->index;
                    data->type = output->type;
                    data->dims = output->dims;
                    data->dims[Dim::Z] *= output->dims[Dim::X];
                    data->dims[Dim::X] = 1;
                    data->strides = calcStrides(data->dims, data->type, data->order);
                },
                output);

                actualInput = addNewData(
                    newDataId(),
                    [curInput, actualOutput, input](VpuData* data) {
                        data->name = curInput->name + "@reshaped";
                        data->index = curInput->index;
                        data->type = curInput->type;
                        data->dims = actualOutput->dims;
                        data->strides = curInput->strides;
                        data->strides[Dim::X] = sizeof(ie_fp16) * input->dims[Dim::X] * input->dims[Dim::Z];
                    },
                    curInput);
        }

        addCopyStage(layer->name + "@" + std::to_string(i), layer, actualInput, actualOutput);
    }
}
