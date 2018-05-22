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

void VpuTileStage::dumpToDot(std::ostream& os) {
    os << "axis=" << axis << "\\n"
       << "tiles=" << tiles;
}

void VpuTileStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<int32_t>(axis));
    writer.write(static_cast<int32_t>(tiles));

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
}

void GraphTransformerImpl::parseTile(const CNNLayerPtr& _layer,
                                     const std::vector<VpuDataHandle>& inputs,
                                     const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<TileLayer>(_layer);
    assert(layer != nullptr);

    assert(layer->outData.size() == 1);
    auto layerOutput = layer->outData[0];
    assert(layerOutput != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    auto axis = layer->axis;
    auto numOutDims = layer->outData[0]->dims.size();

    auto actualInput = input;
    auto actualOutput = output;

    // "conversion" to 4D representation
    axis +=  4 - numOutDims;
    if (numOutDims == 2) {
        actualInput = addNewData(
            newDataId(),
            [input](VpuData* data) {
                data->name = input->name + "@reshaped";
                data->index = input->index;
                data->type = input->type;
                data->dims = VpuDims({input->dims[Dim::Z], input->dims[Dim::Y], input->dims[Dim::X]});
                // NOTE : Strides recalculation is not required for this layer now.
                data->strides = input->strides;
            },
            input);

        actualOutput = addNewData(
            newDataId(),
            [output](VpuData* data) {
                data->name = output->name + "@reshaped";
                data->index = output->index;
                data->type = output->type;
                data->dims = VpuDims({output->dims[Dim::Z], output->dims[Dim::Y], output->dims[Dim::X]});
                // NOTE : Strides recalculation is not required for this layer now.
                data->strides = output->strides;
            },
            output);
    }

    addNewStage<VpuTileStage>(
        layer->name,
        kTile,
        layer,
        [layer, axis](VpuTileStage* stage) {
            stage->axis = axis;
            stage->tiles = layer->tiles;
        },
        {actualInput},
        {actualOutput});
}
