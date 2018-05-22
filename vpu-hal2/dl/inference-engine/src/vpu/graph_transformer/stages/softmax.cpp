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

void VpuSoftMaxStage::dumpToDot(std::ostream& os) {
    os << "axis=" << axis;
}

void VpuSoftMaxStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<char>(axis));
    writer.write(static_cast<char>(0));  // for alignment
    writer.write(static_cast<char>(0));  // for alignment
    writer.write(static_cast<char>(0));  // for alignment

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
}

void GraphTransformerImpl::parseSoftMax(const CNNLayerPtr& _layer,
                                        const std::vector<VpuDataHandle>& inputs,
                                        const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<SoftMaxLayer>(_layer);
    assert(layer != nullptr);

    auto layerInput = layer->insData[0].lock();
    assert(layerInput != nullptr);

    int ax = layer->axis;
    if (layerInput->dims.size() == 3) {
        // For 2D input (3 = 2D + Batch) we add extra dimension in MvTensor,
        // so we need to update the axis.
        ax++;
    }

    char axis = 'c';
    switch (ax) {
    case 2:
        axis = 'h';
        break;
    case 3:
        axis = 'w';
        break;
    case 1:
    default:
        axis = 'c';
        break;
    }

    addNewStage<VpuSoftMaxStage>(
        layer->name,
        kSoftMax,
        layer,
        [axis](VpuSoftMaxStage* stage) {
            stage->axis = axis;
        },
        inputs,
        outputs);
}
