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
#include <cmath>
#include <vector>
#include <limits>

void VpuReluStage::dumpToDot(std::ostream& os) {
    os << "negativeSlope=" << negativeSlope;
}

void VpuReluStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<uint32_t>(inputs.size() == 2));
    writer.write(*(reinterpret_cast<uint32_t *>(&negativeSlope)));

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
    if (inputs.size() == 2) {
        inputs[1]->dumpToBlob(writer);
    }
}

void GraphTransformerImpl::parseReLU(const CNNLayerPtr& _layer,
                                     const std::vector<VpuDataHandle>& inputs,
                                     const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ReLULayer>(_layer);
    assert(layer != nullptr);

    if (std::fabs(layer->negative_slope) < std::numeric_limits<float>::epsilon()) {
        addNewStage<VpuReluStage>(
            layer->name,
            kRelu,
            layer,
            [layer](VpuReluStage* stage) {
                stage->negativeSlope = layer->negative_slope;
            },
            inputs,
            outputs);

    } else {
        addNewStage<VpuReluStage>(
            layer->name,
            kLeakyRelu,
            layer,
            [layer](VpuReluStage* stage) {
                stage->negativeSlope = layer->negative_slope;
            },
            inputs,
            outputs);
    }
}
