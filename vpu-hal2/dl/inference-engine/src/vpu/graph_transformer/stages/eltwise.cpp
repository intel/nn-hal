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
#include <string>

void VpuEltwiseStage::dumpToBlob(BlobWriter& writer) {
    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
    for (size_t i = 1; i < inputs.size(); ++i)
        inputs[i]->dumpToBlob(writer);
}

void GraphTransformerImpl::parseEltwise(const CNNLayerPtr& _layer,
                                        const std::vector<VpuDataHandle>& inputs,
                                        const std::vector<VpuDataHandle>& outputs) {
    if (inputs.size() < 2)
        THROW_IE_EXCEPTION << "[VPU] Incorect inputs number for Eltwise layer.";

    if (outputs.size() != 1)
        THROW_IE_EXCEPTION << "[VPU] Eltwise layer supports only 1 output.";

    auto layer = std::dynamic_pointer_cast<EltwiseLayer>(_layer);
    if (layer == nullptr)
        THROW_IE_EXCEPTION << "[VPU] Cannot case to Eltwise layer.";

    t_MvTensorOpType opType = kNone0;
    switch (layer->_operation) {
    case EltwiseLayer::eOperation::Sum:
        opType = kSum;
        break;
    case EltwiseLayer::eOperation::Prod:
        opType = kProd;
        break;
    case EltwiseLayer::eOperation::Max:
        opType = kMax;
        break;
    default:
        THROW_IE_EXCEPTION << "[VPU] Eltwise operation is not supported";
        break;
    }

    std::vector<VpuDataHandle> newInput(2);
    newInput[0] = inputs[0];
    newInput[1] = inputs[1];

    addNewStage<VpuEltwiseStage>(
        layer->name,
        opType,
        layer,
        [](VpuEltwiseStage* /*stage*/) {
        },
        newInput,
        outputs);

    newInput[0] = outputs[0];
    for (size_t ind = 2; ind < inputs.size(); ++ind) {
        newInput[1] = inputs[ind];

        addNewStage<VpuEltwiseStage>(
            layer->name + "@" + std::to_string(ind - 1),
            opType,
            layer,
            [](VpuEltwiseStage* /*stage*/) {
            },
            newInput,
            outputs);
    }
}
