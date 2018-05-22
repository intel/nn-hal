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

void GraphTransformerImpl::parseBias(const CNNLayerPtr& layer,
                                     const std::vector<VpuDataHandle>& inputs,
                                     const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);

    auto input0 = inputs[0];
    auto input1 = inputs[1];

    if (input1->dims[Dim::X] != input0->dims[Dim::X] ||
        input1->dims[Dim::Y] != input0->dims[Dim::Y] ||
        input1->dims[Dim::Z] != input0->dims[Dim::Z]) {
        THROW_IE_EXCEPTION
                << "[VPU] Current Bias layer realization supports only equal inputs(axis 0, 1 for 4D tensor, axis 0 for other dimensions),"
                << " layer name is " << layer->name;
    }

    addNewStage<VpuEltwiseStage>(
        layer->name,
        kSum,
        layer,
        [](VpuEltwiseStage* /*stage*/) {
        },
        inputs,
        outputs);
}
