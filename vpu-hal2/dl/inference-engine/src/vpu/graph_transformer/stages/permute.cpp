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
#include <utility>

void VpuPermuteStage::dumpToDot(std::ostream& os) {
    os << "order=(" << order0 << ", " << order1 << ", " << order2 << ")";
}

void VpuPermuteStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<int32_t>(order0));
    writer.write(static_cast<int32_t>(order1));
    writer.write(static_cast<int32_t>(order2));

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
}

void GraphTransformerImpl::checkBatchPermute(const CNNLayerPtr& layer,
                                               const std::vector<VpuDataHandle>& inputs,
                                               const std::vector<VpuDataHandle>& outputs) {
    assert(layer != nullptr);

    auto order = layer->GetParamAsFloats("order");
    auto input = inputs[0];
    auto output = outputs[0];

    if (order.size() < 4) {
        THROW_IE_EXCEPTION << "[VPU] Permute has to provide order dimension 4. Layer name is: " << layer->name;
    }
    if (static_cast<int>(order[0]) != 0) {
        // Special case for support LPR
        // This way allows us to replace the four-dimensional tensor with the three-dimensional if we have:
        // out_channels == 1 after original permute
        uint32_t input_dim[4];
        input_dim[2] = input->dims[Dim::Y];
        input_dim[3] = input->dims[Dim::X];

        uint32_t output_dim[4];
        output_dim[2] = input_dim[static_cast<int>(order[2])];
        output_dim[3] = input_dim[static_cast<int>(order[3])];

        if ((output->dims[Dim::N] > 0) && (output->dims[Dim::Z] == 1)) {
        } else {
            THROW_IE_EXCEPTION << "[VPU] Unsupported order value " << layer->name;
        }
    }
}

void GraphTransformerImpl::parsePermute(const CNNLayerPtr& layer,
                                        const std::vector<VpuDataHandle>& inputs,
                                        const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto order = layer->GetParamAsFloats("order");
    auto input = inputs[0];
    auto output = outputs[0];

    if (order.size() < 4) {
        THROW_IE_EXCEPTION << "[VPU] Permute has to provide order dimension 4. Layer name is: " << layer->name;
    }
    if (static_cast<int>(order[0]) != 0) {
        // Special case for support LPR
        // This way allows us to replace the four-dimensional tensor with the three-dimensional if we have:
        // out_channels == 1 after original permute
        uint32_t input_dim[4];
        input_dim[2] = input->dims[Dim::Y];
        input_dim[3] = input->dims[Dim::X];

        uint32_t output_dim[4];
        output_dim[2] = input_dim[static_cast<int>(order[2])];
        output_dim[3] = input_dim[static_cast<int>(order[3])];

        if ((output->dims[Dim::N] > 0) && (output->dims[Dim::Z] == 1)) {
            std::swap(order[0], order[1]);
            std::swap(order[2], order[3]);

            output->dims[Dim::Z] = output->dims[Dim::N];
            output->dims[Dim::N] = 1;
            std::swap(output->dims[Dim::Y], output->dims[Dim::X]);

            output->strides[Dim::X] = sizeof(ie_fp16) * output->dims[Dim::Z];
            output->strides[Dim::Y] = sizeof(ie_fp16) * output->dims[Dim::Z] * output->dims[Dim::X];
            output->strides[Dim::Z] = sizeof(ie_fp16);

        } else {
            THROW_IE_EXCEPTION << "[VPU] Unsupported order value " << layer->name;
        }
    }

    addNewStage<VpuPermuteStage>(
        layer->name,
        kPermute,
        layer,
        [&order](VpuPermuteStage* stage) {
            stage->order0 = (static_cast<int>(order[1]) - 1);
            stage->order1 = (static_cast<int>(order[2]) - 1);
            stage->order2 = (static_cast<int>(order[3]) - 1);
        },
        inputs,
        outputs);
}
