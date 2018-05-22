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

void GraphTransformerImpl::parseInputAndOutputData() {
    uint32_t inputOffset = 0;
    for (const auto& inputInfo : _networkInputs) {
        auto netInput = inputInfo.second;
        assert(netInput != nullptr);

        auto input = addNewData(
            dataId(netInput->getInputData()),
            [netInput, inputOffset, this](VpuData* data) {
                data->name = netInput->name();
                data->index = IndexInput;
                data->type = iePrecisionToVpu(netInput->getInputPrecision());
                data->dims = ieDimsToVpu(netInput->getTensorDesc().getDims());
                data->offset = inputOffset;
                if (_blobConfig.hwOptimization) {
                    data->order = orderZYX;
                } else {
                    data->order = orderYXZ;
                }
                data->strides = calcStrides(data->dims, data->type, data->order);
            });

        if (input == nullptr) {
            THROW_IE_EXCEPTION << "GraphTransformerV2::parseInputAndOutputData(). Could not add new data";
        }

        inputOffset += input->dims.totalSize() * getDataTypeSize(input->type);
    }

    uint32_t outputOffset = 0;
    for (const auto& outputInfo : _networkOutputs) {
        auto netOutput = outputInfo.second;
        assert(netOutput != nullptr);

        auto output = addNewData(
            dataId(netOutput),
            [netOutput, outputOffset, this](VpuData* data) {
                data->name = netOutput->getName();
                data->index = IndexOutput;
                data->type = iePrecisionToVpu(netOutput->getPrecision());
                data->dims = ieDimsToVpu(netOutput->getDims());
                if (_blobConfig.hwOptimization) {
                    data->order = orderZYX;
                } else {
                    data->order = orderYXZ;
                }
                data->strides = calcStrides(data->dims, data->type, data->order);
                data->offset = outputOffset;
            });

        outputOffset += output->dims.totalSize() * getDataTypeSize(output->type);
    }
}
