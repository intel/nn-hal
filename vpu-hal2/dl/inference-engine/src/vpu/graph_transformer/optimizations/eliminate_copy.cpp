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

void GraphTransformerImpl::eliminateCopyStages() {
    for (auto& stage : _stages) {
        if (stage->optimized)
            continue;

        if (stage->type == kCopy) {
            auto input = stage->inputs[0];
            assert(input != nullptr);

            auto output = stage->outputs[0];
            assert(output != nullptr);

            if (input->producer == nullptr) {
                // It might be the following case
                // [CHW] -> ConvertOrder -> [input@reshaped] -> [input] -> Copy -> [output]
                // translate it to
                // [CHW] -> ConvertOrder -> [input@reshaped] -> [output]
                if (input->subData.size() == 1) {
                    auto inputSubData = *input->subData.begin();
                    assert(inputSubData != nullptr);

                    if (inputSubData->producer != nullptr && inputSubData->producer->type == kConvertOrder) {
                        input->subData.erase(inputSubData);
                        inputSubData->parent = output;
                        inputSubData->index = output->index;
                        output->subData.insert(inputSubData);
                        output->producer = nullptr;
                        output->producerOutInd = -1;
                        stage->optimized = true;
                    }
                }

                continue;
            }

            auto producerType = input->producer->type;
            if (producerType != kConv && producerType != kDeconvolution && producerType != kBias &&
                producerType != kRelu && producerType != kReluX && producerType != kLeakyRelu &&
                producerType != kBiasRelu && producerType != kConvertOrder && producerType != kIm2ColConvolution) {
                continue;
            }

            if (input->producer->parentOp == nullptr) {
                if (input->consumers.size() > 1)
                    continue;

                input->producer->outputs[input->producerOutInd] = output;
                output->producer = input->producer;
                output->producerOutInd = input->producerOutInd;
                stage->optimized = true;
            } else {
                // Post-ops are in-place, so input has at least 2 consumers (copy and post-op)
                if (input->consumers.size() > 2)
                    continue;

                if (!input->subData.empty())
                    continue;

                if (!output->consumers.empty())
                    continue;

                auto postOp = input->producer;
                auto mainOp = postOp->parentOp;

                if (mainOp->type != kConv && mainOp->type != kDeconvolution && mainOp->type != kIm2ColConvolution)
                    continue;

                int inInd = -1;
                for (size_t i = 0; i < postOp->inputs.size(); ++i) {
                    if (postOp->inputs[i] == input) {
                        inInd = i;
                        break;
                    }
                }
                if (inInd == -1)
                    continue;

                int mainOutInd = -1;
                for (size_t i = 0; i < mainOp->outputs.size(); ++i) {
                    if (mainOp->outputs[i] == input) {
                        mainOutInd = i;
                        break;
                    }
                }
                if (mainOutInd == -1)
                    continue;

                mainOp->outputs[mainOutInd] = output;
                postOp->inputs[inInd] = output;
                output->consumers.insert(postOp);
                postOp->outputs[input->producerOutInd] = output;
                output->producer = postOp;
                output->producerOutInd = input->producerOutInd;
                stage->optimized = true;
            }
        }
    }
}
