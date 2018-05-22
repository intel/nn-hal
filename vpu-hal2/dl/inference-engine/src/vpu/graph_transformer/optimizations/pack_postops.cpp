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

void GraphTransformerImpl::packPostOps() {
    for (auto stagePos = _stages.begin(); stagePos != _stages.end(); stagePos++) {
        auto stage = *stagePos;
        if (stage->optimized)
            continue;

        // Detect Bias->Relu pattern and replace it with Relu+Bias stage
        // in case of the input of ReLU is not input for other stage
        if ((stage->type == kRelu) || (stage->type == kLeakyRelu)) {
            auto input = stage->inputs[0];

            VpuStageHandle prevStage = input->producer;
            if (prevStage != nullptr && !prevStage->optimized && prevStage->type == kBias) {
                if (prevStage->inputs[0] != prevStage->outputs[0]) {
                    THROW_IE_EXCEPTION << "[VPU] Stage " << prevStage->name << " can be used only in-place";
                }
                if (prevStage->parentOp == nullptr) {
                    THROW_IE_EXCEPTION << "[VPU] Stage " << prevStage->name << " can be used only as post-op";
                }

                bool hasAnotherConsumer = false;
                if (input->consumers.size() != 2) {
                    hasAnotherConsumer = true;
                } else {
                    loopOverSubData(input, [&hasAnotherConsumer](const VpuDataHandle& subData) {
                        if (subData->producer == nullptr) {
                            hasAnotherConsumer = true;
                        }
                    });
                }

                if (!hasAnotherConsumer && prevStage->outputs[0]->index != IndexOutput) {
                    prevStage->inputs[0]->producer = prevStage->parentOp;
                    prevStage->inputs[0]->consumers.erase(prevStage);

                    if (stage->type == kRelu) {
                        stage->type = kBiasRelu;
                    } else {
                        stage->type = kBiasLeakyRelu;
                    }

                    stage->name += "+Bias";
                    stage->inputs.push_back(prevStage->inputs[1]);
                    stage->requiredInputOrder.push_back(prevStage->inputs[1]->order);
                    stage->requiredInputAlignment.push_back(1);

                    prevStage->parentOp->postOp = nullptr;
                    prevStage->parentOp = nullptr;
                    prevStage->optimized = true;
                }
            }
        }

        // Make the following operations in-place (since MvTensor works only in that mode)
        if (stage->type == kBias || stage->type == kElu || stage->type == kRelu || stage->type == kReluX ||
            stage->type == kLeakyRelu || stage->type == kBiasRelu || stage->type == kBiasLeakyRelu) {
            auto input = stage->inputs[0];
            auto output = stage->outputs[0];
            if (output != input) {
                if (_blobConfig.hwOptimization) {
                    // Only Bias and ReLU are supported in-place for HW graph
                    if (stage->type != kBias && stage->type != kRelu && stage->type != kBiasRelu)
                        continue;

                    // In case of HW graph we make this operations in-place only for Convolution, Pooling and FC (HW operations)
                    if (input->producer != nullptr &&
                        !(input->producer->type == kConv || input->producer->type == kIm2ColConvolution ||
                          input->producer->type == kMaxPool || input->producer->type == kAvgPool ||
                          input->producer->type == kFC)) {
                        continue;
                    }
                }

                if (input->index == IndexOutput) {
                    continue;
                }

                // Can't make operation in-place if we have another consumer
                bool hasAnotherConsumer = false;
                if (input->consumers.size() != 1) {
                    hasAnotherConsumer = true;
                } else {
                    loopOverSubData(input, [&hasAnotherConsumer](const VpuDataHandle& subData) {
                        if (subData->producer == nullptr) {
                            hasAnotherConsumer = true;
                        }
                    });
                }

                if (!hasAnotherConsumer) {
                    if (input->producer != nullptr) {
                        if (input->producer->postOp != nullptr) {
                            THROW_IE_EXCEPTION << "[VPU] Can't add " << stage->name << " as post op to " << input->producer->name;
                        }

                        assert(input->producerOutInd >= 0);
                        assert(input->producerOutInd < input->producer->outputs.size());
                        input->producer->outputs[input->producerOutInd] = output;

                        input->producer->postOp = stage;
                        stage->parentOp = input->producer;
                    } else {
                        for (auto& subData : input->subData) {
                            if (subData->producer == nullptr) {
                                THROW_IE_EXCEPTION << "[VPU] Can't add " << stage->name << " as post op";
                            }

                            subData->parent = output;
                            subData->index = output->index;
                            output->subData.insert(subData);
                        }

                        input->subData.clear();
                    }

                    stage->inputs[0] = output;
                    output->consumers.insert(stage);
                }
            }
        }
    }
}
