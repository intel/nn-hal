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

#include "common.hpp"
#include <vector>
#include <unordered_set>

namespace {

bool isStageCHW(const VpuStageHandle& stage) {
    if (stage == nullptr)
        return false;

    if (isHwStage(stage))
        return true;

    if (stage->type == kCopyMakeBorderCHW)
        return true;

    return false;
}

uint32_t getNewOutputDimZ(const VpuStageHandle& stage) {
    if (stage == nullptr)
        return 0u;

    uint32_t newOutputDimZ = 0u;
    if (stage->type == kMyriadXHwConvolution) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwConvolutionStage>();
        assert(hwStage != nullptr);

        newOutputDimZ = hwStage->newOutputDimZ;
    } else if (stage->type == kMyriadXHwFCL) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwFullyConnectedStage>();
        assert(hwStage != nullptr);

        newOutputDimZ = hwStage->newOutputDimZ;
    } else if (stage->type == kMyriadXHwPooling) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwPoolingStage>();
        assert(hwStage != nullptr);

        newOutputDimZ = hwStage->newOutputDimZ;
    } else if (stage->type == kCopyMakeBorderCHW) {
        newOutputDimZ = stage->outputs[0]->dims[Dim::Z];
    } else {
        THROW_IE_EXCEPTION << "[VPU] Internal error - unsupported HW stage " << stage->name;
    }

    return newOutputDimZ;
}

}  // namespace

// Try to detect concat of HW layers and remove copy stages in that case
void GraphTransformerImpl::packHWConcat() {
    std::unordered_set<VpuStage*> processedStages;

    for (auto& stage : _stages) {
        assert(stage != nullptr);

        if (stage->optimized || stage->type != kCopy)
            continue;

        if (processedStages.find(stage.get()) != processedStages.end())
            continue;

        auto concatLayer = std::dynamic_pointer_cast<ConcatLayer>(stage->layer);
        if (concatLayer == nullptr)
            continue;

        // TODO : can we support other axis?
        if (concatLayer->_axis != 1)
            continue;

        auto concatOutput = stage->outputs[0]->parent;
        if (concatOutput == nullptr || concatOutput->parent != nullptr) {
            THROW_IE_EXCEPTION << "[VPU] Invalid concat stage " << stage->name;
        }

        std::vector<VpuStageHandle> concatStages;
        concatStages.reserve(concatOutput->subData.size());
        for (auto subData : concatOutput->subData) {
            if (subData->producer == nullptr)
                continue;

            if (subData->producer->type != kCopy ||
                subData->producer->layer != concatLayer) {
                THROW_IE_EXCEPTION << "[VPU] Invalid concat stage " << subData->producer->name;
            }

            concatStages.push_back(subData->producer);
            processedStages.insert(subData->producer.get());
        }

        bool isAllInputsCHW = true;
        for (auto curCopyStage : concatStages) {
            if (!isAllInputsCHW)
                break;

            auto curInput = curCopyStage->inputs[0];
            assert(curInput != nullptr);

            std::unordered_set<VpuStageHandle, VpuStageHandleHash> producers;
            if (curInput->producer != nullptr && !curInput->producer->optimized) {
                producers.insert(curInput->producer);
            }
            loopOverSubData(curInput, [&producers](VpuDataHandle subData) {
                if (subData->producer != nullptr && !subData->producer->optimized) {
                    producers.insert(subData->producer);
                }
            });

            for (auto producer : producers) {
                if (!isStageCHW(producer)) {
                    isAllInputsCHW = false;
                    break;
                }
            }
        }
        if (!isAllInputsCHW)
            continue;

        concatOutput->order = orderZYX;
        concatOutput->strides = calcStrides(concatOutput->dims, concatOutput->type, concatOutput->order, 16u);
        loopOverSubData(concatOutput, [concatOutput](VpuDataHandle subData) {
            subData->order = concatOutput->order;
            subData->strides = concatOutput->strides;
        });

        for (auto curCopyStage : concatStages) {
            auto curInput = curCopyStage->inputs[0];
            assert(curInput != nullptr);

            auto curOutput = curCopyStage->outputs[0];
            assert(curOutput != nullptr);

            if (curInput->subData.empty()) {
                auto curInputProducer = curInput->producer;
                if (curInputProducer == nullptr) {
                    THROW_IE_EXCEPTION << "[VPU] Can't convert " << curCopyStage->name << " to HW";
                }

                auto newOutputDimZ = getNewOutputDimZ(curInputProducer);
                if (curOutput->dims[Dim::Z] == newOutputDimZ) {
                    auto curInputProducerOutInd = curInput->producerOutInd;

                    curInput->producer = nullptr;
                    curInput->producerOutInd = -1;

                    curInputProducer->outputs[curInputProducerOutInd] = curOutput;
                    curOutput->producer = curInputProducer;
                    curOutput->producerOutInd = curInputProducerOutInd;

                    curCopyStage->optimized = true;
                } else {
                    // The HW stage will write the output with some padding,
                    // so we need to preserve Copy stage to avoid conflicts
                    // between different paths of Concat.

                    // We change the order of input/output for Copy stage
                    // and add reshape to "deceive" SW stage.

                    curInput->order = orderZYX;
                    curInput->strides = calcStrides(curInput->dims, curInput->type, curInput->order, 16u);
                    loopOverSubData(curInput, [curInput](VpuDataHandle subData) {
                        subData->order = curInput->order;
                        subData->strides = curInput->strides;
                    });

                    auto curInputYXZ = reshapeZYXToYXZ(curInput);
                    auto curOutputYXZ = reshapeZYXToYXZ(curOutput);

                    curInput->consumers.erase(curCopyStage);
                    curCopyStage->inputs[0] = curInputYXZ;
                    curInputYXZ->consumers.insert(curCopyStage);

                    curOutput->producer = nullptr;
                    curOutput->producerOutInd = -1;
                    curCopyStage->outputs[0] = curOutputYXZ;
                    curOutputYXZ->producer = curCopyStage;
                    curOutputYXZ->producerOutInd = 0;
                }
            } else {
                if (!(curInput->producer == nullptr || curInput->producer->optimized)) {
                    THROW_IE_EXCEPTION << "[VPU] Can't convert " << curCopyStage->name << " to HW";
                }

                for (auto subData : curInput->subData) {
                    if (subData->producer == nullptr || subData->producer->optimized) {
                        THROW_IE_EXCEPTION << "[VPU] Can't convert " << curCopyStage->name << " to HW";
                    }

                    auto newOutputDimZ = getNewOutputDimZ(subData->producer);
                    if (newOutputDimZ != subData->dims[Dim::Z]) {
                        THROW_IE_EXCEPTION << "[VPU] Can't convert " << curCopyStage->name << " to HW";
                    }

                    subData->parent = curOutput;
                    curOutput->subData.insert(subData);
                }

                curInput->subData.clear();

                curCopyStage->optimized = true;
            }
        }
    }
}
