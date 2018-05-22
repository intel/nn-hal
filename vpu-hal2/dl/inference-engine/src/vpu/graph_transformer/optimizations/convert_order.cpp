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
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <string>

void GraphTransformerImpl::addConvertOrderStages() {
    std::unordered_map<VpuDataHandle, std::list<VpuDataHandle>, VpuDataHandleHash> convertedDataMap;
    std::unordered_map<VpuDataHandle, VpuDataHandle, VpuDataHandleHash> alignedDataMap;
    std::unordered_set<VpuDataHandle, VpuDataHandleHash> topParentVisited;

    for (auto stageIt = _stages.begin(); stageIt != _stages.end(); ++stageIt) {
        auto stage = *stageIt;
        assert(stage != nullptr);

        if (stage->optimized)
            continue;

        // LRN supports both YXZ and ZYX orders, but requires that input and output have the same stride

        if (stage->type == kLRN) {
            auto input = stage->inputs[0];
            assert(input != nullptr);

            if (input->order == orderZYX) {
                stage->requiredInputOrder[0] = orderZYX;
                stage->requiredInputAlignment[0] = 16u;

                stage->requiredOutputOrder[0] = orderZYX;
                stage->requiredOutputAlignment[0] = 16u;

                stage->name += "@" + mvTensorStorageOrderToStr(input->order);
            }
        }

        // Normalize supports both YXZ and ZYX orders

        if (stage->type == kNormalize) {
            auto input = stage->inputs[0];
            assert(input != nullptr);

            if (input->order == orderZYX) {
                stage->requiredInputOrder[0] = orderZYX;

                stage->requiredOutputOrder[0] = orderZYX;
                stage->requiredOutputAlignment[0] = 16u;

                stage->name += "@" + mvTensorStorageOrderToStr(input->order);
            }
        }

        // For SoftMax 1x1xC we can support ZYX order by reshaping input and output

        if (stage->type == kSoftMax) {
            auto softMaxStage = std::dynamic_pointer_cast<VpuSoftMaxStage>(stage);
            assert(softMaxStage != nullptr);

            auto input = stage->inputs[0];
            assert(input != nullptr);

            auto output = stage->outputs[0];
            assert(output != nullptr);

            if (input->order == orderZYX &&
                input->dims[Dim::X] == 1 && input->dims[Dim::Y] == 1) {
                auto inputReshaped = reshapeZYXToYXZ(input);
                auto outputReshaped = reshapeZYXToYXZ(output);

                switch (softMaxStage->axis) {
                case 'c':
                    softMaxStage->axis = 'h';
                    break;
                case 'h':
                    softMaxStage->axis = 'w';
                    break;
                case 'w':
                    softMaxStage->axis = 'c';
                    break;
                default:
                    THROW_IE_EXCEPTION << "[VPU] Incorrect SoftMax axis " << softMaxStage->axis;
                }

                stage->inputs[0] = inputReshaped;
                inputReshaped->consumers.insert(stage);
                input->consumers.erase(stage);

                stage->outputs[0] = outputReshaped;
                outputReshaped->producer = stage;
                outputReshaped->producerOutInd = 0;
                output->producer = nullptr;
                output->producerOutInd = -1;

                stage->name += "@ZYX";
            }
        }

        // For Eltwise we can support ZYX order by reshaping input and output

        if (stage->type == kSum || stage->type == kProd || stage->type == kMax) {
            assert(stage->inputs.size() == 2);

            auto input0 = stage->inputs[0];
            assert(input0 != nullptr);

            auto input1 = stage->inputs[1];
            assert(input1 != nullptr);

            auto output = stage->outputs[0];
            assert(output != nullptr);

            if ((output->index == IndexBSS && output->parent == nullptr) ||
                (output->index == IndexOutput && output->order == orderZYX) ||
                input0 == output) {
                VpuDataHandle input0Reshaped, input1Reshaped;

                bool isAllConsumersHW = true;
                for (auto consumer : output->consumers) {
                    if (!isHwStage(consumer)) {
                        isAllConsumersHW = false;
                        break;
                    }
                }
                loopOverSubData(output, [&isAllConsumersHW](VpuDataHandle subData) {
                    if (!isAllConsumersHW)
                        return;
                    for (auto consumer : subData->consumers) {
                        if (!isHwStage(consumer)) {
                            isAllConsumersHW = false;
                            break;
                        }
                    }
                });

                if (input0->order == orderZYX && input1->order == orderZYX) {
                    input0Reshaped = reshapeZYXToYXZ(input0);
                    input1Reshaped = reshapeZYXToYXZ(input1);
                } else if (input0->order == orderZYX && isAllConsumersHW) {
                    input0Reshaped = reshapeZYXToYXZ(input0);
                    auto input1Converted = findOrCreateConvertedData(convertedDataMap, input1, orderZYX, stage->requiredInputAlignment[1], stageIt);
                    input1Reshaped = reshapeZYXToYXZ(input1Converted);
                } else if (input1->order == orderZYX && isAllConsumersHW) {
                    auto input0Converted = findOrCreateConvertedData(convertedDataMap, input0, orderZYX, stage->requiredInputAlignment[0], stageIt);
                    input0Reshaped = reshapeZYXToYXZ(input0Converted);
                    input1Reshaped = reshapeZYXToYXZ(input1);
                }

                if (input0Reshaped != nullptr && input1Reshaped != nullptr) {
                    VpuDataHandle outputReshaped = nullptr;
                    if (input0 != output) {
                        if (output->index != IndexOutput) {
                            output->order = orderZYX;
                            output->strides = calcStrides(output->dims, output->type, output->order, 16u);

                            loopOverSubData(output, [output](VpuDataHandle subData) {
                                subData->order = orderZYX;
                                subData->strides = output->strides;
                            });
                        }

                        outputReshaped = reshapeZYXToYXZ(output);
                    } else {
                        outputReshaped = input0Reshaped;
                    }

                    stage->inputs[0] = input0Reshaped;
                    input0Reshaped->consumers.insert(stage);
                    input0->consumers.erase(stage);

                    stage->inputs[1] = input1Reshaped;
                    input1Reshaped->consumers.insert(stage);
                    input1->consumers.erase(stage);

                    stage->outputs[0] = outputReshaped;
                    outputReshaped->producer = stage;
                    outputReshaped->producerOutInd = 0;
                    output->producer = nullptr;
                    output->producerOutInd = -1;

                    stage->name += "@ZYX";
                }
            }
        }

        // For ReLU/LeakyReLU we can support ZYX order by reshaping input and output

        if (stage->type == kRelu || stage->type == kLeakyRelu) {
            auto input = stage->inputs[0];
            assert(input != nullptr);

            auto output = stage->outputs[0];
            assert(output != nullptr);

            if (input->order == orderZYX &&
                output->index == IndexBSS && output->parent == nullptr) {
                auto inputReshaped = reshapeZYXToYXZ(input);

                output->order = orderZYX;
                output->strides = calcStrides(output->dims, output->type, output->order, 16u);

                loopOverSubData(output, [output](VpuDataHandle subData) {
                    subData->order = orderZYX;
                    subData->strides = output->strides;
                });

                auto outputReshaped = reshapeZYXToYXZ(output);

                stage->inputs[0] = inputReshaped;
                inputReshaped->consumers.insert(stage);
                input->consumers.erase(stage);

                stage->outputs[0] = outputReshaped;
                outputReshaped->producer = stage;
                outputReshaped->producerOutInd = 0;
                output->producer = nullptr;
                output->producerOutInd = -1;

                stage->name += "@ZYX";
            }
        }

        // [Input CHW] -> Permute (1, 2, 0) -> ToPlaneMajor -> [Output HWC]
        // can be converted to
        // [Input CHW] -> ConvertOrder -> [Output HWC @ reshaped] -> [Output HWC]

        if (stage->type == kPermute) {
            auto permuteStage = std::dynamic_pointer_cast<VpuPermuteStage>(stage);
            assert(permuteStage != nullptr);

            if (permuteStage->order0 == 1 && permuteStage->order1 == 2 && permuteStage->order2 == 0) {
                auto input = stage->inputs[0];
                assert(input != nullptr);

                if (input->order == orderZYX) {
                    auto output = stage->outputs[0];
                    assert(output != nullptr);

                    if (output->index == IndexBSS && output->parent == nullptr && output->consumers.size() == 1) {
                        auto nextStage = *output->consumers.begin();
                        assert(nextStage != nullptr);

                        if (!nextStage->optimized && nextStage->type == kToPlaneMajor) {
                            auto nextOutput = nextStage->outputs[0];
                            assert(nextOutput != nullptr);

                            if (nextOutput->index == IndexBSS && nextOutput->parent == nullptr) {
                                auto newOutput = addNewData(
                                    newDataId(),
                                    [input, output, nextOutput](VpuData* data) {
                                        data->name = nextOutput->name + "@reshaped";
                                        data->index = IndexBSS;
                                        data->type = output->type;
                                        data->order = orderYXZ;
                                        data->dims = input->dims;
                                        data->strides = calcStrides(data->dims, data->type, data->order);
                                    },
                                    nextOutput);

                                auto cvtStage = addConvertStage(stageIt, input, newOutput);
                                cvtStage->name = stage->name + "+" + nextStage->name;

                                input->consumers.erase(stage);
                                stage->optimized = true;

                                nextOutput->producer = nullptr;
                                nextOutput->producerOutInd = -1;
                                nextStage->optimized = true;

                                continue;
                            }
                        }
                    }
                }
            }
        }

        // BiasReLU and BiasLeakyReLU has CHW variants

        if (stage->type == kBiasRelu || stage->type == kBiasLeakyRelu) {
            auto input = stage->inputs[0];
            assert(input != nullptr);

            if (input->order == orderZYX) {
                stage->type = stage->type == kBiasRelu ? kCHWBiasRelu : kCHWBiasLeakyRelu;
                stage->requiredInputOrder[0] = stage->requiredOutputOrder[0] = input->order;
                stage->requiredOutputAlignment[0] = 16u;
                stage->name += "@" + mvTensorStorageOrderToStr(input->order);
            }
        }

        // Bias has CHW variant

        if (stage->type == kBias) {
            auto input = stage->inputs[0];
            assert(input != nullptr);

            if (input->order == orderZYX) {
                stage->type = kCHWBias;
                stage->requiredInputOrder[0] = stage->requiredOutputOrder[0] = input->order;
                stage->requiredOutputAlignment[0] = 16u;
                stage->name += "@" + mvTensorStorageOrderToStr(input->order);
            }
        }

        // Scale/ScaleShift has CHW variant

        if (stage->type == kScale || stage->type == kScaleShift) {
            auto input = stage->inputs[0];
            assert(input != nullptr);

            if (input->order == orderZYX) {
                stage->type = stage->type == kScale ? kCHWScale : kCHWScaleShift;
                stage->requiredInputOrder[0] = stage->requiredOutputOrder[0] = input->order;
                stage->requiredInputAlignment[0] = stage->requiredOutputAlignment[0] = 16u;
                stage->name += "@" + mvTensorStorageOrderToStr(input->order);
            }
        }

        // check inputs

        for (size_t inputIdx = 0; inputIdx < stage->inputs.size(); ++inputIdx) {
            auto input = stage->inputs[inputIdx];
            assert(input != nullptr);

            auto reqOrder = stage->requiredInputOrder[inputIdx];
            if (input->order != reqOrder) {
                auto convertedData = findOrCreateConvertedData(convertedDataMap, input, reqOrder,
                                                               stage->requiredInputAlignment[inputIdx],
                                                               stageIt);

                stage->inputs[inputIdx] = convertedData;
                convertedData->consumers.insert(stage);
                input->consumers.erase(stage);
            } else if (reqOrder == orderZYX && input->strides[Dim::Y] % stage->requiredInputAlignment[inputIdx] != 0) {
                VpuDataHandle inputAligned = nullptr;

                auto it = alignedDataMap.find(input);
                if (it != alignedDataMap.end()) {
                    inputAligned = it->second;
                }

                if (inputAligned == nullptr) {
                    if (input->parent == nullptr) {
                        inputAligned = addAlignedData(input, stage->requiredInputAlignment[inputIdx]);
                        alignedDataMap.insert({input, inputAligned});

                        auto inputReshaped = reshapeZYXToYXZ(input);
                        auto inputAlignedReshaped = reshapeZYXToYXZ(inputAligned);

                        addCopyStage(input->name + "@align", nullptr, inputReshaped, inputAlignedReshaped, &stageIt);
                    } else {
                        auto parent = input->parent;

                        auto parentAligned = addAlignedData(parent, stage->requiredInputAlignment[inputIdx]);
                        alignedDataMap.insert({parent, parentAligned});

                        auto parentReshaped = reshapeZYXToYXZ(parent);
                        auto parentAlignedReshaped = reshapeZYXToYXZ(parentAligned);

                        addCopyStage(parent->name + "@align", nullptr, parentReshaped, parentAlignedReshaped, &stageIt);

                        for (auto subData : parent->subData) {
                            auto subDataAligned = addNewData(
                                newDataId(),
                                [subData, parentAligned](VpuData* data) {
                                    data->name = subData->name + "@aligned";
                                    data->index = parentAligned->index;
                                    data->type = parentAligned->type;
                                    data->order = parentAligned->order;
                                    data->dims = subData->dims;
                                    data->strides = parentAligned->strides;
                                    data->offsetFromParent = subData->offsetFromParent;
                                },
                                parentAligned);

                            alignedDataMap.insert({subData, subDataAligned});
                        }

                        it = alignedDataMap.find(input);
                        if (it == alignedDataMap.end()) {
                            THROW_IE_EXCEPTION << "[VPU] Internal error";
                        }

                        inputAligned = it->second;
                    }
                }

                input->consumers.erase(stage);
                stage->inputs[inputIdx] = inputAligned;
                inputAligned->consumers.insert(stage);
            }
        }

        // check outputs

        for (size_t outputIdx = 0; outputIdx < stage->outputs.size(); ++outputIdx) {
            auto output = stage->outputs[outputIdx];
            assert(output != nullptr);

            auto reqOrder = stage->requiredOutputOrder[outputIdx];
            if (output->order != reqOrder) {
                if (output->index == IndexBSS) {
                    if (output->parent == nullptr) {
                        // Just change the order of output, its consumers will convert it if needed

                        output->order = reqOrder;
                        output->strides = calcStrides(output->dims, output->type, output->order, stage->requiredOutputAlignment[outputIdx]);

                        loopOverSubData(output, [output](VpuDataHandle subData) {
                            subData->order = output->order;
                            subData->strides = output->strides;
                        });
                    } else {
                        // Stage output is a part of bigger data, need to analyze

                        auto topParent = getDataTopParent(output);
                        assert(topParent != nullptr);

                        if (topParentVisited.count(topParent) == 0) {
                            // If we check that top parent first time and there is too much conversion,
                            // change the order of the top parent and all its sub-data.
                            // Other producers and consumers will deal with new order.

                            topParent->order = reqOrder;
                            topParent->strides = calcStrides(topParent->dims, topParent->type, topParent->order, stage->requiredOutputAlignment[outputIdx]);

                            loopOverSubData(topParent, [topParent](VpuDataHandle subData) {
                                subData->order = topParent->order;
                                subData->strides = topParent->strides;
                            });
                        } else {
                            // Insert convert stage otherwise

                            auto nextIt = stageIt;
                            ++nextIt;

                            auto convertedData = addConvertedData(output, reqOrder, stage->requiredOutputAlignment[outputIdx]);
                            addConvertStage(nextIt, convertedData, output);

                            stage->outputs[outputIdx] = convertedData;
                            convertedData->producer = stage;
                            convertedData->producerOutInd = outputIdx;
                            output->producer = nullptr;
                            output->producerOutInd = -1;
                        }

                        topParentVisited.insert(topParent);
                    }
                } else if (output->index == IndexOutput) {
                    // Stage output is a network, need to insert convert stage
                    auto convertedData = addConvertedData(output, reqOrder, stage->requiredOutputAlignment[outputIdx]);
                    addConvertStage(_stages.end(), convertedData, output);

                    stage->outputs[outputIdx] = convertedData;
                    convertedData->producer = stage;
                    convertedData->producerOutInd = outputIdx;
                    if (stage->postOp != nullptr) {
                        output->consumers.erase(stage->postOp);
                        stage->postOp->inputs[0] = convertedData;
                        stage->postOp->outputs[0] = convertedData;
                        convertedData->consumers.insert(stage->postOp);
                    }
                } else {
                    THROW_IE_EXCEPTION << "[VPU] Unsupported data index " << mvDataIndexToStr(output->index) << " for stage output";
                }
            } else if (reqOrder == orderZYX && output->strides[Dim::Y] % stage->requiredOutputAlignment[outputIdx] != 0) {
                if (output->index == IndexBSS) {
                    if (output->parent == nullptr) {
                        // Just change the order of output, its consumers will convert it if needed

                        output->order = reqOrder;
                        output->strides = calcStrides(output->dims, output->type, output->order, stage->requiredOutputAlignment[outputIdx]);

                        loopOverSubData(output, [reqOrder, output](VpuDataHandle subData) {
                            subData->order = reqOrder;
                            subData->strides = output->strides;
                        });
                    } else {
                        // FIXME:
                        THROW_IE_EXCEPTION << "Stage output is a part of bigger data, do not support alignment for this case now";
                    }
                } else if (output->index == IndexOutput) {
                    auto nextIt = stageIt;
                    ++nextIt;
                    auto outputAligned = addAlignedData(output, stage->requiredOutputAlignment[outputIdx]);
                    stage->outputs[outputIdx] = outputAligned;
                    outputAligned->producer = stage;
                    outputAligned->producerOutInd = outputIdx;

                    auto outputAlignedReshaped = reshapeZYXToYXZ(outputAligned);
                    auto outputReshaped = reshapeZYXToYXZ(output);
                    addCopyStage(output->name + "@compact", stage->layer, outputAlignedReshaped, outputReshaped, &nextIt);
                }
            }
        }
    }
}

VpuDataHandle GraphTransformerImpl::findOrCreateConvertedData(
        std::unordered_map<VpuDataHandle, std::list<VpuDataHandle>, VpuDataHandleHash>& convertedDataMap,
        const VpuDataHandle& orig,
        t_MvTensorStorageOrder reqOrder,
        size_t reqAlignment,
        const std::list<VpuStagePtr>::iterator& stageIt) {
    VpuDataHandle convertedData;
    auto it = convertedDataMap.find(orig);
    if (it != convertedDataMap.end()) {
        for (auto data : it->second) {
            if (data->order == reqOrder) {
                convertedData = data;
                break;
            }
        }
    }

    if (convertedData == nullptr) {
        convertedData = addConvertedData(orig, reqOrder, reqAlignment);
        convertedDataMap[orig].push_back(convertedData);
        addConvertStage(stageIt, orig, convertedData);
    }

    return convertedData;
}

VpuDataHandle GraphTransformerImpl::addConvertedData(const VpuDataHandle& orig,
                                                   t_MvTensorStorageOrder order,
                                                   size_t alignment) {
    return addNewData(
        newDataId(),
        [orig, order, alignment](VpuData* data) {
            data->name = orig->name + "@" + mvTensorStorageOrderToStr(order);
            data->index = IndexBSS;
            data->type = orig->type;
            data->order = order;
            data->dims = orig->dims;
            data->strides = calcStrides(data->dims, data->type, data->order, alignment);
       });
}

VpuStageHandle GraphTransformerImpl::addConvertStage(const std::list<VpuStagePtr>::iterator& stageIt,
                                                   const VpuDataHandle& input,
                                                   const VpuDataHandle& output) {
    return addNewStage<VpuConvertHwSwStage>(
        std::string("cvt_order@")
                + input->name
                + "@" + mvTensorStorageOrderToStr(input->order)
                + "->"
                + mvTensorStorageOrderToStr(output->order),
        kConvertOrder,
        nullptr,
        [input, output](VpuConvertHwSwStage* stage) {
            stage->requiredInputOrder[0] = input->order;
            stage->requiredOutputOrder[0] = output->order;
        },
        {input},
        {output},
        nullptr,
        &stageIt);
}

VpuDataHandle GraphTransformerImpl::addAlignedData(const VpuDataHandle& orig, size_t alignment) {
    return addNewData(newDataId(),
                      [orig, alignment](VpuData* data) {
                          data->name = orig->name + "@aligned";
                          data->index = IndexBSS;
                          data->type = orig->type;
                          data->order = orig->order;
                          data->dims = orig->dims;
                          data->strides = calcStrides(data->dims, data->type, data->order, alignment);
                      });
}

VpuDataHandle GraphTransformerImpl::reshapeZYXToYXZ(const VpuDataHandle& origData) {
    return addNewData(
        newDataId(),
        [origData](VpuData* data) {
            data->name = origData->name + "@reshaped";
            data->index = origData->index;
            data->type = origData->type;
            data->order = orderYXZ;
            data->dims.resize(4);
            data->dims[Dim::Z] = origData->dims[Dim::X];
            data->dims[Dim::X] = origData->dims[Dim::Y];
            data->dims[Dim::Y] = origData->dims[Dim::Z];
            data->dims[Dim::N] = 1;
            data->strides.resize(3);
            data->strides[Dim::Z] = origData->strides[Dim::X];
            data->strides[Dim::X] = origData->strides[Dim::Y];
            data->strides[Dim::Y] = origData->strides[Dim::Z];
        },
        origData);
}

VpuDataHandle GraphTransformerImpl::reshapeYXZToZYX(const VpuDataHandle& origData) {
    return addNewData(
        newDataId(),
        [origData](VpuData* data) {
            data->name = origData->name + "@reshaped";
            data->index = origData->index;
            data->type = origData->type;
            data->order = orderZYX;
            data->dims.resize(4);
            data->dims[Dim::Z] = origData->dims[Dim::Y];
            data->dims[Dim::Y] = origData->dims[Dim::X];
            data->dims[Dim::X] = origData->dims[Dim::Z];
            data->dims[Dim::N] = 1;
            data->strides.resize(3);
            data->strides[Dim::Z] = origData->strides[Dim::Y];
            data->strides[Dim::Y] = origData->strides[Dim::X];
            data->strides[Dim::X] = origData->strides[Dim::Z];
        },
        origData);
}
