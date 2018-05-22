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
#include <tuple>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <vector>

void VpuMyriadXHwPoolingStage::dumpToDot(std::ostream& os) {
    os << "radixX=" << radixX << "\\n"
       << "radixY=" << radixY << "\\n"
       << "stride=" << stride << "\\n"
       << "poolType=" << poolType << "\\n"
       << "pad.enable=" << pad.enable << "\\n"
       << "pad.top=" << pad.top << "\\n"
       << "pad.bottom=" << pad.bottom << "\\n"
       << "pad.left=" << pad.left << "\\n"
       << "pad.right=" << pad.right << "\\n"
       << "newOutputDimZ=" << newOutputDimZ << "\\n"
       << "hasParallelCopy=" << hasParallelCopy << "\\n"
       << "descriptors.size=" << descriptors.size();
}

void VpuMyriadXHwPoolingStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<uint32_t>(hasParallelCopy));

    writer.write(static_cast<uint32_t>(descriptors.size()));
    for (auto& d : descriptors) {
        writer.write(d);
    }

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);

    VpuData fakeData;
    fakeData.dumpToBlob(writer);  // taps
    fakeData.dumpToBlob(writer);  // biases

    if (hasParallelCopy) {
        inputs[1]->dumpToBlob(writer);
        outputs[1]->dumpToBlob(writer);
    }
}

namespace {

std::tuple<uint32_t, VpuMyriadXHwPoolingStage::Tiles> splitPooling(uint32_t outZ) {
    VpuMyriadXHwPoolingStage::Tiles tiles;

    auto newOutZ = alignVal(outZ, 16u);

    for (uint32_t i = 0; i < newOutZ / 16u; ++i) {
        tiles.push_back(std::make_tuple(16u, MODE_16_16));
    }

    return std::make_tuple(newOutZ, std::move(tiles));
}

}  // namespace

void GraphTransformerImpl::processHWPool(
        const std::list<VpuStagePtr>::iterator& stageIt,
        uint32_t cmxLimit) {
    auto stage = *stageIt;
    assert(stage != nullptr);

    auto swStage = std::dynamic_pointer_cast<VpuPoolStage>(stage);
    assert(swStage != nullptr);

    // HW supports only same strides for X and Y
    if (swStage->strideX != swStage->strideY)
        return;

    // 3x3s2 Avg is not supported by HW
    if (swStage->radixX == 3 && swStage->radixY == 3 && swStage->strideX == 2) {
        if (stage->type == kAvgPool || swStage->padX != 0 || swStage->padY != 0) {
            return;
        }
    }

    auto input = stage->inputs[0];
    auto output = stage->outputs[0];

    VpuStageHandle postOp;
    VpuDataHandle biases;
    std::string hwStageNameSuffix;
    std::tie(postOp, biases, hwStageNameSuffix) = getPostOpInfoForHW(stage);

    auto pad = getPadding(input->dims, output->dims,
                          swStage->radixX, swStage->radixY,
                          swStage->strideX, swStage->strideY);

    uint32_t newOutputDimZ = 0;
    VpuMyriadXHwPoolingStage::Tiles tiles;
    std::tie(newOutputDimZ, tiles) = splitPooling(output->dims[Dim::Z]);

    if (tiles.empty()) {
        THROW_IE_EXCEPTION << "[VPU] Can't convert " << stage->name << " to HW stage";
    }

    auto hasRelu = isReluPostOp(postOp);

    bool splitOverHeight = input->dims[Dim::X] * input->dims[Dim::Y] / 1024.0 > 128;

    if (!splitOverHeight) {
        auto outBufSize = estimateHwBufferSize(output->dims);
        if (outBufSize > cmxLimit)
            splitOverHeight = true;
    }

    if (!splitOverHeight) {
        addNewStage<VpuMyriadXHwPoolingStage>(
            stage->name + "@HW" + hwStageNameSuffix,
            kMyriadXHwPooling,
            stage->layer,
            [swStage, pad, newOutputDimZ, &tiles, hasRelu](VpuMyriadXHwPoolingStage* stage) {
                stage->radixX = swStage->radixX;
                stage->radixY = swStage->radixY;
                stage->stride = swStage->strideX;

                stage->poolType = swStage->type == kMaxPool ? POOL_MAX : POOL_AVERAGE;

                stage->pad = pad;

                stage->newOutputDimZ = newOutputDimZ;
                stage->tiles = std::move(tiles);

                stage->hasRelu = hasRelu;

                stage->requiredInputOrder[0] = orderZYX;
                stage->requiredInputAlignment[0] = 16u;

                stage->requiredOutputOrder[0] = orderZYX;
                stage->requiredOutputAlignment[0] = 16u;
            },
            {input},
            {output},
            nullptr,
            &stageIt);
    } else {
        auto maxOutputLines = output->dims[Dim::Y];
        if (!tiles.empty()) {
            auto maxOutputChannelsInDescr = std::numeric_limits<uint32_t>::min();
            for (const auto& t : tiles) {
                maxOutputChannelsInDescr = std::max(maxOutputChannelsInDescr, std::get<0>(t));
            }

            uint32_t bytesPerFullDepthSlice = sizeof(ie_fp16) * maxOutputChannelsInDescr * alignVal(output->dims[Dim::X], 8u);

            maxOutputLines = cmxLimit / bytesPerFullDepthSlice;
        }

        auto heightSplits = heightSolution(
            input->dims[Dim::Y],
            swStage->radixY,
            swStage->strideY,
            std::make_tuple(pad.top > 0 ? swStage->radixY / 2 : 0,
                            pad.bottom > 0 ? swStage->radixY / 2 : 0),
            maxOutputLines);

        if (heightSplits.empty() || heightSplits.size() == 1) {
            addNewStage<VpuMyriadXHwPoolingStage>(
                stage->name + "@HW" + hwStageNameSuffix,
                kMyriadXHwPooling,
                stage->layer,
                [swStage, pad, newOutputDimZ, &tiles, hasRelu](VpuMyriadXHwPoolingStage* stage) {
                    stage->radixX = swStage->radixX;
                    stage->radixY = swStage->radixY;
                    stage->stride = swStage->strideX;

                    stage->poolType = swStage->type == kMaxPool ? POOL_MAX : POOL_AVERAGE;

                    stage->pad = pad;

                    stage->newOutputDimZ = newOutputDimZ;
                    stage->tiles = std::move(tiles);

                    stage->hasRelu = hasRelu;

                    stage->requiredInputOrder[0] = orderZYX;
                    stage->requiredInputAlignment[0] = 16u;

                    stage->requiredOutputOrder[0] = orderZYX;
                    stage->requiredOutputAlignment[0] = 16u;
                },
                {input},
                {output},
                nullptr,
                &stageIt);
        } else {
            output->producer = nullptr;
            output->producerOutInd = -1;

            std::vector<VpuDataHandle> copyInputs;
            std::vector<VpuDataHandle> copyOutputs;

            int tileInd = 0;
            for (const auto& heightSplitSol : heightSplits) {
                int inputWithJunk, outputWithJunk;
                int outputJunkBefore, outputJunkAfter;
                int inputStartIndex, inputEndIndex;
                int outputStartIndex, outputEndIndex;
                std::tie(inputWithJunk, outputWithJunk,
                         outputJunkBefore, outputJunkAfter,
                         inputStartIndex, inputEndIndex,
                         outputStartIndex, outputEndIndex) =
                    heightSplitSol;

                auto subInput = addNewData(
                    newDataId(),
                    [input, inputWithJunk, inputStartIndex, tileInd](VpuData* data) {
                        data->name = input->name + "@sub" + std::to_string(tileInd);
                        data->index = input->index;
                        data->type = input->type;
                        data->order = input->order;
                        data->dims = VpuDims({input->dims[Dim::X], static_cast<uint32_t>(inputWithJunk), input->dims[Dim::Z]});
                        data->strides = input->strides;
                        data->offsetFromParent = VpuDims({0u, static_cast<uint32_t>(inputStartIndex), 0u});
                    },
                    input);

                if (outputJunkBefore == 0 && outputJunkAfter == 0) {
                    auto subOutput = addNewData(
                        newDataId(),
                        [output, outputWithJunk, outputStartIndex, tileInd](VpuData* data) {
                            data->name = output->name + "@sub" + std::to_string(tileInd);
                            data->index = output->index;
                            data->type = output->type;
                            data->order = output->order;
                            data->dims = VpuDims({output->dims[Dim::X], static_cast<uint32_t>(outputWithJunk), output->dims[Dim::Z]});
                            data->strides = output->strides;
                            data->offsetFromParent = VpuDims({0u, static_cast<uint32_t>(outputStartIndex), 0u});
                        },
                        output);

                    if (copyInputs.empty() || copyInputs.back() == nullptr) {
                        addNewStage<VpuMyriadXHwPoolingStage>(
                            stage->name + "@HW" + hwStageNameSuffix + "@soh" + std::to_string(tileInd),
                            kMyriadXHwPooling,
                            stage->layer,
                            [swStage, pad, newOutputDimZ, &tiles, hasRelu](VpuMyriadXHwPoolingStage* stage) {
                                stage->radixX = swStage->radixX;
                                stage->radixY = swStage->radixY;
                                stage->stride = swStage->strideX;

                                stage->poolType = swStage->type == kMaxPool ? POOL_MAX : POOL_AVERAGE;

                                stage->pad = pad;

                                stage->newOutputDimZ = newOutputDimZ;
                                stage->tiles = tiles;

                                stage->hasRelu = hasRelu;

                                stage->requiredInputOrder[0] = orderZYX;
                                stage->requiredInputAlignment[0] = 16u;

                                stage->requiredOutputOrder[0] = orderZYX;
                                stage->requiredOutputAlignment[0] = 16u;
                            },
                            {subInput},
                            {subOutput},
                            nullptr,
                            &stageIt);
                    } else {
                        addNewStage<VpuMyriadXHwPoolingStage>(
                            stage->name + "@HW" + hwStageNameSuffix + "@soh" + std::to_string(tileInd) + "+Copy",
                            kMyriadXHwPooling,
                            stage->layer,
                            [swStage, pad, newOutputDimZ, &tiles, hasRelu](VpuMyriadXHwPoolingStage* stage) {
                                stage->radixX = swStage->radixX;
                                stage->radixY = swStage->radixY;
                                stage->stride = swStage->strideX;

                                stage->poolType = swStage->type == kMaxPool ? POOL_MAX : POOL_AVERAGE;

                                stage->pad = pad;

                                stage->newOutputDimZ = newOutputDimZ;
                                stage->tiles = tiles;

                                stage->hasRelu = hasRelu;

                                stage->hasParallelCopy = true;

                                stage->requiredInputOrder[0] = orderZYX;
                                stage->requiredInputAlignment[0] = 16u;

                                stage->requiredInputOrder[1] = orderZYX;
                                stage->requiredInputAlignment[1] = 16u;

                                stage->requiredOutputOrder[0] = orderZYX;
                                stage->requiredOutputAlignment[0] = 16u;

                                stage->requiredOutputOrder[1] = orderZYX;
                                stage->requiredOutputAlignment[1] = 16u;
                            },
                            {subInput, copyInputs.back()},
                            {subOutput, copyOutputs.back()},
                            nullptr,
                            &stageIt);
                    }

                    copyInputs.push_back(nullptr);
                    copyOutputs.push_back(nullptr);
                } else {
                    auto subPoolOutput = addNewData(
                        newDataId(),
                        [output, outputWithJunk, tileInd](VpuData* data) {
                            data->name = output->name + "@subPool" + std::to_string(tileInd);
                            data->index = IndexBSS;
                            data->type = output->type;
                            data->order = orderZYX;
                            data->dims = VpuDims({output->dims[Dim::X], static_cast<uint32_t>(outputWithJunk), output->dims[Dim::Z]});
                            data->strides = calcStrides(data->dims, data->type, data->order, 16u);
                        });

                    auto subPoolOutputInner = addNewData(
                        newDataId(),
                        [subPoolOutput, outputJunkBefore, outputStartIndex, outputEndIndex](VpuData* data) {
                            data->name = subPoolOutput->name + "@inner";
                            data->index = subPoolOutput->index;
                            data->type = subPoolOutput->type;
                            data->order = subPoolOutput->order;
                            uint32_t outTileHeight = outputEndIndex - outputStartIndex;
                            data->dims = VpuDims({subPoolOutput->dims[Dim::X], outTileHeight, subPoolOutput->dims[Dim::Z]});
                            data->strides = subPoolOutput->strides;
                            data->offsetFromParent = VpuDims({0u, static_cast<uint32_t>(outputJunkBefore), 0u});
                        },
                        subPoolOutput);

                    auto subOutput = addNewData(
                        newDataId(),
                        [output, subPoolOutputInner, outputStartIndex, tileInd](VpuData* data) {
                            data->name = output->name + "@sub" + std::to_string(tileInd);
                            data->index = output->index;
                            data->type = output->type;
                            data->order = output->order;
                            data->dims = VpuDims({output->dims[Dim::X], subPoolOutputInner->dims[Dim::Y], output->dims[Dim::Z]});
                            data->strides = output->strides;
                            data->offsetFromParent = VpuDims({0u, static_cast<uint32_t>(outputStartIndex), 0u});
                        },
                        output);

                    if (copyInputs.empty() || copyInputs.back() == nullptr) {
                        addNewStage<VpuMyriadXHwPoolingStage>(
                            stage->name + "@HW" + hwStageNameSuffix + "@soh" + std::to_string(tileInd),
                            kMyriadXHwPooling,
                            stage->layer,
                            [swStage, pad, newOutputDimZ, &tiles, hasRelu](VpuMyriadXHwPoolingStage* stage) {
                                stage->radixX = swStage->radixX;
                                stage->radixY = swStage->radixY;
                                stage->stride = swStage->strideX;

                                stage->poolType = swStage->type == kMaxPool ? POOL_MAX : POOL_AVERAGE;

                                stage->pad = pad;

                                stage->newOutputDimZ = newOutputDimZ;
                                stage->tiles = tiles;

                                stage->hasRelu = hasRelu;

                                stage->requiredInputOrder[0] = orderZYX;
                                stage->requiredInputAlignment[0] = 16u;

                                stage->requiredOutputOrder[0] = orderZYX;
                                stage->requiredOutputAlignment[0] = 16u;
                            },
                            {subInput},
                            {subPoolOutput},
                            nullptr,
                            &stageIt);
                    } else {
                        addNewStage<VpuMyriadXHwPoolingStage>(
                            stage->name + "@HW" + hwStageNameSuffix + "@soh" + std::to_string(tileInd) + "+Copy",
                            kMyriadXHwPooling,
                            stage->layer,
                            [swStage, pad, newOutputDimZ, &tiles, hasRelu](VpuMyriadXHwPoolingStage* stage) {
                                stage->radixX = swStage->radixX;
                                stage->radixY = swStage->radixY;
                                stage->stride = swStage->strideX;

                                stage->poolType = swStage->type == kMaxPool ? POOL_MAX : POOL_AVERAGE;

                                stage->pad = pad;

                                stage->newOutputDimZ = newOutputDimZ;
                                stage->tiles = tiles;

                                stage->hasRelu = hasRelu;

                                stage->hasParallelCopy = true;

                                stage->requiredInputOrder[0] = orderZYX;
                                stage->requiredInputAlignment[0] = 16u;

                                stage->requiredInputOrder[1] = orderZYX;
                                stage->requiredInputAlignment[1] = 16u;

                                stage->requiredOutputOrder[0] = orderZYX;
                                stage->requiredOutputAlignment[0] = 16u;

                                stage->requiredOutputOrder[1] = orderZYX;
                                stage->requiredOutputAlignment[1] = 16u;
                            },
                            {subInput, copyInputs.back()},
                            {subPoolOutput, copyOutputs.back()},
                            nullptr,
                            &stageIt);
                    }

                    copyInputs.push_back(subPoolOutputInner);
                    copyOutputs.push_back(subOutput);
                }

                ++tileInd;
            }

            if (!copyInputs.empty() && copyInputs.back() != nullptr) {
                addNewStage<VpuCopyStage>(
                    stage->name + "@copy",
                    kCopyMakeBorderCHW,
                    stage->layer,
                    [](VpuCopyStage* stage) {
                        stage->requiredInputOrder[0] = orderZYX;
                        stage->requiredInputAlignment[0] = 16u;

                        stage->requiredOutputOrder[0] = orderZYX;
                        stage->requiredOutputAlignment[0] = 16u;
                    },
                    {copyInputs.back()},
                    {copyOutputs.back()},
                    nullptr,
                    &stageIt);
            }
        }
    }

    input->consumers.erase(stage);
    stage->optimized = true;

    if (postOp != nullptr) {
        postOp->optimized = true;
    }
}
