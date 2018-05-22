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
#include <utility>
#include <memory>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <vector>

void VpuMyriadXHwConvolutionStage::dumpToDot(std::ostream& os) {
    os << "radixX=" << radixX << "\\n"
       << "radixY=" << radixY << "\\n"
       << "stride=" << stride << "\\n"
       << "pad.enable=" << pad.enable << "\\n"
       << "pad.top=" << pad.top << "\\n"
       << "pad.bottom=" << pad.bottom << "\\n"
       << "pad.left=" << pad.left << "\\n"
       << "pad.right=" << pad.right << "\\n"
       << "newInputDimZ=" << newInputDimZ << "\\n"
       << "newOutputDimZ=" << newOutputDimZ << "\\n"
       << "withPool=" << withPool << "\\n"
       << "poolRadX=" << poolRadX << "\\n"
       << "poolRadY=" << poolRadY << "\\n"
       << "hasParallelCopy=" << hasParallelCopy << "\\n"
       << "descriptors.size=" << descriptors.size();
}

void VpuMyriadXHwConvolutionStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<uint32_t>(hasParallelCopy));

    writer.write(static_cast<uint32_t>(descriptors.size()));
    for (auto& d : descriptors) {
        writer.write(d);
    }

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
    inputs[1]->dumpToBlob(writer);  // taps
    inputs[2]->dumpToBlob(writer);  // biases

    if (hasParallelCopy) {
        inputs[3]->dumpToBlob(writer);
        outputs[1]->dumpToBlob(writer);
    }
}

namespace {

std::tuple<bool, uint32_t> checkModeForValidity(uint32_t iX, uint32_t iY, uint32_t iZ,
                                                uint32_t oX, uint32_t oY, uint32_t oZ,
                                                uint32_t kX, uint32_t kY, uint32_t kS,
                                                cnnDataMode dataType, cnnCoefficientMode coeffType,
                                                cnnOperationMode mode) {
    const uint32_t CNNHW_INTERNAL_MEMORY_SIZE = 128 * 1024;
    std::array<uint32_t, 6> COEFF_PER_WORD_VALUES{{1u, 2u, 4u, 8u, 16u, 16u}};
    std::array<uint32_t, 2> BYTES_PER_PIXEL{{2u, 1u}};
    std::array<uint32_t, 5> MODES_COST{{0u, 5u, 11u, 19u, 31u}};

    auto noOfBlocks = 1u << mode;
    auto inChansPerBlock = iZ / noOfBlocks;
    auto coeffSetSize = kX * kY;

#if 0
    auto bytesPerLine = alignVal(iX * BYTES_PER_PIXEL[dataType], 16u);
    auto linesPerChannel = std::min(CNNHW_INTERNAL_MEMORY_SIZE / (bytesPerLine * iZ), std::min(iY, 512u));
#endif

    auto coeffPerWord = COEFF_PER_WORD_VALUES[coeffType];
    auto coeffLPB = (inChansPerBlock * coeffSetSize + coeffPerWord - 1) / coeffPerWord;

    if (iX > 4096 || iY > 4096 || iZ > 2048 || oZ > 2048)
        return std::make_tuple(false, 0);
    if (kX > 16 || kY > 16 || kS > 16)
        return std::make_tuple(false, 0);
    if (inChansPerBlock > 2048)
        return std::make_tuple(false, 0);
    if (coeffLPB > 256)
        return std::make_tuple(false, 0);

#if 0
    if (((oX / kS) <= 4) && (linesPerChannel <= (kY + 2 * (kS + 1) + 1)))
        return std::make_tuple(false, 0);
    if (((oX / kS) > 4) && (linesPerChannel <= (kY + kS + 1 + 1)))
        return std::make_tuple(false, 0);
#endif

    if (noOfBlocks > iZ)
        return std::make_tuple(false, 0);

    // TODO : this check fixes VGG network
    if (iY > kY + 1) {
        auto bytesPerPixel = BYTES_PER_PIXEL[dataType];
        auto pixelsPerCMXLine = 128u / (bytesPerPixel * 8u);
        auto localLineStride = (iX + (pixelsPerCMXLine - 1)) / pixelsPerCMXLine;
        auto bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel;
        auto sizeOfBlock = CNNHW_INTERNAL_MEMORY_SIZE >> mode;
        auto chanPerBlock = iZ / noOfBlocks;
        auto availableBytesPerChan = sizeOfBlock / chanPerBlock;
        auto linesPerChan = std::min(availableBytesPerChan / bytesPerLine, iY);
        auto minLines = kY + 1;

        if (minLines > linesPerChan - 1u)
            return std::make_tuple(false, 0);
    }

    return std::make_tuple(true, (iZ / noOfBlocks) * kX * kY + MODES_COST[mode]);
}

// This function splits the convolution operation into uniform pieces
std::tuple<uint32_t, uint32_t, VpuMyriadXHwConvolutionStage::Tiles> splitConvolution(
        uint32_t iX, uint32_t iY, uint32_t iZ,
        uint32_t oX, uint32_t oY, uint32_t oZ,
        uint32_t kX, uint32_t kY, uint32_t kS,
        cnnDataMode dataType, cnnCoefficientMode coeffType,
        const std::vector<cnnOperationMode>& modes = {MODE_1_256, MODE_2_128, MODE_4_64, MODE_8_32, MODE_16_16}) {
    using ConvSolution = std::tuple<cnnOperationMode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>;
    using ConvSolutions = std::vector<ConvSolution>;

    ConvSolutions solutions;
    for (auto mode : modes) {
        auto ramBlocks = 1u << mode;
        auto newInZ = alignVal(iZ, ramBlocks);
        auto newOutZ = alignVal(oZ, 8u);
        auto maxOc = std::min(256u / ramBlocks, newOutZ);

        for (uint32_t i = maxOc / 8u; i >= 1; --i) {
            auto oChansPerDescr = 8u * i;

            bool valid;
            uint32_t cost;
            std::tie(valid, cost) = checkModeForValidity(iX, iY, newInZ,
                                                         oX, oY, oChansPerDescr,
                                                         kX, kY, kS,
                                                         dataType, coeffType,
                                                         mode);

            if (valid) {
                auto numDescr = newOutZ / oChansPerDescr;
                auto remOChans = newOutZ % oChansPerDescr;
                if (remOChans != 0) {
                    ++numDescr;
                }
                solutions.push_back(std::make_tuple(mode, newInZ, newOutZ, numDescr,
                                                    oChansPerDescr, remOChans, numDescr * cost));
            }
        }
    }

    if (!solutions.empty()) {
        auto minNumDesc = std::numeric_limits<uint32_t>::max();
        for (const auto& s : solutions) {
            minNumDesc = std::min(minNumDesc, std::get<3>(s));
        }

        // filter out solutions != minNumDesc
        {
            ConvSolutions tmpSolutions;
            for (const auto& s : solutions) {
                if (std::get<3>(s) == minNumDesc)
                    tmpSolutions.push_back(s);
            }
            solutions.swap(tmpSolutions);
        }

        auto minCost = std::numeric_limits<uint32_t>::max();
        for (const auto& s : solutions) {
            minCost = std::min(minCost, std::get<6>(s));
        }

        // filter out solutions != minCost
        {
            ConvSolutions tmpSolutions;
            for (const auto& s : solutions) {
                if (std::get<6>(s) == minCost)
                    tmpSolutions.push_back(s);
            }
            solutions.swap(tmpSolutions);
        }

        cnnOperationMode mode;
        uint32_t newInZ, newOutZ, oChansPerDescr, remOChans;
        uint32_t _;
        std::tie(mode, newInZ, newOutZ, _, oChansPerDescr, remOChans, _) = solutions[0];

        VpuMyriadXHwConvolutionStage::Tiles tiles(newOutZ / oChansPerDescr,
                                                  std::make_tuple(oChansPerDescr, mode));
        if (remOChans != 0) {
            tiles.push_back(std::make_tuple(remOChans, mode));
        }

        // Sum the tiles. If it is more than oZ, go to the last tile and change it to mode 0
        uint32_t totalOutChans = 0;
        for (const auto& t : tiles) {
            totalOutChans += std::get<0>(t);
        }
        if (totalOutChans > oZ) {
            VpuMyriadXHwConvolutionStage::Tiles newTiles;
            if (tiles.size() == 1) {
                newTiles.push_back(std::make_tuple(oZ, std::get<1>(tiles[0])));
            } else {
                for (size_t i = 0; i < tiles.size() - 1; ++i) {
                    newTiles.push_back(tiles[i]);
                }
                uint32_t almostTotalOutChans = 0;
                for (const auto& t : newTiles) {
                    almostTotalOutChans += std::get<0>(t);
                }
                newTiles.push_back(std::make_tuple(oZ - almostTotalOutChans, std::get<1>(tiles[0])));
            }
            tiles = newTiles;
        }

        return std::make_tuple(newInZ, newOutZ, std::move(tiles));
    }

    return std::make_tuple(0, 0, VpuMyriadXHwConvolutionStage::Tiles());
}

}  // namespace

void GraphTransformerImpl::addHWConv(const std::list<VpuStagePtr>::iterator& stageIt,
                                     VpuDataHandle input,
                                     VpuDataHandle output,
                                     float scale,
                                     Handle<VpuPoolStage> postPoolStage,
                                     bool isLastTile,
                                     const std::string& extraSuffix,
                                     VpuDataHandle copyInput,
                                     VpuDataHandle copyOutput) {
    auto stage = *stageIt;
    assert(stage != nullptr);

    auto swStage = std::dynamic_pointer_cast<VpuConvStage>(stage);
    assert(swStage != nullptr);

    auto convLayer = std::dynamic_pointer_cast<ConvolutionLayer>(stage->layer);
    assert(convLayer != nullptr);

    // For HW 1x1s1 with padding we add CopyMakeBorder stage to pad input with zeros,
    // since built-in padding seems doesn't work in this case
    if (swStage->radixX == 1 && swStage->radixY == 1 &&
        swStage->strideX == 1 && swStage->strideY == 1 &&
        (swStage->padX != 0 || swStage->padY != 0)) {
        auto paddedInput = addNewData(
            newDataId(),
            [input, output, extraSuffix](VpuData* data) {
                data->name = input->name + "@padded" + extraSuffix;
                data->index = IndexBSS;
                data->type = VpuDataType::FP16;
                data->order = orderZYX;
                data->dims = VpuDims({output->dims[Dim::X], output->dims[Dim::Y], input->dims[Dim::Z]});
                data->strides = calcStrides(data->dims, data->type, data->order, 16u);
            });

        addNewStage<VpuCopyStage>(
            stage->name + "@padding" + extraSuffix,
            kCopyMakeBorderCHW,
            stage->layer,
            [](VpuCopyStage* stage) {
                stage->requiredInputOrder[0] = orderZYX;
                stage->requiredInputAlignment[0] = 16u;

                stage->requiredOutputOrder[0] = orderZYX;
                stage->requiredOutputAlignment[0] = 16u;
            },
            {input},
            {paddedInput},
            nullptr,
            &stageIt);

        input = paddedInput;
    }

    auto weights = stage->inputs[1];
    assert(weights != nullptr);

    VpuStageHandle postOp;
    VpuDataHandle biases;
    std::string hwStageNameSuffix;
    std::tie(postOp, biases, hwStageNameSuffix) = getPostOpInfoForHW(stage);

    auto pad = getPadding(input->dims, output->dims,
                          swStage->radixX, swStage->radixY,
                          swStage->strideX, swStage->strideY);

    uint32_t newInputDimZ = 0, newOutputDimZ = 0;
    VpuMyriadXHwConvolutionStage::Tiles tiles;
    std::tie(newInputDimZ, newOutputDimZ, tiles)
            = splitConvolution(input->dims[Dim::X], input->dims[Dim::Y], input->dims[Dim::Z],
                               output->dims[Dim::X], output->dims[Dim::Y], output->dims[Dim::Z],
                               swStage->radixX, swStage->radixY, swStage->strideX,
                               MODE_FP16, FP16_COEFF);

    uint32_t inputTileDimZ = newInputDimZ;
    uint32_t numInputTiles = 1;

    if (tiles.empty()) {
        // Split over output failed - try to split over input too.

        if (postPoolStage != nullptr) {
            THROW_IE_EXCEPTION << "[VPU] Internal error";
        }

        std::array<uint32_t, 8> TILE_SIZE_CANDIDATES{{512u, 256u, 128u, 64u, 32u, 16u, 8u, 4u}};

        for (auto curTileSize : TILE_SIZE_CANDIDATES) {
            // TODO : support any number of input channels
            if (input->dims[Dim::Z] > curTileSize &&
                input->dims[Dim::Z] % curTileSize == 0) {
                inputTileDimZ = curTileSize;
                numInputTiles = input->dims[Dim::Z] / inputTileDimZ;

                std::tie(newInputDimZ, newOutputDimZ, tiles)
                        = splitConvolution(input->dims[Dim::X], input->dims[Dim::Y], inputTileDimZ,
                                           output->dims[Dim::X], output->dims[Dim::Y], output->dims[Dim::Z],
                                           swStage->radixX, swStage->radixY, swStage->strideX,
                                           MODE_FP16, FP16_COEFF);

                // TODO : support any number of output channels
                if (newInputDimZ == inputTileDimZ && !tiles.empty()) {
                    break;
                } else {
                    tiles.clear();
                }
            }
        }

        if (tiles.empty()) {
            THROW_IE_EXCEPTION << "[VPU] Can't convert " << stage->name << " to HW stage";
        }
    }

    VpuDims origWeightsDims(4);
    origWeightsDims[Dim::X] = swStage->radixY;
    origWeightsDims[Dim::Y] = swStage->radixX;
    origWeightsDims[Dim::Z] = input->dims[Dim::Z];
    origWeightsDims[Dim::N] = output->dims[Dim::Z];

    VpuDims hwWeigthsDims(4);
    hwWeigthsDims[Dim::X] = 8;
    hwWeigthsDims[Dim::Y] = origWeightsDims[Dim::X] * origWeightsDims[Dim::Y];
    hwWeigthsDims[Dim::Z] = newInputDimZ;
    hwWeigthsDims[Dim::N] = newOutputDimZ / 8;

    weights->index = IndexNone;
    weights->writer = nullptr;

    auto hasRelu = isReluPostOp(postOp);

    auto biasesHW = biases;
    if (scale != 1.0f && biases != nullptr && biases->index != IndexNone) {
        biasesHW = addNewData(
            newDataId(),
            [biases, extraSuffix, convLayer, scale](VpuData* data) {
                data->name = biases->name + "@HW" + extraSuffix;
                data->index = IndexBlob;
                data->type = VpuDataType::FP16;
                data->dims = biases->dims;
                data->order = biases->order;
                data->strides = biases->strides;
                data->writer = std::make_shared<ScaledBiasesWriter>(convLayer->_biases, scale);
            });

        if (isLastTile) {
            biases->index = IndexNone;
            biases->writer = nullptr;
        }
    }

    if (postPoolStage != nullptr) {
        hwStageNameSuffix += "+Pool";
    }

    if (numInputTiles == 1) {
        auto weightsHW = addNewData(
            newDataId(),
            [weights, origWeightsDims, hwWeigthsDims, convLayer, numInputTiles, extraSuffix, scale](VpuData* data) {
                data->name = weights->name + "@HW" + extraSuffix;
                data->index = IndexBlob;
                data->type = VpuDataType::FP16;
                data->dims = hwWeigthsDims;
                data->order = orderZYX;
                data->strides.resize(4);
                data->strides[Dim::X] = getDataTypeSize(data->type);
                data->strides[Dim::Y] = hwWeigthsDims[Dim::X] * data->strides[Dim::X];
                data->strides[Dim::Z] = hwWeigthsDims[Dim::Y] * data->strides[Dim::Y];
                data->strides[Dim::N] = hwWeigthsDims[Dim::Z] * data->strides[Dim::Z];
                data->writer = std::make_shared<HwWeightsWriter>(convLayer->_weights,
                                                                 origWeightsDims, hwWeigthsDims,
                                                                 1, 0,
                                                                 scale);
            });

        if (copyInput == nullptr) {
            addNewStage<VpuMyriadXHwConvolutionStage>(
                stage->name + "@HW" + hwStageNameSuffix + extraSuffix,
                kMyriadXHwConvolution,
                stage->layer,
                [swStage, pad, newInputDimZ, newOutputDimZ, &tiles, hasRelu, postPoolStage](VpuMyriadXHwConvolutionStage* stage) {
                    stage->radixX = swStage->radixX;
                    stage->radixY = swStage->radixY;
                    stage->stride = swStage->strideX;

                    stage->pad = pad;

                    stage->newInputDimZ = newInputDimZ;
                    stage->newOutputDimZ = newOutputDimZ;
                    stage->tiles = std::move(tiles);

                    stage->hasRelu = hasRelu;

                    if (postPoolStage != nullptr) {
                        stage->withPool = true;
                        stage->poolRadX = postPoolStage->radixX;
                        stage->poolRadY = postPoolStage->radixY;
                    }

                    stage->requiredInputOrder[0] = orderZYX;
                    stage->requiredInputAlignment[0] = 16u;

                    stage->requiredOutputOrder[0] = orderZYX;
                    stage->requiredOutputAlignment[0] = 16u;
                },
                {input, weightsHW, biasesHW},
                {output},
                nullptr,
                &stageIt);
        } else {
            assert(copyOutput != nullptr);

            addNewStage<VpuMyriadXHwConvolutionStage>(
                stage->name + "@HW" + hwStageNameSuffix + extraSuffix + "+Copy",
                kMyriadXHwConvolution,
                stage->layer,
                [swStage, pad, newInputDimZ, newOutputDimZ, &tiles, hasRelu, postPoolStage](VpuMyriadXHwConvolutionStage* stage) {
                    stage->radixX = swStage->radixX;
                    stage->radixY = swStage->radixY;
                    stage->stride = swStage->strideX;

                    stage->pad = pad;

                    stage->newInputDimZ = newInputDimZ;
                    stage->newOutputDimZ = newOutputDimZ;
                    stage->tiles = std::move(tiles);

                    stage->hasRelu = hasRelu;

                    if (postPoolStage != nullptr) {
                        stage->withPool = true;
                        stage->poolRadX = postPoolStage->radixX;
                        stage->poolRadY = postPoolStage->radixY;
                    }

                    stage->hasParallelCopy = true;

                    stage->requiredInputOrder[0] = orderZYX;
                    stage->requiredInputAlignment[0] = 16u;

                    stage->requiredInputOrder[3] = orderZYX;
                    stage->requiredInputAlignment[3] = 16u;

                    stage->requiredOutputOrder[0] = orderZYX;
                    stage->requiredOutputAlignment[0] = 16u;

                    stage->requiredOutputOrder[1] = orderZYX;
                    stage->requiredOutputAlignment[1] = 16u;
                },
                {input, weightsHW, biasesHW, copyInput},
                {output, copyOutput},
                nullptr,
                &stageIt);
        }
    } else {
        auto fakeBiases = addNewData(
            newDataId(),
            [](VpuData* data) {
                data->name = "fake";
            });

        std::vector<VpuDataHandle> subConvOutputs;
        std::vector<VpuDataHandle> subSumOutputs;

        for (int inputTileInd = 0; inputTileInd < numInputTiles; ++inputTileInd) {
            auto subInput = addNewData(
                newDataId(),
                [input, inputTileInd, inputTileDimZ, extraSuffix](VpuData* data) {
                    data->name = input->name + "@sub@" + std::to_string(inputTileInd) + extraSuffix;
                    data->index = input->index;
                    data->type = input->type;
                    data->order = input->order;
                    data->dims = VpuDims({input->dims[Dim::X], input->dims[Dim::Y], static_cast<uint32_t>(inputTileDimZ)});
                    data->strides = input->strides;
                    data->offsetFromParent = VpuDims({0u, 0u, static_cast<uint32_t>(inputTileInd * inputTileDimZ)});
                },
                input);

            auto subWeightsHW = addNewData(
                newDataId(),
                [weights, origWeightsDims, hwWeigthsDims, convLayer, inputTileInd, numInputTiles, extraSuffix, scale](VpuData* data) {
                    data->name = weights->name + "@HW@sub@" + std::to_string(inputTileInd) + extraSuffix;
                    data->index = IndexBlob;
                    data->type = VpuDataType::FP16;
                    data->dims = hwWeigthsDims;
                    data->order = orderZYX;
                    data->strides.resize(4);
                    data->strides[Dim::X] = getDataTypeSize(data->type);
                    data->strides[Dim::Y] = hwWeigthsDims[Dim::X] * data->strides[Dim::X];
                    data->strides[Dim::Z] = hwWeigthsDims[Dim::Y] * data->strides[Dim::Y];
                    data->strides[Dim::N] = hwWeigthsDims[Dim::Z] * data->strides[Dim::Z];
                    data->writer = std::make_shared<HwWeightsWriter>(convLayer->_weights,
                                                                     origWeightsDims, hwWeigthsDims,
                                                                     numInputTiles, inputTileInd,
                                                                     scale);
                });

            auto subConvOutput = addNewData(
                newDataId(),
                [output, inputTileInd, extraSuffix](VpuData* data) {
                    data->name = output->name + "@subConv@" + std::to_string(inputTileInd) + extraSuffix;
                    data->index = IndexBSS;
                    data->type = output->type;
                    data->dims = output->dims;
                    data->order = orderZYX;
                    data->strides = calcStrides(data->dims, data->type, data->order, 16u);
                });
            subConvOutputs.push_back(subConvOutput);

            if (copyInput == nullptr || inputTileInd > 0) {
                addNewStage<VpuMyriadXHwConvolutionStage>(
                    stage->name + "@HW@sod" + std::to_string(inputTileInd) + extraSuffix,
                    kMyriadXHwConvolution,
                    stage->layer,
                    [swStage, pad, newInputDimZ, newOutputDimZ, &tiles](VpuMyriadXHwConvolutionStage* stage) {
                        stage->radixX = swStage->radixX;
                        stage->radixY = swStage->radixY;
                        stage->stride = swStage->strideX;

                        stage->pad = pad;

                        stage->newInputDimZ = newInputDimZ;
                        stage->newOutputDimZ = newOutputDimZ;
                        stage->tiles = tiles;

                        stage->hasRelu = false;

                        stage->requiredInputOrder[0] = orderZYX;
                        stage->requiredInputAlignment[0] = 16u;

                        stage->requiredOutputOrder[0] = orderZYX;
                        stage->requiredOutputAlignment[0] = 16u;
                    },
                    {subInput, subWeightsHW, inputTileInd == 0 ? biasesHW : fakeBiases},
                    {subConvOutput},
                    nullptr,
                    &stageIt);
            } else {
                assert(copyOutput != nullptr);

                addNewStage<VpuMyriadXHwConvolutionStage>(
                    stage->name + "@HW@sod" + std::to_string(inputTileInd) + extraSuffix + "+Copy",
                    kMyriadXHwConvolution,
                    stage->layer,
                    [swStage, pad, newInputDimZ, newOutputDimZ, &tiles](VpuMyriadXHwConvolutionStage* stage) {
                        stage->radixX = swStage->radixX;
                        stage->radixY = swStage->radixY;
                        stage->stride = swStage->strideX;

                        stage->pad = pad;

                        stage->newInputDimZ = newInputDimZ;
                        stage->newOutputDimZ = newOutputDimZ;
                        stage->tiles = tiles;

                        stage->hasRelu = false;

                        stage->hasParallelCopy = true;

                        stage->requiredInputOrder[0] = orderZYX;
                        stage->requiredInputAlignment[0] = 16u;

                        stage->requiredOutputOrder[0] = orderZYX;
                        stage->requiredOutputAlignment[0] = 16u;
                    },
                    {subInput, subWeightsHW, inputTileInd == 0 ? biasesHW : fakeBiases, copyInput},
                    {subConvOutput, copyOutput},
                    nullptr,
                    &stageIt);
            }

            if (inputTileInd > 0) {
                auto subSumOutput = output;
                if (inputTileInd < numInputTiles - 1 || hasRelu) {
                    subSumOutput = addNewData(
                        newDataId(),
                        [output, inputTileInd, extraSuffix](VpuData *data) {
                            data->name = output->name + "@subSum@" + std::to_string(inputTileInd) + extraSuffix;
                            data->index = IndexBSS;
                            data->type = output->type;
                            data->dims = output->dims;
                            data->order = orderZYX;
                            data->strides = calcStrides(data->dims, data->type, data->order, 16u);
                        });
                }
                subSumOutputs.push_back(subSumOutput);

                VpuDataHandle firstInput, secondInput;
                if (inputTileInd == 1) {
                    firstInput = subConvOutputs[inputTileInd - 1];
                } else {
                    firstInput = subSumOutputs[inputTileInd - 2];
                }
                secondInput = subConvOutputs[inputTileInd];

                addNewStage<VpuEltwiseStage>(
                    stage->name + "@accum@" + std::to_string(inputTileInd) + extraSuffix,
                    kSum,
                    stage->layer,
                    [](VpuEltwiseStage* stage) {
                        stage->requiredInputOrder[0] = orderYXZ;
                        stage->requiredInputOrder[1] = orderYXZ;
                        stage->requiredOutputOrder[0] = orderYXZ;
                    },
                    {firstInput, secondInput},
                    {subSumOutput},
                    nullptr,
                    &stageIt);

                if (inputTileInd == numInputTiles - 1 && hasRelu) {
                    addNewStage<VpuReluStage>(
                        stage->name + "@ReLU" + extraSuffix,
                        kRelu,
                        stage->layer,
                        [](VpuReluStage* stage) {
                            stage->negativeSlope = 0;

                            stage->requiredInputOrder[0] = orderYXZ;
                            stage->requiredOutputOrder[0] = orderYXZ;
                        },
                        {subSumOutput},
                        {output},
                        nullptr,
                        &stageIt);
                }
            }
        }
    }
}

void GraphTransformerImpl::processHWConv(
        const std::list<VpuStagePtr>::iterator& stageIt,
        uint32_t cmxLimit,
        bool isYoloNetwork,
        bool isOriginalYolo) {
    auto stage = *stageIt;
    assert(stage != nullptr);

    auto swStage = std::dynamic_pointer_cast<VpuConvStage>(stage);
    assert(swStage != nullptr);

    auto convLayer = std::dynamic_pointer_cast<ConvolutionLayer>(stage->layer);
    assert(convLayer != nullptr);

    // HW doesn't support dilation
    if (swStage->dilationX != 1 || swStage->dilationY != 1)
        return;

    // HW supports only same strides for X and Y
    if (swStage->strideX != swStage->strideY)
        return;

    // TODO : what to do with grouped convolution?
    if (convLayer->_group != 1)
        return;

    auto input = stage->inputs[0];
    auto weights = stage->inputs[1];
    auto output = stage->outputs[0];

    VpuStageHandle postOp;
    VpuDataHandle biases;
    std::string hwStageNameSuffix;
    std::tie(postOp, biases, hwStageNameSuffix) = getPostOpInfoForHW(stage);

    // Try to merge convolution with max pooling
    // TODO : check which convolution and pooling parameters are supported

    Handle<VpuPoolStage> postPoolStage;
    auto actualOutput = output;
    if (swStage->radixX == 3 && swStage->radixY == 3 &&
        swStage->strideX == 1 && swStage->strideY == 1 &&
        swStage->padX == 1 && swStage->padY == 1) {
        auto outputConsumers = output->consumers;
        if (stage->postOp != nullptr)
            outputConsumers.erase(stage->postOp);

        if (outputConsumers.size() == 1) {
            auto nextStage = *outputConsumers.begin();

            if (nextStage->type == kMaxPool) {
                postPoolStage = nextStage.dynamicCast<VpuPoolStage>();
                assert(postPoolStage != nullptr);

                if (postPoolStage->radixX == 2 && postPoolStage->radixY == 2 &&
                    postPoolStage->strideX == 2 && postPoolStage->strideY == 2 &&
                    postPoolStage->padX == 0 && postPoolStage->padY == 0) {
                    actualOutput = postPoolStage->outputs[0];
                } else {
                    postPoolStage = nullptr;
                }
            }
        }
    }

    uint32_t newInputDimZ = 0, newOutputDimZ = 0;
    VpuMyriadXHwConvolutionStage::Tiles tiles;
    std::tie(newInputDimZ, newOutputDimZ, tiles)
            = splitConvolution(input->dims[Dim::X], input->dims[Dim::Y], input->dims[Dim::Z],
                               actualOutput->dims[Dim::X], actualOutput->dims[Dim::Y], actualOutput->dims[Dim::Z],
                               swStage->radixX, swStage->radixY, swStage->strideX,
                               MODE_FP16, FP16_COEFF);

    if (tiles.empty()) {
        postPoolStage = nullptr;
        actualOutput = output;
    }

    bool splitOverHeight = input->dims[Dim::X] * input->dims[Dim::Y] / 1024.0 > 128;

    if (!splitOverHeight) {
        auto outBufSize = estimateHwBufferSize(actualOutput->dims);
        if (outBufSize > cmxLimit)
            splitOverHeight = true;
    }

    // HACK : enable split-over-height for YOLO convolutions (conv1, conv2, conv3)

    if (isYoloNetwork &&
        (stage->name == "conv1" || stage->name == "conv2" || stage->name == "conv3")) {
        splitOverHeight = true;
    }

    // HACK : scale too small weights in YOLO convolutions (conv8) to avoid FP16 precision errors

    auto scale = 1.0f;
    if (isOriginalYolo &&
        stage->name == "conv8") {
        scale = 16.0f;
    }

    if (!splitOverHeight) {
        addHWConv(stageIt, input, actualOutput, scale, postPoolStage);
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

        std::vector<TileSoH> heightSplits;

        if (postPoolStage != nullptr) {
            // For conv3x3s1p1 and fused 2x2s2 pooling, we need 4 extra lines
            // Also, for this specific case, the maxOutputLines is doubled, because the output is reduced by a factor of 2
            heightSplits = heightSolutionWithPooling(
                input->dims[Dim::Y],
                swStage->radixY,
                swStage->strideY,
                swStage->padY,
                maxOutputLines);
        } else {
            // For convolution without fused pooling
            // The following is not correct for convolution. We cannot have selective zero padding
            // pad = (stage.radixY // 2 if pad_top > 0 else 0, stage.radixY // 2 if pad_bottom > 0 else 0)

            if ((swStage->padY == swStage->padX) &&
                (swStage->padY == 0 || swStage->padY == (swStage->radixY / 2))) {
                heightSplits = heightSolution(
                    input->dims[Dim::Y],
                    swStage->radixY,
                    swStage->strideY,
                    std::make_tuple(swStage->padY, swStage->padX),
                    maxOutputLines);
            }
        }

        if (heightSplits.empty() || heightSplits.size() == 1) {
            addHWConv(stageIt, input, actualOutput, scale, postPoolStage);
        } else {
            actualOutput->producer = nullptr;
            actualOutput->producerOutInd = -1;

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
                        [actualOutput, outputWithJunk, outputStartIndex, tileInd](VpuData* data) {
                            data->name = actualOutput->name + "@sub" + std::to_string(tileInd);
                            data->index = actualOutput->index;
                            data->type = actualOutput->type;
                            data->order = actualOutput->order;
                            data->dims = VpuDims({actualOutput->dims[Dim::X], static_cast<uint32_t>(outputWithJunk), actualOutput->dims[Dim::Z]});
                            data->strides = actualOutput->strides;
                            data->offsetFromParent = VpuDims({0u, static_cast<uint32_t>(outputStartIndex), 0u});
                        },
                        actualOutput);

                    if (copyInputs.empty() || copyInputs.back() == nullptr) {
                        addHWConv(stageIt, subInput, subOutput, scale, postPoolStage,
                                  tileInd == heightSplits.size() - 1u,
                                  "@soh" + std::to_string(tileInd),
                                  nullptr, nullptr);
                    } else {
                        addHWConv(stageIt, subInput, subOutput, scale, postPoolStage,
                                  tileInd == heightSplits.size() - 1u,
                                  "@soh" + std::to_string(tileInd),
                                  copyInputs.back(), copyOutputs.back());
                    }

                    copyInputs.push_back(nullptr);
                    copyOutputs.push_back(nullptr);
                } else {
                    auto subConvOutput = addNewData(
                        newDataId(),
                        [actualOutput, outputWithJunk, tileInd](VpuData* data) {
                            data->name = actualOutput->name + "@subConv" + std::to_string(tileInd);
                            data->index = IndexBSS;
                            data->type = actualOutput->type;
                            data->order = orderZYX;
                            data->dims = VpuDims({actualOutput->dims[Dim::X], static_cast<uint32_t>(outputWithJunk), actualOutput->dims[Dim::Z]});
                            data->strides = calcStrides(data->dims, data->type, data->order, 16u);
                        });

                    auto subConvOutputInner = addNewData(
                        newDataId(),
                        [subConvOutput, outputJunkBefore, outputStartIndex, outputEndIndex](VpuData* data) {
                            data->name = subConvOutput->name + "@inner";
                            data->index = subConvOutput->index;
                            data->type = subConvOutput->type;
                            data->order = subConvOutput->order;
                            uint32_t outTileHeight = outputEndIndex - outputStartIndex;
                            data->dims = VpuDims({subConvOutput->dims[Dim::X], outTileHeight, subConvOutput->dims[Dim::Z]});
                            data->strides = subConvOutput->strides;
                            data->offsetFromParent = VpuDims({0u, static_cast<uint32_t>(outputJunkBefore), 0u});
                        },
                        subConvOutput);

                    auto subOutput = addNewData(
                        newDataId(),
                        [actualOutput, subConvOutputInner, outputStartIndex, tileInd](VpuData* data) {
                            data->name = actualOutput->name + "@sub" + std::to_string(tileInd);
                            data->index = actualOutput->index;
                            data->type = actualOutput->type;
                            data->order = actualOutput->order;
                            data->dims = VpuDims({actualOutput->dims[Dim::X], subConvOutputInner->dims[Dim::Y], actualOutput->dims[Dim::Z]});
                            data->strides = actualOutput->strides;
                            data->offsetFromParent = VpuDims({0u, static_cast<uint32_t>(outputStartIndex), 0u});
                        },
                        actualOutput);

                    if (copyInputs.empty() || copyInputs.back() == nullptr) {
                        addHWConv(stageIt, subInput, subConvOutput, scale, postPoolStage,
                                  tileInd == heightSplits.size() - 1u,
                                  "@soh" + std::to_string(tileInd),
                                  nullptr, nullptr);
                    } else {
                        addHWConv(stageIt, subInput, subConvOutput, scale, postPoolStage,
                                  tileInd == heightSplits.size() - 1u,
                                  "@soh" + std::to_string(tileInd),
                                  copyInputs.back(), copyOutputs.back());
                    }

                    copyInputs.push_back(subConvOutputInner);
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
    weights->consumers.erase(stage);
    stage->optimized = true;

    if (postOp != nullptr) {
        postOp->optimized = true;
    }

    if (postPoolStage != nullptr) {
        postPoolStage->optimized = true;
    }
}
