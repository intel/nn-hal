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
#include <vector>
#include <limits>
#include <algorithm>
#include <list>
#include <string>
#include <memory>
#include <utility>

void VpuMyriadXHwFullyConnectedStage::dumpToDot(std::ostream& os) {
    os << "newInputDimZ=" << newInputDimZ << "\\n"
       << "newOutputDimZ=" << newOutputDimZ << "\\n"
       << "descriptors.size=" << descriptors.size();
}

void VpuMyriadXHwFullyConnectedStage::dumpToBlob(BlobWriter& writer) {
    // HW FC never has parallel copy
    writer.write(static_cast<uint32_t>(0));

    writer.write(static_cast<uint32_t>(descriptors.size()));
    for (auto& d : descriptors) {
        writer.write(d);
    }

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
    inputs[1]->dumpToBlob(writer);  // taps
    inputs[2]->dumpToBlob(writer);  // biases
}

namespace {

std::tuple<uint32_t, uint32_t, VpuMyriadXHwFullyConnectedStage::Tiles> splitFullyConnected(
        uint32_t inN, uint32_t outN,
        const std::vector<cnnOperationMode>& modes = {MODE_1_256, MODE_2_128, MODE_4_64, MODE_8_32, MODE_16_16}) {
    using FCSolution = std::tuple<cnnOperationMode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>;
    using FCSolutions = std::vector<FCSolution>;

    std::array<uint32_t, 5> COST_PER_MODE{{0u, 5u, 11u, 19u, 31u}};

    FCSolutions solutions;
    for (auto mode : modes) {
        auto ramBlocks = 1u << mode;
        auto maxInN = ramBlocks * 256u;
        auto maxOutN = 256u / ramBlocks;
        auto newInN = alignVal(inN, ramBlocks);
        auto newOutN = alignVal(outN, 8u);
        auto workInN = std::min(newInN, maxInN);
        auto workOutN = std::min(newOutN, maxOutN);

        while (workInN >= ramBlocks) {
            auto countIn = static_cast<uint32_t>(std::ceil(static_cast<double>(newInN) / workInN));
            auto countOut = static_cast<uint32_t>(std::ceil(static_cast<double>(newOutN) / workOutN));
            auto cost = countIn * countOut * (workInN / ramBlocks + COST_PER_MODE[mode]);
            solutions.push_back(std::make_tuple(mode, newInN, newOutN, workInN, workOutN, countIn, countOut, cost));
            workInN /= 2;
        }
    }

    auto minCountIn = std::numeric_limits<uint32_t>::max();
    for (const auto& s : solutions) {
        minCountIn = std::min(minCountIn, std::get<5>(s));
    }

    // filter out solutions != minCountIn
    {
        FCSolutions tmpSolutions;
        for (const auto& s : solutions) {
            if (std::get<5>(s) == minCountIn)
                tmpSolutions.push_back(s);
        }
        solutions.swap(tmpSolutions);
    }

    std::sort(
        solutions.begin(), solutions.end(),
        [](const FCSolution& a, const FCSolution& b) {
            return std::get<7>(a) < std::get<7>(b);
        });

    if (!solutions.empty()) {
        cnnOperationMode mode;
        uint32_t newInN, newOutN, workInN, workOutN, countIn, countOut;
        uint32_t _;
        std::tie(mode, newInN, newOutN, workInN, workOutN, countIn, countOut, _) = solutions[0];
        VpuMyriadXHwFullyConnectedStage::SubTiles subTiles(countIn, std::make_tuple(workInN, workOutN, mode));
        VpuMyriadXHwFullyConnectedStage::Tiles tiles(countOut, subTiles);
        return std::make_tuple(newInN, newOutN, std::move(tiles));
    }

    return std::make_tuple(0, 0, VpuMyriadXHwFullyConnectedStage::Tiles());
}

}  // namespace

void GraphTransformerImpl::processHWFC(
        const std::list<VpuStagePtr>::iterator& stageIt,
        bool isYoloNetwork,
        bool isOriginalYolo) {
    auto stage = *stageIt;
    assert(stage != nullptr);

    auto swStage = std::dynamic_pointer_cast<VpuFullyConnectedStage>(stage);
    assert(swStage != nullptr);

    Blob::Ptr ieWeights, ieBiases;
    if (auto fcLayer = std::dynamic_pointer_cast<WeightableLayer>(stage->layer)) {
        ieWeights = fcLayer->_weights;
        ieBiases = fcLayer->_biases;
    } else {
        if (stage->layer == nullptr) {
            THROW_IE_EXCEPTION << "[VPU] Layer is NULL for FC stage";
        } else {
            THROW_IE_EXCEPTION << "[VPU] Unknown layer type for FC stage : " << stage->layer->type;
        }
    }

    VpuStageHandle postOp;
    VpuDataHandle biases;
    std::string hwStageNameSuffix;
    std::tie(postOp, biases, hwStageNameSuffix) = getPostOpInfoForHW(stage);

    auto input = stage->inputs[0];
    auto weights = stage->inputs[1];
    auto output = stage->outputs[0];

    if (output->dims[Dim::X] != 1 || output->dims[Dim::Y] != 1)
        return;

    VpuDataHandle inputAsVec;
    if (input->dims[Dim::X] == 1u && input->dims[Dim::Y] == 1u) {
         inputAsVec = input;
    } else {
        inputAsVec = addNewData(
            newDataId(),
            [input](VpuData* data) {
                data->name = input->name + "@asVec";
                data->index = IndexBSS;
                data->type = input->type;
                data->order = orderZYX;
                data->dims = VpuDims({1u, 1u, input->dims.totalSize()});
                data->strides = calcStrides(data->dims, data->type, data->order, 16u);
            });

        addNewStage<VpuCopyStage>(
            stage->name + "@inputRelayout",
            kHwFcRelayout,
            stage->layer,
            [](VpuCopyStage* stage) {
                stage->requiredInputOrder[0] = orderZYX;

                stage->requiredOutputOrder[0] = orderZYX;
                stage->requiredOutputAlignment[0] = 16u;
            },
            {input},
            {inputAsVec},
            nullptr,
            &stageIt);
    }

    // Split in the best mode possible. However, if there are more than one tiles,
    // we force mode 0 and split again, because of a hardware bug (which does not reset
    // the accumulator)
    uint32_t newInputDimZ = 0, newOutputDimZ = 0;
    VpuMyriadXHwFullyConnectedStage::Tiles tiles;
    std::tie(newInputDimZ, newOutputDimZ, tiles) = splitFullyConnected(inputAsVec->dims[Dim::Z], output->dims[Dim::Z]);
    if (tiles.size() > 1) {
        std::tie(newInputDimZ, newOutputDimZ, tiles) = splitFullyConnected(inputAsVec->dims[Dim::Z], output->dims[Dim::Z], {MODE_1_256});
    }

    if (tiles.empty()) {
        THROW_IE_EXCEPTION << "[VPU] Can't convert " << stage->name << " to HW stage";
    }

    // HACK : scale too small weights in YOLO FC (fc9) to avoid FP16 precision errors

    float wScale = 1.0f, bScale = 1.0f;
    if (isOriginalYolo &&
        stage->name == "fc9") {
        wScale = 16.0f;
        bScale = 256.0f;
    }

    auto tempOutput = output;
    if (bScale != 1.0f) {
        tempOutput = addNewData(
            newDataId(),
            [output](VpuData* data) {
                data->name = output->name + "@scaled";
                data->index = IndexBSS;
                data->type = output->type;
                data->order = orderZYX;
                data->dims = output->dims;
                data->strides = calcStrides(data->dims, data->type, data->order, 16u);
            });
    }

    auto biasesHW = biases;
    if (bScale != 1.0f && biases != nullptr && biases->index != IndexNone) {
        biasesHW = addNewData(
            newDataId(),
            [biases, ieBiases, bScale](VpuData* data) {
                data->name = biases->name + "@HW";
                data->index = IndexBlob;
                data->type = VpuDataType::FP16;
                data->dims = biases->dims;
                data->order = biases->order;
                data->strides = biases->strides;
                data->writer = std::make_shared<ScaledBiasesWriter>(ieBiases, bScale);
            });

        biases->index = IndexNone;
        biases->writer = nullptr;
    }

    VpuDims origWeightsDims(4);
    origWeightsDims[Dim::X] = 1u;
    origWeightsDims[Dim::Y] = 1u;
    origWeightsDims[Dim::Z] = inputAsVec->dims[Dim::Z];
    origWeightsDims[Dim::N] = output->dims[Dim::Z];

    VpuDims hwWeigthsDims(4);
    hwWeigthsDims[Dim::X] = 8u;
    hwWeigthsDims[Dim::Y] = 1u;
    hwWeigthsDims[Dim::Z] = newInputDimZ;
    hwWeigthsDims[Dim::N] = newOutputDimZ / 8u;

    auto weightsHW = addNewData(
        newDataId(),
        [weights, origWeightsDims, hwWeigthsDims, ieWeights, wScale](VpuData* data) {
            data->name = weights->name + "@HW";
            data->index = IndexBlob;
            data->type = VpuDataType::FP16;
            data->dims = hwWeigthsDims;
            data->order = orderZYX;
            data->strides.resize(4);
            data->strides[Dim::X] = getDataTypeSize(data->type);
            data->strides[Dim::Y] = hwWeigthsDims[Dim::X] * data->strides[Dim::X];
            data->strides[Dim::Z] = hwWeigthsDims[Dim::Y] * data->strides[Dim::Y];
            data->strides[Dim::N] = hwWeigthsDims[Dim::Z] * data->strides[Dim::Z];
            data->writer = std::make_shared<HwWeightsWriter>(ieWeights, origWeightsDims, hwWeigthsDims, 1, 0, wScale);
        });

    weights->index = IndexNone;
    weights->writer = nullptr;

    auto hasRelu = isReluPostOp(postOp);

    addNewStage<VpuMyriadXHwFullyConnectedStage>(
        stage->name + "@HW" + hwStageNameSuffix,
        kMyriadXHwFCL,
        stage->layer,
        [newInputDimZ, newOutputDimZ, &tiles, hasRelu](VpuMyriadXHwFullyConnectedStage* stage) {
            stage->newInputDimZ = newInputDimZ;
            stage->newOutputDimZ = newOutputDimZ;
            stage->tiles = std::move(tiles);

            stage->hasRelu = hasRelu;

            stage->requiredInputOrder[0] = orderZYX;
            stage->requiredInputAlignment[0] = 16u;

            stage->requiredOutputOrder[0] = orderZYX;
            stage->requiredOutputAlignment[0] = 16u;
        },
        {inputAsVec, weightsHW, biasesHW},
        {tempOutput},
        nullptr,
        &stageIt);

    if (bScale != 1.0f) {
        addNewStage<VpuPowerStage>(
            stage->name + "@scale",
            kCHWPower,
            stage->layer,
            [bScale](VpuPowerStage* stage) {
                stage->offset = 0.0f;
                stage->scale = 1.0f / bScale;
                stage->power = 1.0f;

                stage->requiredInputOrder[0] = orderZYX;
                stage->requiredInputAlignment[0] = 16u;

                stage->requiredOutputOrder[0] = orderZYX;
                stage->requiredOutputAlignment[0] = 16u;
            },
            {tempOutput},
            {output},
            nullptr,
            &stageIt);
    }

    input->consumers.erase(stage);
    weights->consumers.erase(stage);
    stage->optimized = true;

    if (postOp != nullptr) {
        postOp->optimized = true;
    }
}
