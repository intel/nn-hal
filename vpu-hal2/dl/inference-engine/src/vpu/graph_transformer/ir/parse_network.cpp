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
#include <cassert>
#include <set>
#include <list>

#ifdef NNLOG
#include <android/log.h>
#include <cutils/log.h>
#endif

void GraphTransformerImpl::parseNetwork(ICNNNetwork& network) {
    char tmpBuf[1024];
    network.getName(tmpBuf, sizeof(tmpBuf));
    _networkName.assign(tmpBuf);

    LOG_DEBUG("[VPU] GraphTransformer : transform network %s", _networkName.c_str());

#ifdef NNLOG
    ALOGI("[VPU] GraphTransformer : transform network %s", _networkName.c_str());
#endif

    network.getInputsInfo(_networkInputs);
    network.getOutputsInfo(_networkOutputs);

    // Check inputs

    if (_networkInputs.empty()) {
        THROW_IE_EXCEPTION << "[VPU] No inputs detected in network " << _networkName;
    }

    for (const auto& inputInfo : _networkInputs) {
        assert(inputInfo.second != nullptr);
        auto inputDims = inputInfo.second->getDims();
        auto inputPrecision = inputInfo.second->getInputPrecision();

        if (inputDims.size() > 4) {
            THROW_IE_EXCEPTION << "[VPU] Plugin supports blobs with number of dimensions <= 4. Requested "
                               << inputDims.size()
                               << " for input "
                               << inputInfo.first;
        }

        // verify that batchSize == 1
        if (inputDims.size() == 4 && inputDims[3] != 1) {
            THROW_IE_EXCEPTION << "[VPU] Plugin supports only batch size 1. Requested "
                               << inputDims[3]
                               << " for input "
                               << inputInfo.first;
        }

        if (inputPrecision != Precision::U8 &&
            inputPrecision != Precision::FP16 &&
            inputPrecision != Precision::FP32) {
            THROW_IE_EXCEPTION << "[PARAMETER_MISMATCH] Unsupported input precision: " << inputPrecision.name() << "!";
        }
    }

    for (const auto& outputInfo : _networkOutputs) {
        assert(outputInfo.second != nullptr);
        auto outputPrecision = outputInfo.second->getPrecision();

        if (outputPrecision != Precision::FP16 &&
            outputPrecision != Precision::FP32) {
            THROW_IE_EXCEPTION << "[PARAMETER_MISMATCH] Unsupported output precision: " << outputPrecision.name() << "!";
        }
    }

    // Traversing the topology

    std::set<DataPtr> availableData;
    std::list<CNNLayerPtr> layersToHandle;

    // init by layers connected to inputs
    // layers that are not reachable from inputs will be ignored
    for (const auto& inputInfo : _networkInputs) {
        assert(inputInfo.second != nullptr);
        auto inputData = inputInfo.second->getInputData();
        assert(inputData != nullptr);

        availableData.insert(inputData);
        for (const auto& layerInfo : inputData->inputTo) {
            assert(layerInfo.second != nullptr);
            layersToHandle.push_back(layerInfo.second);
        }
    }

    std::set<CNNLayerPtr> parsedLayers;
    _orderedLayers.clear();

    size_t loopTracker = 0;
    while (!layersToHandle.empty()) {
        auto layer = layersToHandle.front();

        if (layersToHandle.size() == loopTracker) {
            THROW_IE_EXCEPTION << "[VPU] Inputs for layer " << layer->name
                               << "(and " << loopTracker - 1 << " more layers) can not be computed";
        }

        layersToHandle.pop_front();

        bool allInputsAvailable = true;
        for (const auto& in : layer->insData) {
            auto inData = in.lock();
            assert(inData != nullptr);

            if (availableData.find(inData) == availableData.end()) {
                allInputsAvailable = false;
                break;
            }
        }

        if (!allInputsAvailable) {
            auto it = std::find(layersToHandle.begin(), layersToHandle.end(), layer);
            if (it == layersToHandle.end()) {
                layersToHandle.push_back(layer);
            }
            loopTracker++;
            continue;
        }

        if (parsedLayers.find(layer) == parsedLayers.end()) {
            _orderedLayers.push_back(layer);
            parsedLayers.insert(layer);
        }

        // adding children to the list to verify
        for (const auto& out : layer->outData) {
            assert(out != nullptr);
            availableData.insert(out);

            // new data added -> have to reset loop tracking
            loopTracker = 0;

            for (const auto& layerInfo : out->inputTo) {
                assert(layerInfo.second != nullptr);
                auto it = std::find(layersToHandle.begin(), layersToHandle.end(), layerInfo.second);
                if (it == layersToHandle.end()) {
                    layersToHandle.push_back(layerInfo.second);
                }
            }
        }
    }
}
