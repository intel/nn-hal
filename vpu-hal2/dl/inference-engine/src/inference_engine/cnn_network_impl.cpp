/*
 * INTEL CONFIDENTIAL
 * Copyright 2016 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#include <ie_common.h>
#include "cnn_network_impl.hpp"
#include <memory>
#include <map>
#include <string>
#include <cassert>
#include "debug.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

CNNNetworkImpl::CNNNetworkImpl(): _targetDevice(TargetDevice::eDefault) {
}

void CNNNetworkImpl::getOutputsInfo(std::map<std::string, DataPtr> &out) const noexcept {
    out = _outputData;
}

void CNNNetworkImpl::getInputsInfo(InputsDataMap& inputs) const noexcept {
    inputs = _inputData;
}

void CNNNetworkImpl::addLayer(const CNNLayerPtr& layer) noexcept {
    _layers[layer->name] = layer;
}

StatusCode CNNNetworkImpl::getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp)  noexcept {
    auto it = _layers.find(layerName);
    if (it == _layers.end())
        return DescriptionBuffer(NOT_FOUND, resp) << "Layer " << layerName << " not found in network";
    out = it->second;
    return OK;
}

StatusCode CNNNetworkImpl::addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept {
    CNNLayerPtr outLayer;
    auto rc = getLayerByName(layerName.c_str(), outLayer, resp);
    if (rc != OK) return rc;

    if (outputIndex >= outLayer->outData.size())
        return  DescriptionBuffer(OUT_OF_BOUNDS, resp) << "port index " << outputIndex
                << " exceeds layer's outputs which is "<< outLayer->outData.size();
    shared_ptr<Data> outData = outLayer->outData[outputIndex];
    _outputData[outData->getName()] = outData;
    return OK;
}

void CNNNetworkImpl::resolveOutput() {
    // check orphan nodes...
    for (auto kvp : _data) {
        if (!kvp.second->isInitialized())
            THROW_IE_EXCEPTION << "data name [" << kvp.first << "] dimensions is not known";

        // data nodes not going to any layer are basically graph output...
        if (kvp.second->getInputTo().empty()) {
            _outputData[kvp.first] = kvp.second;
        }
    }
}

void CNNNetworkImpl::addOutput(const string& dataName) {
    auto it = _data.find(dataName);
    if (it == _data.end()) {
        THROW_IE_EXCEPTION << "data [" << dataName << "] doesn't exist";
    }
    auto data = it->second;
    assert(data->getName() == dataName);
    _outputData[dataName] = data;
}

StatusCode CNNNetworkImpl::setBatchSize(const size_t size) noexcept {
    auto originalBatchSize = getBatchSize();
    if (originalBatchSize == size)
        return OK;
    for (auto layer : _data) {
        SizeVector dims = layer.second->getDims();
        // Calculates original size for batch = 1
        size_t diff = dims.at(0) / originalBatchSize;
        dims.at(0) = size * diff;
        layer.second->setDims(dims);
    }
    return OK;
}

size_t CNNNetworkImpl::getBatchSize() const noexcept {
    if (!_inputData.size())
        return 0;
    // currently CNNNetworkImpl::setBatchSize set the same values
    // for the latest dim as a batch, we can take the first input
    // and return batch size for it
    SizeVector dims = _inputData.cbegin()->second->getDims();
    return dims.at(dims.size() - 1);
}
