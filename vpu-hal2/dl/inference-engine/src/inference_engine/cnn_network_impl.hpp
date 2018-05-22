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

#pragma once

#include <map>
#include <memory>
#include <ie_icnn_network.hpp>
#include "ie_common.h"
#include "ie_data.h"
#include "ie_blob.h"
#include "ie_api.h"
#include "description_buffer.hpp"
#include <string>
#include <vector>

namespace InferenceEngine {
namespace details {
class INFERENCE_ENGINE_API_CLASS(CNNNetworkImpl) : public ICNNNetwork {
public:
    CNNNetworkImpl();
    Precision getPrecision() noexcept override {
        return precision;
    }

    void setPrecision(Precision::ePrecision  prec) {
        precision = prec;
    }

    void getOutputsInfo(std::map<std::string, DataPtr> &out) const noexcept override;

    void getInputsInfo(InputsDataMap& inputs) const noexcept override;

    InputInfo::Ptr getInput(const std::string& inputName) noexcept override {
        return _inputData[inputName];
    }

    void setInputInfo(InputInfo::Ptr data) {
        _inputData[data->name()] = data;
    }

    void getName(char* pName, size_t len)  noexcept override {
        // Description buffer will preserve garbage if external pointer not initialized
        if (len < 1) return;
        memset(pName, 0, len);
        DescriptionBuffer(pName, len) << _name;
    }

    const std::string& getName() const {
        return _name;
    }

    void setName(const std::string& name) {
        _name = name;
    }

    const std::map<std::string, CNNLayerPtr>& allLayers() const {
        return _layers;
    }

    size_t layerCount()  noexcept override {
        return _layers.size();
    }

    DataPtr& getData(const char* name) noexcept override {
        return _data[name];
    }

    DataPtr& getData(const std::string& name) {
        return getData(name.c_str());
    }

    void addLayer(const CNNLayerPtr& layer) noexcept override;

    StatusCode getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) noexcept override;

    StatusCode setBatchSize(const size_t size) noexcept override;

    size_t getBatchSize() const noexcept override;

    void setTargetDevice(TargetDevice device) noexcept override {
        _targetDevice = device;
    }

    TargetDevice getTargetDevice() noexcept override {
        return _targetDevice;
    }

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void resolveOutput();

    void addOutput(const std::string& dataName);

    void Release() noexcept override {
        delete this;
    }

protected:
    Precision precision {Precision::MIXED};
    std::map<std::string, DataPtr> _data;
    std::map<std::string, CNNLayerPtr> _layers;
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    std::string _name;
    /// @brief
    TargetDevice _targetDevice;
    DataPtr _emptyData;
};


typedef std::shared_ptr<CNNNetworkImpl> CNNNetworkImplPtr;
}  // namespace details
}  // namespace InferenceEngine
