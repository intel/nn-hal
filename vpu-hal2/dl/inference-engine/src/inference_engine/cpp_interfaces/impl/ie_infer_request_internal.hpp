// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <ie_input_info.hpp>
#include <ie_icnn_network.hpp>
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "debug.h"
#include "cpp_interfaces/exception2status.hpp"

namespace InferenceEngine {

class ExecutableNetworkInternal;

typedef std::shared_ptr<ExecutableNetworkInternal> ExecutableNetworkInternalPtr;

/**
 * @brief optional implementation of IInferRequestInternal to avoid duplication in all plugins
 */
class InferRequestInternal : virtual public IInferRequestInternal {
public:
    typedef std::shared_ptr<InferRequestInternal> Ptr;

    InferRequestInternal(InputsDataMap networkInputs, OutputsDataMap networkOutputs)  {
        // We should copy maps in order to avoid modifications in the future.
        for (const auto& it : networkInputs) {
            InputInfo::Ptr newPtr;
            if (it.second) {
                newPtr.reset(new InputInfo());
                DataPtr newData(new Data(*it.second->getInputData()));
                newPtr->getPreProcess() = it.second->getPreProcess();
                newPtr->setInputData(newData);
            }
            _networkInputs[it.first] = newPtr;
        }

        for (const auto& it : networkOutputs) {
            DataPtr newData;
            if (it.second) {
                newData.reset(new Data(*it.second));
            }
            _networkOutputs[it.first] = newData;
        }
    }

    /**
     * @brief Given optional implementation of setting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void SetBlob(const char *name, const Blob::Ptr &data) override {
        if (!data)
            THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
        if (data->buffer() == nullptr)
            THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
        if (name == nullptr) {
            THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
        }
        InputInfo::Ptr foundInput;
        DataPtr foundOutput;
        size_t dataSize = data->size();
        if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
            size_t inputSize = details::product(foundInput->getDims());
            if (dataSize != inputSize) {
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }
            if (foundInput->getInputPrecision() != data->precision()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                                   << "Failed to set Blob with precision not corresponding user input precision";
            }
            _inputs[name] = data;
        } else {
            size_t outputSize = details::product(foundOutput->getDims());
            if (dataSize != outputSize) {
                THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                                   << dataSize << "!=" << outputSize << ").";
            }
            if (foundOutput->getPrecision() != data->precision()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                                   << "Failed to set Blob with precision not corresponding user output precision";
            }
            _outputs[name] = data;
        }
    }

    /**
     * @brief Given optional implementation of getting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void GetBlob(const char *name, Blob::Ptr &data) override {
        InputInfo::Ptr foundInput;
        DataPtr foundOutput;
        if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
            data = _inputs[name];
        } else {
            data = _outputs[name];
        }
    }

    void setPointerToExecutableNetworkInternal(ExecutableNetworkInternalPtr exeNetwork) {
        _exeNetwork = exeNetwork;
    }

protected:
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;
    InferenceEngine::BlobMap _inputs;
    InferenceEngine::BlobMap _outputs;
    ExecutableNetworkInternalPtr _exeNetwork;

protected:
    /**
     * @brief helper to find input or output blob by name
     * @param name - a name of input or output blob.
     * @return true - if loaded network has input with provided name,
     *         false - if loaded network has output with provided name
     * @throws [parameter_mismatch] exception if input and output has the same name
     * @throws [not_found] exception if there is no input and output layers with given name
     */
    bool findInputAndOutputBlobByName(const char *name, InputInfo::Ptr &foundInput, DataPtr &foundOutput) {
        foundInput = nullptr;
        foundOutput = nullptr;
        if (_networkInputs.empty() || _networkOutputs.empty()) {
            THROW_IE_EXCEPTION << "Internal error: network inputs and outputs is not set";
        }
        auto foundInputPair = std::find_if(std::begin(_networkInputs),
                                           std::end(_networkInputs),
                                           [&](const std::pair<std::string, InputInfo::Ptr> &pair) {
                                               return pair.first == name;
                                           });
        auto foundOutputPair = std::find_if(std::begin(_networkOutputs),
                                            std::end(_networkOutputs),
                                            [&](const std::pair<std::string, DataPtr> &pair) {
                                                return pair.first == name;
                                            });
        if (foundOutputPair == std::end(_networkOutputs) && (foundInputPair == std::end(_networkInputs))) {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to find input and output with name: \'" << name << "\'";
        }
        if (foundInputPair != std::end(_networkInputs)) {
            foundInput = foundInputPair->second;
            return true;
        } else {
            foundOutput = foundOutputPair->second;
            return false;
        }
    }
};

}  // namespace InferenceEngine
