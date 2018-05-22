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

/**
 * \brief inference engine plugin API wrapper, to be used by particular implementors
 * \file ie_plugin_base.hpp
 */

#pragma once

#include <memory>
#include <map>
#include <string>
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "cpp_interfaces/base/ie_executable_network_base.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"

namespace InferenceEngine {

/**
 * @brief optional implementation of IInferencePluginInternal to avoid duplication in all plugins
 */
class InferencePluginInternal
        : public IInferencePluginInternal, public std::enable_shared_from_this<InferencePluginInternal> {
public:
    /**
     * Given optional implementation of deprecated load to avoid need for it to be implemented by plugin
     */
    void LoadNetwork(ICNNNetwork &network) override {
        _isDeprecatedLoad = true;
        network.getInputsInfo(_networkInputs);
        network.getOutputsInfo(_networkOutputs);
        if (_networkInputs.empty() || _networkOutputs.empty()) {
            THROW_IE_EXCEPTION << "The network doesn't have inputs/outputs.";
        }
        _createdInferRequest = nullptr;  // first release the infer request
        _loadedNetwork = nullptr;  // first release the loaded network

        _firstInput = _networkInputs.begin()->first;
        _firstOutput = _networkOutputs.begin()->first;
        LoadNetwork(_loadedNetwork, network, {});

        ResponseDesc resp;
        StatusCode sts = _loadedNetwork->CreateInferRequest(_createdInferRequest, &resp);
        if (sts != OK) THROW_IE_EXCEPTION << resp.msg;
    }

    /**
     * @brief Creates an executable network from an pares network object, users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the HW resources)
     * @param network - a network object acquired from CNNNetReader
     * @param config string-string map of config parameters relevant only for this load operation
     * @return shared_ptr to the ExecutableNetwork object
     */
    virtual ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(ICNNNetwork &network,
                                                              const std::map<std::string, std::string> &config) = 0;

    /**
     * Given optional implementation of load executable network to avoid need for it to be implemented by plugin
     */
    void LoadNetwork(IExecutableNetwork::Ptr &executableNetwork,
                     ICNNNetwork &network,
                     const std::map<std::string, std::string> &config) override {
        network.getInputsInfo(_networkInputs);
        network.getOutputsInfo(_networkOutputs);
        auto impl = LoadExeNetworkImpl(network, config);
        impl->setNetworkInputs(_networkInputs);
        impl->setNetworkOutputs(_networkOutputs);
        // skip setting shared ptr to avoid curricular dependency: ExecutableNetworkBase -> IExecutableNetworkInternal -> InferencePluginInternal
        if (!_isDeprecatedLoad) {
            impl->SetPointerToPluginInternal(shared_from_this());
        }

        executableNetwork.reset(new ExecutableNetworkBase<ExecutableNetworkInternal>(impl), [](details::IRelease *p) {
            p->Release();
        });
        _isDeprecatedLoad = false;
    };

    /**
     * Given optional implementation of deprecated infer to avoid need for it to be implemented by plugin
     */
    void Infer(const Blob &input, Blob &result) override {
        const BlobMap inputs = {{_firstInput, std::shared_ptr<Blob>(const_cast<Blob *>(&input), [](Blob *ptr) {})}};
        BlobMap results = {{_firstOutput, std::shared_ptr<Blob>(&result, [](Blob *ptr) {})}};
        return Infer(inputs, results);
    }

    /**
     * Given optional implementation of deprecated infer to avoid need for it to be implemented by plugin
     */
    void Infer(const BlobMap &input, BlobMap &result) override {
        if (_createdInferRequest == nullptr) {
            THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;
        }
        ResponseDesc resp;
        StatusCode sts;

        auto setBlobs = [&](const BlobMap &blobMap) {
            for (auto pair : blobMap) {
                auto blobName = pair.first;
                auto blobPtr = pair.second;
                sts = _createdInferRequest->SetBlob(blobName.c_str(), blobPtr, &resp);
                if (sts != OK) THROW_IE_EXCEPTION << resp.msg;
            }
        };
        setBlobs(input);
        setBlobs(result);

        sts = _createdInferRequest->Infer(&resp);
        if (sts != OK) THROW_IE_EXCEPTION << resp.msg;
    }

    /**
     * Given optional implementation of deprecated infer to avoid need for it to be implemented by plugin
     */
    void GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) override {
        if (_createdInferRequest == nullptr) {
            THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;
        }
        ResponseDesc resp;
        StatusCode sts = _createdInferRequest->GetPerformanceCounts(perfMap, &resp);
        if (sts != OK) THROW_IE_EXCEPTION << resp.msg;
    }

    /**
     * Given optional implementation of ImportNetwork to avoid need for it to be implemented by plugin
     */
    void ImportNetwork(IExecutableNetwork::Ptr &ret, const std::string &modelFileName) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    /**
     * Given optional implementation of SetConfig to avoid need for it to be implemented by plugin
     */
    void SetConfig(const std::map<std::string, std::string> &config) override {}

    /**
     * Given optional implementation of SetLogCallback to avoid need for it to be implemented by plugin
     */
    void SetLogCallback(IErrorListener &listener) override {}

    /**
     * Given optional implementation of AddExtension to avoid need for it to be implemented by plugin
     */
    void AddExtension(InferenceEngine::IExtensionPtr extension) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

protected:
    IExecutableNetwork::Ptr _loadedNetwork;
    std::string _firstInput;
    std::string _firstOutput;
    IInferRequest::Ptr _createdInferRequest;
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;
    std::map<std::string, std::string> _config;
    bool _isDeprecatedLoad = false;
};

}  // namespace InferenceEngine
