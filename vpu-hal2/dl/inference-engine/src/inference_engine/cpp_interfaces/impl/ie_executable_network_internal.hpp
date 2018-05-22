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

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <ie_plugin_ptr.hpp>
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_async_request_internal.hpp"

namespace InferenceEngine {

class InferencePluginInternal;

typedef std::shared_ptr<InferencePluginInternal> InferencePluginInternalPtr;

/**
 * @brief minimum API to be implemented by plugin, which is used in ExecutableNetworkBase forwarding mechanism
 */
class ExecutableNetworkInternal : public IExecutableNetworkInternal {
public:
    typedef std::shared_ptr<ExecutableNetworkInternal> Ptr;

    virtual void setNetworkInputs(const InferenceEngine::InputsDataMap networkInputs) {
        _networkInputs = networkInputs;
    }

    virtual void setNetworkOutputs(const InferenceEngine::OutputsDataMap networkOutputs) {
        _networkOutputs = networkOutputs;
    }

    void Export(const std::string &modelFileName) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void SetPointerToPluginInternal(InferencePluginInternalPtr plugin) {
        _plugin = plugin;
    }

protected:
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;
    InferencePluginInternalPtr _plugin;
};

}  // namespace InferenceEngine
