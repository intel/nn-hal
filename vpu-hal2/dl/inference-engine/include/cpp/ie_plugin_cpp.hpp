// Copyright (c) 2018 Intel Corporation
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
 * @brief This is a header file for the Inference Engine plugin C++ API
 * @file ie_plugin_cpp.hpp
 */
#pragma once

#include <map>
#include <string>

#include "ie_plugin.hpp"
#include "details/ie_exception_conversion.hpp"
#include "cpp/ie_executable_network.hpp"
#include "ie_plugin_ptr.hpp"
#include <memory>


namespace InferenceEngine {

/**
 * @class InferencePlugin
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 * It can throw exceptions safely for the application, where it is properly handled.
 */

class InferencePlugin {
    InferenceEnginePluginPtr actual;

public:
    /**
    * @brief A constructor.
    * Constructs an InferencePlugin instance from the given pointer.
    */
    explicit InferencePlugin(const InferenceEnginePluginPtr &pointer) : actual(pointer) {}

    /**
    * @brief Gets plugin version information
    * @return A Version instance initialized by plugin
    */
    const Version *GetVersion() {
        const Version *versionInfo = nullptr;
        actual->GetVersion(versionInfo);
        return versionInfo;
    }

    /**
     * @brief Loads a pre-built network with weights to the engine.
     * After that the network is ready for inference.
     * @param network Network object acquired from CNNNetReader
     */
    void LoadNetwork(ICNNNetwork &network) {
        CALL_STATUS_FNC(LoadNetwork, network);
    }

    /**
     * @brief Loads a pre-built network with weights to the engine.
     * After that the network is ready for inference.
     * @param network Network object acquired from CNNNetReader
     * @param config Map of pairs: (parameter name, parameter value)
     * @return Executable network object
    */
    ExecutableNetwork LoadNetwork(ICNNNetwork &network, const std::map<std::string, std::string> &config) {
        IExecutableNetwork::Ptr ret;
        CALL_STATUS_FNC(LoadNetwork, ret, network, config);
        return ExecutableNetwork(ret);
    }

    /**
     * @brief Infers an image(s)
     * Input and output dimensions depend on the topology.
     *     As an example for classification topologies use a 4D Blob as input (batch, channels, width,
     *             height) and get a 1D blob as output (scoring probability vector). To infer a batch,
     *             use a 4D Blob as input and get a 2D blob as output in both cases the method
     *             allocates the resulted blob
     * @param input Map of input blobs accessed by input names
     * @param result Map of output blobs accessed by output names
     */
    void Infer(const BlobMap &input, BlobMap &result) {
        CALL_STATUS_FNC(Infer, input, result);
    }

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     * Note: not all plugins provide meaningful data.
     * @return perfMap Map of pair: (layer name, layer profiling information)
     */
    std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const {
        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        CALL_STATUS_FNC(GetPerformanceCounts, perfMap);
        return perfMap;
    }

    /**
     * @brief Registers extension within the plugin
     * @param extension Pointer to already loaded extension
     */
    void AddExtension(InferenceEngine::IExtensionPtr extension) {
        CALL_STATUS_FNC(AddExtension, extension);
    }

    /**
    * @brief Sets configuration for plugin, acceptable keys can be found in ie_plugin_config.hpp
    * @param config Map of pairs: (config parameter name, config parameter value)
    */
    void SetConfig(const std::map<std::string, std::string> &config) {
        CALL_STATUS_FNC(SetConfig, config);
    }

    /**
    * @brief Creates an executable network from a previously exported network
    * @param ret Reference to a shared ptr of the returned network interface
    * @param modelFileName Path to the location of the exported file
    * @param resp Pointer to the response message that holds a description of an error if any occurred
    */
    void ImportNetwork(IExecutableNetwork::Ptr &ret, const std::string &modelFileName) {
        CALL_STATUS_FNC(ImportNetwork, ret, modelFileName);
    }

    void QueryNetwork(const ICNNNetwork &network, QueryNetworkResult &res) const {
        actual->QueryNetwork(network, res);
        if (res.rc != OK) THROW_IE_EXCEPTION << res.resp.msg;
    }


    /**
    * @return wrapped object
    */
    operator InferenceEngine::InferenceEnginePluginPtr() {
        return actual;
    }

    typedef std::shared_ptr<InferencePlugin> Ptr;
};
}  // namespace InferenceEngine
