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

// Interface to register custom hetero functionality

#pragma once
#include <map>
#include <string>
#include <memory>
#include <ie_icnn_network.hpp>
#include <ie_iexecutable_network.hpp>
#include <ie_plugin.hpp>

namespace InferenceEngine {

/**
 * This interface describes a mechanism of custom loaders to be used in heterogeneous
 * plugin during setting of affinity and loading of split subnetwork to the plugins
 * The custom loader can define addition settings for the plugins or network loading
 * Examples of cases when this interface should be implemented in the application:
 * 1. add custom layers to existing plugins if it is not pointed to the heterogeneous plugin
 *  or registration of custom layer is different than supported in available public plugins
 * 2. set affinity manually for the same plugin being initialized by different parameters,
 *  e.g different device id
 *  In this case there will be mapping of
 *    Device1 > HeteroDeviceLoaderImpl1
 *    Device2 > HeteroDeviceLoaderImpl2
 *  the affinity should be pointed manually, the implementation of HeteroDeviceLoaderImpl1 and
 *  HeteroDeviceLoaderImpl2 should be in the application, and these device loaders should be registred
 *  through calling of
 *  IHeteroInferencePlugin::SetDeviceLoader("Device1", HeteroDeviceLoaderImpl1)
 *  IHeteroInferencePlugin::SetDeviceLoader("Device2", HeteroDeviceLoaderImpl2)
*/
class IHeteroDeviceLoader {
public:
    typedef std::shared_ptr<IHeteroDeviceLoader> Ptr;
    virtual ~IHeteroDeviceLoader() = default;

    /**
     * Loads network to the device. The instantiation of plugin should be in the implementation
     * of the IHeteroDeviceLoader. As well setting of special config option should happen in the
     * implementation as well
     * @param device Loading of network should happen for this device
     * @param ret Reference to a shared ptr of the returned network interface
     * @param network Network object acquired from CNNNetReader
     * @param config Map of pairs: (config parameter name, config parameter value) relevant only for this load operation
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode LoadNetwork(
        const std::string& device,
        IExecutableNetwork::Ptr &ret,
        ICNNNetwork &network,
        const std::map<std::string, std::string> &config,
        ResponseDesc *resp) noexcept = 0;

    /**
     * This function calls plugin function QueryNetwork for the plugin being instantiated
     * in the implementation of IHeteroDeviceLoader
     * @param device QueryNetwork will be executed for this device
     * @param network Network object acquired from CNNNetReader
     * @param res
     */
    virtual void QueryNetwork(const std::string &device,
                              const ICNNNetwork &network,
                              QueryNetworkResult &res)noexcept  = 0;
};

typedef std::map<std::string, InferenceEngine::IHeteroDeviceLoader::Ptr> MapDeviceLoaders;

/**
 * This interface extends regular plugin interface for heterogeneous case. Not all plugins
 * implements it. The main purpose of this interface - to register loaders and have an ability
 * to get default settings for affinity on certain devices.
 */
class IHeteroInferencePlugin : public IInferencePlugin {
public:
    /**
     * Registers device loader for the device
     * @param device - the device name being used in CNNNLayer::affinity
     * @param loader - helper class allowing to analyze if layers are supported and allow
     * to load network to the plugin being defined in the IHeteroDeviceLoader implementation
     */
    virtual void SetDeviceLoader(const std::string &device, IHeteroDeviceLoader::Ptr loader)noexcept = 0;

    /**
     * The main goal of this function to set affinity according to the options set for the plugin\
     * implemnenting IHeteroInferencePlugin.
     * This function works only if all affinity in the network are empty.
     * @param network Network object acquired from CNNNetReader
     * @param config Map of pairs: (config parameter name, config parameter value) relevant only for this load operation
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode SetAffinity(
        ICNNNetwork& network,
        const std::map<std::string, std::string> &config,
        ResponseDesc *resp) noexcept = 0;
};

}  // namespace InferenceEngine
