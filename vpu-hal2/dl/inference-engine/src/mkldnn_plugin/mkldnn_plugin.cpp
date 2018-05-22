//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
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
#include "mkldnn_plugin.h"
#include "mkldnn_extension_mngr.h"
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <memory>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

InferenceEngine::ExecutableNetworkInternal::Ptr
Engine::LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network, const std::map<std::string, std::string> &config) {
    auto specifiedDevice = network.getTargetDevice();
    auto supportedDevice = InferenceEngine::TargetDevice::eCPU;
    if (specifiedDevice != InferenceEngine::TargetDevice::eDefault && specifiedDevice != supportedDevice) {
        THROW_IE_EXCEPTION << "The plugin doesn't support target device: " << getDeviceName(specifiedDevice) << ".\n" <<
                           "Supported target device: " << getDeviceName(supportedDevice);
    }

    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getInputPrecision();
        if (input_precision != InferenceEngine::Precision::U16 && input_precision != InferenceEngine::Precision::I16
            && input_precision != InferenceEngine::Precision::FP32 && input_precision != InferenceEngine::Precision::U8) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "Input image format " << input_precision << " is not supported yet...";
        }
    }

    // TODO: handle input precision differently - per input and not one per network...

    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;
    conf.readProperties(config);

    return std::make_shared<MKLDNNExecNetwork>(network, conf, extensionManager);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    // accumulate config parameters on engine level
    engConfig.readProperties(config);

    // Pass config to already loaded network
    // TODO: Clarify the behavior of SetConfig method. Should it pass data to already loaded networks?
    if (_loadedNetwork) {
        // ugly casting. can we avoid it?
        auto exe_network =
                dynamic_cast<ExecutableNetworkBase<ExecutableNetworkInternal>*>(_loadedNetwork.get());
        auto exe_network_impl = dynamic_cast<MKLDNNExecNetwork*>(exe_network->getImpl().get());

        exe_network_impl->setProperty(config);
    }
}

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    extensionManager->AddExtension(extension);
}

void Engine::QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const {
    details::CNNNetworkIterator i(const_cast<ICNNNetwork *>(&network));
    while (i != details::CNNNetworkIterator()) {
        try {
            // if we can create and have not thrown exception, then layer is supported
            MKLDNNNode::CreateNode(*i, extensionManager);
            res.supportedLayers.insert((*i)->name);
        } catch (InferenceEngine::details::InferenceEngineException) {
        }
        i++;
    }
}

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin(
                {{1, 0},
#ifdef MKL_VERSION
                 MKL_VERSION,
#else
                 CI_BUILD_NUMBER,
#endif
                 "MKLDNNPlugin"}, std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
