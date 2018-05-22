//
// INTEL CONFIDENTIAL
// Copyright 2017 Intel Corporation.
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


#include <memory>
#include <vector>

#include "ie_plugin.hpp"
#include "inference_engine.hpp"
#include <vpu/vpu_plugin_config.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>

#include "hddl_plugin.h"
#include "hddl_executable_network.h"

using namespace InferenceEngine;
using namespace InferenceEngine::VPUConfigParams;
using namespace InferenceEngine::PluginConfigParams;
using namespace VPU::HDDLPlugin;

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin({0, 1, CI_BUILD_NUMBER, "HDDLPlugin"},
                                           std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

ExecutableNetworkInternal::Ptr
Engine::LoadExeNetworkImpl(ICNNNetwork &network, const std::map<std::string, std::string> &config) {
    InferenceEngine::InputsDataMap networkInputs;
    InferenceEngine::OutputsDataMap networkOutputs;

    network.getInputsInfo(networkInputs);
    network.getOutputsInfo(networkOutputs);

    auto specifiedDevice = network.getTargetDevice();
    auto supportedDevice = InferenceEngine::TargetDevice::eHDDL;
    if (specifiedDevice != InferenceEngine::TargetDevice::eDefault && specifiedDevice != supportedDevice) {
        THROW_IE_EXCEPTION << "The plugin doesn't support target device: " << getDeviceName(specifiedDevice) << ".\n" <<
                           "Supported target device: " << getDeviceName(supportedDevice);
    }

    for (auto networkInput : networkInputs) {
        auto input_precision = networkInput.second->getInputPrecision();

        if (input_precision != InferenceEngine::Precision::FP32 && input_precision != InferenceEngine::Precision::U8
            && input_precision != InferenceEngine::Precision::FP16) {
            THROW_IE_EXCEPTION << "Input image format " << input_precision << " is not supported yet.\n"
                               << "Supported formats: FP16, FP32 and U8.";
        }
    }

    if (network.getPrecision() != Precision::FP16) {
        THROW_IE_EXCEPTION << "The plugin does not support networks with " << network.getPrecision() << " format.\n"
                           << "Supported format: FP16.";
    }


    // override what was set globally for plugin, otherwise - override default config without touching config for plugin
    auto configCopy = _config;
    for (auto i = config.begin(); i != config.end(); i++) {
        configCopy[i->first] = i->second;
    }

    return std::make_shared<ExecutableNetwork>(network, configCopy, _hddlAllocatorPtr);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    // override default config
    for (auto i = config.begin(); i != config.end(); i++) {
        _config[i->first] = i->second;
    }
    Common::ParsedConfig::validate(_config, MYRIAD_X);
    _log->init(Common::ParsedConfig::parseLogLevel(_config.at(CONFIG_KEY(LOG_LEVEL))));
}

Engine::Engine() {
    _config = Common::ParsedConfig::getDefaultConfig(MYRIAD_X);
    _log = std::make_shared<Common::Logger>();
    _log->init(Common::ParsedConfig::parseLogLevel(_config.at(CONFIG_KEY(LOG_LEVEL))));
    _hddlAllocatorPtr = std::make_shared<HDDLAllocator>(_log);
}

Engine::~Engine() {
    _hddlAllocatorPtr->Release();
}
