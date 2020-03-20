// Copyright (c) 2017-2018 Intel Corporation
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
 * @brief A header that defines advanced related properties for CPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods
 *
 * @file ie_helpers.hpp
 */


#include "IRDocument.h"
#include "IRLayers.h"
#include "GnaNetwork.h"
#include "IRBuilder.h"
#include "gna_config.hpp"
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>

#include "ie_iinfer_request.hpp"
#include "ie_infer_request.hpp"
#include "ie_plugin_cpp.hpp"
#include "ie_exception_conversion.hpp"
#include "ie_builders.hpp"
#include "ie_network.hpp"
//#include "debug.h"
#include <fstream>

#include <android/log.h>
#include <log/log.h>

using namespace InferenceEngine::details;
using namespace InferenceEngine;

Blob::Ptr generateBlob(Precision pr, SizeVector dims, Layout la) {
    auto blob = make_shared_blob<float>({pr, dims, la});
    blob->allocate();
    return blob;
}

void GnaNetwork::loadNetwork(InferenceEngine::CNNNetwork& passed_network)
{
    ALOGI("IENetwork.h void loadNetwork() GNA device");
    InferencePlugin plugin(enginePtr);

    /** Specifying the precision and layout of input data provided by the user.
      * This should be called before load of the network to the plugin **/
    ALOGI("IENetwork.h TargetDevice eGNA");
    std::map<std::string, std::string> config;
    //config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
    std::map<std::string, std::string> gnaPluginConfig;
    gnaPluginConfig[GNAConfigParams::KEY_GNA_DEVICE_MODE] = "GNA_HW";
    gnaPluginConfig[GNAConfigParams::KEY_GNA_PRECISION] = "I16";
    std::string scaleFactorConfigKey_1 = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(1);
    gnaPluginConfig[scaleFactorConfigKey_1] = std::to_string(2048);
    gnaPluginConfig[GNA_CONFIG_KEY(COMPACT_MODE)] = CONFIG_VALUE(NO);
    //gnaPluginConfig[GNA_CONFIG_KEY(LIB_N_THREADS)] = "3";
    config.insert(std::begin(gnaPluginConfig), std::end(gnaPluginConfig));
    ALOGI("IENetwork.h Create plugin");

	  InferenceEngine::Core ie;
	  executable_network = ie.LoadNetwork(passed_network, "GNA", config);
    inputInfo = passed_network.getInputsInfo();
    // if (inputInfo.size() != 1) {
    //     throw std::logic_error("Sample supports topologies only with 1 input");
    // }
    outputInfo = passed_network.getOutputsInfo();
    inferRequest = executable_network.CreateInferRequest();
}

void GnaNetwork::prepareInput()
{
  #ifdef NNLOG
    ALOGI("Prepare input blob");
  #endif

    Precision inputPrecision = Precision::FP32;
    inputInfo.begin()->second->setPrecision(inputPrecision);

    auto inputDims = inputInfo.begin()->second->getTensorDesc().getDims();
    if (inputDims.size() == 4)
      inputInfo.begin()->second->setLayout(Layout::NCHW);
    else if (inputDims.size() == 2)
      inputInfo.begin()->second->setLayout(Layout::NC);
    else
      inputInfo.begin()->second->setLayout(Layout::C);
}

void GnaNetwork::prepareOutput()
{
  #ifdef NNLOG
    ALOGI("Prepare output blob");
  #endif
    Precision outputPrecision = Precision::FP32;
    outputInfo.begin()->second->setPrecision(outputPrecision);

    auto outputDims = outputInfo.begin()->second->getDims();
    if (outputDims.size() == 4)
      outputInfo.begin()->second->setLayout(Layout::NHWC);
    else if (outputDims.size() == 2)
      outputInfo.begin()->second->setLayout(Layout::NC);
    else
      outputInfo.begin()->second->setLayout(Layout::C);

    #ifdef NNLOG
    //auto dims = inputInfo.begin()->second->getDims();
    //ALOGI("inputInfo dims size = %d\n", dims.size());
    //ALOGI("outputInfo dims size = %d\n", outputDims.size());
    #endif
}

//setBlob input/output blob for infer request
void GnaNetwork::setBlob(const std::string& inName, const Blob::Ptr& inputBlob)
{
    ALOGI("setBlob input or output blob name : %s", inName.c_str());
    ALOGI("Blob size %d and size in bytes %d bytes element size %d bytes", inputBlob->size(), inputBlob->byteSize(), inputBlob->element_size());
    inferRequest.SetBlob(inName, inputBlob);
}

//for non aync infer request
TBlob<float>::Ptr GnaNetwork::getBlob(const std::string& outName) {
    Blob::Ptr outputBlob;
    outputBlob = inferRequest.GetBlob(outName);
    #ifdef NNLOG
      ALOGI("Get input/output blob, name : ", outName.c_str());
    #endif
    return android::hardware::neuralnetworks::nnhal::As<TBlob<float>>(outputBlob);
}

void GnaNetwork::Infer() {
    #ifdef NNLOG
        ALOGI("Infer Network\n");
        ALOGI("StartAsync scheduled");
    #endif
    inferRequest.StartAsync();  //for async infer

    //ALOGI("async wait");
    //inferRequest.Wait(1000);
    //inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);
    inferRequest.Wait(10000); //check right value to infer

    #ifdef NNLOG
    ALOGI("infer request completed");
    #endif
}
