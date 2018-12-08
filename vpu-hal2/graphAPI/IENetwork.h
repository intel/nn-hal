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

#pragma once

#include "IRDocument.h"
#include "IRLayers.h"
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>
#include "vpu_plugin_config.hpp"
#include "ie_iinfer_request.hpp"
#include "ie_infer_request.hpp"
#include "ie_plugin_cpp.hpp"
#include "ie_exception_conversion.hpp"
#include "debug.h"
#include <fstream>

#include <android/log.h>
#include <log/log.h>

using namespace InferenceEngine::details;
using namespace IRBuilder;
using namespace InferenceEngine;

template <typename T>
inline std::ostream & operator << (std::ostream &out, const std::vector<T> &vec) {
    if (vec.empty()) return std::operator<<(out, "[]");
    out << "[" << vec[0];
    for (unsigned i=1; i < vec.size(); i++) {
        out << ", " << vec[i];
    }
    return out << "]";
}

//aks
/*
void dumpBlob(const std::string &prefix, size_t len, TBlob<short>::Ptr blob)
{
    auto dims = blob->getTensorDesc().getDims();
    //std::cout << prefix << dims;
    ALOGI("prefix %s", prefix.c_str());

    auto mem = blob->readOnly();

    const float *pf = mem.as<const float*>();

    if (len > blob->size()) len = blob->size();

    for (unsigned int i=0; i<len; i++)
    {
        if (0==i % 16)
        {
            //std::cout << std::endl<< i<< ": ";
            ALOGI("i : %d", i);
        }
        //std::cout << pf[i] << ", ";
        ALOGI(", %1.0f",pf[i]);
    }
    //std::cout << std::endl;
    ALOGI("-end");
}

*/
static void setConfig(std::map<std::string, std::string> &config) {
    //config[VPUConfigParams::FIRST_SHAVE] = "0";
    //config[VPUConfigParams::LAST_SHAVE] = "11";
    //config[VPUConfigParams::MEMORY_OPTIMIZATION] = CONFIG_VALUE(NO);//InferenceEngine::PluginConfigParams::YES;
    //config[VPUConfigParams::COPY_OPTIMIZATION] = CONFIG_VALUE(NO);//InferenceEngine::PluginConfigParams::YES;
/* //enable below for VPU logs
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
    config[VPUConfigParams::KEY_VPU_LOG_LEVEL] = CONFIG_VALUE(LOG_DEBUG);
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_DEBUG);
*/
    //config[VPU_CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_DEBUG);
    //config[InferenceEngine::PluginConfigParams::CONFIG_KEY(LOG_LEVEL)] = InferenceEngine::PluginConfigParams::LOG_DEBUG;
    //config[VPUConfigParams::VPU_LOG_LEVEL] = CONFIG_VALUE(LOG_DEBUG);
    //config[InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL] = InferenceEngine::PluginConfigParams::LOG_DEBUG /*LOG_WARNING*/;
    //config[InferenceEngine::VPUConfigParams::IGNORE_UNKNOWN_LAYERS] = InferenceEngine::PluginConfigParams::NO;
}

class ExecuteNetwork
{
    InferenceEnginePluginPtr enginePtr;
    ICNNNetwork *network;
    //IExecutableNetwork::Ptr pExeNet;
    ExecutableNetwork executable_network;
    InputsDataMap inputInfo;
    OutputsDataMap outputInfo;
    IInferRequest::Ptr req;
    InferRequest inferRequest;
    ResponseDesc resp;

public:
    ExecuteNetwork(){}
    ExecuteNetwork(IRDocument &doc, TargetDevice target = TargetDevice::eCPU)
    {
        InferenceEngine::PluginDispatcher dispatcher({"/vendor/lib64","/vendor/lib","/system/lib64","/system/lib","","./"});
        enginePtr = dispatcher.getSuitablePlugin(target);

        network = doc.getNetwork();
        network->getInputsInfo(inputInfo);
        network->getOutputsInfo(outputInfo);

        //size_t batch = 1;
        //network->setBatchSize(batch);

        #ifdef NNLOG
        ALOGI("%s Plugin loaded",InferenceEngine::TargetDeviceInfo::name(target));
        #endif
    }

    ExecuteNetwork(ExecutableNetwork& exeNet) : ExecuteNetwork(){
    executable_network = exeNet;
    inferRequest = executable_network.CreateInferRequest();
    ALOGI("infer request created");

    }

    //~ExecuteNetwork(){ }
    void loadNetwork()
    {

        std::map<std::string, std::string> networkConfig;
        setConfig(networkConfig);

        InferencePlugin plugin(enginePtr);
        executable_network = plugin.LoadNetwork(*network, networkConfig);
        //std::cout << "Network loaded" << std::endl;
	ALOGI("Network loaded");

        inferRequest = executable_network.CreateInferRequest();
        //std::cout << "infer request created" << std::endl;
      }

    void prepareInput()
    {
	  #ifdef NNLOG
      ALOGI("Prepare input blob");
	  #endif
      Precision inputPrecision = Precision::FP32;
      inputInfo.begin()->second->setPrecision(inputPrecision);
      inputInfo.begin()->second->setLayout(Layout::NC);

    }

    void prepareOutput()
    {
	  #ifdef NNLOG
      ALOGI("Prepare output blob");
	  #endif
      Precision inputPrecision = Precision::FP32;
      outputInfo.begin()->second->setPrecision(inputPrecision);
      //outputInfo.begin()->second->setLayout(Layout::NC);

      #ifdef NNLOG
      auto dims = inputInfo.begin()->second->getDims();
      ALOGI("inputInfo dims size = %d\n", dims.size());
      //outputInfo.begin()->second->setDims(dims);
      auto outputDims = outputInfo.begin()->second->getDims();
      ALOGI("outputInfo dims size = %d\n", outputDims.size());
      #endif
    }

    //setBlob input/output blob for infer request
    void setBlob(const std::string& inName, const Blob::Ptr& inputBlob)
    {
        #ifdef NNLOG
        ALOGI("setBlob input or output blob name : %s", inName.c_str());
        ALOGI("Blob size %d and size in bytes %d bytes element size %d bytes", inputBlob->size(), inputBlob->byteSize(), inputBlob->element_size());
        #endif

        //inferRequest.SetBlob(inName.c_str(), inputBlob);
        inferRequest.SetBlob(inName, inputBlob);

        //std::cout << "setBlob input or output name : " << inName << std::endl;

    }

     //for non aync infer request
    TBlob<float>::Ptr getBlob(const std::string& outName) {
       Blob::Ptr outputBlob;
       outputBlob = inferRequest.GetBlob(outName);
       //std::cout << "GetBlob input or output name : " << outName << std::endl;
       #ifdef NNLOG
       ALOGI("Get input/output blob, name : ", outName.c_str());
       #endif
       return As<TBlob<float>>(outputBlob);
       //return outputBlob;
    }

    void Infer() {
        #ifdef NNLOG
        ALOGI("Infer Network\n");
        #endif
//        inferRequest = executable_network.CreateInferRequest();
/*
        auto inName = inputInfo.begin()->first;
        ALOGI("set input blob\n");
        inferRequest.SetBlob(inName, in);

        ALOGI("aks prepare output blob\n");
        const std::string firstOutName = outputInfo.begin()->first;
        InferenceEngine::TBlob<PrecisionTrait<Precision::FP32>::value_type>::Ptr outputBlob;
        outputBlob = InferenceEngine::make_shared_blob<PrecisionTrait<Precision::FP32>::value_type,
                InferenceEngine::SizeVector>(Precision::FP32, outputInfo.begin()->second->getDims());
        outputBlob->allocate();

        ALOGI("set output blob\n");
        inferRequest.SetBlob(firstOutName, outputBlob);

*/
        #ifdef NNLOG
        ALOGI("StartAsync scheduled");
        #endif
        inferRequest.StartAsync();  //for async infer
        //ALOGI("async wait");
        //inferRequest.Wait(1000);
        inferRequest.Wait(10000); //check right value to infer
        //inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);

        //std::cout << "output name : " << firstOutName << std::endl;
        #ifdef NNLOG
        ALOGI("infer request completed");
        #endif

        return;
    }
};
