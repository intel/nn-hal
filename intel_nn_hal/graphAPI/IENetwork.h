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

#define IE_LEGACY

#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>
#include "IRDocument.h"
#include "IRLayers.h"
#ifdef USE_NGRAPH
#undef IE_LEGACY
#endif

#include <fstream>
#include "ie_exception_conversion.hpp"
#include "ie_iinfer_request.hpp"
#include "ie_infer_request.hpp"
#ifdef IE_LEGACY
#include "ie_plugin_cpp.hpp"
#else
#include <ie_core.hpp>
#endif // IE_LEGACY

#include <android/log.h>
#include <log/log.h>

#ifdef ENABLE_MYRIAD
#include "vpu_plugin_config.hpp"
#endif

#ifdef USE_NGRAPH
#include <cutils/properties.h>
#endif

using namespace InferenceEngine::details;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
template <typename T>
inline std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
    if (vec.empty()) return std::operator<<(out, "[]");
    out << "[" << vec[0];
    for (unsigned i = 1; i < vec.size(); i++) {
        out << ", " << vec[i];
    }
    return out << "]";
}

// aks
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
    // config[VPUConfigParams::FIRST_SHAVE] = "0";
    // config[VPUConfigParams::LAST_SHAVE] = "11";
    // config[VPUConfigParams::MEMORY_OPTIMIZATION] =
    // CONFIG_VALUE(NO);//InferenceEngine::PluginConfigParams::YES;
    // config[VPUConfigParams::COPY_OPTIMIZATION] =
    // CONFIG_VALUE(NO);//InferenceEngine::PluginConfigParams::YES;
    /* //enable below for VPU logs
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
        config[VPUConfigParams::KEY_VPU_LOG_LEVEL] = CONFIG_VALUE(LOG_DEBUG);
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_DEBUG);
    */
    // config[VPU_CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_DEBUG);
    // config[InferenceEngine::PluginConfigParams::CONFIG_KEY(LOG_LEVEL)] =
    // InferenceEngine::PluginConfigParams::LOG_DEBUG; config[VPUConfigParams::VPU_LOG_LEVEL] =
    // CONFIG_VALUE(LOG_DEBUG); config[InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL] =
    // InferenceEngine::PluginConfigParams::LOG_DEBUG /*LOG_WARNING*/;
    // config[InferenceEngine::VPUConfigParams::IGNORE_UNKNOWN_LAYERS] =
    // InferenceEngine::PluginConfigParams::NO; config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] =
    // VPU_CONFIG_VALUE(NHWC);
}

#ifdef USE_NGRAPH
static bool isNgraphPropSet() {
    const char ngIrProp[] = "nn.hal.ngraph";
    return property_get_bool(ngIrProp, false);
}
#endif

class ExecuteNetwork {
#ifdef IE_LEGACY
    InferenceEnginePluginPtr enginePtr;
#else
    CNNNetwork mCnnNetwork;
#endif // IE_LEGACY
    ICNNNetwork *network;
    // IExecutableNetwork::Ptr pExeNet;
    ExecutableNetwork executable_network;
    InputsDataMap inputInfo = {};
    OutputsDataMap outputInfo = {};
    IInferRequest::Ptr req;
    InferRequest inferRequest;
    ResponseDesc resp;
#ifdef USE_NGRAPH
    bool mNgraphProp = false;
#endif

public:
    ExecuteNetwork() : network(nullptr) {
    }
#ifdef USE_NGRAPH
    ExecuteNetwork(CNNNetwork ngraphNetwork, IRDocument &doc,  std::string target = "CPU") : network(nullptr) {
        mNgraphProp = isNgraphPropSet();
#else
    ExecuteNetwork(IRDocument &doc,  std::string target = "CPU") : network(nullptr) {
#endif
#ifdef IE_LEGACY
        InferenceEngine::PluginDispatcher dispatcher(
            {"/vendor/lib64", "/vendor/lib", "/system/lib64", "/system/lib", "", "./"});
        enginePtr = dispatcher.getPluginByDevice(target);
#endif // IE_LEGACY
#ifdef USE_NGRAPH
        if(mNgraphProp) {
            inputInfo = ngraphNetwork.getInputsInfo();
            outputInfo = ngraphNetwork.getOutputsInfo();
        } else
#endif
        {
            network = doc.getNetwork();
            network->getInputsInfo(inputInfo);
            network->getOutputsInfo(outputInfo);
        }

#ifndef IE_LEGACY
#ifdef USE_NGRAPH
        if(!mNgraphProp)
#endif
        {
            std::shared_ptr<InferenceEngine::ICNNNetwork> sp_cnnNetwork;
            sp_cnnNetwork.reset(network);
            mCnnNetwork = InferenceEngine::CNNNetwork(sp_cnnNetwork);
        }
#endif // IE_LEGACY
        // size_t batch = 1;
        // network->setBatchSize(batch);

#ifdef NNLOG
#ifdef IE_LEGACY
        ALOGI("%s Plugin loaded", InferenceEngine::TargetDeviceInfo::name(target));
#endif // IE_LEGACY
#endif
    }

    ExecuteNetwork(ExecutableNetwork &exeNet) : ExecuteNetwork() {
        executable_network = exeNet;
        inferRequest = executable_network.CreateInferRequest();
        ALOGI("infer request created");
    }

    //~ExecuteNetwork(){ }
#ifdef USE_NGRAPH
    void loadNetwork(CNNNetwork ngraphNetwork) {
#else
    void loadNetwork() {
#endif
#ifdef IE_LEGACY
        std::map<std::string, std::string> networkConfig;
        InferencePlugin plugin(enginePtr);

        setConfig(networkConfig);

        ALOGI("%s before plugin.LoadNetwork()", __func__);
        executable_network = plugin.LoadNetwork(*network, networkConfig);

        ALOGI("%s before CreateInferRequest", __func__);
        inferRequest = executable_network.CreateInferRequest();
#else
        Core ie_core(std::string("/vendor/etc/openvino/plugins.xml"));

#ifdef USE_NGRAPH
    try {
        if(mNgraphProp == true) {
            ALOGI("%s LoadNetwork actually using ngraphNetwork", __func__);
            executable_network = ie_core.LoadNetwork(ngraphNetwork, std::string("CPU"));
        } else {
            ALOGI("%s LoadNetwork actually using mCnnNetwork", __func__);
            executable_network = ie_core.LoadNetwork(mCnnNetwork, std::string("CPU"));
        }
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
    }
#else
        ALOGI("%s Loading network to IE", __func__);
        executable_network = ie_core.LoadNetwork(mCnnNetwork, std::string("CPU"));
#endif

        ALOGI("%s Calling CreateInferRequest", __func__);
        inferRequest = executable_network.CreateInferRequest();
#endif // IE_LEGACY
    }

    void prepareInput() {
#ifdef NNLOG
        ALOGI("Prepare input blob");
#endif
        Precision inputPrecision = Precision::FP32;
        inputInfo.begin()->second->setPrecision(inputPrecision);
        // inputInfo.begin()->second->setPrecision(Precision::U8);

        auto inputDims = inputInfo.begin()->second->getTensorDesc().getDims();
        if (inputDims.size() == 4)
            inputInfo.begin()->second->setLayout(Layout::NCHW);
        else if (inputDims.size() == 2)
            inputInfo.begin()->second->setLayout(Layout::NC);
        else
            inputInfo.begin()->second->setLayout(Layout::C);

        // inputInfo.begin()->second->setPrecision(Precision::U8);
        // inputInfo.begin()->second->setLayout(Layout::NCHW);
    }

    void prepareOutput() {
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
// auto dims = inputInfo.begin()->second->getDims();
// ALOGI("inputInfo dims size = %d\n", dims.size());
// ALOGI("outputInfo dims size = %d\n", outputDims.size());
#endif
    }

    // setBlob input/output blob for infer request
    void setBlob(const std::string &inName, const Blob::Ptr &inputBlob) {
#ifdef NNLOG
        ALOGI("setBlob input or output blob name : %s", inName.c_str());
        ALOGI("Blob size %d and size in bytes %d bytes element size %d bytes", inputBlob->size(),
              inputBlob->byteSize(), inputBlob->element_size());
#endif

        // inferRequest.SetBlob(inName.c_str(), inputBlob);
        inferRequest.SetBlob(inName, inputBlob);

        // std::cout << "setBlob input or output name : " << inName << std::endl;
    }

    // for non aync infer request
    TBlob<float>::Ptr getBlob(const std::string &outName) {
        Blob::Ptr outputBlob;
        outputBlob = inferRequest.GetBlob(outName);
// std::cout << "GetBlob input or output name : " << outName << std::endl;
#ifdef NNLOG
        ALOGI("Get input/output blob, name : ", outName.c_str());
#endif
        return As<TBlob<float>>(outputBlob);
        // return outputBlob;
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
                outputBlob =
           InferenceEngine::make_shared_blob<PrecisionTrait<Precision::FP32>::value_type,
                        InferenceEngine::SizeVector>(Precision::FP32,
           outputInfo.begin()->second->getDims()); outputBlob->allocate();

                ALOGI("set output blob\n");
                inferRequest.SetBlob(firstOutName, outputBlob);

        */
#ifdef NNLOG
        ALOGI("StartAsync scheduled");
#endif
        inferRequest.StartAsync();  // for async infer
        // ALOGI("async wait");
        // inferRequest.Wait(1000);
        inferRequest.Wait(10000);  // check right value to infer
// inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);

// std::cout << "output name : " << firstOutName << std::endl;
#ifdef NNLOG
        ALOGI("infer request completed");
#endif

        return;
    }
};
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
