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

#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>

#include "ie_iinfer_request.hpp"
#include "ie_infer_request.hpp"
#include "ie_plugin_cpp.hpp"
#include "ie_exception_conversion.hpp"
#include "impl/ie_memory_state_internal.hpp"
#include "ie_memory_state.hpp"
//#include "debug.h"
#include <fstream>

#include <android/log.h>
#include <log/log.h>
#include "IRDocument.h"
#include "IRLayers.h"
#include "Utils.h"

#include <fstream>
#include <sys/stat.h>

#ifdef ENABLE_MYRIAD
#include "vpu_plugin_config.hpp"
#endif
#include "gna_config.hpp"
using namespace InferenceEngine::details;
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

using BaseOp = ::android::hardware::neuralnetworks::nnhal::BaseOp;

class GnaNetwork : public BaseOp
{
    //InferenceEnginePluginPtr enginePtr;
    std::shared_ptr<ICNNNetwork> network;
    //IExecutableNetwork::Ptr pExeNet;
    ExecutableNetwork executable_network;
    IInferRequest::Ptr req;
    InferRequest inferRequest;
    ResponseDesc resp;
    std::vector<InferenceEngine::MemoryState> memoryStates;

public:
    InputsDataMap inputInfo = {};
    OutputsDataMap outputInfo = {};
    std::map<std::string, InferenceEngineProfileInfo> perfCounters;
    int noInferCall = 0;
    double totalTime = 0.0f;
    std::vector<double> inferTime;
    std::vector<double> inferTimeGNA;
    std::map<std::string, InferenceEngineProfileInfo> totalPerfCounters;
    bool isCpuOp() { return false; }

    std::vector<InferenceEngine::Blob::Ptr> ptrInputBlobs;
    GnaNetwork() : network(nullptr){}

    GnaNetwork(std::shared_ptr<ICNNNetwork> curr_network, std::string target = "GNA")
    {
        if (curr_network) {
            InferenceEngine::CNNNetwork cnnnetwork({curr_network});
            network = curr_network;
        }
        #ifdef NNLOG
            ALOGI("%s Plugin loaded",InferenceEngine::TargetDeviceInfo::name(target));
        #endif
    }

    GnaNetwork(ExecutableNetwork& exeNet) : GnaNetwork(){
	    executable_network = exeNet;
	    inferRequest = executable_network.CreateInferRequest();
	    ALOGI("infer request created");
    }

#ifdef CACHING
    void importNetwork(const std::string& gnaModel, bool isDecoderNw);

    void exportGraph(const std::string& fileName) {
        executable_network.Export(fileName);
    }
#endif
    void loadNetwork(InferenceEngine::CNNNetwork& passed_network, bool isDecoderNw);

    void prepareInput();

    void prepareOutput();

    void setBlob(const std::string& inName, const Blob::Ptr& inputBlob);

     //for non aync infer request
    TBlob<float>::Ptr getBlob(const std::string& outName);

    InferRequest getInferRequest() {
        return inferRequest;
    }

    void queryState() {
        memoryStates = executable_network.QueryState();
    }

    void setMemoryState(std::string index, InferenceEngine::Blob::Ptr blob) {
        for (auto&& state: memoryStates) {
            if (state.GetName() == index) {
                state.SetState(blob);
                break;
            }
        }
    }

    void reset() {
        for (auto&& memState: memoryStates) {
            memState.Reset();
        }
    }

    void Infer();

    ConstInputsDataMap getInputsInfo() {
        return executable_network.GetInputsInfo();
    }

    ConstOutputsDataMap getOutputsInfo() {
        return executable_network.GetOutputsInfo();
    }
};
