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

#ifndef IENETWORK_H
#define IENETWORK_H

#include <android/log.h>
#include <cutils/properties.h>
#include <log/log.h>
#include <fstream>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_core.hpp"
#include "inference_engine.hpp"

using namespace InferenceEngine::details;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

typedef InferenceEngine::Blob IRBlob;
typedef InferenceEngine::SizeVector TensorDims;

template <typename T, typename S>
std::shared_ptr<T> As(const std::shared_ptr<S> &src) {
    return /*std::dynamic_pointer_cast<T>(src)*/ std::static_pointer_cast<T>(src);
}  // aks

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
    if (vec.empty()) return std::operator<<(out, "[]");
    out << "[" << vec[0];
    for (unsigned i = 1; i < vec.size(); i++) {
        out << ", " << vec[i];
    }
    return out << "]";
}

class ExecuteNetwork {
    CNNNetwork mCnnNetwork;
    ExecutableNetwork executable_network;
    InputsDataMap inputInfo = {};
    OutputsDataMap outputInfo = {};
    InferRequest inferRequest;
    ResponseDesc resp;

public:
    ExecuteNetwork() {}
    ExecuteNetwork(CNNNetwork ngraphNetwork, std::string target = "CPU") {
        inputInfo = ngraphNetwork.getInputsInfo();
        outputInfo = ngraphNetwork.getOutputsInfo();
    }

    ExecuteNetwork(ExecutableNetwork &exeNet) : ExecuteNetwork() {
        executable_network = exeNet;
        inferRequest = executable_network.CreateInferRequest();
        ALOGI("infer request created");
    }

    void loadNetwork(CNNNetwork ngraphNetwork) {
        ALOGV("Entering %s", __FUNCTION__);
        Core ie_core(std::string("/vendor/etc/openvino/plugins.xml"));

        try {
            ALOGD("%s LoadNetwork using ngraphNetwork", __func__);
            executable_network = ie_core.LoadNetwork(ngraphNetwork, std::string("CPU"));
        } catch (const std::exception &ex) {
            ALOGE("%s Exception !!! %s", __func__, ex.what());
        }

        ALOGD("%s Calling CreateInferRequest", __func__);
        inferRequest = executable_network.CreateInferRequest();
        ALOGV("Exiting %s", __FUNCTION__);
    }

    void prepareInput() {
        ALOGV("Entering %s", __FUNCTION__);
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
        ALOGV("Exiting %s", __FUNCTION__);
    }

    void prepareOutput() {
        ALOGV("Entering %s", __FUNCTION__);
        Precision outputPrecision = Precision::FP32;
        outputInfo.begin()->second->setPrecision(outputPrecision);

        auto outputDims = outputInfo.begin()->second->getDims();
        if (outputDims.size() == 4)
            outputInfo.begin()->second->setLayout(Layout::NHWC);
        else if (outputDims.size() == 2)
            outputInfo.begin()->second->setLayout(Layout::NC);
        else
            outputInfo.begin()->second->setLayout(Layout::C);
        ALOGV("Exiting %s", __FUNCTION__);
    }

    // setBlob input/output blob for infer request
    void setBlob(const std::string &inName, const Blob::Ptr &inputBlob) {
        ALOGD("setBlob input or output blob name : %s", inName.c_str());
        ALOGD("Blob size %d and size in bytes %d bytes element size %d bytes", inputBlob->size(),
              inputBlob->byteSize(), inputBlob->element_size());
        inferRequest.SetBlob(inName, inputBlob);
    }

    // for non aync infer request
    TBlob<float>::Ptr getBlob(const std::string &outName) {
        Blob::Ptr outputBlob;
        outputBlob = inferRequest.GetBlob(outName);
        ALOGD("Get input/output blob, name : ", outName.c_str());
        return As<TBlob<float>>(outputBlob);
    }

    void Infer() {
        ALOGV("Entering %s", __FUNCTION__);
        ALOGI("StartAsync scheduled");
        inferRequest.StartAsync();  // for async infer
        inferRequest.Wait(10000);   // check right value to infer
        ALOGI("infer request completed");
        ALOGV("Exiting %s", __FUNCTION__);
        return;
    }
};  // namespace nnhal
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // IENETWORK_H
