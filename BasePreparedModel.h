/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_BASEPREPAREDMODEL_H
#define ANDROID_ML_NN_BASEPREPAREDMODEL_H

#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hardware/hardware.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include <NgraphNetworkCreator.hpp>
#include "Driver.h"
#include "IENetwork.h"
#include "utils.h"
#include "ModelManager.h"

using ::android::hardware::MQDescriptorSync;
using ::android::hidl::memory::V1_0::IMemory;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace {

InferenceEngine::Precision g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;

using time_point = std::chrono::steady_clock::time_point;

auto now() { return std::chrono::steady_clock::now(); };

auto microsecondsDuration(decltype(now()) end, decltype(now()) start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
};

}  // namespace

template <class T>
using vec = std::vector<T>;
typedef uint8_t* memory;

class BasePreparedModel : public V1_2::IPreparedModel {
public:
    BasePreparedModel(const Model& model)
        : mTargetDevice("CPU"), mEnginePtr(nullptr) {
        g_layer_precision = InferenceEngine::Precision::FP16;
        mModelInfo = std::make_shared<NnapiModelInfo>(model);
        mNgc = std::make_shared<NgraphNetworkCreator>(mModelInfo.get(), mTargetDevice);
    }
    BasePreparedModel(const std::string device, const Model& model)
        : mTargetDevice(device), mEnginePtr(nullptr) {
        if (mTargetDevice == "CPU" || mTargetDevice == "GPU")
            g_layer_precision = InferenceEngine::Precision::FP32;
        else if (mTargetDevice == "MYRIAD")
            g_layer_precision = InferenceEngine::Precision::FP16;
        else
            g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;
        mModelInfo = std::make_shared<NnapiModelInfo>(model);
        mNgc = std::make_shared<NgraphNetworkCreator>(mModelInfo.get(), mTargetDevice);
    }

    virtual ~BasePreparedModel() { deinitialize(); }

    Return<ErrorStatus> execute(const Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override;
    Return<ErrorStatus> execute_1_2(const Request& request, MeasureTiming measure,
                                    const sp<V1_2::IExecutionCallback>& callback) override;
    Return<void> executeSynchronously(const Request& request, MeasureTiming measure,
                                      executeSynchronously_cb cb) override;
    Return<void> configureExecutionBurst(
        const sp<V1_2::IBurstCallback>& callback,
        const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
        const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
        configureExecutionBurst_cb cb) override;

    static bool isOperationSupported(const Operation& operation, const Model& model);
    virtual bool initialize(const Model& model);
protected:
    virtual void deinitialize();
    template <typename T_IExecutionCallback>
    Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
                                    const sp<T_IExecutionCallback>& callback);
    template <typename T_IExecutionCallback>
    void asyncExecute(const Request& request, MeasureTiming measure, time_point driverStart,
                      const sp<T_IExecutionCallback>& callback);

    std::shared_ptr<NgraphNetworkCreator> mNgc;
    std::string mTargetDevice;
    std::shared_ptr<NnapiModelInfo> mModelInfo;
    std::vector<RunTimeOperandInfo> mOperands;
    std::vector<RunTimePoolInfo> mPoolInfos;
    ExecuteNetwork* mEnginePtr;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
