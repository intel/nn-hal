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
#include <android/hardware/neuralnetworks/1.3/IExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.3/IFencedExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.3/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.3/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include <NgraphNetworkCreator.hpp>
#include "Driver.h"
#include "IENetwork.h"
#include "ModelManager.h"
#include "DetectionClient.h"
#include "utils.h"

#if __ANDROID__
#include <hardware/hardware.h>
#endif

using ::android::hardware::MQDescriptorSync;
using ::android::hidl::memory::V1_0::IMemory;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

template <class T>
using vec = std::vector<T>;
typedef uint8_t* memory;
extern bool mRemoteCheck;
extern std::shared_ptr<DetectionClient> mDetectionClient;
class BasePreparedModel : public V1_3::IPreparedModel {
public:
    BasePreparedModel(const Model& model) : mTargetDevice(IntelDeviceType::CPU) {
        mModelInfo = std::make_shared<NnapiModelInfo>(model);
    }
    BasePreparedModel(const IntelDeviceType device, const Model& model) : mTargetDevice(device) {
        mModelInfo = std::make_shared<NnapiModelInfo>(model);
    }

    virtual ~BasePreparedModel() { deinitialize(); }

    Return<ErrorStatus> execute(const Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override;
    Return<ErrorStatus> execute_1_2(const Request& request, MeasureTiming measure,
                                    const sp<V1_2::IExecutionCallback>& callback) override;
    Return<V1_3::ErrorStatus> execute_1_3(const V1_3::Request& request, V1_2::MeasureTiming measure,
                                          const V1_3::OptionalTimePoint& deadline,
                                          const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                          const sp<V1_3::IExecutionCallback>& callback) override;
    Return<void> executeSynchronously(const Request& request, V1_2::MeasureTiming measure,
                                      executeSynchronously_cb cb) override;
    Return<void> executeSynchronously_1_3(const V1_3::Request& request, V1_2::MeasureTiming measure,
                                          const V1_3::OptionalTimePoint& deadline,
                                          const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                          executeSynchronously_1_3_cb cb) override;
    Return<void> configureExecutionBurst(
        const sp<V1_2::IBurstCallback>& callback,
        const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
        const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
        configureExecutionBurst_cb cb) override;
    Return<void> executeFenced(const V1_3::Request& request, const hidl_vec<hidl_handle>& waitFor,
                               V1_2::MeasureTiming measure, const V1_3::OptionalTimePoint& deadline,
                               const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                               const V1_3::OptionalTimeoutDuration& duration,
                               executeFenced_cb cb) override;

    virtual bool initialize();
    virtual bool checkRemoteConnection();
    virtual bool loadRemoteModel();

    std::shared_ptr<NnapiModelInfo> getModelInfo() { return mModelInfo; }

    std::shared_ptr<NgraphNetworkCreator> getNgraphNwCreator() { return mNgraphNetCreator; }

    std::shared_ptr<IIENetwork> getPlugin() { return mPlugin; }

    std::shared_ptr<InferenceEngine::CNNNetwork> cnnNetworkPtr;

protected:
    virtual void deinitialize();

    IntelDeviceType mTargetDevice;
    std::shared_ptr<NnapiModelInfo> mModelInfo;
    std::shared_ptr<NgraphNetworkCreator> mNgraphNetCreator;
    std::shared_ptr<IIENetwork> mPlugin;
};

class BaseFencedExecutionCallback : public V1_3::IFencedExecutionCallback {
public:
    BaseFencedExecutionCallback(Timing timingSinceLaunch, Timing timingAfterFence,
                                V1_3::ErrorStatus error)
        : kTimingSinceLaunch(timingSinceLaunch),
          kTimingAfterFence(timingAfterFence),
          kErrorStatus(error) {}
    Return<void> getExecutionInfo(getExecutionInfo_cb callback) override {
        callback(kErrorStatus, kTimingSinceLaunch, kTimingAfterFence);
        return Void();
    }

private:
    const Timing kTimingSinceLaunch;
    const Timing kTimingAfterFence;
    const V1_3::ErrorStatus kErrorStatus;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
