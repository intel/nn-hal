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

#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.1/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include <NgraphNetworkCreator.hpp>
#include "Driver.h"
#include "IENetwork.h"
#include "ModelManager.h"
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

class BasePreparedModel : public IPreparedModel {
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

    virtual bool initialize();

    std::shared_ptr<NnapiModelInfo> getModelInfo() { return mModelInfo; }

    std::shared_ptr<NgraphNetworkCreator> getNgraphNwCreator() { return mNgc; }

    std::shared_ptr<IIENetwork> getPlugin() { return mPlugin; }

    std::shared_ptr<InferenceEngine::CNNNetwork> cnnNetworkPtr;

protected:
    virtual void deinitialize();

    IntelDeviceType mTargetDevice;
    std::shared_ptr<NnapiModelInfo> mModelInfo;
    std::shared_ptr<NgraphNetworkCreator> mNgc;
    std::shared_ptr<IIENetwork> mPlugin;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
