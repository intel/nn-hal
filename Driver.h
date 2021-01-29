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

#ifndef ANDROID_ML_NN_DRIVER_H
#define ANDROID_ML_NN_DRIVER_H
#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModelCallback.h>
#include <android/hardware/neuralnetworks/1.0/types.h>
#include <android/hardware/neuralnetworks/1.1/IDevice.h>
#include <android/hardware/neuralnetworks/1.1/types.h>
#include <android/hardware/neuralnetworks/1.2/IDevice.h>
#include <android/hardware/neuralnetworks/1.2/IExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/IPreparedModelCallback.h>
#include <android/hardware/neuralnetworks/1.2/types.h>

#include <string>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// For HAL-1.0 version
using namespace ::android::hardware::neuralnetworks::V1_0;
using V1_0_Model = ::android::hardware::neuralnetworks::V1_0::Model;
using V1_0_Operation = ::android::hardware::neuralnetworks::V1_0::Operation;
using V1_0_Capabilities = ::android::hardware::neuralnetworks::V1_0::Capabilities;

// For HAL-1.1 version
using namespace ::android::hardware::neuralnetworks::V1_1;
using V1_1_Model = ::android::hardware::neuralnetworks::V1_1::Model;
using V1_1_Operation = ::android::hardware::neuralnetworks::V1_1::Operation;
using V1_1_Capabilities = ::android::hardware::neuralnetworks::V1_1::Capabilities;

// For HAL-1.2 version
using namespace ::android::hardware::neuralnetworks::V1_2;
using Model = ::android::hardware::neuralnetworks::V1_2::Model;
using Operand = ::android::hardware::neuralnetworks::V1_2::Operand;
using Operation = ::android::hardware::neuralnetworks::V1_2::Operation;
using OperationType = ::android::hardware::neuralnetworks::V1_2::OperationType;
using OperandType = ::android::hardware::neuralnetworks::V1_2::OperandType;
using Capabilities = ::android::hardware::neuralnetworks::V1_2::Capabilities;

using ::android::hardware::MQDescriptorSync;
using HidlToken = android::hardware::hidl_array<uint8_t, 32>;

// Base class used to create vpu drivers for the NN HAL.  This class
// provides some implementation of the more common functions.
//
// Since these drivers simulate hardware, they must run the computations
// on the CPU.  An actual driver would not do that.
class Driver : public ::android::hardware::neuralnetworks::V1_2::IDevice {
public:
    Driver() {}
    Driver(const char* name) : mDeviceName(name) {}

    ~Driver() override {}

    // For HAL-1.0 version
    Return<void> getCapabilities(getCapabilities_cb cb) override;
    Return<void> getSupportedOperations(const V1_0_Model& model,
                                        getSupportedOperations_cb cb) override;
    Return<ErrorStatus> prepareModel(const V1_0_Model& model,
                                     const sp<V1_0::IPreparedModelCallback>& callback) override;

    // For HAL-1.1 version
    Return<void> getCapabilities_1_1(getCapabilities_1_1_cb cb) override;
    Return<void> getSupportedOperations_1_1(const V1_1_Model& model,
                                            getSupportedOperations_1_1_cb cb) override;
    Return<ErrorStatus> prepareModel_1_1(const V1_1_Model& model, ExecutionPreference preference,
                                         const sp<V1_0::IPreparedModelCallback>& callback) override;

    // For HAL-1.2 version
    Return<void> getCapabilities_1_2(getCapabilities_1_2_cb cb) override;
    Return<void> getSupportedOperations_1_2(const Model& model,
                                            getSupportedOperations_1_2_cb cb) override;
    Return<ErrorStatus> prepareModel_1_2(const Model& model, ExecutionPreference preference,
                                         const hidl_vec<hidl_handle>& modelCache,
                                         const hidl_vec<hidl_handle>& dataCache,
                                         const HidlToken& token,
                                         const sp<V1_2::IPreparedModelCallback>& callback) override;
    Return<ErrorStatus> prepareModelFromCache(
        const hidl_vec<hidl_handle>& modelCache, const hidl_vec<hidl_handle>& dataCache,
        const HidlToken& token, const sp<V1_2::IPreparedModelCallback>& callback) override;
    Return<DeviceStatus> getStatus() override;
    Return<void> getVersionString(getVersionString_cb cb) override;
    Return<void> getType(getType_cb cb) override;
    Return<void> getSupportedExtensions(getSupportedExtensions_cb) override;
    Return<void> getNumberOfCacheFilesNeeded(getNumberOfCacheFilesNeeded_cb cb) override;

protected:
    std::string mDeviceName;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_DRIVER_H
