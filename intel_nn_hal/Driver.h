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

#ifndef ANDROID_ML_NN_VPU_DRIVER_H
#define ANDROID_ML_NN_VPU_DRIVER_H

#include <android/hardware/neuralnetworks/1.0/types.h>
#include <android/hardware/neuralnetworks/1.1/IDevice.h>
#include <android/hardware/neuralnetworks/1.1/types.h>
#include <hardware/hardware.h>
#include <string>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using Model = ::android::hardware::neuralnetworks::V1_1::Model;
using V10_Model = ::android::hardware::neuralnetworks::V1_0::Model;
using Operation = ::android::hardware::neuralnetworks::V1_1::Operation;
using V10_Operation = ::android::hardware::neuralnetworks::V1_0::Operation;
using OperationType = ::android::hardware::neuralnetworks::V1_1::OperationType;
using OperandType = ::android::hardware::neuralnetworks::V1_0::OperandType;
using Capabilities = ::android::hardware::neuralnetworks::V1_1::Capabilities;
using V10_Capabilities = ::android::hardware::neuralnetworks::V1_0::Capabilities;

using namespace ::android::hardware::neuralnetworks::V1_1;
using namespace ::android::hardware::neuralnetworks::V1_0;

// Base class used to create vpu drivers for the NN HAL.  This class
// provides some implementation of the more common functions.
//
// Since these drivers simulate hardware, they must run the computations
// on the CPU.  An actual driver would not do that.
class Driver : public ::android::hardware::neuralnetworks::V1_1::IDevice {
public:
    Driver() {}
    Driver(const char* name) : mName(name) {}

    ~Driver() override {}
    Return<void> getCapabilities(getCapabilities_cb cb) override;
    Return<void> getCapabilities_1_1(getCapabilities_1_1_cb cb) override;
    Return<void> getSupportedOperations(const V10_Model& model,
                                        getSupportedOperations_cb cb) override;
    Return<void> getSupportedOperations_1_1(const Model& model,
                                            getSupportedOperations_1_1_cb cb) override;
    Return<ErrorStatus> prepareModel(const V10_Model& model,
                                     const sp<IPreparedModelCallback>& callback) override;
    Return<ErrorStatus> prepareModel_1_1(const Model& model, ExecutionPreference preference,
                                         const sp<IPreparedModelCallback>& callback) override;
    Return<DeviceStatus> getStatus() override;

protected:
    std::string mName;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_VPU_DRIVER_H
