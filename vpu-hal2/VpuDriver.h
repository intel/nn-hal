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

#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <hardware/hardware.h>
#include <string>
namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace driver {

using ::android::hardware::neuralnetworks::V1_0::IDevice;
// Base class used to create vpu drivers for the NN HAL.  This class
// provides some implementation of the more common functions.
//
// Since these drivers simulate hardware, they must run the computations
// on the CPU.  An actual driver would not do that.
class Driver : public IDevice {
public:
    Driver() {}
    Driver(const char* name) : mName(name) {}

    ~Driver() override {}
    Return<ErrorStatus> prepareModel(const Model& model,
                                     const sp<IPreparedModelCallback>& callback) override;
    Return<DeviceStatus> getStatus() override;
    Return<void> getCapabilities(getCapabilities_cb _hidl_cb) override;
    Return<void> getSupportedOperations(const Model& model, getSupportedOperations_cb cb) override;
protected:
    std::string mName;
};


}  // namespace driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif // ANDROID_ML_NN_VPU_DRIVER_H
