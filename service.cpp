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

#include <Utils.h>
#include <log/log.h>
#include <nnapi/IDevice.h>
#include <nnapi/hal/1.3/Device.h>
#include "Driver.h"
#define MAX_LENGTH (255)

#if __ANDROID__
#include <hidl/HidlTransportSupport.h>
#include <hidl/LegacySupport.h>

#undef LOG_TAG
#define LOG_TAG "neuralnetworks-hal-service"

using android::hardware::configureRpcThreadpool;
using android::hardware::joinRpcThreadpool;
using android::hardware::neuralnetworks::nnhal::Driver;

int main(int argc, char* argv[]) {
    if (argc > 2 && argv[2] != NULL && strnlen(argv[2], MAX_LENGTH) > 0) {
        if (strcmp(argv[1], "-D") != 0) return 0;
        const char* deviceType = argv[2];
        android::sp<Driver> device;

        if (strncmp(deviceType, "GNA", 3) == 0)
            device = new Driver(android::hardware::neuralnetworks::nnhal::IntelDeviceType::GNA);
        else if (strncmp(deviceType, "VPU", 3) == 0)
            device = new Driver(android::hardware::neuralnetworks::nnhal::IntelDeviceType::VPU);
        else if (strncmp(deviceType, "GPU", 3) == 0)
            device = new Driver(android::hardware::neuralnetworks::nnhal::IntelDeviceType::GPU);
        else
            device = new Driver(android::hardware::neuralnetworks::nnhal::IntelDeviceType::CPU);

        ALOGD("NN-HAL-1.3(%s) is ready.", deviceType);
        configureRpcThreadpool(4, true);
        android::status_t status = device->registerAsService(deviceType);
        LOG_ALWAYS_FATAL_IF(status != android::OK, "Error while registering as service for %s: %d",
                            deviceType, status);
        joinRpcThreadpool();
    }

    return 0;
}
#else
// This registers the SampleDriverFull into the DeviceManager.
namespace android {
namespace nn {

GeneralResult<SharedDevice> getService(const std::string& serviceName) {
    auto driver = new android::hardware::neuralnetworks::nnhal::Driver(
        android::hardware::neuralnetworks::nnhal::IntelDeviceType::CPU);
    return V1_3::utils::Device::create(serviceName, std::move(driver));
}

}  // namespace nn
}  // namespace android
#endif
