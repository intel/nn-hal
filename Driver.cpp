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

#include "Driver.h"

#include <android-base/logging.h>
#include <thread>
#include "BasePreparedModel.h"
#include "CpuPreparedModel.h"
#include "GnaPreparedModel.h"
#include "ModelManager.h"
#include "ValidateHal.h"

#define LOG_TAG "Driver"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;

// For HAL-1.0 version
Return<void> Driver::getCapabilities(getCapabilities_cb cb) {
    ALOGV("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getSupportedOperations(const V1_0_Model& model, getSupportedOperations_cb cb) {
    ALOGV("Entering %s", __func__);

    return Void();
}

Return<ErrorStatus> Driver::prepareModel(const V1_0_Model& model,
                                         const sp<V1_0::IPreparedModelCallback>& callback) {
    ALOGV("Entering %s", __func__);

    return ErrorStatus::NONE;
}

static sp<BasePreparedModel> ModelFactory(IntelDeviceType deviceType, const Model& model) {
    sp<BasePreparedModel> driverPreparedModel = NULL;

    if (deviceType == IntelDeviceType::CPU)
        driverPreparedModel = new CpuPreparedModel(model);
    else if (deviceType == IntelDeviceType::GNA)
        driverPreparedModel = new GnaPreparedModel(model);
    return driverPreparedModel;
}

// For HAL-1.1 version
Return<void> Driver::getCapabilities_1_1(getCapabilities_1_1_cb cb) {
    ALOGV("Entering %s", __func__);

    if (mDeviceType == IntelDeviceType::CPU) {
        ALOGI("CPU driver getCapabilities()");
        // Setting operandPerformance value to base value for all operand types
        Capabilities capabilities = {
            .float32Performance = {.execTime = 0.9f, .powerUsage = 0.9f},
            .quantized8Performance = {.execTime = 0.9f, .powerUsage = 0.9f},
            .relaxedFloat32toFloat16Performance = {.execTime = 0.9f, .powerUsage = 0.9f}};

        ALOGI("CPU MKLDNN driver Capabilities .execTime = 0.9f, .powerUsage = 0.9f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceType == IntelDeviceType::GPU) {
        ALOGI("GPU driver getCapabilities()");
        Capabilities capabilities = {
            .float32Performance = {.execTime = 1.1f, .powerUsage = 1.1f},
            .quantized8Performance = {.execTime = 1.1f, .powerUsage = 1.1f}};

        ALOGI("GPU clDNN driver Capabilities .execTime = 1.1f, .powerUsage = 1.1f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceType == IntelDeviceType::GNA) {
        ALOGI("GNA driver getCapabilities()");
        Capabilities capabilities = {
            .float32Performance = {.execTime = 1.2f, .powerUsage = 1.2f},
            .quantized8Performance = {.execTime = 1.2f, .powerUsage = 1.2f}};

        ALOGI("GNA driver Capabilities .execTime = 1.2f, .powerUsage = 1.2f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceType == IntelDeviceType::VPU) {
        ALOGI("Myriad driver getCapabilities()");
        Capabilities capabilities = {
            .float32Performance = {.execTime = 1.3f, .powerUsage = 1.3f},
            .quantized8Performance = {.execTime = 1.3f, .powerUsage = 1.3f}};

        ALOGI("Myriad driver Capabilities .execTime = 1.3f, .powerUsage = 1.3f");
        cb(ErrorStatus::NONE, capabilities);
    } else {
        Capabilities capabilities;
        cb(ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
    }
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<void> Driver::getSupportedOperations_1_1(const Model& model,
                                                getSupportedOperations_1_1_cb cb) {
    ALOGV("Entering %s", __func__);

    int count = model.operations.size();
    std::vector<bool> supported(count, true);

    if (!validateModel(model)) {
        ALOGE("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    auto modelInfo = std::make_shared<NnapiModelInfo>(model);
    NgraphNetworkCreator ngraphCreatorInst(modelInfo, mDeviceType);
    ngraphCreatorInst.getSupportedOperations(supported);

    cb(ErrorStatus::NONE, supported);
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<ErrorStatus> Driver::prepareModel_1_1(const Model& model, ExecutionPreference preference,
                                             const sp<V1_0::IPreparedModelCallback>& callback) {
    ALOGV("Entering %s", __func__);

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateModel(model) || !validateExecutionPreference(preference)) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<BasePreparedModel> driverPreparedModel = ModelFactory(mDeviceType, model);
    if (driverPreparedModel == NULL) {
        ALOGE("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    for (auto opn : model.operations) dumpOperation(opn);

    if (!driverPreparedModel->initialize()) {
        ALOGE("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::NONE;
    }

    callback->notify(ErrorStatus::NONE, driverPreparedModel);
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

Return<DeviceStatus> Driver::getStatus() {
    ALOGI("DeviceStatus::AVAILABLE");
    return DeviceStatus::AVAILABLE;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
