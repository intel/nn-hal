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

#define LOG_TAG "MklDnnDriver"

#include "MklDnnDriver.h"
#include "MklDnnPreparedModel.h"
#include <android/hidl/allocator/1.0/IAllocator.h>
#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace mkldnn_driver {

Return<ErrorStatus> MklDnnDriver::prepareModel(const Model& model,
                                               const sp<IPreparedModelCallback>& callback) {
    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!MklDnnPreparedModel::validModel(model)) {
        ALOGE("model is not valid");
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<MklDnnPreparedModel> preparedModel = new MklDnnPreparedModel(model);
    if (!preparedModel->initialize()) {
        ALOGE("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::GENERAL_FAILURE, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    callback->notify(ErrorStatus::NONE, preparedModel);
    return ErrorStatus::NONE;
}

Return<DeviceStatus> MklDnnDriver::getStatus() {
    return DeviceStatus::AVAILABLE;
}

Return<void> MklDnnDriver::getCapabilities(getCapabilities_cb cb) {
    Capabilities capabilities = {.float32Performance = {.execTime = 0.9f, .powerUsage = 1.1f},
                                 .quantized8Performance = {.execTime = 0.9f, .powerUsage = 1.1f}};
    cb(ErrorStatus::NONE, capabilities);
    return Void();
}

Return<void> MklDnnDriver::getSupportedOperations(const Model& model,
                                                  getSupportedOperations_cb cb) {
    int count = model.operations.size();
    std::vector<bool> supported(count, false);

    if (!MklDnnPreparedModel::validModel(model)) {
        ALOGE("model is not valid");
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    for (int i = 0; i < count; i++) {
        const auto& operation = model.operations[i];
        supported[i] = MklDnnPreparedModel::isOperationSupported(operation, model);
    }

    cb(ErrorStatus::NONE, supported);
    return Void();
}

}  // namespace mkldnn_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
