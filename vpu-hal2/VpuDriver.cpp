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

#define LOG_TAG "Driver"

#include "VpuDriver.h"
#include "VpuPreparedModel.h"
#include <android-base/logging.h>
#include <thread>


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace driver {

static sp<PreparedModel> ModelFactory(const char* name, const Model& model) {
    sp<PreparedModel> preparedModel = NULL;
    if (strcmp(name, "CPU") ==0)
        preparedModel = new CpuPreparedModel(model);
    else if (strcmp(name, "VPU") ==0)
        preparedModel = new VpuPreparedModel(model);

    return preparedModel;
}

Return<ErrorStatus> Driver::prepareModel(const Model& model,
                                             const sp<IPreparedModelCallback>& callback)
{
    ALOGI("Driver::prepareModel");

    if (callback.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!PreparedModel::validModel(model)) {
        ALOGI("model is not valid");
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<PreparedModel> preparedModel = ModelFactory(mName.c_str(), model);
    if (preparedModel == NULL) {
        ALOGI("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!preparedModel->initialize()) {
        ALOGI("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::GENERAL_FAILURE, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    callback->notify(ErrorStatus::NONE, preparedModel);
    return ErrorStatus::NONE;
}

Return<DeviceStatus> Driver::getStatus() {
    ALOGI("DeviceStatus::AVAILABLE");
    return DeviceStatus::AVAILABLE;
}

Return<void> Driver::getCapabilities(getCapabilities_cb cb) {
    if (mName.compare("CPU") ==0) {
        ALOGI("Cpu driver getCapabilities()");
        Capabilities capabilities = {.float32Performance = {.execTime = 1.1f, .powerUsage = 1.1f}};
        cb(ErrorStatus::NONE, capabilities);
    } else {
        ALOGI("Myriad driver getCapabilities()");
        Capabilities capabilities = {.float32Performance = {.execTime = 1.1f, .powerUsage = 1.1f},
                                 .quantized8Performance = {.execTime = 1.1f, .powerUsage = 1.1f}};
        cb(ErrorStatus::NONE, capabilities);
    }
    return Void();
}

Return<void> Driver::getSupportedOperations(const Model& model,
                                                     getSupportedOperations_cb cb) {
    //VLOG(DRIVER) << "getSupportedOperations()";
    ALOGI("Driver getSupportedOperations()");
    int count = model.operations.size();
    std::vector<bool> supported(count, false);

    if (!PreparedModel::validModel(model)) {
        ALOGI("model is not valid");
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    for (int i = 0; i < count; i++) {
        const auto& operation = model.operations[i];
        supported[i] = PreparedModel::isOperationSupported(operation, model);
    }

    cb(ErrorStatus::NONE, supported);
    return Void();
}


}  // namespace driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
