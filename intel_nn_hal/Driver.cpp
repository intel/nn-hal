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

#include "Driver.h"
#ifndef AT_RUNTIME
#include "PreparedModel.h"
#else
#include "Executor.h"
#endif
#include <android-base/logging.h>
#include <cutils/properties.h>
#include <thread>
#include "ValidateHal.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace driver {

using namespace android::nn;

#ifndef AT_RUNTIME
static sp<PreparedModel> ModelFactory(const char* name, const Model& model) {
    sp<PreparedModel> preparedModel = NULL;

    if (strcmp(name, "CPU") == 0)
        preparedModel = new CpuPreparedModel(model);
    else if (strcmp(name, "VPU") == 0)
        preparedModel = new VpuPreparedModel(model);

    return preparedModel;
}

#else
static sp<executor::PreparedModel> ModelFactory(const char* name, const Model& model) {
    sp<executor::PreparedModel> preparedModel = NULL;

    if (strcmp(name, "CPU") == 0)
        preparedModel = new executor::CpuPreparedModel(model);
    else if (strcmp(name, "VPU") == 0)
        preparedModel = new executor::VpuPreparedModel(model);

    return preparedModel;
}

#endif

Return<ErrorStatus> Driver::prepareModel(const Model& model,
                                         const sp<IPreparedModelCallback>& callback) {
    ALOGI("Driver::prepareModel");

    if (callback.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!validateModel(model)) {
        ALOGI("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
#ifndef AT_RUNTIME
    sp<PreparedModel> preparedModel = ModelFactory(mName.c_str(), model);
#else
    sp<executor::PreparedModel> preparedModel = ModelFactory(mName.c_str(), model);
#endif

    if (preparedModel == NULL) {
        ALOGI("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!preparedModel->initialize()) {
        ALOGI("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::GENERAL_FAILURE, nullptr);
        return ErrorStatus::NONE;
    }

    callback->notify(ErrorStatus::NONE, preparedModel);
    return ErrorStatus::NONE;
}

Return<DeviceStatus> Driver::getStatus() {
    ALOGI("DeviceStatus::AVAILABLE");
    return DeviceStatus::AVAILABLE;
}

Return<void> Driver::getCapabilities(getCapabilities_cb cb) {
    if (mName.compare("CPU") == 0) {
        ALOGI("Cpu driver getCapabilities()");
        Capabilities capabilities = {
            .float32Performance = {.execTime = 0.9f, .powerUsage = 0.9f},
            .quantized8Performance = {.execTime = 0.9f, .powerUsage = 0.9f}};

        ALOGI("CPU MKLDNN driver Capabilities .execTime = 0.9f, .powerUsage = 0.9f");
        cb(ErrorStatus::NONE, capabilities);
    } else { /* mName.compare("VPU") == 0 */
        ALOGI("Myriad driver getCapabilities()");

        Capabilities capabilities = {
            .float32Performance = {.execTime = 1.1f, .powerUsage = 1.1f},
            .quantized8Performance = {.execTime = 1.1f, .powerUsage = 1.1f}};

        ALOGI("Myriad driver Capabilities .execTime = 1.1f, .powerUsage = 1.1f");
        cb(ErrorStatus::NONE, capabilities);
    }
    return Void();
}

Return<void> Driver::getSupportedOperations(const Model& model, getSupportedOperations_cb cb) {
    ALOGI("Driver getSupportedOperations()");
    int count = model.operations.size();
    std::vector<bool> supported(count, false);
    char value[PROPERTY_VALUE_MAX];

    if (!validateModel(model)) {
        ALOGI("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    property_get("persist.vendor.nn_hal.disable", value, "0");
    if (atoi(value)) {
        ALOGI("NNHAL Disabled - CPU fallback\n");
        for (int i = 0; i < count; i++) {
            supported[i] = false;
        }
        cb(ErrorStatus::NONE, supported);
        return Void();
    }
#ifndef AT_RUNTIME
    for (int i = 0; i < count; i++) {
        const auto& operation = model.operations[i];
        supported[i] = PreparedModel::isOperationSupported(operation, model);
    }
#else
    for (int i = 0; i < count; i++) {
        const auto& operation = model.operations[i];
        supported[i] = executor::PreparedModel::isOperationSupported(operation, model);
    }
#endif

    cb(ErrorStatus::NONE, supported);
    return Void();
}

}  // namespace driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
