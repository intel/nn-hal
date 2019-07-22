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
#include <android-base/logging.h>
#include <thread>
#include "PreparedModel.h"
#include "ValidateHal.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;

static sp<PreparedModel> ModelFactory(const char* name, const Model& model) {
    sp<PreparedModel> preparedModel = NULL;

    if (strcmp(name, "CPU") == 0)
        preparedModel = new CpuPreparedModel(model);
    else if (strcmp(name, "VPU") == 0)
        preparedModel = new VpuPreparedModel(model);

    return preparedModel;
}

Return<ErrorStatus> Driver::prepareModel(const V10_Model& model,
                                         const sp<IPreparedModelCallback>& callback) {
    ALOGI("Entering %s", __func__);

    return ErrorStatus::NONE;
}

Return<ErrorStatus> Driver::prepareModel_1_1(const Model& model, ExecutionPreference preference,
                                             const sp<IPreparedModelCallback>& callback) {
    ALOGI("Entering %s", __func__);

    if (callback.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateModel(model) || !validateExecutionPreference(preference)) {
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
    ALOGI("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getCapabilities_1_1(getCapabilities_1_1_cb cb) {
    ALOGI("Entering %s", __func__);
    if (mName.compare("CPU") == 0) {
        ALOGI("Cpu driver getCapabilities()");
        Capabilities capabilities = {
            .float32Performance = {.execTime = 0.9f, .powerUsage = 0.9f},
            .quantized8Performance = {.execTime = 0.9f, .powerUsage = 0.9f},
            .relaxedFloat32toFloat16Performance = {.execTime = 0.9f, .powerUsage = 0.9f}};
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

Return<void> Driver::getSupportedOperations(const V10_Model& model, getSupportedOperations_cb cb) {
    ALOGI("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getSupportedOperations_1_1(const Model& model,
                                                getSupportedOperations_1_1_cb cb) {
    ALOGI("Entering %s", __func__);

    int count = model.operations.size();
    std::vector<bool> supported(count, true);

    if (!validateModel(model)) {
        ALOGI("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
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

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
