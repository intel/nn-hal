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

#define LOG_TAG "VpuDriver"

#include "VpuDriver.h"
#include "VpuPreparedModel.h"
//#include "Utils.h"
#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

Return<ErrorStatus> VpuDriver::prepareModel(const Model& model,
                                             const sp<IPreparedModelCallback>& callback)
{
    ALOGI("VpuDriver::prepareModel");

    if (callback.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!VpuPreparedModel::validModel(model)) {
        ALOGI("model is not valid");
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<VpuPreparedModel> preparedModel = new VpuPreparedModel(model);
    if (!preparedModel->initialize()) {
        ALOGI("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::GENERAL_FAILURE, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    callback->notify(ErrorStatus::NONE, preparedModel);
    return ErrorStatus::NONE;
/*
    if (VLOG_IS_ON(DRIVER)) {
        VLOG(DRIVER) << "prepareModel";
        logModelToInfo(model);
    }
    if (callback.get() == nullptr) {
        LOG(ERROR) << "invalid callback passed to prepareModel";
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!VpuPreparedModel::validModel(model)) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<VpuPreparedModel> preparedModel = new VpuPreparedModel(model);
    if (!preparedModel->initialize()) {
       callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
       return ErrorStatus::INVALID_ARGUMENT;
    }
    callback->notify(ErrorStatus::NONE, preparedModel);
    return ErrorStatus::NONE;
*/
}

Return<DeviceStatus> VpuDriver::getStatus() {

    ALOGI("DeviceStatus::AVAILABLE");
    return DeviceStatus::AVAILABLE;
}

/*
int VpuDriver::run() {
    android::hardware::configureRpcThreadpool(4, true);
    if (registerAsService(mName) != android::OK) {
        LOG(ERROR) << "Could not register service";
        return 1;
    }
    android::hardware::joinRpcThreadpool();
    LOG(ERROR) << "Service exited!";
    return 1;
}
*/

Return<void> VpuDriver::getCapabilities(getCapabilities_cb cb) {
    //android::nn::initVLogMask();
    //VLOG(DRIVER) << "getCapabilities()";
    ALOGI("vpu driver getCapabilities()");
    Capabilities capabilities = {.float32Performance = {.execTime = 1.1f, .powerUsage = 1.1f},
                                 .quantized8Performance = {.execTime = 1.1f, .powerUsage = 1.1f}};
    cb(ErrorStatus::NONE, capabilities);
    return Void();
}

Return<void> VpuDriver::getSupportedOperations(const Model& model,
                                                     getSupportedOperations_cb cb) {
    //VLOG(DRIVER) << "getSupportedOperations()";
    ALOGI("vpu driver getSupportedOperations()");
    int count = model.operations.size();
    std::vector<bool> supported(count, false);

    if (!VpuPreparedModel::validModel(model)) {
        ALOGI("model is not valid");
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    for (int i = 0; i < count; i++) {
        const auto& operation = model.operations[i];
        supported[i] = VpuPreparedModel::isOperationSupported(operation, model);
    }

    cb(ErrorStatus::NONE, supported);
    return Void();

/*
    if (!VpuPreparedModel::validModel(model)) {
        const size_t count = model.operations.size();
        std::vector<bool> supported(count, true);
        cb(ErrorStatus::NONE, supported);
    } else {
        std::vector<bool> supported;
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
    }
    return Void();
*/

}


}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
