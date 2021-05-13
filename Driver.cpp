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

hidl_vec<Capabilities::OperandPerformance> nonExtensionOperandPerformance(PerformanceInfo perf) {
    using OpPerf = Capabilities::OperandPerformance;

    // Note: range presents enumerators in declaration order, not in numerical order.
    static constexpr ::android::hardware::hidl_enum_range<OperandType> kOperandTypeRange;

    hidl_vec<OpPerf> ret(kOperandTypeRange.end() - kOperandTypeRange.begin());

    std::transform(kOperandTypeRange.begin(), kOperandTypeRange.end(), ret.begin(),
                   [perf](OperandType type) {
                       return Capabilities::OperandPerformance{type, perf};
                   });
    std::sort(ret.begin(), ret.end(),
              [](const OpPerf& a, const OpPerf& b) { return a.type < b.type; });

    return ret;
}

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

// For HAL-1.1 version
Return<void> Driver::getCapabilities_1_1(getCapabilities_1_1_cb cb) {
    ALOGV("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getSupportedOperations_1_1(const V1_1_Model& model,
                                                getSupportedOperations_1_1_cb cb) {
    ALOGV("Entering %s", __func__);

    return Void();
}

Return<ErrorStatus> Driver::prepareModel_1_1(const V1_1_Model& model,
                                             ExecutionPreference preference,
                                             const sp<V1_0::IPreparedModelCallback>& callback) {
    ALOGV("Entering %s", __func__);

    return ErrorStatus::NONE;
}

// For HAL-1.2 version
Return<void> Driver::getCapabilities_1_2(getCapabilities_1_2_cb cb) {
    ALOGV("Entering %s", __func__);
    if (mDeviceName.compare("CPU") == 0) {
        ALOGI("CPU driver getCapabilities()");
        // Setting operandPerformance value to base value for all operand types
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.9f, .powerUsage = 0.9f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.9f, .powerUsage = 0.9f},
            .operandPerformance = nonExtensionOperandPerformance({0.9f, 0.9f})};

        ALOGI("CPU MKLDNN driver Capabilities .execTime = 0.9f, .powerUsage = 0.9f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("GPU") == 0) {
        ALOGI("GPU driver getCapabilities()");
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.95f, .powerUsage = 0.85f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.95f, .powerUsage = 0.85f},
            .operandPerformance = nonExtensionOperandPerformance({0.95f, 0.95f})};

        ALOGI("GPU clDNN driver Capabilities .execTime = 0.95f, .powerUsage = 0.85f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("GNA") == 0) {
        ALOGI("GPU driver getCapabilities()");
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.8f, .powerUsage = 0.8f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.8f, .powerUsage = 0.8f},
            .operandPerformance = nonExtensionOperandPerformance({0.8f, 0.8f})};

        ALOGI("GPU clDNN driver Capabilities .execTime = 0.95f, .powerUsage = 0.85f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("VPU") == 0) {
        ALOGI("Myriad driver getCapabilities()");
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 1.1f, .powerUsage = 1.1f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 1.1f, .powerUsage = 1.1f},
            .operandPerformance = nonExtensionOperandPerformance({1.1f, 1.1f})};

        ALOGI("Myriad driver Capabilities .execTime = 1.1f, .powerUsage = 1.1f");
        cb(ErrorStatus::NONE, capabilities);
    } else {
        Capabilities capabilities;
        cb(ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
    }
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<DeviceStatus> Driver::getStatus() {
    ALOGI("DeviceStatus::AVAILABLE");
    return DeviceStatus::AVAILABLE;
}

Return<void> Driver::getVersionString(getVersionString_cb cb) {
    ALOGV("Entering %s", __func__);
    cb(ErrorStatus::NONE, "intel_nn_hal");
    return Void();
}

Return<void> Driver::getType(getType_cb cb) {
    ALOGV("Entering %s", __func__);
    cb(ErrorStatus::NONE, V1_2::DeviceType::CPU);
    return Void();
}

Return<void> Driver::getSupportedExtensions(getSupportedExtensions_cb cb) {
    ALOGV("Entering %s", __func__);
    cb(ErrorStatus::NONE, {/* No extensions. */});
    return Void();
}

Return<void> Driver::getSupportedOperations_1_2(const Model& model,
                                                getSupportedOperations_1_2_cb cb) {
    ALOGV("Entering %s", __func__);

    int count = model.operations.size();
    std::vector<bool> supported(count, true);

    if (!validateModel(model)) {
        ALOGE("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    auto modelInfo = std::make_shared<NnapiModelInfo>(model);
    NgraphNetworkCreator ngraphCreatorInst(modelInfo, mDeviceName.c_str());
    ngraphCreatorInst.getSupportedOperations(supported);

    cb(ErrorStatus::NONE, supported);
    ALOGV("Exiting %s", __func__);
    return Void();
}

static sp<BasePreparedModel> ModelFactory(const char* name, const Model& model) {
    sp<BasePreparedModel> driverPreparedModel = NULL;

    if (strcmp(name, "CPU") == 0)
        driverPreparedModel = new CpuPreparedModel(model);
    else if (strcmp(name, "GNA") == 0)
        driverPreparedModel = new GnaPreparedModel(model);
    return driverPreparedModel;
}

Return<ErrorStatus> Driver::prepareModel_1_2(const Model& model, ExecutionPreference preference,
                                             const hidl_vec<hidl_handle>&,
                                             const hidl_vec<hidl_handle>&, const HidlToken&,
                                             const sp<V1_2::IPreparedModelCallback>& callback) {
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
    sp<BasePreparedModel> driverPreparedModel = ModelFactory(mDeviceName.c_str(), model);
    if (driverPreparedModel == NULL) {
        ALOGE("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!driverPreparedModel->initialize(model)) {
        ALOGE("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::NONE;
    }

    callback->notify(ErrorStatus::NONE, driverPreparedModel);
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

Return<void> Driver::getNumberOfCacheFilesNeeded(getNumberOfCacheFilesNeeded_cb cb) {
    ALOGV("Entering %s", __func__);
    // Set both numbers to be 0 for cache not supported.
    cb(ErrorStatus::NONE, /*numModelCache=*/0, /*numDataCache=*/0);
    return Void();
}

Return<ErrorStatus> Driver::prepareModelFromCache(
    const hidl_vec<hidl_handle>&, const hidl_vec<hidl_handle>&, const HidlToken&,
    const sp<V1_2::IPreparedModelCallback>& callback) {
    ALOGV("Entering %s", __func__);
    callback->notify_1_2(ErrorStatus::GENERAL_FAILURE, nullptr);
    return ErrorStatus::GENERAL_FAILURE;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android