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
#include <string>

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

hidl_vec<V1_2::Capabilities::OperandPerformance> nonExtensionOperandPerformanceV1_2(
    V1_0::PerformanceInfo perf) {
    using OpPerf = V1_2::Capabilities::OperandPerformance;

    // Note: range presents enumerators in declaration order, not in numerical order.
    static constexpr ::android::hardware::hidl_enum_range<V1_2::OperandType> kOperandTypeRange;

    hidl_vec<OpPerf> ret(kOperandTypeRange.end() - kOperandTypeRange.begin());

    std::transform(kOperandTypeRange.begin(), kOperandTypeRange.end(), ret.begin(),
                   [perf](V1_2::OperandType type) {
                       return V1_2::Capabilities::OperandPerformance{(type), perf};
                   });
    std::sort(ret.begin(), ret.end(),
              [](const OpPerf& a, const OpPerf& b) { return a.type < b.type; });

    return ret;
}

hidl_vec<Capabilities::OperandPerformance> nonExtensionOperandPerformance(
    V1_0::PerformanceInfo perf) {
    using OpPerf = Capabilities::OperandPerformance;

    // Note: range presents enumerators in declaration order, not in numerical order.
    static constexpr hidl_enum_range<OperandType> kOperandTypeRange;

    std::vector<OpPerf> ret;
    ret.reserve(kOperandTypeRange.end() - kOperandTypeRange.begin());
    for (OperandType type : kOperandTypeRange) {
        if (static_cast<OperandType>(type) != OperandType::SUBGRAPH) {
            ret.push_back(OpPerf{type, perf});
        }
    }
    std::sort(ret.begin(), ret.end(),
              [](const OpPerf& a, const OpPerf& b) { return a.type < b.type; });
    hidl_vec<OpPerf> ret1;
    ret1 = ret;

    return ret1;
}

static sp<BasePreparedModel> ModelFactory(const char* name, const Model& model,
                                          const Driver* driver) {
    sp<BasePreparedModel> driverPreparedModel = NULL;

    if (strcmp(name, "CPU") == 0)
        driverPreparedModel = new CpuPreparedModel(model, driver);
    else if (strcmp(name, "GNA") == 0)
        driverPreparedModel = new GnaPreparedModel(model);
    return driverPreparedModel;
}

// For HAL-1.0 version
Return<void> Driver::getCapabilities(getCapabilities_cb cb) {
    ALOGV("Entering %s", __func__);
    return getCapabilities_1_3(
        [&](V1_3::ErrorStatus error, const V1_3::Capabilities& capabilities) {
            // TODO(dgross): Do we need to check compliantWithV1_0(capabilities)?
            cb(convertToV1_0(error), convertToV1_0(capabilities));
        });
}

Return<void> Driver::getSupportedOperations(const V1_0_Model& model, getSupportedOperations_cb cb) {
    ALOGV("Entering %s", __func__);
    if (!validateModel(model)) {
        ALOGE("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(V1_0::ErrorStatus::INVALID_ARGUMENT, {});
        return Void();
    }
    return getSupportedOperations_1_3(
        convertToV1_3(model), [&](V1_3::ErrorStatus status, const hidl_vec<bool>& supported) {
            cb(convertToV1_0(status), supported);
        });
}

Return<ErrorStatus> Driver::prepareModel(const V1_0_Model& model,
                                         const sp<V1_0::IPreparedModelCallback>& callback) {
    ALOGV("Entering %s", __func__);
    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateModel(model)) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<BasePreparedModel> driverPreparedModel =
        ModelFactory(mDeviceName.c_str(), convertToV1_3(model), this);
    if (driverPreparedModel == NULL) {
        ALOGE("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    for (auto opn : model.operations) dumpOperation(opn);

    if (!driverPreparedModel->initialize(convertToV1_3(model))) {
        ALOGE("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::NONE;
    }

    callback->notify(ErrorStatus::NONE, driverPreparedModel);
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

// For HAL-1.1 version
Return<void> Driver::getCapabilities_1_1(getCapabilities_1_1_cb cb) {
    ALOGV("Entering %s", __func__);
    return getCapabilities_1_3(
        [&](V1_3::ErrorStatus error, const V1_3::Capabilities& capabilities) {
            // TODO(dgross): Do we need to check compliantWithV1_1(capabilities)?
            cb(convertToV1_0(error), convertToV1_1(capabilities));
        });
}

Return<void> Driver::getSupportedOperations_1_1(const V1_1_Model& model,
                                                getSupportedOperations_1_1_cb cb) {
    ALOGV("Entering %s", __func__);
    if (!validateModel(model)) {
        ALOGE("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(V1_0::ErrorStatus::INVALID_ARGUMENT, {});
        return Void();
    }
    return getSupportedOperations_1_3(
        convertToV1_3(model), [&](V1_3::ErrorStatus status, const hidl_vec<bool>& supported) {
            cb(convertToV1_0(status), supported);
        });
}

Return<ErrorStatus> Driver::prepareModel_1_1(const V1_1_Model& model,
                                             ExecutionPreference preference,
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
    sp<BasePreparedModel> driverPreparedModel =
        ModelFactory(mDeviceName.c_str(), convertToV1_3(model), this);
    if (driverPreparedModel == NULL) {
        ALOGE("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    for (auto opn : model.operations) dumpOperation(opn);

    if (!driverPreparedModel->initialize(convertToV1_3(model))) {
        ALOGE("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::NONE;
    }

    callback->notify(ErrorStatus::NONE, driverPreparedModel);
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

// For HAL-1.2 version
Return<void> Driver::getCapabilities_1_2(getCapabilities_1_2_cb cb) {
    ALOGV("Entering %s", __func__);
    if (mDeviceName.compare("CPU") == 0) {
        ALOGI("CPU driver getCapabilities()");
        // Setting operandPerformance value to base value for all operand types
        V1_2::Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.9f, .powerUsage = 0.9f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.9f, .powerUsage = 0.9f},
            .operandPerformance = nonExtensionOperandPerformanceV1_2({0.9f, 0.9f})};

        ALOGI("CPU MKLDNN driver Capabilities .execTime = 0.9f, .powerUsage = 0.9f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("GPU") == 0) {
        ALOGI("GPU driver getCapabilities()");
        V1_2::Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.95f, .powerUsage = 0.85f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.95f, .powerUsage = 0.85f},
            .operandPerformance = nonExtensionOperandPerformanceV1_2({0.95f, 0.95f})};

        ALOGI("GPU clDNN driver Capabilities .execTime = 0.95f, .powerUsage = 0.85f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("GNA") == 0) {
        ALOGI("GPU driver getCapabilities()");
        V1_2::Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.8f, .powerUsage = 0.8f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.8f, .powerUsage = 0.8f},
            .operandPerformance = nonExtensionOperandPerformanceV1_2({0.8f, 0.8f})};

        ALOGI("GPU clDNN driver Capabilities .execTime = 0.95f, .powerUsage = 0.85f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("VPU") == 0) {
        ALOGI("Myriad driver getCapabilities()");
        V1_2::Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 1.1f, .powerUsage = 1.1f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 1.1f, .powerUsage = 1.1f},
            .operandPerformance = nonExtensionOperandPerformanceV1_2({1.1f, 1.1f})};

        ALOGI("Myriad driver Capabilities .execTime = 1.1f, .powerUsage = 1.1f");
        cb(ErrorStatus::NONE, capabilities);
    } else {
        V1_2::Capabilities capabilities;
        cb(ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
    }
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<void> Driver::getSupportedOperations_1_2(const V1_2_Model& model,
                                                getSupportedOperations_1_2_cb cb) {
    ALOGV("Entering %s", __func__);

    int count = model.operations.size();
    std::vector<bool> supported(count, true);

    if (!validateModel(model)) {
        ALOGE("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    auto modelInfo = std::make_shared<NnapiModelInfo>(convertToV1_3(model));
    NgraphNetworkCreator ngraphCreatorInst(modelInfo, mDeviceName.c_str());
    ngraphCreatorInst.getSupportedOperations(supported);

    cb(ErrorStatus::NONE, supported);
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<ErrorStatus> Driver::prepareModel_1_2(const V1_2_Model& model,
                                             ExecutionPreference preference,
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
    sp<BasePreparedModel> driverPreparedModel =
        ModelFactory(mDeviceName.c_str(), convertToV1_3(model), this);
    if (driverPreparedModel == NULL) {
        ALOGE("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    for (auto opn : model.operations) dumpOperation(opn);

    if (!driverPreparedModel->initialize(convertToV1_3(model))) {
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

// For HAL-1.3 version
Return<void> Driver::getCapabilities_1_3(getCapabilities_1_3_cb cb) {
    ALOGV("Entering %s", __func__);
    if (mDeviceName.compare("CPU") == 0) {
        ALOGI("CPU driver getCapabilities()");
        // Setting operandPerformance value to base value for all operand types
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.9f, .powerUsage = 0.9f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.9f, .powerUsage = 0.9f},
            .operandPerformance = nonExtensionOperandPerformance({0.9f, 0.9f}),
            .ifPerformance = {.execTime = 0.9f, .powerUsage = 0.9f},
            .whilePerformance = {.execTime = 0.9f, .powerUsage = 0.9f}};

        ALOGI("CPU MKLDNN driver Capabilities .execTime = 0.9f, .powerUsage = 0.9f");
        cb(V1_3::ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("GPU") == 0) {
        ALOGI("GPU driver getCapabilities()");
        V1_3::Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.95f, .powerUsage = 0.85f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.95f, .powerUsage = 0.85f},
            .operandPerformance = nonExtensionOperandPerformance({0.95f, 0.95f}),
            .ifPerformance = {.execTime = 0.95f, .powerUsage = 0.85f},
            .whilePerformance = {.execTime = 0.95f, .powerUsage = 0.85f}};

        ALOGI("GPU clDNN driver Capabilities .execTime = 0.95f, .powerUsage = 0.85f");
        cb(V1_3::ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("GNA") == 0) {
        ALOGI("GPU driver getCapabilities()");
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.8f, .powerUsage = 0.8f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.8f, .powerUsage = 0.8f},
            .operandPerformance = nonExtensionOperandPerformance({0.8f, 0.8f}),
            .ifPerformance = {.execTime = 0.8f, .powerUsage = 0.8f},
            .whilePerformance = {.execTime = 0.8f, .powerUsage = 0.8f}};

        ALOGI("GPU clDNN driver Capabilities .execTime = 0.95f, .powerUsage = 0.85f");
        cb(V1_3::ErrorStatus::NONE, capabilities);
    } else if (mDeviceName.compare("VPU") == 0) {
        ALOGI("Myriad driver getCapabilities()");
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 1.1f, .powerUsage = 1.1f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 1.1f, .powerUsage = 1.1f},
            .operandPerformance = nonExtensionOperandPerformance({1.1f, 1.1f}),
            .ifPerformance = {.execTime = 1.1f, .powerUsage = 1.1f},
            .whilePerformance = {.execTime = 1.1f, .powerUsage = 1.1f}};

        ALOGI("Myriad driver Capabilities .execTime = 1.1f, .powerUsage = 1.1f");
        cb(V1_3::ErrorStatus::NONE, capabilities);
    } else {
        Capabilities capabilities;
        cb(V1_3::ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
    }
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<void> Driver::getSupportedOperations_1_3(const Model& model,
                                                getSupportedOperations_1_3_cb cb) {
    ALOGV("Entering %s", __func__);

    int count = model.main.operations.size();
    std::vector<bool> supported(count, true);

    if (!validateModel(model)) {
        ALOGE("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(V1_3::ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    auto modelInfo = std::make_shared<NnapiModelInfo>(model);
    NgraphNetworkCreator ngraphCreatorInst(modelInfo, mDeviceName.c_str());
    ngraphCreatorInst.getSupportedOperations(supported);

    cb(V1_3::ErrorStatus::NONE, supported);
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<V1_3::ErrorStatus> Driver::prepareModel_1_3(
    const Model& model, V1_1::ExecutionPreference preference, V1_3::Priority priority,
    const V1_3::OptionalTimePoint&,
    const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
    const android::hardware::hidl_vec<android::hardware::hidl_handle>&, const HidlToken&,
    const android::sp<V1_3::IPreparedModelCallback>& cb) {
    ALOGV("Entering %s", __func__);

    if (cb.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!validateModel(model) || !validateExecutionPreference(preference) ||
        !validatePriority(priority)) {
        cb->notify_1_3(V1_3::ErrorStatus::INVALID_ARGUMENT, nullptr);
        ALOGI("validatemodel failed");
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

    sp<BasePreparedModel> driverPreparedModel = ModelFactory(mDeviceName.c_str(), model, this);
    if (!driverPreparedModel->initialize(model)) {
        ALOGI("Failed to initialize prepared model");
        cb->notify_1_3(convertToV1_3(ErrorStatus::INVALID_ARGUMENT), nullptr);
        return V1_3::ErrorStatus::NONE;
    }
    cb->notify_1_3((V1_3::ErrorStatus::NONE), driverPreparedModel);
    ALOGV("Exiting %s", __func__);

    return convertToV1_3(ErrorStatus::NONE);
}

Return<V1_3::ErrorStatus> Driver::prepareModelFromCache_1_3(
    const V1_3::OptionalTimePoint& timing,
    const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
    const android::hardware::hidl_vec<android::hardware::hidl_handle>&, const HidlToken&,
    const sp<V1_3::IPreparedModelCallback>& callback) {
    ALOGV("V1_3::Driver::prepareModelFromCache_1_3()");

    if (callback.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

    const auto ret = callback->notify_1_3(V1_3::ErrorStatus::GENERAL_FAILURE, nullptr);
    if (!ret.isOk()) {
        LOG(ERROR) << "Error when calling IPreparedModelCallback::notify_1_3: "
                   << ret.description();
    }
    ALOGV("Exiting %s", __func__);
    return V1_3::ErrorStatus::GENERAL_FAILURE;
}

Return<void> Driver::allocate(const V1_3::BufferDesc& desc,
                              const hidl_vec<sp<V1_3::IPreparedModel>>& preparedModels,
                              const hidl_vec<V1_3::BufferRole>& inputRoles,
                              const hidl_vec<V1_3::BufferRole>& outputRoles,
                              V1_3::IDevice::allocate_cb cb) {
    ALOGV("Entering %s", __func__);
    cb(V1_3::ErrorStatus::GENERAL_FAILURE, nullptr, 0);
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

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android