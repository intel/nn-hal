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

#include <android-base/logging.h>
#include <thread>

#include "Driver.h"
#include "PreparedModel.h"
#include "ValidateHal.h"
#include "GnaPreparedModel.h"

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

static sp<PreparedModel> ModelFactory(const char* name, const Model& model) {
    sp<PreparedModel> preparedModel = NULL;

    if (strcmp(name, "CPU") == 0)
        preparedModel = new CpuPreparedModel(model);
    else if (strcmp(name, "VPU") == 0)
        preparedModel = new VpuPreparedModel(model);
    else if (strcmp(name, "GPU") == 0)
        preparedModel = new GpuPreparedModel(model);
    else if (strcmp(name, "GNA") == 0)
        preparedModel = new GnaPreparedModel(model);

    return preparedModel;
}

static sp<PreparedModel> ModelFactory(const char* name) {
    sp<PreparedModel> preparedModel = NULL;
    
    if (strcmp(name, "GNA") == 0)
        preparedModel = new GnaPreparedModel();

    return preparedModel;
}

Return<V1_0_ErrorStatus> Driver::prepareModel(const V1_0_Model& model,
                                         const sp<V1_0::IPreparedModelCallback>& callback) {
    VLOG("Entering %s", __func__);

    return V1_0_ErrorStatus::NONE;
}

Return<V1_0_ErrorStatus> Driver::prepareModel_1_1(const V1_1_Model& model,
                                             ExecutionPreference preference,
                                             const sp<V1_0::IPreparedModelCallback>& callback) {
    VLOG("Entering %s", __func__);

    return V1_0_ErrorStatus::NONE;
}
Return<V1_0_ErrorStatus> Driver::prepareModel_1_2(const V1_2_Model& model,
                                             ExecutionPreference preference,
                                             const hidl_vec<hidl_handle>&,
                                             const hidl_vec<hidl_handle>&, const HidlToken&,
                                             const sp<V1_2::IPreparedModelCallback>& callback) {
    VLOG("Entering %s", __func__);

    return V1_0_ErrorStatus::NONE;
}

Return<ErrorStatus> Driver::prepareModel_1_3(const Model& model, ExecutionPreference preference,
					     Priority priority,
					     const OptionalTimePoint& deadline,
                                             const hidl_vec<hidl_handle>& modelCache,
                                             const hidl_vec<hidl_handle>& dataCache, const HidlToken& token,
                                             const sp<V1_3::IPreparedModelCallback>& callback) {
    VLOG("Entering %s", __func__);

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateModel(model) || !validateExecutionPreference(preference)) {
        callback->notify(V1_0_ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<PreparedModel> preparedModel = ModelFactory(mName.c_str(), model);
    if (preparedModel == NULL) {
        ALOGE("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!preparedModel->initialize(modelCache, token)) {
        ALOGE("failed to initialize preparedmodel");
        callback->notify(V1_0_ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::NONE;
    }

    callback->notify(V1_0_ErrorStatus::NONE, preparedModel);
    return ErrorStatus::NONE;
}

Return<DeviceStatus> Driver::getStatus() {
    VLOG("DeviceStatus::AVAILABLE");
    return DeviceStatus::AVAILABLE;
}

Return<void> Driver::getVersionString(getVersionString_cb cb) {
    VLOG("Entering %s", __func__);
    cb(V1_0_ErrorStatus::NONE, "intel_nn_hal");
    return Void();
}

Return<void> Driver::getType(getType_cb cb) {
    VLOG("Entering %s", __func__);
    cb(V1_0_ErrorStatus::NONE, V1_2::DeviceType::CPU);
    return Void();
}

Return<void> Driver::getSupportedExtensions(getSupportedExtensions_cb cb) {
    VLOG("Entering %s", __func__);
    cb(V1_0_ErrorStatus::NONE, {/* No extensions. */});
    return Void();
}

Return<void> Driver::getNumberOfCacheFilesNeeded(getNumberOfCacheFilesNeeded_cb cb) {
    VLOG("Entering %s", __func__);
    // Set both numbers to be 0 for cache not supported.
    cb(V1_0_ErrorStatus::NONE, 3, 0);
    return Void();
}

Return<ErrorStatus> Driver::prepareModelFromCache_1_3(
    const OptionalTimePoint&,
    const hidl_vec<hidl_handle>& modelCache,
    const hidl_vec<hidl_handle>&,
    const HidlToken& token,
    const sp<V1_3::IPreparedModelCallback>& callback) {
    VLOG("Entering %s", __func__);
	
    if (callback.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<PreparedModel> preparedModel = ModelFactory(mName.c_str());
    if (preparedModel == NULL) {
        ALOGI("failed to create preparedmodel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    bool success = preparedModel->initializeFromCache(modelCache, token);
    if (success) {
        callback->notify(V1_0_ErrorStatus::NONE, preparedModel);
        return ErrorStatus::NONE;
    } else {
        callback->notify(V1_0_ErrorStatus::GENERAL_FAILURE, preparedModel);
        return ErrorStatus::GENERAL_FAILURE;
    }
}

Return<V1_0_ErrorStatus> Driver::prepareModelFromCache(
        const hidl_vec<hidl_handle>& modelCache,
		const hidl_vec<hidl_handle>& dataCache,
        const HidlToken& token,
		const sp<V1_2::IPreparedModelCallback>& callback) {
        VLOG("Entering %s", __func__);
        return V1_0_ErrorStatus::GENERAL_FAILURE;
}

Return<void> Driver::getCapabilities(getCapabilities_cb cb) {
    VLOG("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getCapabilities_1_1(getCapabilities_1_1_cb cb) {
    VLOG("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getCapabilities_1_2(getCapabilities_1_2_cb cb) {
    VLOG("Entering %s", __func__);
    return Void();
}

Return<void> Driver::allocate(const BufferDesc& desc,
		          const hidl_vec<sp<V1_3::IPreparedModel>>& preparedModels,
		          const hidl_vec<BufferRole>& inputRoles,
			  const hidl_vec<BufferRole>& outputRoles,
	                    allocate_cb cb) {
    VLOG("Entering %s", __func__);
    return Void();

}

Return<void> Driver::getCapabilities_1_3(getCapabilities_1_3_cb cb) {
    VLOG("Entering %s", __func__);
    if (mName.compare("CPU") == 0) {
        VLOG("CPU driver getCapabilities()");
        // Setting operandPerformance value to base value for all operand types
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.9f, .powerUsage = 0.9f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.9f, .powerUsage = 0.9f},
            .operandPerformance = nonExtensionOperandPerformance({0.9f, 0.9f})};

        VLOG("CPU MKLDNN driver Capabilities .execTime = 0.9f, .powerUsage = 0.9f");
        cb(ErrorStatus::NONE, capabilities);
    } else if (mName.compare("GPU") == 0) {
        VLOG("GPU driver getCapabilities()");
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.95f, .powerUsage = 0.85f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.95f, .powerUsage = 0.85f},
            .operandPerformance = nonExtensionOperandPerformance({0.95f, 0.95f})};

        VLOG("GPU clDNN driver Capabilities .execTime = 0.95f, .powerUsage = 0.85f");
        cb(ErrorStatus::NONE, capabilities);
    }
    else if (mName.compare("GNA") == 0){
       VLOG("GNA driver getCapabilities()");
       Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.8f, .powerUsage = 0.8f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.8f, .powerUsage = 0.8f},
            .operandPerformance = nonExtensionOperandPerformance({0.8f, 0.8f})};

       VLOG("GNA driver Capabilities .execTime = 0.8f, .powerUsage = 0.8f");
       cb(ErrorStatus::NONE, capabilities);
    } else {
        // mName.compare("VPU") == 0
        VLOG("Myriad driver getCapabilities()");
        Capabilities capabilities = {
            .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 1.1f, .powerUsage = 1.1f},
            .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 1.1f, .powerUsage = 1.1f},
            .operandPerformance = nonExtensionOperandPerformance({1.1f, 1.1f})};

        VLOG("Myriad driver Capabilities .execTime = 1.1f, .powerUsage = 1.1f");
        cb(ErrorStatus::NONE, capabilities);
    }
    return Void();
}

Return<void> Driver::getSupportedOperations(const V1_0_Model& model, getSupportedOperations_cb cb) {
    VLOG("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getSupportedOperations_1_1(const V1_1_Model& model,
                                                getSupportedOperations_1_1_cb cb) {
    VLOG("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getSupportedOperations_1_2(const V1_2_Model& model,
                                                getSupportedOperations_1_2_cb cb) {
    VLOG("Entering %s", __func__);

    return Void();
}

Return<void> Driver::getSupportedOperations_1_3(const Model& model,
                                                getSupportedOperations_1_3_cb cb) {
    VLOG("Entering %s", __func__);

    int count = model.main.operations.size();
    std::vector<bool> supported(count, true);

    if (!validateModel(model)) {
        VLOG("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    for (int i = 0; i < count; i++) {
        const auto& operation = model.main.operations[i];
        supported[i] = PreparedModel::isOperationSupported(operation, model, mName);
    }
    cb(ErrorStatus::NONE, supported);
    return Void();
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
