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
#include "BasePreparedModel.h"

#include <android-base/logging.h>
#include <android/log.h>
#include <cutils/properties.h>
#include <log/log.h>
#include <thread>
#include "ExecutionBurstServer.h"
#include "ValidateHal.h"

#define DISABLE_ALL_QUANT
#define LOG_TAG "BasePreparedModel"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;

static const Timing kNoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

void BasePreparedModel::deinitialize() {
    ALOGV("Entering %s", __func__);
    mModelInfo->unmapRuntimeMemPools();

    ALOGV("Exiting %s", __func__);
}

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
    // TODO: Check buffer is at least as long as size of data.
    T* data = reinterpret_cast<T*>(info.buffer);
    return data[0];
}

bool BasePreparedModel::initialize(const Model& model) {
    ALOGV("Entering %s", __func__);
    return true;
}

static Return<void> notify(const sp<V1_0::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>&, Timing) {
    return callback->notify(status);
}

static Return<void> notify(const sp<V1_2::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>& outputShapes, Timing timing) {
    return callback->notify_1_2(status, outputShapes, timing);
}

template <typename T_IExecutionCallback>
Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
                                BasePreparedModel* preparedModel,
                                const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, preparedModel->getModelInfo()->getModel())) {
        notify(callback, ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([preparedModel, request, measure, driverStart, callback] {
        asyncExecute(request, measure, preparedModel, driverStart, callback);
    }).detach();
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

static void floatToUint8(const float* src, uint8_t* dst, size_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        dst[i] = static_cast<uint8_t>(src[i]);
        ALOGV("%s input: %f output: %d ", __func__, src[i], dst[i]);
    }
}

template <typename T_IExecutionCallback>
void asyncExecute(const Request& request, MeasureTiming measure, BasePreparedModel* preparedModel,
                  time_point driverStart, const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();
    time_point driverEnd, deviceStart, deviceEnd;
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto inIndex = modelInfo->getModelInputIndex(i);
        auto srcBlob = modelInfo->getBlobFromMemoryPoolIn(request, i);

        const std::string& inputNodeName = ngraphNw->getNodeName(inIndex);
        if (inputNodeName == "") {
            ALOGD("Ignorning input at index(%d), since it is invalid", inIndex);
            continue;
        }
        ALOGD("Input index: %d layername : %s", inIndex, inputNodeName.c_str());
        auto destBlob = plugin->getBlob(inputNodeName);
        uint8_t* dest = destBlob->buffer().as<uint8_t*>();
        uint8_t* src = srcBlob->buffer().as<uint8_t*>();
        std::memcpy(dest, src, srcBlob->byteSize());
        writeBufferToFile(inputNodeName, srcBlob->buffer().as<float*>(), srcBlob->size());
    }
    ALOGD("Run");

    if (measure == MeasureTiming::YES) deviceStart = now();
    plugin->infer();
    if (measure == MeasureTiming::YES) deviceEnd = now();

    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = modelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        const std::string& outputNodeName = ngraphNw->getNodeName(outIndex);
        if (outputNodeName == "") {
            ALOGD("Ignorning output at index(%d), since it is invalid", outIndex);
            continue;
        }
        ALOGD("Output index: %d layername : %s", outIndex, outputNodeName.c_str());
        auto srcBlob = plugin->getBlob(outputNodeName);
        auto operandType = modelInfo->getOperandType(outIndex);
        uint32_t expectedLength = srcBlob->byteSize();
        uint32_t rActualLength = 0;
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i, rActualLength);
        auto outDims = srcBlob->getTensorDesc().getDims();
        if (operandType == OperandType::TENSOR_BOOL8)
            expectedLength /= 4;  // 8bit expected instead of 32bit
        if (rActualLength != expectedLength) {
            ALOGE("%s Invalid length(%d) at outIndex(%d)", __func__, rActualLength, outIndex);
            // Notify Insufficient Buffer Length to modelInfo
            modelInfo->updateOutputshapes(i, outDims, false);
            notify(callback, ErrorStatus::OUTPUT_INSUFFICIENT_SIZE, modelInfo->getOutputShapes(),
                   kNoTiming);
            return;
        } else {
            modelInfo->updateOutputshapes(i, outDims);
        }
        switch (operandType) {
            case OperandType::TENSOR_INT32:
            case OperandType::TENSOR_FLOAT32: {
                std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(),
                            srcBlob->byteSize());
                break;
            }
            case OperandType::TENSOR_BOOL8: {
                floatToUint8(srcBlob->buffer().as<float*>(), (uint8_t*)destPtr, srcBlob->size());
                break;
            }
            default:
                std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(),
                            srcBlob->byteSize());
                break;
        }
        writeBufferToFile(outputNodeName, srcBlob->buffer().as<float*>(), srcBlob->size());
    }

    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
    }

    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        returned = notify(callback, ErrorStatus::NONE, modelInfo->getOutputShapes(), timing);
    } else {
        returned = notify(callback, ErrorStatus::NONE, modelInfo->getOutputShapes(), kNoTiming);
    }
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
    ALOGV("Exiting %s", __func__);
}

static std::tuple<ErrorStatus, hidl_vec<V1_2::OutputShape>, Timing> executeSynchronouslyBase(
    const Request& request, MeasureTiming measure, BasePreparedModel* preparedModel,
    time_point driverStart) {
    ALOGV("Entering %s", __func__);
    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();
    time_point driverEnd, deviceStart, deviceEnd;
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        ALOGE("Failed to set runtime pool info from HIDL memories");
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto inIndex = modelInfo->getModelInputIndex(i);
        auto srcBlob = modelInfo->getBlobFromMemoryPoolIn(request, i);

        const std::string& inputNodeName = ngraphNw->getNodeName(inIndex);
        if (inputNodeName == "") {
            ALOGD("Ignorning input at index(%d), since it is invalid", inIndex);
            continue;
        }
        ALOGD("Input index: %d layername : %s", inIndex, inputNodeName.c_str());
        auto destBlob = plugin->getBlob(inputNodeName);
        uint8_t* dest = destBlob->buffer().as<uint8_t*>();
        uint8_t* src = srcBlob->buffer().as<uint8_t*>();
        std::memcpy(dest, src, srcBlob->byteSize());
        writeBufferToFile(inputNodeName, srcBlob->buffer().as<float*>(), srcBlob->size());
    }

    ALOGD("Run");

    if (measure == MeasureTiming::YES) deviceStart = now();
    plugin->infer();
    if (measure == MeasureTiming::YES) deviceEnd = now();

    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = modelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        const std::string& outputNodeName = ngraphNw->getNodeName(outIndex);
        if (outputNodeName == "") {
            ALOGD("Ignorning output at index(%d), since it is invalid", outIndex);
            continue;
        }
        ALOGD("Output index: %d layername : %s", outIndex, outputNodeName.c_str());
        auto srcBlob = plugin->getBlob(outputNodeName);
        auto operandType = modelInfo->getOperandType(outIndex);
        uint32_t expectedLength = srcBlob->byteSize();
        uint32_t rActualLength = 0;
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i, rActualLength);
        auto outDims = srcBlob->getTensorDesc().getDims();
        if (operandType == OperandType::TENSOR_BOOL8)
            expectedLength /= 4;  // 8bit expected instead of 32bit
        if (rActualLength != expectedLength) {
            ALOGE("%s Invalid length(%d) at outIndex(%d)", __func__, rActualLength, outIndex);
            // Notify Insufficient Buffer Length to modelInfo
            modelInfo->updateOutputshapes(i, outDims, false);
            return {ErrorStatus::OUTPUT_INSUFFICIENT_SIZE, modelInfo->getOutputShapes(), kNoTiming};
        } else
            modelInfo->updateOutputshapes(i, outDims);
        switch (operandType) {
            case OperandType::TENSOR_INT32:
            case OperandType::TENSOR_FLOAT32: {
                std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(),
                            srcBlob->byteSize());
                break;
            }
            case OperandType::TENSOR_BOOL8: {
                floatToUint8(srcBlob->buffer().as<float*>(), (uint8_t*)destPtr, srcBlob->size());
                break;
            }
            default:
                std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(),
                            srcBlob->byteSize());
                break;
        }
        writeBufferToFile(outputNodeName, srcBlob->buffer().as<float*>(), srcBlob->size());
    }

    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        return {ErrorStatus::NONE, modelInfo->getOutputShapes(), timing};
    }
    return {ErrorStatus::NONE, modelInfo->getOutputShapes(), kNoTiming};
    ALOGV("Exiting %s", __func__);
}

Return<void> BasePreparedModel::executeSynchronously(const Request& request, MeasureTiming measure,
                                                     executeSynchronously_cb cb) {
    ALOGV("Entering %s", __func__);
    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (!validateRequest(request, mModelInfo->getModel())) {
        cb(ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return Void();
    }
    auto [status, outputShapes, timing] =
        executeSynchronouslyBase(request, measure, this, driverStart);
    cb(status, std::move(outputShapes), timing);
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<void> BasePreparedModel::configureExecutionBurst(
    const sp<V1_2::IBurstCallback>& callback,
    const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
    const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel, configureExecutionBurst_cb cb) {
    ALOGV("Entering %s", __func__);
    const sp<V1_2::IBurstContext> burst =
        ExecutionBurstServer::create(callback, requestChannel, resultChannel, this);

    if (burst == nullptr) {
        cb(ErrorStatus::GENERAL_FAILURE, {});
        ALOGI("%s GENERAL_FAILURE", __func__);
    } else {
        cb(ErrorStatus::NONE, burst);
        ALOGI("%s burst created", __func__);
    }
    return Void();
}

Return<ErrorStatus> BasePreparedModel::execute(const Request& request,
                                               const sp<V1_0::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, MeasureTiming::NO, this, callback);
}

Return<ErrorStatus> BasePreparedModel::execute_1_2(const Request& request, MeasureTiming measure,
                                                   const sp<V1_2::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, measure, this, callback);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
