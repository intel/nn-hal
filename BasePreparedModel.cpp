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
#include "ValidateHal.h"

#define DISABLE_ALL_QUANT
#define LOG_TAG "BasePreparedModel"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;

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

bool BasePreparedModel::initialize() {
    ALOGV("Entering %s", __func__);
    return true;
}

static Return<void> notify(const sp<V1_0::IExecutionCallback>& callback,
                           const ErrorStatus& status) {
    return callback->notify(status);
}

static void floatToUint8(const float* src, uint8_t* dst, size_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        dst[i] = static_cast<uint8_t>(src[i]);
        ALOGV("%s input: %f output: %d ", __func__, src[i], dst[i]);
    }
}

namespace {
using time_point = std::chrono::steady_clock::time_point;
auto now() { return std::chrono::steady_clock::now(); };
auto microsecondsDuration(decltype(now()) end, decltype(now()) start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
};
}  // namespace

template <typename T_IExecutionCallback>
Return<ErrorStatus> executeBase(const Request& request, BasePreparedModel* preparedModel,
                                const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, preparedModel->getModelInfo()->getModel())) {
        notify(callback, ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([preparedModel, request, callback] {
        asyncExecute(request, preparedModel, callback);
    }).detach();
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

template <typename T_IExecutionCallback>
void asyncExecute(const Request& request, BasePreparedModel* preparedModel,
                  const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();
    time_point driverEnd, deviceStart, deviceEnd;
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        notify(callback, ErrorStatus::GENERAL_FAILURE);
        return;
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        uint32_t len;
        auto inIndex = modelInfo->getModelInputIndex(i);
        void* srcPtr = modelInfo->getBlobFromMemoryPoolIn(request, i, len);

        const std::string& inputNodeName = ngraphNw->getNodeName(inIndex);
        if (inputNodeName == "") {
            ALOGD("Ignorning input at index(%d), since it is invalid", inIndex);
            continue;
        }
        ALOGD("Input index: %d layername : %s", inIndex, inputNodeName.c_str());
        auto destBlob = plugin->getBlob(inputNodeName);

        uint8_t* dest = destBlob->buffer().as<uint8_t*>();
        std::memcpy(dest, (uint8_t*)srcPtr, len);
    }
    ALOGD("%s Run", __func__);
    try {
        plugin->infer();
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
        notify(callback, ErrorStatus::GENERAL_FAILURE);
        return;
    }
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
        uint32_t actualLength = srcBlob->byteSize();
        uint32_t expectedLength = 0;
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i, expectedLength);
        auto outputBlobDims = srcBlob->getTensorDesc().getDims();

        ALOGD("output precision: %d", static_cast<int>(srcBlob->getTensorDesc().getPrecision()));

        switch (operandType) {
            case OperandType::TENSOR_QUANT8_ASYMM:
                actualLength /= 4;
                break;
            default:
                ALOGV("Operand type is 4 bytes !!");
                break;
        }

        switch (operandType) {
            case OperandType::TENSOR_INT32:
            case OperandType::TENSOR_FLOAT32: {
                std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(),
                            srcBlob->byteSize());
                break;
            }
            case OperandType::TENSOR_QUANT8_ASYMM: {
                floatToUint8(srcBlob->buffer().as<float*>(), (uint8_t*)destPtr, srcBlob->size());
                break;
            }
            default:
                std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(),
                            srcBlob->byteSize());
                break;
        }
    }

    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
    }

    Return<void> returned;
    returned = notify(callback, ErrorStatus::NONE);
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
    ALOGV("Exiting %s", __func__);
}

Return<ErrorStatus> BasePreparedModel::execute(const Request& request,
                                               const sp<V1_0::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, this, callback);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
