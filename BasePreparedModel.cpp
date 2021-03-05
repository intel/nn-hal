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
#define LOG_TAG "BasePreparedModel"
#include "BasePreparedModel.h"

#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <thread>
#include "ValidateHal.h"

#include <cutils/properties.h>

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
    delete mEnginePtr;
    mEnginePtr = nullptr;
    ALOGI("free engine");
    ALOGV("Exiting %s", __func__);
}
bool BasePreparedModel::isOperationSupported(const Operation& operation, const Model& model) {
    ALOGV("Entering %s", __func__);

    ALOGD("Check operation %d", operation.type);

#define VLOG_CHECKFAIL(fail) ALOGE("Check failed: %s", fail)

    // sp<NgraphNetworkCreator> mNgraphNwCreator;

    switch (operation.type) {
        case OperationType::ADD:
        case OperationType::CONCATENATION: {
            // if(!mNgraphNwCreator->validateOperations())
            //     return false;
        } break;
        case OperationType::RESHAPE:
            break;
        default:
            ALOGI("unsupport operation %d", operation.type);
            return false;
    }

#ifdef DISABLE_ALL_QUANT
    for (auto i : operation.inputs) {
        const auto input = model.operands[i];
        if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            VLOG_CHECKFAIL("input quant");
            return false;
        }
    }
    for (auto i : operation.outputs) {
        const auto output = model.operands[i];
        if (output.type == OperandType::TENSOR_QUANT8_ASYMM) {
            VLOG_CHECKFAIL("output quant");
            return false;
        }
    }
#endif

    ALOGI("Operation %d supported by driver", operation.type);
    ALOGV("Exiting %s", __func__);
    return true;
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
Return<ErrorStatus> BasePreparedModel::executeBase(const Request& request, MeasureTiming measure,
                                                   const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, mModelInfo->getModel())) {
        notify(callback, ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([this, request, measure, driverStart, callback] {
        asyncExecute(request, measure, driverStart, callback);
    }).detach();
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

template <typename T_IExecutionCallback>
void BasePreparedModel::asyncExecute(const Request& request, MeasureTiming measure,
                                     time_point driverStart,
                                     const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!mModelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }

    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

    auto inOutData = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                                               const hidl_vec<RequestArgument>& arguments,
                                               bool inputFromRequest, ExecuteNetwork* enginePtr) {
        // do memcpy for input data
        for (size_t i = 0; i < indexes.size(); i++) {
            RunTimeOperandInfo& operand = mOperands[indexes[i]];
            const RequestArgument& arg = arguments[i];
            auto poolIndex = arg.location.poolIndex;
            nnAssert(poolIndex < requestPoolInfos.size());
            auto& r = requestPoolInfos[poolIndex];
            if (arg.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // from.dimensions only modifies the dimensions that were
                // unspecified in the model.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                operand.dimensions = arg.dimensions;
            }
            operand.buffer = r.buffer + arg.location.offset;  // r.getBuffer()
            operand.length = arg.location.length;  // sizeOfData(operand.type, operand.dimensions);

            ALOGI("Copy request input/output to model input/output");
            if (inputFromRequest) {
                auto inputBlob = mModelInfo->GetInOutOperandAsBlob(
                    operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                    operand.length);  // if not doing memcpy
                ALOGD("Copy inputBlob for mNgc->getNodeName([%d])->name %s", indexes[i],
                      mNgc->getNodeName(indexes[i]).c_str());

                auto destBlob = enginePtr->getBlob(mNgc->getNodeName(indexes[i]));
                uint8_t* dest = destBlob->buffer().as<uint8_t*>();
                uint8_t* src = inputBlob->buffer().as<uint8_t*>();
                std::memcpy(dest, src, inputBlob->byteSize());
            } else {
                auto outputBlob = mModelInfo->GetInOutOperandAsBlob(
                    operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                    operand.length);  // if not doing memcpy
                ALOGD("copyData from IE to Android blob for mNgc->getNodeName([%d])->name %s",
                      indexes[i], mNgc->getNodeName(indexes[i]).c_str());
                auto srcBlob = enginePtr->getBlob(mNgc->getNodeName(indexes[i]));
                uint8_t* dest = outputBlob->buffer().as<uint8_t*>();
                uint8_t* src = srcBlob->buffer().as<uint8_t*>();
                std::memcpy(dest, src, outputBlob->byteSize());
            }
        }
    };

    ALOGI("pass request inputs buffer to network/model respectively");

    inOutData(mModelInfo->getModelInputIndexes(), request.inputs, true, mEnginePtr);
    ALOGD("Run");

    mEnginePtr->Infer();

    ALOGI("pass request outputs buffer to network/model respectively");
    inOutData(mModelInfo->getModelOutputIndexes(), request.outputs, false, mEnginePtr);

    if (measure == MeasureTiming::YES) deviceEnd = now();

    ALOGI("update shared memories");
    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

    InferenceEngine::TBlob<float>::Ptr outBlob =
        mEnginePtr->getBlob(mNgc->getNodeName(mModelInfo->getModelOutputIndex(0)));

    InferenceEngine::TBlob<float>::Ptr inBlob =
        mEnginePtr->getBlob(mNgc->getNodeName(mModelInfo->getModelInputIndex(0)));
    hidl_vec<OutputShape> outputShapes;
#ifdef NN_DEBUG
    {
        ALOGI("Model output0 are:");

        auto nelem = (outBlob->size() > 20 ? 20 : outBlob->size());
        for (int i = 0; i < nelem; i++) {
            ALOGD("outBlob elements %d = %f", i, outBlob->readOnly()[i]);
        }

        ALOGI("Model input0 are:");

        nelem = (inBlob->size() > 20 ? 20 : inBlob->size());
        for (int i = 0; i < nelem; i++) {
            ALOGD("inBlob elements %d = %f", i, inBlob->readOnly()[i]);
        }
    }
#endif
    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        // VLOG(L1, "Driver::asyncExecute timing = %s", toString(timing));
        returned = notify(callback, ErrorStatus::NONE, outputShapes, timing);
    } else {
        returned = notify(callback, ErrorStatus::NONE, outputShapes, kNoTiming);
    }
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
    ALOGV("Exiting %s", __func__);
}

Return<void> BasePreparedModel::executeSynchronously(const Request& request, MeasureTiming measure,
                                                     executeSynchronously_cb cb) {
    ALOGV("Entering %s", __func__);
    time_point driverStart, driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (!validateRequest(request, mModelInfo->getModel())) {
        cb(ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return Void();
    }
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!mModelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        cb(ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return Void();
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto inIndex = mModelInfo->getModelInputIndex(i);
        auto srcBlob = mModelInfo->getBlobFromMemoryPoolIn(request, i);

            ALOGD("Found input index: %d layername : %s", inIndex, mNgc->getNodeName(inIndex).c_str());
            auto destBlob = mEnginePtr->getBlob(mNgc->getNodeName(inIndex));
            uint8_t* dest = destBlob->buffer().as<uint8_t*>();
            uint8_t* src = srcBlob->buffer().as<uint8_t*>();
            std::memcpy(dest, src, srcBlob->byteSize());
    }

    ALOGI("pass request inputs buffer to network/model respectively");

    //inOutData(mModelInfo->getModelInputIndexes(), request.inputs, true, mEnginePtr);
    ALOGD("Run");

    mEnginePtr->Infer();

    ALOGI("pass request outputs buffer to network/model respectively");
    //inOutData(mModelInfo->getModelOutputIndexes(), request.outputs, false, mEnginePtr);

    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = mModelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        void* destPtr = mModelInfo->getBlobFromMemoryPoolOut(request, i);

            ALOGD("Found output index: %d layername : %s", outIndex, mNgc->getNodeName(outIndex).c_str());
            auto srcBlob = mEnginePtr->getBlob(mNgc->getNodeName(outIndex));
            std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(), srcBlob->byteSize());
            float* a = static_cast<float*>(destPtr);
            ALOGD("########### -- %f", *a);
    }
    if (!mModelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
        cb(ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return Void();
    }

    if (measure == MeasureTiming::YES) deviceEnd = now();


    InferenceEngine::TBlob<float>::Ptr outBlob =
        mEnginePtr->getBlob(mNgc->getNodeName(mModelInfo->getModelOutputIndex(0)));

    InferenceEngine::TBlob<float>::Ptr inBlob =
        mEnginePtr->getBlob(mNgc->getNodeName(mModelInfo->getModelInputIndex(0)));
    hidl_vec<OutputShape> outputShapes;
#ifdef NN_DEBUG
    {
        ALOGI("Model output0 are:");

        auto nelem = (outBlob->size() > 20 ? 20 : outBlob->size());
        for (int i = 0; i < nelem; i++) {
            ALOGI("outBlob elements %d = %f", i, outBlob->readOnly()[i]);
        }

        ALOGI("Model input0 are:");

        nelem = (inBlob->size() > 20 ? 20 : inBlob->size());
        for (int i = 0; i < nelem; i++) {
            ALOGI("inBlob elements %d = %f", i, inBlob->readOnly()[i]);
        }
    }
#endif
    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        ALOGD("Driver::executeSynchronously timing = %s", timing);
        cb(ErrorStatus::NONE, outputShapes, timing);
    } else {
        cb(ErrorStatus::NONE, outputShapes, kNoTiming);
    }
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<void> BasePreparedModel::configureExecutionBurst(
    const sp<V1_2::IBurstCallback>& callback,
    const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
    const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel, configureExecutionBurst_cb cb) {
    ALOGV("Entering %s", __func__);

    cb(ErrorStatus::GENERAL_FAILURE, {});
    return Void();
}

Return<ErrorStatus> BasePreparedModel::execute(const Request& request,
                                               const sp<V1_0::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, MeasureTiming::NO, callback);
}

Return<ErrorStatus> BasePreparedModel::execute_1_2(const Request& request, MeasureTiming measure,
                                                   const sp<V1_2::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, measure, callback);
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
