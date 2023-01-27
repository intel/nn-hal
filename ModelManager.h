#ifndef ANDROID_ML_NN_MODELMANAGER_H
#define ANDROID_ML_NN_MODELMANAGER_H

#include <android/hidl/memory/1.0/IMemory.h>
#include <android/log.h>
#include <hidlmemory/mapping.h>
#include <log/log.h>
#include "ie_blob.h"

#include "Driver.h"
#include "utils.h"

#undef LOG_TAG
#define LOG_TAG "ModelManager"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using ::android::hidl::memory::V1_0::IMemory;

using Blob = InferenceEngine::Blob;

// Utility class that provides functions and methods around NNAPI Model
class NnapiModelInfo {
public:
    NnapiModelInfo(const Model& model) : mModel(model) {}

    bool initRuntimeInfo() {
        mPoolInfos.resize(mModel.pools.size());
        for (size_t i = 0; i < mModel.pools.size(); i++) {
            auto& poolInfo = (mPoolInfos)[i];
            if (!poolInfo.set(mModel.pools[i])) {
                ALOGE("Could not map pool");
                return false;
            }
        }
        if (!initializeRunTimeOperandInfo()) return false;

        return true;
    }
    // Copy model input indices to a seperate vector
    const auto& getModelInputIndexes() { return mModel.main.inputIndexes; }

    uint32_t getModelInputIndex(uint32_t index) { return mModel.main.inputIndexes[index]; }

    uint32_t getModelOutputIndex(uint32_t index) { return mModel.main.outputIndexes[index]; }

    size_t getModelOutputsSize() { return mModel.main.outputIndexes.size(); }

    // Index into the operand vector
    V1_3::OperandLifeTime getOperandLifetime(uint32_t operandIdx) {
        auto tmpOperand = mModel.main.operands[operandIdx];
        return tmpOperand.lifetime;
    }
    OperandType getOperandType(uint32_t operandIdx) {
        auto tmpOperand = mModel.main.operands[operandIdx];
        return tmpOperand.type;
    }

    bool isOperandLifeTimeTemp(uint32_t operandIdx) {
        if (getOperandLifetime(operandIdx) == OperandLifeTime::TEMPORARY_VARIABLE) return true;
        return false;
    }
    bool isOperandLifeTimeConst(uint32_t operandIdx) {
        if (getOperandLifetime(operandIdx) == OperandLifeTime::CONSTANT_COPY ||
            getOperandLifetime(operandIdx) == OperandLifeTime::CONSTANT_REFERENCE)
            return true;
        return false;
    }

    template <typename T>
    T GetConstOperand(uint32_t index) {
        uint32_t len;
        const uint8_t* buf = GetOperandMemory(index, len);
        return GetConstFromBuffer<T>(buf, len);
    }

    const auto& getOperations() { return mModel.main.operations; }
    const auto& getOperationOutput(int operationIndex, uint32_t outputIndex) {
        return mModel.main.operations[operationIndex].outputs[outputIndex];
    }
    const auto& getOperationInput(int operationIndex, uint32_t inputIndex) {
        return mModel.main.operations[operationIndex].inputs[inputIndex];
    }
    size_t getOperationInputsSize(int operationIndex) {
        return mModel.main.operations[operationIndex].inputs.size();
    }
    size_t getOperationOutputsSize(int operationIndex) {
        return mModel.main.operations[operationIndex].outputs.size();
    }

    size_t getOperationsSize() { return mModel.main.operations.size(); }

    const auto& getOperationType(int index) { return mModel.main.operations[index].type; }

    const Operand& getOperand(int index) { return mModel.main.operands[index]; }

    size_t getOperandsSize() { return mModel.main.operands.size(); }

    float getOperandScale(int index) {
        auto operand = getOperand(index);
        return operand.scale;
    }

    int32_t getOperandZeroPoint(int index) {
        auto operand = getOperand(index);
        return operand.zeroPoint;
    }

    RunTimeOperandInfo& getRuntimeOperand(uint32_t index) {
        return mOperands[mModel.main.inputIndexes[index]];
    }

    bool isConstOperand(int index) {
        ALOGD("---------------------------------------------");
        ALOGD("Operand index: %d", index);
        const auto op = mModel.main.operands[index];
        ALOGD(" %s", toString(op).c_str());
        bool ret = (op.lifetime == OperandLifeTime::CONSTANT_COPY ||
                    op.lifetime == OperandLifeTime::CONSTANT_REFERENCE);
        ALOGD("%s", ret ? "Const" : "Non-Const");
        ALOGD("---------------------------------------------");
        return ret;
    }

    const uint8_t* GetOperandMemory(int index, uint32_t& lenOut);

    template <typename T>
    T ParseOperationInput(int operationIndex, uint32_t index) {
        uint32_t inputIndex = mModel.main.operations[operationIndex].inputs[index];
        const auto operand = mModel.main.operands[inputIndex];
        const auto value = GetConstOperand<T>(inputIndex);
        ALOGV("Operation input index: %d, operand index: %d", index, inputIndex);
        ALOGV("Operation: %s", toString(mModel.main.operations[operationIndex]).c_str());
        printHelper<T>::print(value, toString(operand).c_str());

        return value;
    }

    // TODO: Move it to Utils class
    template <typename T>
    std::vector<T> GetConstVecFromBuffer(const uint8_t* buf, uint32_t len) {
        int n = len / sizeof(T);
        if (n * sizeof(T) != len) {
            ALOGE("typeid(T).name() should be  multiples of %lu bytes", sizeof(T));
            nnAssert(false);
        }

        std::vector<T> ret;
        for (int i = 0; i < n; i++) {
            ret.push_back(*(T*)buf);
            buf += sizeof(T);
        }
        return ret;
    }

    template <typename T>
    std::vector<T> GetConstVecOperand(uint32_t index) {
        // dumpOperand(index, mModel);
        uint32_t len;
        const uint8_t* buf = GetOperandMemory(index, len);
        return GetConstVecFromBuffer<T>(buf, len);
    }

    template <typename T>
    T GetConstFromBuffer(const uint8_t* buf, uint32_t len);

    void* getBlobFromMemoryPoolIn(const Request& request, uint32_t index, uint32_t& rBufferLength);
    void* getBlobFromMemoryPoolOut(const Request& request, uint32_t index, uint32_t& rBufferLength);

    Model getModel() { return mModel; }

    ErrorStatus setRunTimePoolInfosFromHidlMemories(const hidl_vec<hidl_memory>& pools);
    V1_3::ErrorStatus setRunTimePoolInfosFromHidlMemories(
        const hidl_vec<V1_3::Request::MemoryPool>& pools);

    bool updateRequestPoolInfos() {
        for (auto& runtimeInfo : mRequestPoolInfos) {
            runtimeInfo.update();
        }

        return true;
    }

    std::vector<V1_2::OutputShape> getOutputShapes() { return mOutputShapes; }

    void unmapRuntimeMemPools() {
        for (auto& runtimeInfo : mRequestPoolInfos) {
            runtimeInfo.unmap_mem();
        }
    }

    bool isOmittedInput(int operationIndex, uint32_t index);
    bool updateOutputshapes(size_t outputIndex, std::vector<size_t>& outputShape,
                            bool isLengthSufficient = true);

private:
    bool initializeRunTimeOperandInfo();

    Model mModel;  // TODO: Do we need a new copy of model??
    std::vector<RunTimePoolInfo> mPoolInfos;
    std::vector<RunTimeOperandInfo> mOperands;
    std::vector<RunTimePoolInfo> mRequestPoolInfos;
    std::vector<V1_2::OutputShape> mOutputShapes;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif
