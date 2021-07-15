#include "ModelManager.h"

#define LOG_TAG "ModelManager"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

bool NnapiModelInfo::updateOutputshapes(size_t outputIndex, std::vector<size_t>& outputDims,
                                        bool isLengthSufficient) {
    auto& outputShapeDims = mOutputShapes[outputIndex].dimensions;
    mOutputShapes[outputIndex].isSufficient = isLengthSufficient;
    if (outputDims.size() < outputShapeDims.size()) {
        return false;
    }
    for (size_t i = 0; i < outputShapeDims.size(); i++) {
        if (outputShapeDims[i] != outputDims[i]) {
            ALOGD("%s Updating dim(%zu) at Output index(%zu)", __func__, i, outputIndex);
            outputShapeDims[i] = outputDims[i];
        }
    }
    return true;
}

bool NnapiModelInfo::initializeRunTimeOperandInfo() {
    // initialize runtime operand info from model.
    const size_t count = mModel.main.operands.size();
    ALOGD("Operand size = %zu\n", count);
    if (!count) {
        ALOGE("NNERR:Operand Count is 0");
        return false;
    }
    mOperands.resize(count);
    mOutputShapes.resize(mModel.main.outputIndexes.size());

    // Start by setting the runtime info to what's in the model.
    for (size_t i = 0; i < count; i++) {
        const Operand& from = mModel.main.operands[i];
        dumpOperand(i, mModel.main);
        RunTimeOperandInfo& to = mOperands[i];
        to.dimensions.resize(from.dimensions.size());
        for (size_t j = 0; j < from.dimensions.size(); j++) {
            to.dimensions[j] = from.dimensions[j];
        }

        to.scale = from.scale;
        switch (from.type) {
            case OperandType::TENSOR_FLOAT32:
            case OperandType::FLOAT32:
                to.type = OperandType::TENSOR_FLOAT32;
                ALOGD("OperandType = %d\n", from.type);
                break;
            case OperandType::INT32:
            case OperandType::UINT32:
            case OperandType::BOOL:
                nnAssert(to.scale == 0);
                FALLTHROUGH_INTENDED;
            case OperandType::TENSOR_INT32:
                to.type = from.type;
                break;
            case OperandType::TENSOR_FLOAT16:
            case OperandType::FLOAT16:
                to.type = from.type;
                break;
            case OperandType::TENSOR_BOOL8:
                to.type = from.type;
                break;
            case OperandType::TENSOR_QUANT8_ASYMM:
            case OperandType::TENSOR_QUANT8_SYMM:
            case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
                to.type = from.type;
                break;
            default:
                ALOGE("wrong operand type %d", from.type);
                return false;
        }

        to.length = from.location.length;
        to.lifetime = from.lifetime;
        to.zeroPoint = from.zeroPoint;

        switch (from.lifetime) {
            case OperandLifeTime::TEMPORARY_VARIABLE:
                to.buffer = nullptr;
                to.length = sizeOfData(to.type, to.dimensions);
                to.numberOfUsesLeft = from.numberOfConsumers;
                break;
            case OperandLifeTime::CONSTANT_COPY:
                to.buffer = const_cast<uint8_t*>(&mModel.operandValues[from.location.offset]);
                to.numberOfUsesLeft = 0;
                break;
            case OperandLifeTime::CONSTANT_REFERENCE: {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < mPoolInfos.size());
                auto& r = mPoolInfos[poolIndex];
                to.buffer = r.buffer + from.location.offset;
                to.numberOfUsesLeft = 0;
                break;
            }
            case OperandLifeTime::SUBGRAPH_INPUT:
            case OperandLifeTime::SUBGRAPH_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                to.buffer = nullptr;
                to.numberOfUsesLeft = 0;
                break;
            default:
                return false;
                break;
        }
    }

    for (uint32_t i = 0; i < mModel.main.outputIndexes.size(); i++) {
        const uint32_t operandIndex = mModel.main.outputIndexes[i];
        const RunTimeOperandInfo& from = mOperands[operandIndex];
        mOutputShapes[i].dimensions = from.dimensions;
        mOutputShapes[i].isSufficient = true;
    }

    return true;
}

// TODO: Move it to Utils class
template <typename T>
T NnapiModelInfo::GetConstFromBuffer(const uint8_t* buf, uint32_t len) {
    // ALOGD("buf: %p, len: %d", buf, len);
    if (len != sizeof(T)) {
        ALOGE("fix me: typeid(T).name() should be %lu bytes", sizeof(T));
        // fix me if buffer is of type float and if float and OperandLifeTime::CONSTANT_REFERENCE
        nnAssert(false);
    }
    return *(T*)(buf);
}

const uint8_t* NnapiModelInfo::GetOperandMemory(int index, uint32_t& lenOut) {
    ALOGV("%s", __func__);
    const auto op = mModel.main.operands[index];
    lenOut = op.location.length;
    if (op.lifetime == OperandLifeTime::CONSTANT_COPY) {
        ALOGV("operand lifetime OperandLifeTime::CONSTANT_COPY");
        if (op.location.poolIndex != 0) {
            // ALOGE("CONSTANT_COPY expects poolIndex to be 0");
            nnAssert(false);
        }
        return (const_cast<uint8_t*>(&mModel.operandValues[op.location.offset]));
    } else if (op.lifetime == OperandLifeTime::CONSTANT_REFERENCE) {
        ALOGV("operand lifetime OperandLifeTime::CONSTANT_REFERENCE");
        auto poolIndex = op.location.poolIndex;
        auto& r = mPoolInfos[poolIndex];
        return (const_cast<uint8_t*>(r.buffer + op.location.offset));
    } else if (op.lifetime == OperandLifeTime::TEMPORARY_VARIABLE ||
               op.lifetime == OperandLifeTime::SUBGRAPH_INPUT ||
               op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT ||
               op.lifetime == OperandLifeTime::NO_VALUE) {
        // ALOGD(
        //     "operand lifetime "
        //     "OperandLifeTime::MODEL_INPUT||MODEL_OUTPUT||NO_VALUE||TEMPORARY_VARIABLE");
        lenOut = sizeOfData(op.type, op.dimensions);
        ALOGV("operand lifetime(%d), type(%d), lenOut(%d)", op.lifetime, op.type, lenOut);
        return nullptr;
    }
    ALOGE("operand is expected to be const, but lifetime is %d", op.lifetime);
    nnAssert(false);  // temp fix since some time const operand set as TEMPORARY_VARIABLE
    return nullptr;
}

Blob::Ptr NnapiModelInfo::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                                uint32_t& len) {
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32 ||
        op.type == OperandType::TENSOR_FLOAT16) {
        if (op.lifetime == OperandLifeTime::SUBGRAPH_INPUT) {
            ALOGD("Create input blob !!!!");
            vec<unsigned int> order;
            InferenceEngine::Layout layout;
            bool isOrderSet = false;
            if (op.dimensions.size() == 4) {
                order = {0, 3, 1, 2};  // nhwc -> nchw
                layout = InferenceEngine::Layout::NCHW;
                isOrderSet = true;
            } else if (op.dimensions.size() == 2) {
                order = {0, 1};
                layout = InferenceEngine::Layout::NC;
                isOrderSet = true;
            } else if (op.dimensions.size() == 1) {
                layout = InferenceEngine::Layout::C;
                isOrderSet = false;
            } else {
                layout = InferenceEngine::Layout::ANY;
                isOrderSet = false;
            }

            auto inputDims = toDims(op.dimensions);

            TensorDims inputOpDims;

            if (!isOrderSet)
                inputOpDims = inputDims;
            else
                inputOpDims = permuteDims(inputDims, order);

            InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, inputOpDims, layout);

            if (buf == nullptr) {
                ALOGD("SUBGRAPH_INPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                return blob;
            }
        } else if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            ALOGD("Create output blob !!!!");
            vec<unsigned int> order;
            InferenceEngine::Layout layout;
            if (op.dimensions.size() == 4) {
                layout = InferenceEngine::Layout::NHWC;
            } else if (op.dimensions.size() == 2) {
                layout = InferenceEngine::Layout::NC;
            } else if (op.dimensions.size() == 3) {
                layout = InferenceEngine::Layout::CHW;
                ALOGI("Anoob : GetInOutOperandAsBlob output already transposed to NHWC");
            } else if (op.dimensions.size() == 1) {
                layout = InferenceEngine::Layout::C;
            } else {
                layout = InferenceEngine::Layout::ANY;
            }
            InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions),
                                           layout);  // nhwc
            if (buf == nullptr) {
                ALOGD("SUBGRAPH_OUTPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    InferenceEngine::make_shared_blob<float>(td, (float*)buf);
                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32 || op.type == OperandType::INT32) {
        ALOGV("check if const tensors of type IN32 supported");
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            ALOGD("TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<int32_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<int32_t>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<int32_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t*)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_BOOL8 || op.type == OperandType::BOOL) {
        ALOGV("check if const tensors of type BOOL supported");
        // nnAssert(true);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::BOOL, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            ALOGD("TENSOR_BOOL8 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<uint8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<uint8_t>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<uint8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<uint8_t>>(td, (uint8_t*)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_QUANT8_ASYMM) {
        ALOGV("check if tensors of type TENSOR_QUANT8_ASYMM supported");
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::U8, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            ALOGD("TENSOR_QUANT8_ASYMM buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<uint8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<uint8_t>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<uint8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<uint8_t>>(td, (uint8_t*)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_QUANT8_SYMM ||
               op.type == OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL) {
        ALOGV(
            "check if tensors of type TENSOR_QUANT8_SYMM/TENSOR_QUANT8_SYMM_PER_CHANNEL  "
            "supported");
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I8, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            ALOGD("TENSOR_QUANT8_SYMM/TENSOR_QUANT8_SYMM_PER_CHANNEL buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<int8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<int8_t>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<int8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<int8_t>>(td, (int8_t*)buf, len);
            return blob;
        }
    }
    return nullptr;
}

IRBlob::Ptr NnapiModelInfo::GetConstOperandAsTensor(int operand_idx, int operation_idx) {
    const auto op = mModel.main.operands[operand_idx];
    uint32_t len;

    const uint8_t* buf = GetOperandMemory(operand_idx, len);
    ALOGD("NnapiModelInfo:: operand_index: %d, operation_index :%d,len: %d, buf: %p", operand_idx,
          operation_idx, len, buf);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        InferenceEngine::Layout layout;
        bool isOrderSet = false;
        if (op.dimensions.size() == 4) {
            order = {0, 3, 1, 2};                    // nhwc -> nchw
            layout = InferenceEngine::Layout::OIHW;  // weights layout
            isOrderSet = true;
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = InferenceEngine::Layout::NC;
            isOrderSet = true;
        } else if (op.dimensions.size() == 1) {
            layout = InferenceEngine::Layout::C;
            isOrderSet = false;
        } else {
            layout = InferenceEngine::Layout::ANY;
            isOrderSet = false;
        }

        auto inputDims = toDims(op.dimensions);

        TensorDims inputOpDims;

        if (!isOrderSet)
            inputOpDims = inputDims;
        else
            inputOpDims = permuteDims(inputDims, order);

        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, inputOpDims, layout);

        if (buf == nullptr) {
            ALOGD("TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_INT32 || op.type == OperandType::INT32) {
        ALOGV("check if const tensors of type IN32 supported");
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            ALOGD("TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<int32_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<int32_t>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<int32_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t*)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_BOOL8 || op.type == OperandType::BOOL) {
        ALOGV("check if const tensors of type BOOL supported");
        // nnAssert(true);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::BOOL, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            ALOGD("TENSOR_BOOL8 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<uint8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<uint8_t>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<uint8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<uint8_t>>(td, (uint8_t*)buf, len);
            return blob;
        }
    }
    return nullptr;
}

// Redundant.. Remove the code
IRBlob::Ptr NnapiModelInfo::GetConstWeightsOperandAsTensor(uint32_t index) {
    const auto op = mModel.main.operands[index];
    uint32_t len;
    const uint8_t* buf = GetOperandMemory(index, len);
    ALOGD("NnapiModelInfo:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        InferenceEngine::Layout layout;
        bool isOrderSet = false;
        if (op.dimensions.size() == 4) {
            // order = {0,3,1,2};  //nhwc -> nchw
            order = {3, 0, 1, 2};                    // IHWO -> OIHW for depth conv
            layout = InferenceEngine::Layout::OIHW;  // weights layout
            isOrderSet = true;
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = InferenceEngine::Layout::NC;
            isOrderSet = true;
        } else if (op.dimensions.size() == 1) {
            layout = InferenceEngine::Layout::C;
            isOrderSet = false;
        } else {
            layout = InferenceEngine::Layout::ANY;
            isOrderSet = false;
        }

        auto inputDims = toDims(op.dimensions);

        TensorDims inputOpDims;

        if (!isOrderSet)
            inputOpDims = inputDims;
        else
            inputOpDims = permuteDims(inputDims, order);

        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, inputOpDims, layout);

        if (buf == nullptr) {
            ALOGD("TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32 || op.type == OperandType::INT32) {
        ALOGV("check if const tensors of type IN32 supported");
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            ALOGD("TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<int32_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<int32_t>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<int32_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t*)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_BOOL8 || op.type == OperandType::BOOL) {
        ALOGV("check if const tensors of type BOOL supported");
        // nnAssert(true);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::BOOL, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            ALOGD("TENSOR_BOOL8 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<uint8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<uint8_t>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<uint8_t>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<uint8_t>>(td, (uint8_t*)buf, len);
            return blob;
        }
    }
    return nullptr;
}

V1_3::ErrorStatus NnapiModelInfo::setRunTimePoolInfosFromHidlMemories(
    const hidl_vec<V1_3::Request::MemoryPool>& pools) {
    ALOGD("Number of pools: %zu", pools.size());
    mRequestPoolInfos.resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = mRequestPoolInfos[i];
        switch (pools[i].getDiscriminator()) {
            case V1_3::Request::MemoryPool::hidl_discriminator::hidlMemory:
                if (!poolInfo.set(pools[i].hidlMemory())) {
                    ALOGE("Could not map memory pool !!!");
                    return V1_3::ErrorStatus::GENERAL_FAILURE;
                }
                break;

            case V1_3::Request::MemoryPool::hidl_discriminator::token:
                ALOGE(
                    "%s NNHAL 1.3 driver does not yet support driver buffer allocation. Returning "
                    "failure",
                    __func__);
                return V1_3::ErrorStatus::INVALID_ARGUMENT;
                break;
        }
    }

    return V1_3::ErrorStatus::NONE;
}

ErrorStatus NnapiModelInfo::setRunTimePoolInfosFromHidlMemories(
    const hidl_vec<hidl_memory>& pools) {
    ALOGD("Number of pools: %d", pools.size());

    mRequestPoolInfos.resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = mRequestPoolInfos[i];
        if (!poolInfo.set(pools[i])) {
            ALOGE("Could not map memory pool !!!");
            return ErrorStatus::GENERAL_FAILURE;
        }
    }

    return ErrorStatus::NONE;
}

Blob::Ptr NnapiModelInfo::getBlobFromMemoryPoolIn(const Request& request, uint32_t index) {
    RunTimeOperandInfo& operand = mOperands[mModel.main.inputIndexes[index]];
    const V1_0::RequestArgument& arg = request.inputs[index];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRequestPoolInfos.size());
    auto& r = mRequestPoolInfos[poolIndex];

    if (arg.dimensions.size() > 0) {
        // It's the responsibility of the caller to validate that
        // from.dimensions only modifies the dimensions that were
        // unspecified in the model.  That's the case in SampleDriver.cpp
        // with the call to validateRequest().
        operand.dimensions = arg.dimensions;
    }

    operand.buffer = r.buffer + arg.location.offset;
    operand.length = arg.location.length;
    ALOGI("%s Operand length:%d pointer:%p offset:%d pool index: %d", __func__, operand.length,
          (r.buffer + arg.location.offset), arg.location.offset, poolIndex);
    rBufferLength = operand.length;

    return (r.buffer + arg.location.offset);
}

void* NnapiModelInfo::getBlobFromMemoryPoolOut(const Request& request, uint32_t index,
                                               uint32_t& rBufferLength) {
    RunTimeOperandInfo& operand = mOperands[mModel.main.outputIndexes[index]];
    const V1_0::RequestArgument& arg = request.outputs[index];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRequestPoolInfos.size());
    auto& r = mRequestPoolInfos[poolIndex];

    ALOGD("%s lifetime:%d location offset:%d length:%d pool index:%d", __func__, operand.lifetime,
          arg.location.offset, arg.location.length, poolIndex);

    if (arg.dimensions.size() > 0) {
        // It's the responsibility of the caller to validate that
        // from.dimensions only modifies the dimensions that were
        // unspecified in the model.  That's the case in SampleDriver.cpp
        // with the call to validateRequest().
        operand.dimensions = arg.dimensions;
    }

    operand.buffer = r.buffer + arg.location.offset;
    operand.length = arg.location.length;
    rBufferLength = operand.length;
    ALOGI("%s Operand length:%d pointer:%p", __func__, operand.length,
          (r.buffer + arg.location.offset));
    return (r.buffer + arg.location.offset);
}

bool NnapiModelInfo::isOmittedInput(int operationIndex, uint32_t index) {
    uint32_t inputIndex = mModel.main.operations[operationIndex].inputs[index];
    const auto op = mModel.main.operands[inputIndex];
    if (op.lifetime == OperandLifeTime::NO_VALUE) {
        ALOGD("index %d has life time NO_VALUE", index);
        return true;
    }

    return false;
}

template int NnapiModelInfo::GetConstOperand<int>(unsigned int);
template float NnapiModelInfo::GetConstOperand<float>(unsigned int);
template uint8_t NnapiModelInfo::GetConstOperand<uint8_t>(unsigned int);
template int8_t NnapiModelInfo::GetConstOperand<int8_t>(unsigned int);
template uint32_t NnapiModelInfo::GetConstOperand<uint32_t>(unsigned int);
template _Float16 NnapiModelInfo::GetConstOperand<_Float16>(unsigned int);
template int NnapiModelInfo::GetConstFromBuffer<int>(unsigned char const*, unsigned int);
template float NnapiModelInfo::GetConstFromBuffer<float>(unsigned char const*, unsigned int);
template uint8_t NnapiModelInfo::GetConstFromBuffer<uint8_t>(unsigned char const*, unsigned int);
template int8_t NnapiModelInfo::GetConstFromBuffer<int8_t>(unsigned char const*, unsigned int);
template uint32_t NnapiModelInfo::GetConstFromBuffer<uint32_t>(unsigned char const*, unsigned int);
template _Float16 NnapiModelInfo::GetConstFromBuffer<_Float16>(unsigned char const*, unsigned int);
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
