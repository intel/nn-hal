/*
 * Copyright (C) 2018 The Android Open Source Project
 * Copyright (c) 2018 Intel Corporation
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

#define LOG_TAG "VpuUtils"

#include "VpuUtils.h"
#include "NeuralNetworks.h"

#include <android-base/logging.h>
#include <android-base/properties.h>
#include <android-base/strings.h>
#include <sys/system_properties.h>
#include <unordered_map>


//using ::android::hidl::allocator::V1_0::IAllocator;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

const char kVLogPropKey[] = "debug.nn.vpu.vlog"; //"debug.nn.vlog"
int vLogMask = ~0;

// Split the space separated list of tags from verbose log setting and build the
// logging mask from it. note that '1' and 'all' are special cases to enable all
// verbose logging.
//
// NN API verbose logging setting comes from system property debug.nn.vlog.
// Example:
// setprop debug.nn.vlog 1 : enable all logging tags.
// setprop debug.nn.vlog "model compilation" : only enable logging for MODEL and
//                                             COMPILATION tags.
void initVLogMask() {
    vLogMask = 0;
    const std::string vLogSetting = android::base::GetProperty(kVLogPropKey, "");
    if (vLogSetting.empty()) {
        return;
    }

    std::unordered_map<std::string, int> vLogFlags = {
        {"1", -1},
        {"all", -1},
        {"model", MODEL},
        {"compilation", COMPILATION},
        {"execution", EXECUTION},
        {"vpuexe", VPUEXE},
        {"manager", MANAGER},
        {"driver", DRIVER}};

    std::vector<std::string> elements = android::base::Split(vLogSetting, " ");
    for (const auto& elem : elements) {
        const auto& flag = vLogFlags.find(elem);
        if (flag == vLogFlags.end()) {
            LOG(ERROR) << "Unknown trace flag: " << elem;
            continue;
        }

        if (flag->second == -1) {
            // -1 is used for the special values "1" and "all" that enable all
            // tracing.
            vLogMask = ~0;
            return;
        } else {
            vLogMask |= 1 << flag->second;
            vLogMask = ~0; //TODO remove later
        }
    }
}

#define COUNT(X) (sizeof(X) / sizeof(X[0]))

const char* kTypeNames[kNumberOfDataTypes] = {
        "FLOAT32",        "INT32",        "UINT32",
        "TENSOR_FLOAT32", "TENSOR_INT32", "TENSOR_QUANT8_ASYMM",
};

static_assert(COUNT(kTypeNames) == kNumberOfDataTypes, "kTypeNames is incorrect");

const char* kTypeNamesOEM[kNumberOfDataTypesOEM] = {
        "OEM",            "TENSOR_OEM_BYTE",
};

static_assert(COUNT(kTypeNamesOEM) == kNumberOfDataTypesOEM, "kTypeNamesOEM is incorrect");

// TODO Check if this useful
const char* kErrorNames[] = {
        "NO_ERROR", "OUT_OF_MEMORY", "INCOMPLETE", "NULL", "BAD_DATA",
};

namespace {

template <typename EntryType, uint32_t entryCount, uint32_t entryCountOEM>
EntryType tableLookup(const EntryType (&table)[entryCount],
                      const EntryType (&tableOEM)[entryCountOEM],
                      uint32_t code) {
    if (code < entryCount) {
        return table[code];
    } else if (code >= kOEMCodeBase && (code - kOEMCodeBase) < entryCountOEM) {
        return tableOEM[code - kOEMCodeBase];
    } else {
        nnAssert(!"tableLookup: bad code");
        return EntryType();
    }
}

};  // anonymous namespace

const char* kOperationNames[kNumberOfOperationTypes] = {
        "ADD",
        "AVERAGE_POOL",
        "CONCATENATION",
        "CONV",
        "DEPTHWISE_CONV",
        "DEPTH_TO_SPACE",
        "DEQUANTIZE",
        "EMBEDDING_LOOKUP",
        "FLOOR",
        "FULLY_CONNECTED",
        "HASHTABLE_LOOKUP",
        "L2_NORMALIZATION",
        "L2_POOL",
        "LOCAL_RESPONSE_NORMALIZATION",
        "LOGISTIC",
        "LSH_PROJECTION",
        "LSTM",
        "MAX_POOL",
        "MUL",
        "RELU",
        "RELU1",
        "RELU6",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RNN",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SVDF",
        "TANH",
};

static_assert(COUNT(kOperationNames) == kNumberOfOperationTypes, "kOperationNames is incorrect");

const char* kOperationNamesOEM[kNumberOfOperationTypesOEM] = {
        "OEM_OPERATION",
};

static_assert(COUNT(kOperationNamesOEM) == kNumberOfOperationTypesOEM,
              "kOperationNamesOEM is incorrect");

const char* getOperationName(OperationType type) {
    uint32_t n = static_cast<uint32_t>(type);
    return tableLookup(kOperationNames, kOperationNamesOEM, n);
}

const uint32_t kSizeOfDataType[]{
        4, // ANEURALNETWORKS_FLOAT32
        4, // ANEURALNETWORKS_INT32
        4, // ANEURALNETWORKS_UINT32
        4, // ANEURALNETWORKS_TENSOR_FLOAT32
        4, // ANEURALNETWORKS_TENSOR_INT32
        1  // ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8
};

static_assert(COUNT(kSizeOfDataType) == kNumberOfDataTypes, "kSizeOfDataType is incorrect");

const bool kScalarDataType[]{
        true,  // ANEURALNETWORKS_FLOAT32
        true,  // ANEURALNETWORKS_INT32
        true,  // ANEURALNETWORKS_UINT32
        false, // ANEURALNETWORKS_TENSOR_FLOAT32
        false, // ANEURALNETWORKS_TENSOR_INT32
        false, // ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8
};

static_assert(COUNT(kScalarDataType) == kNumberOfDataTypes, "kScalarDataType is incorrect");

const uint32_t kSizeOfDataTypeOEM[]{
        0, // ANEURALNETWORKS_OEM
        1, // ANEURALNETWORKS_TENSOR_OEM_BYTE
};

static_assert(COUNT(kSizeOfDataTypeOEM) == kNumberOfDataTypesOEM,
              "kSizeOfDataTypeOEM is incorrect");

const bool kScalarDataTypeOEM[]{
        true,  // ANEURALNETWORKS_OEM
        false, // ANEURALNETWORKS_TENSOR_OEM_BYTE
};

static_assert(COUNT(kScalarDataTypeOEM) == kNumberOfDataTypesOEM,
              "kScalarDataTypeOEM is incorrect");

uint32_t sizeOfData(OperandType type, const std::vector<uint32_t>& dimensions) {
    int n = static_cast<int>(type);

    uint32_t size = tableLookup(kSizeOfDataType, kSizeOfDataTypeOEM, n);

    if (tableLookup(kScalarDataType, kScalarDataTypeOEM, n) == true) {
        return size;
    }

    for (auto d : dimensions) {
        size *= d;
    }
    return size;
}


hidl_memory allocateSharedMemory(int64_t size) {
    hidl_memory memory;

    // TODO: should we align memory size to nearest page? doesn't seem necessary...
    const std::string& type = "ashmem";
    sp<IAllocator> allocator = IAllocator::getService(type);
    allocator->allocate(size, [&](bool success, const hidl_memory& mem) {
        if (!success) {
            LOG(ERROR) << "unable to allocate " << size << " bytes of " << type;
        } else {
            memory = mem;
        }
    });

    return memory;
}

uint32_t alignBytesNeeded(uint32_t index, size_t length) {
    uint32_t pattern;
    if (length < 2) {
        pattern = 0; // No alignment necessary
    } else if (length < 4) {
        pattern = 1; // Align on 2-byte boundary
    } else {
        pattern = 3; // Align on 4-byte boundary
    }
    uint32_t extra = (~(index - 1)) & pattern;
    return extra;
}

void logModelToInfo(const Model& model) {
    LOG(INFO) << "Model start";
    LOG(INFO) << "operands" << toString(model.operands);
    LOG(INFO) << "operations" << toString(model.operations);
    LOG(INFO) << "inputIndexes" << toString(model.inputIndexes);
    LOG(INFO) << "outputIndexes" << toString(model.outputIndexes);
    LOG(INFO) << "operandValues size" << model.operandValues.size();
    LOG(INFO) << "pools" << toString(model.pools);
}

// Validates the type. The used dimensions can be underspecified.
int validateOperandType(const ANeuralNetworksOperandType& type, const char* tag,
                        bool allowPartial) {
    if (!allowPartial) {
        for (uint32_t i = 0; i < type.dimensionCount; i++) {
            if (type.dimensions[i] == 0) {
                LOG(ERROR) << tag << " OperandType invalid dimensions[" << i
                           << "] = " << type.dimensions[i];
                return ANEURALNETWORKS_BAD_DATA;
            }
        }
    }
    if (!validCode(kNumberOfDataTypes, kNumberOfDataTypesOEM, type.type)) {
        LOG(ERROR) << tag << " OperandType invalid type " << type.type;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM) {
        if (type.zeroPoint < 0 || type.zeroPoint > 255) {
            LOG(ERROR) << tag << " OperandType invalid zeroPoint " << type.zeroPoint;
            return ANEURALNETWORKS_BAD_DATA;
        }
        if (type.scale < 0.f) {
            LOG(ERROR) << tag << " OperandType invalid scale " << type.scale;
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int validateOperandList(uint32_t count, const uint32_t* list, uint32_t operandCount,
                        const char* tag) {
    for (uint32_t i = 0; i < count; i++) {
        if (list[i] >= operandCount) {
            LOG(ERROR) << tag << " invalid operand index at " << i << " = " << list[i]
                       << ", operandCount " << operandCount;
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

static bool validOperandIndexes(const hidl_vec<uint32_t> indexes, size_t operandCount) {
    for (uint32_t i : indexes) {
        if (i >= operandCount) {
            LOG(ERROR) << "Index out of range " << i << "/" << operandCount;
            return false;
        }
    }
    return true;
}

static bool validOperands(const hidl_vec<Operand>& operands, const hidl_vec<uint8_t>& operandValues,
                          size_t poolCount) {
    for (auto& operand : operands) {
        if (!validCode(kNumberOfDataTypes, kNumberOfDataTypesOEM,
                       static_cast<uint32_t>(operand.type))) {
            LOG(ERROR) << "Invalid operand type " << toString(operand.type);
            return false;
        }
        /* TODO validate dim with type
        if (!validOperandIndexes(operand.dimensions, mDimensions)) {
            return false;
        }
        */
        switch (operand.lifetime) {
            case OperandLifeTime::CONSTANT_COPY:
                if (operand.location.offset + operand.location.length > operandValues.size()) {
                    LOG(ERROR) << "OperandValue location out of range.  Starts at "
                               << operand.location.offset << ", length " << operand.location.length
                           << ", max " << operandValues.size();
                    return false;
                }
                break;
            case OperandLifeTime::TEMPORARY_VARIABLE:
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                if (operand.location.offset != 0 || operand.location.length != 0) {
                    LOG(ERROR) << "Unexpected offset " << operand.location.offset << " or length "
                               << operand.location.length << " for runtime location.";
                    return false;
                }
                break;
            case OperandLifeTime::CONSTANT_REFERENCE:
                if (operand.location.poolIndex >= poolCount) {
                    LOG(ERROR) << "Invalid poolIndex " << operand.location.poolIndex << "/"
                               << poolCount;
                    return false;
                }
                break;
            // TODO: Validate that we are within the pool.
            default:
                LOG(ERROR) << "Invalid lifetime";
                return false;
        }
    }
    return true;
}

static bool validOperations(const hidl_vec<Operation>& operations, size_t operandCount) {
    for (auto& op : operations) {
        if (!validCode(kNumberOfOperationTypes, kNumberOfOperationTypesOEM,
                       static_cast<uint32_t>(op.type))) {
            LOG(ERROR) << "Invalid operation type " << toString(op.type);
            return false;
        }
        if (!validOperandIndexes(op.inputs, operandCount) ||
            !validOperandIndexes(op.outputs, operandCount)) {
            return false;
        }
    }
    return true;
}

// TODO doublecheck
bool validateModel(const Model& model) {
    const size_t operandCount = model.operands.size();
    return (validOperands(model.operands, model.operandValues, model.pools.size()) &&
            validOperations(model.operations, operandCount) &&
            validOperandIndexes(model.inputIndexes, operandCount) &&
            validOperandIndexes(model.outputIndexes, operandCount));
}

bool validRequestArguments(const hidl_vec<RequestArgument>& arguments,
                           const hidl_vec<uint32_t>& operandIndexes,
                           const hidl_vec<Operand>& operands, size_t poolCount,
                           const char* type) {
    const size_t argumentCount = arguments.size();
    if (argumentCount != operandIndexes.size()) {
        LOG(ERROR) << "Request specifies " << argumentCount << " " << type << "s but the model has "
                   << operandIndexes.size();
        return false;
    }
    for (size_t argumentIndex = 0; argumentIndex < argumentCount; argumentIndex++) {
        const RequestArgument& argument = arguments[argumentIndex];
        const uint32_t operandIndex = operandIndexes[argumentIndex];
        const Operand& operand = operands[operandIndex];
        if (argument.hasNoValue) {
            if (argument.location.poolIndex != 0 ||
                argument.location.offset != 0 ||
                argument.location.length != 0 ||
                argument.dimensions.size() != 0) {
                LOG(ERROR) << "Request " << type << " " << argumentIndex
                           << " has no value yet has details.";
                return false;
            }
        }
        if (argument.location.poolIndex >= poolCount) {
            LOG(ERROR) << "Request " << type << " " << argumentIndex << " has an invalid poolIndex "
                       << argument.location.poolIndex << "/" << poolCount;
            return false;
        }
        // TODO: Validate that we are within the pool.
        uint32_t rank = argument.dimensions.size();
        if (rank > 0) {
            if (rank != operand.dimensions.size()) {
                LOG(ERROR) << "Request " << type << " " << argumentIndex
                           << " has number of dimensions (" << rank
                           << ") different than the model's (" << operand.dimensions.size() << ")";
                return false;
            }
            for (size_t i = 0; i < rank; i++) {
                if (argument.dimensions[i] != operand.dimensions[i] &&
                    operand.dimensions[i] != 0) {
                    LOG(ERROR) << "Request " << type << " " << argumentIndex
                               << " has dimension " << i << " of " << operand.dimensions[i]
                               << " different than the model's " << operand.dimensions[i];
                    return false;
                }
                if (argument.dimensions[i] == 0) {
                    LOG(ERROR) << "Request " << type << " " << argumentIndex
                               << " has dimension " << i << " of zero";
                    return false;
                }
            }
        }
    }
    return true;
}

// TODO doublecheck
bool validateRequest(const Request& request, const Model& model) {
    const size_t poolCount = request.pools.size();
    return (validRequestArguments(request.inputs, model.inputIndexes, model.operands, poolCount,
                                  "input") &&
            validRequestArguments(request.outputs, model.outputIndexes, model.operands, poolCount,
                                  "output"));
}


uint32_t getProp(const char* str, uint32_t defaultValue) {
    const std::string propStr = android::base::GetProperty(str, "");
    if (propStr.size() > 0) {
        return std::stoi(propStr);
    } else {
        return defaultValue;
    }
}

}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
