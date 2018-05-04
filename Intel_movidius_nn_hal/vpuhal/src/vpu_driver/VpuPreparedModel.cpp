/*
 * Copyright (C) 2018 The Android Open Source Project
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

 #define LOG_TAG "VpuPreparedModel"

#include "VpuPreparedModel.h"
#include "VpuUtils.h"
#include <string>
#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>

#define DISABLE_ALL_QUANT

/*

Logging related functions

start of VpuDriver namespace

setRunTimePoolInfosFromHidlMemories()

initialize()

isOperationSupported()

validModel()

execute()

deinitialize()

*/

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

  // setRunTimePoolInfosFromHidlMemories() function
/*
  bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                           const hidl_vec<hidl_memory>& pools) {
      poolInfos->resize(pools.size());
      for (size_t i = 0; i < pools.size(); i++) {
          auto& poolInfo = (*poolInfos)[i];
          if (!poolInfo.set(pools[i])) {
              ALOGE("Could not map pool");
              return false;
          }
      }
      return true;
  }  //TODO Move to VpuExecutor.cpp
*/

// initialize() function

bool VpuPreparedModel::initialize() {
    bool success = false;
    success = setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);

    return success;
  }

template <typename T>
T getOperandConstVal(const Model& model, const Operand& operand)
{
    const T* data = reinterpret_cast<const T *>(&model.operandValues[operand.location.offset]);
    return data[0];
}

//
std::string get_operation_string(int operation){
  std::string operation_name =NULL;
  switch (operation) {
    case 0:	  operation_name = "ANEURALNETWORKS_ADD"; break;
    case 1:	  operation_name = "ANEURALNETWORKS_AVERAGE_POOL_2D"; break;
    case 2:	  operation_name = "ANEURALNETWORKS_CONCATENATION"; break;
    case 3:	  operation_name = "ANEURALNETWORKS_CONV_2D"; break;
    case 4:	  operation_name = "ANEURALNETWORKS_DEPTHWISE_CONV_2D"; break;
    case 5:	  operation_name = "ANEURALNETWORKS_DEPTH_TO_SPACE"; break;
    case 6:	  operation_name = "ANEURALNETWORKS_DEQUANTIZE"; break;
    case 7:	  operation_name = "ANEURALNETWORKS_EMBEDDING_LOOKUP"; break;
    case 8:	  operation_name = "ANEURALNETWORKS_FLOOR"; break;
    case 9:	  operation_name = "ANEURALNETWORKS_FULLY_CONNECTED"; break;
    case 10:	operation_name = "ANEURALNETWORKS_HASHTABLE_LOOKUP"; break;
    case 11:	operation_name = "ANEURALNETWORKS_L2_NORMALIZATION"; break;
    case 12:	operation_name = "ANEURALNETWORKS_L2_POOL_2D"; break;
    case 13:	operation_name = "ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION"; break;
    case 14:	operation_name = "ANEURALNETWORKS_LOGISTIC"; break;
    case 15:	operation_name = "ANEURALNETWORKS_LSH_PROJECTION"; break;
    case 16:	operation_name = "ANEURALNETWORKS_LSTM"; break;
    case 17:	operation_name = "ANEURALNETWORKS_MAX_POOL_2D"; break;
    case 18:	operation_name = "ANEURALNETWORKS_MUL"; break;
    case 19:	operation_name = "ANEURALNETWORKS_RELU"; break;
    case 20:	operation_name = "ANEURALNETWORKS_RELU1"; break;
    case 21:	operation_name = "ANUERALNETWORKS_RELU6"; break;
    case 22:	operation_name = "ANEURALNETWOKRS_RESHAPE"; break;
    case 23:	operation_name = "ANEURALNETWORKS_RESIZE_BILINEAR"; break;
    case 24:	operation_name = "ANEURALNETWORKS_RNN"; break;
    case 25:	operation_name = "ANEURALNETOWORKS_SOFTMAX"; break;
    case 26:	operation_name = "ANEURALNETWORKS_SPACE_TO_DEPTH"; break;
    case 27:	operation_name = "ANEURALNETWORKS_SVDF"; break;
    case 28:	operation_name = "ANEURALNETWORKS_TANH"; break;
    default: operation_name = "Not a operation"; break;
  }
  return operation_name;
}

//isOperationSupported() function

bool VpuPreparedModel::isOperationSupported(const Operation& operation, const Model& model)
{

  VLOG(DRIVER) << "Check for Operation support on VPU:  " << getOperationName(operation.type);

#define VLOG_CHECKFAIL(fail) ALOGD("Check failed: %s", fail)

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
#else
    for (auto i : operation.inputs) {
        const auto input = model.operands[i];
        if (input.type == OperandType::TENSOR_QUANT8_ASYMM &&
            input.zeroPoint != 0) {
            VLOG_CHECKFAIL("input quant");
            return false;
        }getOperandConstVal
    }
    for (auto i : operation.outputs) {
        const auto output = model.operands[i];
        if (output.type == OperandType::TENSOR_QUANT8_ASYMM &&
            output.zeroPoint != 0) {
            VLOG_CHECKFAIL("output quant");
            return false;
        }
    }
#endif

    switch(operation.type) {

        case OperationType::RELU:
        {
          VLOG(DRIVER) << "RELU is supported operation ";
          break;
        }
        case OperationType::RELU1:
        {
          VLOG(DRIVER) << "RELU1 is supported operation ";
          break;
        }
        case OperationType::RELU6:
        {
          VLOG(DRIVER) << "RELU6 is supported operation ";
          break;
        }
        case OperationType::TANH:
        {
          VLOG(DRIVER) << "TANH is supported operation ";
          break;
        }
        case OperationType::LOGISTIC:
        {
          VLOG(DRIVER) << "LOGISTIC is supported operation ";
          break;
        }
        case OperationType::CONV_2D:
        {
          VLOG(DRIVER) << "CONV_2D is supported operation ";
          break;
        }
        case OperationType::DEPTHWISE_CONV_2D:
        {
          VLOG(DRIVER) << "DEPTHWISE_CONV_2D is supported operation ";
          break;
        }
        case OperationType::AVERAGE_POOL_2D:
        {
          VLOG(DRIVER) << "AVERAGE_POOL_2D is supported operation ";
          break;
        }
        case OperationType::L2_POOL_2D:
        {
          VLOG(DRIVER) << "L2_POOL_2D is supported operation ";
          break;
        }
        case OperationType::MAX_POOL_2D:
        {
          VLOG(DRIVER) << "MAX_POOL_2D is supported operation ";
          break;
        }
        case OperationType::SOFTMAX:
        {
          VLOG(DRIVER) << "SOFTMAX is supported operation ";
          break;
        }/*
        case OperationType::FULLY_CONNECTED:
        {
          VLOG(DRIVER) << "FULLY_CONNECTED is supported operation ";
          break;
        }
        case OperationType::L2_NORMALIZATION:
        {
          VLOG(DRIVER) << "L2_NORMALIZATION is supported operation ";
          break;
        }*/
        case OperationType::RESHAPE:
        {
          VLOG(DRIVER) << "RESHAPE is supported operation "; //ANEURALNETWOKRS_RESHAPE
          break;
        }

        default:
           VLOG(DRIVER) << getOperationName(operation.type) << " Operation not supported on VPU";
           return false;
    }

    return true;
}


// validOperands() function

static bool validOperands(const hidl_vec<Operand>& operands, const hidl_vec<uint8_t>& operandValues,
                          size_t poolCount) {
    for (auto& operand : operands) {
        if (!validCode(kNumberOfDataTypes, kNumberOfDataTypesOEM,
                       static_cast<uint32_t>(operand.type))) {
                         //ALOGE("Invalid operand type: %s",operand.type);
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
                  //ALOGE("OperandValue location out of range.  Starts at %d, length %d, max %d", operand.location.offset, operand.location.length, operandValues.size());
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
                    //ALOGE("Unexpected offset %d, or length %d for runtime location.", operand.location.offset, operand.location.length);
                    LOG(ERROR) << "Unexpected offset " << operand.location.offset << " or length "
                               << operand.location.length << " for runtime location.";
                    return false;
                }
                break;
            case OperandLifeTime::CONSTANT_REFERENCE:
                if (operand.location.poolIndex >= poolCount) {
                  //ALOGE("Invalid poolIndex %d poolCount %d", operand.location.poolIndex, poolCount);
                    LOG(ERROR) << "Invalid poolIndex " << operand.location.poolIndex << "/"
                               << poolCount;
                    return false;
                }
                break;
            // TODO: Validate that we are within the pool.
            default:
                //ALOGE("Invalid lifetime");
                LOG(ERROR) << "Invalid lifetime";
                return false;
        }
    }
    return true;
}

// validOperandIndexes() function

static bool validOperandIndexes(const hidl_vec<uint32_t> indexes, size_t operandCount) {
    for (uint32_t i : indexes) {
        if (i >= operandCount) {
            //ALOGE("Index out of range %d / %d",i,operandCount);
            LOG(ERROR) << "Index out of range " << i << "/" << operandCount;
            return false;
        }
    }
    return true;
}

// validOperations() function

static bool validOperations(const hidl_vec<Operation>& operations, size_t operandCount) {
    for (auto& op : operations) {
        if (!validCode(kNumberOfOperationTypes, kNumberOfOperationTypesOEM,
                       static_cast<uint32_t>(op.type))) {
            //ALOGE("Invalid operation type %s", op.type);
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


// validModel() function
bool VpuPreparedModel::validModel(const Model& model)
{
  const size_t operandCount = model.operands.size();

  return (validOperands(model.operands, model.operandValues, model.pools.size()) &&
        validOperations(model.operations, operandCount) &&
        validOperandIndexes(model.inputIndexes, operandCount) &&
        validOperandIndexes(model.outputIndexes, operandCount));
}

/*
// validRequestArguments() function
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
}*/


// TODO doublecheck
// validateRequest() function
/*
bool validateRequest(const Request& request, const Model& model) {
    const size_t poolCount = request.pools.size();
    return (validRequestArguments(request.inputs, model.inputIndexes, model.operands, poolCount,
                                  "input") &&
            validRequestArguments(request.outputs, model.outputIndexes, model.operands, poolCount,
                                  "output"));
}
*/

// execute() function

Return<ErrorStatus> VpuPreparedModel::execute(const Request& request,
                                                 const sp<IExecutionCallback>& callback) {

        ALOGD("Begin to execute on VPU");

        if (callback.get() == nullptr) {
            ALOGE("invalid callback passed to execute");
            return ErrorStatus::INVALID_ARGUMENT;
        }

        if (!validateRequest(request, mModel)) {
            callback->notify(ErrorStatus::INVALID_ARGUMENT);
            return ErrorStatus::INVALID_ARGUMENT;
        }

        // This thread is intentionally detached because the sample driver service
        // is expected to live forever.
        std::thread([this, request, callback]{ asyncExecute(request, callback); }).detach();

        ALOGD("Start execute thread done on VPU");
        return ErrorStatus::NONE;

}
//deinitialize() function
void VpuPreparedModel::deinitialize()
{

    ALOGD("deinitialize");
    /*
    for (const auto& operand : mOperands) {
        for (const auto& pmem : operand.stub_pmems) {
            VLOG(L1, "free stub pmems %p of operand %p", pmem, &operand);
            delete pmem;
        }
        VLOG(L1, "free pmems %p of operand %p", operand.pmem, &operand);
        if (operand.pmem)
            delete operand.pmem;
    }
    VLOG(L1, "free cpu engine");
    */
}

void VpuPreparedModel::asyncExecute(const Request& request,
                                       const sp<IExecutionCallback>& callback) {
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        callback->notify(ErrorStatus::GENERAL_FAILURE);
        return;
    }

    VpuExecutor executor;
    int n = executor.run(mModel, request, mPoolInfos, requestPoolInfos);

    VLOG(DRIVER) << "executor.run returned " << n;
    ErrorStatus executionStatus =
            n == ANEURALNETWORKS_NO_ERROR ? ErrorStatus::NONE : ErrorStatus::GENERAL_FAILURE;
    Return<void> returned = callback->notify(executionStatus);
    if (!returned.isOk()) {
        LOG(ERROR) << " hidl callback failed to return properly: " << returned.description();
    }
}

}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
