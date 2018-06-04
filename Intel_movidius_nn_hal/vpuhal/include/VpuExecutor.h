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

#ifndef ANDROID_ML_NN_HAL_VPU_EXECUTOR_H
#define ANDROID_ML_NN_HAL_VPU_EXECUTOR_H


#include "HalInterfaces.h"

//TODO check this files needed or not
#include "VpuOperationsUtils.h"
#include "VpuUtils.h"
#include "Blob.h"


#include <algorithm>
#include <vector>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo {
    // TODO Storing the type here is redundant, as it won't change during execution.
    OperandType type;
    // The type and dimensions of the operand.  The dimensions can
    // change at runtime.  We include the type because it's useful
    // to pass together with the dimension to the functions implementing
    // the operators.
    std::vector<uint32_t> dimensions;

    float scale;
    int32_t zeroPoint;
    // Where the operand's data is stored.  Check the corresponding
    // location information in the model to figure out if this points
    // to memory we have allocated for an temporary operand.
    uint8_t* buffer;
    // The length of the buffer.
    uint32_t length;
    // Whether this is a temporary variable, a model input, a constant, etc.
    OperandLifeTime lifetime;
    // Keeps track of how many operations have yet to make use
    // of this temporary variable.  When the count is decremented to 0,
    // we free the buffer.  For non-temporary variables, this count is
    // always 0.
    uint32_t numberOfUsesLeft;

    Shape shape() const {
        return Shape{.type = type, .dimensions = dimensions, .scale = scale, .offset = zeroPoint};
    }
};

// Used to keep a pointer to each of the memory pools.
struct RunTimePoolInfo {
    sp<IMemory> memory;
    hidl_memory hidlMemory;
    uint8_t* buffer;

    bool set(const hidl_memory& hidlMemory);
    bool update();
};

typedef std::vector<OperationType> Oertaion_vector;

bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools);


// This class is used to execute a model on the VPU.
class VpuExecutor { // VpuExecutor
public:
    // Executes the model. The results will be stored at the locations
    // specified in the constructor.
    // The model must outlive the executor.  We prevent it from being modified
    // while this is executing.

    int run(const Model& model, const Request& request,
            const std::vector<RunTimePoolInfo>& modelPoolInfos,
            const std::vector<RunTimePoolInfo>& requestPoolInfos);

private:

    bool initializeRunTimeInfo(const std::vector<RunTimePoolInfo>& modelPoolInfos,
                               const std::vector<RunTimePoolInfo>& requestPoolInfos);
    // Runs one operation of the graph.
    int executeOperation(const Operation& entry);
    Operation_inputs_info get_operation_operands_info(const Operation& operation);

    // Decrement the usage count for the operands listed.  Frees the memory
    // allocated for any temporary variable with a count of zero.
    void freeNoLongerUsedOperands(const std::vector<uint32_t>& inputs);

    // The model and the request that we'll execute. Only valid while run()
    // is being executed.
    const Model* mModel = nullptr;
    const Request* mRequest = nullptr;
    float *kernel_data_buffer;

    std::vector<RunTimeOperandInfo> mOperands;
};

namespace {

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
  // TODO: Check buffer is at least as long as size of data.
  T* data = reinterpret_cast<T*>(info.buffer);
  return data[0];
}

inline bool IsNullInput(const RunTimeOperandInfo *input) {
    return input->lifetime == OperandLifeTime::NO_VALUE;
}

inline int NumInputsWithValues(const Operation &operation,
                               std::vector<RunTimeOperandInfo> &operands) {
  const std::vector<uint32_t> &inputs = operation.inputs;
  return std::count_if(inputs.begin(), inputs.end(),
                       [&operands](uint32_t i) {
                         return !IsNullInput(&operands[i]);
                       });
}

inline int NumOutputs(const Operation &operation) {
  return operation.outputs.size();
}

inline size_t NumDimensions(const RunTimeOperandInfo *operand) {
  return operand->shape().dimensions.size();
}

inline uint32_t SizeOfDimension(const RunTimeOperandInfo *operand, int i) {
  return operand->shape().dimensions[i];
}

inline RunTimeOperandInfo *GetInput(const Operation &operation,
                                    std::vector<RunTimeOperandInfo> &operands,
                                    int index) {
  return &operands[operation.inputs[index]];
}

inline RunTimeOperandInfo *GetOutput(const Operation &operation,
                                     std::vector<RunTimeOperandInfo> &operands,
                                     int index) {
  return &operands[operation.outputs[index]];
}

}  // anonymous namespace


}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif // ANDROID_ML_NN_HAL_VPU_EXECUTOR_H
