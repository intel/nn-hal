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

#define LOG_TAG "VpuExecutor"

#include "VpuExecutor.h"
#include <log/log.h>
#include "Blob.h"
#include <stdio.h>
#include "ncs_lib.h"


#include "VpuOperations.h"

#include "NeuralNetworks.h"


#include <sys/mman.h>

#define file_dump false

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

/*

RunTimePoolInfo::set()

RunTimePoolInfo::update()

setRunTimePoolInfosFromHidlMemories()

setInfoAndAllocateIfNeeded()

VpuExecutor

VpuExecutor::run()

VpuExecutor::initializeRunTimeInfo()

VpuExecutor::freeNoLongerUsedOperands()

*/

  // TODO: short term, make share memory mapping and updating a utility function.
  // TODO: long term, implement mmap_fd as a hidl IMemory service.
  bool RunTimePoolInfo::set(const hidl_memory& hidlMemory) {
      this->hidlMemory = hidlMemory;
      auto memType = hidlMemory.name();
      if (memType == "ashmem") {
          memory = mapMemory(hidlMemory);
          if (memory == nullptr) {
              LOG(ERROR) << "Can't map shared memory.";
              return false;
          }
          memory->update();
          buffer = reinterpret_cast<uint8_t*>(static_cast<void*>(memory->getPointer()));
          if (buffer == nullptr) {
              LOG(ERROR) << "Can't access shared memory.";
              return false;
          }
          return true;
      } else if (memType == "mmap_fd") {
          size_t size = hidlMemory.size();
          int fd = hidlMemory.handle()->data[0];
          int prot = hidlMemory.handle()->data[1];
          size_t offset = getSizeFromInts(hidlMemory.handle()->data[2],
                                          hidlMemory.handle()->data[3]);
          buffer = static_cast<uint8_t*>(mmap(nullptr, size, prot, MAP_SHARED, fd, offset));
          if (buffer == MAP_FAILED) {
              LOG(ERROR) << "Can't mmap the file descriptor.";
              return false;
          }
          return true;
      } else {
          LOG(ERROR) << "unsupported hidl_memory type";
          return false;
      }
  }

  // Making sure the output data are correctly updated after execution.
bool RunTimePoolInfo::update() {
    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory->commit();
        return true;
    } else if (memType == "mmap_fd") {
        int prot = hidlMemory.handle()->data[1];
        if (prot & PROT_WRITE) {
            size_t size = hidlMemory.size();
            return msync(buffer, size, MS_SYNC) == 0;
        }
    }
    // No-op for other types of memory.
    return true;
}

bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools) {
    poolInfos->resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = (*poolInfos)[i];
        if (!poolInfo.set(pools[i])) {
            LOG(ERROR) << "Could not map pool";
            return false;
        }
    }
    return true;
}


// Updates the RunTimeOperandInfo with the newly calculated shape.
// Allocate the buffer if we need to.
static bool setInfoAndAllocateIfNeeded(RunTimeOperandInfo* info, const Shape& shape) {
    // For user-provided model output operands, the parameters must match the Shape
    // calculated from the preparation step.
    if (info->lifetime == OperandLifeTime::MODEL_OUTPUT) {
        if (info->type != shape.type ||
            info->dimensions != shape.dimensions) {
            LOG(ERROR) << "Invalid type or dimensions for model output";
            return false;
        }
        if (info->type == OperandType::TENSOR_QUANT8_ASYMM &&
            (info->scale != shape.scale || info->zeroPoint != shape.offset)) {
            LOG(ERROR) << "Invalid scale or zeroPoint for model output";
            return false;
        }
    }
    info->type = shape.type;
    info->dimensions = shape.dimensions;
    info->scale = shape.scale;
    info->zeroPoint = shape.offset;
    if (info->lifetime == OperandLifeTime::TEMPORARY_VARIABLE && info->buffer == nullptr) {
        uint32_t length = sizeOfData(info->type, info->dimensions);
        info->buffer = new uint8_t[length];
        if (info->buffer == nullptr) {
            return false;
        }
    }
    return true;
}


// Ignore the .pools entry in model and request.  This will have been taken care of
// by the caller.
int VpuExecutor::run(const Model& model, const Request& request,
                     const std::vector<RunTimePoolInfo>& modelPoolInfos,
                     const std::vector<RunTimePoolInfo>& requestPoolInfos) {
    VLOG(VPUEXE) << "VpuExecutor::run()";
    VLOG(VPUEXE) << "model: " << toString(model);
    VLOG(VPUEXE) << "request: " << toString(request);

    mModel = &model;
    mRequest = &request; // TODO check if mRequest is needed

    initializeRunTimeInfo(modelPoolInfos, requestPoolInfos);

    Oertaion_vector nn_ops_vectors;
    for (const auto& operation : model.operations) {
      nn_ops_vectors.push_back(operation.type);
    }

    const hidl_vec<uint32_t>& network_inputs = model.operations[0].inputs;
    const RunTimeOperandInfo& network_input = mOperands[network_inputs[0]];
    Shape nw_input_shape = network_input.shape();
    uint32_t input_num_elements = getNumberOfElements(nw_input_shape);
    VLOG(VPUEXE) << "Input Num of Elements: " << input_num_elements;

    float *network_input_buffer;
    network_input_buffer = (float *)malloc(sizeof(float) * input_num_elements);
    if(network_input_buffer == NULL)
    LOG(ERROR) << "Unable to allocate network_input_buffer";
    memset(network_input_buffer, 0, sizeof(float) * input_num_elements);
    memcpy(network_input_buffer,reinterpret_cast<float*>(network_input.buffer), sizeof(float) * input_num_elements);

    const hidl_vec<uint32_t>& network_outputs = model.operations[nn_ops_vectors.size()-1].outputs;
    RunTimeOperandInfo& network_output = mOperands[network_outputs[0]];
    Shape nw_output_shape = network_output.shape();
    uint32_t output_num_elements = getNumberOfElements(nw_output_shape);
    float *network_output_buffer;
    network_output_buffer = (float *)malloc(sizeof(float) * output_num_elements);
    if(network_output_buffer == NULL)
    LOG(ERROR) << "Unable to allocate network_output_buffer";
    memset(network_output_buffer,0,sizeof(float) * output_num_elements);

    VLOG(VPUEXE) << "Output Num of Elements: " << output_num_elements;

    VLOG(VPUEXE) << "Got the input data request Starting to execute on VPU!";

    int val = ncs_execute((float*)network_input_buffer,input_num_elements,network_output_buffer, output_num_elements);

    if(val != 0)
      return ANEURALNETWORKS_OP_FAILED;


    memcpy(reinterpret_cast<float*>(network_output.buffer),network_output_buffer,output_num_elements*sizeof(float));
    VLOG(VPUEXE) << "Got the output result fro VPU!";

    free(network_input_buffer);
    free(network_output_buffer);
    nn_ops_vectors.clear();

    

    for (auto runtimeInfo : modelPoolInfos) {
        runtimeInfo.update();
    }

    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

    mModel = nullptr;
    mRequest = nullptr;
    VLOG(VPUEXE) << "Completed run normally";

    return ANEURALNETWORKS_NO_ERROR;
}


bool VpuExecutor::initializeRunTimeInfo(const std::vector<RunTimePoolInfo>& modelPoolInfos,
                                        const std::vector<RunTimePoolInfo>& requestPoolInfos) {
    VLOG(VPUEXE) << "VpuExecutor::initializeRunTimeInfo";
    const size_t count = mModel->operands.size();
    mOperands.resize(count);

    // Start by setting the runtime info to what's in the model.
    for (size_t i = 0; i < count; i++) {
        const Operand& from = mModel->operands[i];
        RunTimeOperandInfo& to = mOperands[i];
        to.type = from.type;
        to.dimensions = from.dimensions;
        to.scale = from.scale;
        to.zeroPoint = from.zeroPoint;
        to.length = from.location.length;
        to.lifetime = from.lifetime;
        switch (from.lifetime) {
            case OperandLifeTime::TEMPORARY_VARIABLE:
                to.buffer = nullptr;
                to.numberOfUsesLeft = from.numberOfConsumers;
                break;
            case OperandLifeTime::CONSTANT_COPY:
                to.buffer = const_cast<uint8_t*>(&mModel->operandValues[from.location.offset]);
                to.numberOfUsesLeft = 0;
                break;
            case OperandLifeTime::CONSTANT_REFERENCE: {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < modelPoolInfos.size());
                auto& r = modelPoolInfos[poolIndex];
                to.buffer = r.buffer + from.location.offset;
                to.numberOfUsesLeft = 0;
                break;
            }
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                to.buffer = nullptr;
                to.numberOfUsesLeft = 0;
                break;
            default:
                nnAssert(false);
                break;
        }
    }

    // Adjust the runtime info for the arguments passed to the model,
    // modifying the buffer location, and possibly the dimensions.
    auto updateForArguments = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                                  const hidl_vec<RequestArgument>& arguments) {
        nnAssert(indexes.size() == arguments.size());
        for (size_t i = 0; i < indexes.size(); i++) {
            const uint32_t operandIndex = indexes[i];
            const RequestArgument& from = arguments[i];
            RunTimeOperandInfo& to = mOperands[operandIndex];
            if (from.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // from.dimensions only modifies the dimensions that were
                // unspecified in the model.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                // TODO make sure that's the case for the default CPU path.
                to.dimensions = from.dimensions;
            }
            if (from.hasNoValue) {
                to.lifetime = OperandLifeTime::NO_VALUE;
                nnAssert(to.buffer == nullptr);
            } else {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < requestPoolInfos.size());
                auto& r = requestPoolInfos[poolIndex];
                to.buffer = r.buffer + from.location.offset;
            }
        }
    };
    updateForArguments(mModel->inputIndexes, mRequest->inputs);
    updateForArguments(mModel->outputIndexes, mRequest->outputs);

    return true;
}

void VpuExecutor::freeNoLongerUsedOperands(const std::vector<uint32_t>& inputs) {

    for (uint32_t i : inputs) {
        auto& info = mOperands[i];
        // Check if it's a static or model input/output.
        if (info.numberOfUsesLeft == 0) {
            continue;
        }
        info.numberOfUsesLeft--;
        if (info.numberOfUsesLeft == 0) {
            nnAssert(info.buffer != nullptr);
            delete[] info.buffer;
            info.buffer = nullptr;
        }
    }
}



}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
