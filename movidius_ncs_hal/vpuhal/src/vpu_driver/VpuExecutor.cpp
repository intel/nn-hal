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



#define LOG_TAG "VpuExecutor"

#include "VpuExecutor.h"
#include <log/log.h>
#include "Blob.h"
#include "vpu_lib.h"
//TODO check this file is required or not

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

VpuExecutor::executeOperation()


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

Operation_inputs_info VpuExecutor::get_operation_operands_info(const Operation& operation){
  Operation_inputs_info stage_info;
  const hidl_vec<uint32_t>& ins = operation.inputs;
  const hidl_vec<uint32_t>& outs = operation.outputs;
  bool success = false;


  auto allParametersPresent = [&operation, &ins, &outs, this](size_t requiredIns,
                                                                size_t requiredOuts) -> bool {
        auto verify = [&operation, this](size_t requiredCount, const hidl_vec<uint32_t>& indexes,
                          const char* type) -> bool {
            size_t actualCount = indexes.size();
            if (actualCount != requiredCount) {
                LOG(ERROR) << getOperationName(operation.type)
                           << ": Invalid number of " << type << " operands. Got " << actualCount
                           << " of " << requiredCount;
                return false;
            }
            for (size_t i = 0; i < actualCount; i++) {
                if (mOperands[indexes[i]].lifetime == OperandLifeTime::NO_VALUE) {
                    LOG(ERROR) << getOperationName(operation.type) << " " << type
                               << " operand " << i << " is required but missing.";
                    return false;
                }
            }
            return true;
        };
        return verify(requiredIns, ins, "in") && verify(requiredOuts, outs, "out");
    };
  switch (operation.type) { //switch start
    case OperationType::RELU6:{
      VLOG(VPUEXE) << toString(operation);
      /*
      if (!allParametersPresent(1, 1)) {
          return ANEURALNETWORKS_BAD_DATA;
      }*/
      const RunTimeOperandInfo& input = mOperands[ins[0]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      Shape inputShape = input.shape();
      Shape outShape = output.shape();
      if (input.type == OperandType::TENSOR_FLOAT32){
        success = genericActivationPrepare(input.shape(), &outShape);
        if(!success)
            nnAssert(false);
      }

      stage_info.main_operation = RELU6;
      stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
      stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
      stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
      stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

      stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
      stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
      stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
      stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);

      stage_info.post_operation = NONE; break;
    } break;
    case OperationType::RELU1:{
      VLOG(VPUEXE) << toString(operation);
      /*
      if (!allParametersPresent(1, 1)) {
          return ANEURALNETWORKS_BAD_DATA;
      }*/
      const RunTimeOperandInfo& input = mOperands[ins[0]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      Shape inputShape = input.shape();
      Shape outShape = output.shape();
      if (input.type == OperandType::TENSOR_FLOAT32){
        success = genericActivationPrepare(input.shape(), &outShape);
        if(!success)
            nnAssert(false);
      }

      stage_info.main_operation = RELU1;
      stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
      stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
      stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
      stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

      stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
      stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
      stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
      stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);

      stage_info.post_operation = NONE; break;
    } break;
    case OperationType::RELU:{
      VLOG(VPUEXE) << toString(operation);
      /*
      if (!allParametersPresent(1, 1)) {
          return ANEURALNETWORKS_BAD_DATA;
      }*/
      const RunTimeOperandInfo& input = mOperands[ins[0]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      Shape inputShape = input.shape();
      Shape outShape = output.shape();
      if (input.type == OperandType::TENSOR_FLOAT32){
        success = genericActivationPrepare(input.shape(), &outShape);
        if(!success)
            nnAssert(false);
      }

      stage_info.main_operation = RELU;
      stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
      stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
      stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
      stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

      stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
      stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
      stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
      stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);

      stage_info.post_operation = NONE; break;
    } break;
    case OperationType::LOGISTIC:{
      VLOG(VPUEXE) << toString(operation);
      /*
      if (!allParametersPresent(1, 1)) {
          return ANEURALNETWORKS_BAD_DATA;
      }*/
      const RunTimeOperandInfo& input = mOperands[ins[0]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      Shape inputShape = input.shape();
      Shape outShape = output.shape();
      if (input.type == OperandType::TENSOR_FLOAT32){
        success = genericActivationPrepare(input.shape(), &outShape);
        if(!success)
            nnAssert(false);
      }

      stage_info.main_operation = LOGISTIC;
      stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
      stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
      stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
      stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

      stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
      stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
      stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
      stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);

      stage_info.post_operation = NONE;
    } break;
    case OperationType::TANH:{
      VLOG(VPUEXE) << toString(operation);
      /*
      if (!allParametersPresent(1, 1)) {
          return ANEURALNETWORKS_BAD_DATA;
      }*/
      const RunTimeOperandInfo& input = mOperands[ins[0]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      Shape inputShape = input.shape();
      Shape outShape = output.shape();
      if (input.type == OperandType::TENSOR_FLOAT32){
        success = genericActivationPrepare(input.shape(), &outShape);
        if(!success)
            nnAssert(false);
      }

      stage_info.main_operation = TANH;
      stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
      stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
      stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
      stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

      stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
      stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
      stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
      stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);

      stage_info.post_operation = NONE; break;
    } break;
    case OperationType::CONV_2D: {
      VLOG(VPUEXE) << toString(operation);
      const size_t inCount = ins.size();
      /*
      if ((inCount != 10 && inCount != 7) ||
                    !allParametersPresent(inCount, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }*/
      const RunTimeOperandInfo& input  = mOperands[ins[0]];
      const RunTimeOperandInfo& filter = mOperands[ins[1]];
      const RunTimeOperandInfo& bias  = mOperands[ins[2]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      int32_t padding_left, padding_right;
      int32_t padding_top, padding_bottom;
      int32_t stride_width, stride_height;
      int32_t activation;
      Shape inputShape = input.shape();
      Shape filterShape = filter.shape();
      Shape biasShape = bias.shape();
      Shape outShape = output.shape();

      if (inCount == 10) {
            padding_left     = getScalarData<int32_t>(mOperands[ins[3]]);
            padding_right    = getScalarData<int32_t>(mOperands[ins[4]]);
            padding_top      = getScalarData<int32_t>(mOperands[ins[5]]);
            padding_bottom   = getScalarData<int32_t>(mOperands[ins[6]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[7]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[8]]);
            activation       = getScalarData<int32_t>(mOperands[ins[9]]);
          }
      else {
        int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[3]]);
        stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
        stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
        activation       = getScalarData<int32_t>(mOperands[ins[6]]);



        int32_t input_width  = getSizeOfDimension(inputShape, 2);
        int32_t input_height = getSizeOfDimension(inputShape, 1);
        int32_t filter_width  = getSizeOfDimension(filterShape, 2);
        int32_t filter_height = getSizeOfDimension(filterShape, 1);

        calculateExplicitPadding(input_width, stride_width,
                             filter_width, padding_implicit,
                             &padding_left, &padding_right);
        calculateExplicitPadding(input_height, stride_height,
                             filter_height, padding_implicit,
                             &padding_top, &padding_bottom);
     }
     if (input.type == OperandType::TENSOR_FLOAT32){
       success = convPrepare(input.shape(), filter.shape(), bias.shape(),
                             padding_left, padding_right,
                             padding_top, padding_bottom,
                             stride_width, stride_height,
                             &outShape);
        if(!success)
            nnAssert(false);
     }
     stage_info.main_operation = CONV_2D;
     stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
     stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
     stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
     stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

     stage_info.kernel_shape[0] = getSizeOfDimension(filterShape, 1);
     stage_info.kernel_shape[1] = getSizeOfDimension(filterShape, 2);
     stage_info.kernel_shape[2] = getSizeOfDimension(filterShape, 3);
     stage_info.kernel_shape[3] = getSizeOfDimension(filterShape, 0);

     stage_info.bias_shape[0] = getSizeOfDimension(biasShape, 0);
     stage_info.bias_shape[1] = getSizeOfDimension(biasShape, 1);
     stage_info.bias_shape[2] = getSizeOfDimension(biasShape, 2);
     stage_info.bias_shape[3] = getSizeOfDimension(biasShape, 3);

     stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
     stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
     stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
     stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);

     stage_info.kernel_buffer = reinterpret_cast<float*>(filter.buffer);
     stage_info.bias_buffer = reinterpret_cast<float*>(bias.buffer);

     stage_info.stride_width = stride_width;
     stage_info.stride_height = stride_height;

     stage_info.padding_left = padding_left;
     stage_info.padding_right = padding_right;
     stage_info.padding_top = padding_top;
     stage_info.padding_bottom = padding_bottom;

     switch (activation) {
       case 0: stage_info.post_operation = NONE; break;
       case 1: stage_info.post_operation = RELU; break;
       case 2: stage_info.post_operation = RELU1; break;
       case 3: stage_info.post_operation = RELU6; break;
       default: stage_info.post_operation = NONE; break;
     }
     stage_info.kernel_data = true;
     stage_info.bias_data = true;
     stage_info.op_params_data = false;

#if file_dump
     //TODO temporary fix
     stage_info.kernel_shape[0] = (stage_info.kernel_shape[0] == 0 ) ? 1: stage_info.kernel_shape[0];
     stage_info.kernel_shape[1] = (stage_info.kernel_shape[1] == 0 ) ? 1: stage_info.kernel_shape[1];
     stage_info.kernel_shape[2] = (stage_info.kernel_shape[2] == 0 ) ? 1: stage_info.kernel_shape[2];
     stage_info.kernel_shape[3] = (stage_info.kernel_shape[3] == 0 ) ? 1: stage_info.kernel_shape[3];
     uint32_t nk_ele = stage_info.kernel_shape[0] * stage_info.kernel_shape[1] * stage_info.kernel_shape[2] * stage_info.kernel_shape[3];

     stage_info.bias_shape[0] = (stage_info.bias_shape[0] == 0 ) ? 1: stage_info.bias_shape[0];
     stage_info.bias_shape[1] = (stage_info.bias_shape[1] == 0 ) ? 1: stage_info.bias_shape[1];
     stage_info.bias_shape[2] = (stage_info.bias_shape[2] == 0 ) ? 1: stage_info.bias_shape[2];
     stage_info.bias_shape[3] = (stage_info.bias_shape[3] == 0 ) ? 1: stage_info.bias_shape[3];
     uint32_t nb_ele = stage_info.bias_shape[0] * stage_info.bias_shape[1] * stage_info.bias_shape[2] * stage_info.bias_shape[3];

     FILE *fp;

     if(stage_info.kernel_data){
       fp = fopen("/data/ncs_graph_data","ab+");
       if(!fp) ALOGE("unable to open the file /data/ncs_graph_data ");
       VLOG(VPUEXE) << "CONV_2D nk_ele: " << nk_ele;
       fseek(fp, 0, SEEK_END);
       fwrite(reinterpret_cast<const float*>(filter.buffer),sizeof(float),nk_ele,fp);
       fclose(fp);
     }

     if(stage_info.bias_data){
       fp = fopen("/data/ncs_graph_data","ab+");
       if(!fp) ALOGE("unable to open the file /data/ncs_graph_data ");
       VLOG(VPUEXE) << "CONV_2D nb_ele: " << nb_ele;
       fseek(fp, 0, SEEK_END);
       fwrite(reinterpret_cast<const float*>(bias.buffer),sizeof(float),nb_ele,fp);
       fclose(fp);
     }
#endif
     bool DEBUG_CONV_2D = false;
     //DEBUG_CONV_2D = true;  //un comment this line to get CONV_2D layer debug data
     if(DEBUG_CONV_2D){

       VLOG(VPUEXE) << " CONV_2D padding_left: " << padding_left;
       VLOG(VPUEXE) << "CONV_2D padding_right: " << padding_right;
       VLOG(VPUEXE) << " CONV_2D padding_top: " << padding_top;
       VLOG(VPUEXE) << "CONV_2D padding_bottom: " << padding_bottom;

       VLOG(VPUEXE) << "CONV_2D input_shape[0]: " << stage_info.input_shape[0];
       VLOG(VPUEXE) << "CONV_2D input_shape[1]: " << stage_info.input_shape[1];
       VLOG(VPUEXE) << "CONV_2D input_shape[2]: " << stage_info.input_shape[2];
       VLOG(VPUEXE) << "CONV_2D input_shape[3]: " << stage_info.input_shape[3];

       VLOG(VPUEXE) << "CONV_2D kernel_shape[0]: " << stage_info.kernel_shape[0];
       VLOG(VPUEXE) << "CONV_2D kernel_shape[1]: " << stage_info.kernel_shape[1];
       VLOG(VPUEXE) << "CONV_2D kernel_shape[2]: " << stage_info.kernel_shape[2];
       VLOG(VPUEXE) << "CONV_2D kernel_shape[3]: " << stage_info.kernel_shape[3];

       VLOG(VPUEXE) << "CONV_2D bias_shape[0]: " << stage_info.bias_shape[0];
       VLOG(VPUEXE) << "CONV_2D bias_shape[1]: " << stage_info.bias_shape[1];
       VLOG(VPUEXE) << "CONV_2D bias_shape[2]: " << stage_info.bias_shape[2];
       VLOG(VPUEXE) << "CONV_2D bias_shape[3]: " << stage_info.bias_shape[3];

       VLOG(VPUEXE) << "CONV_2D output_shape[0]: " << stage_info.output_shape[0];
       VLOG(VPUEXE) << "CONV_2D output_shape[1]: " << stage_info.output_shape[1];
       VLOG(VPUEXE) << "CONV_2D output_shape[2]: " << stage_info.output_shape[2];
       VLOG(VPUEXE) << "CONV_2D output_shape[3]: " << stage_info.output_shape[3];
     }
    } break;//CONV_2D end

    case OperationType::DEPTHWISE_CONV_2D: {
      VLOG(VPUEXE) << toString(operation);
      const size_t inCount = ins.size();
      /*
      if ((inCount != 11 && inCount != 8) ||
                    !allParametersPresent(inCount, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }*/
      const RunTimeOperandInfo& input  = mOperands[ins[0]];
      const RunTimeOperandInfo& filter = mOperands[ins[1]];
      const RunTimeOperandInfo& bias   = mOperands[ins[2]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      int32_t padding_left, padding_right;
      int32_t padding_top, padding_bottom;
      int32_t stride_width, stride_height;
      int32_t depth_multiplier;
      int32_t activation;
      Shape inputShape = input.shape();
      Shape filterShape = filter.shape();
      Shape biasShape = bias.shape();
      Shape outShape = output.shape();

      if (inCount == 11) {
            padding_left     = getScalarData<int32_t>(mOperands[ins[3]]);
            padding_right    = getScalarData<int32_t>(mOperands[ins[4]]);
            padding_top      = getScalarData<int32_t>(mOperands[ins[5]]);
            padding_bottom   = getScalarData<int32_t>(mOperands[ins[6]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[7]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[8]]);
            depth_multiplier = getScalarData<int32_t>(mOperands[ins[9]]);
            activation       = getScalarData<int32_t>(mOperands[ins[10]]);
          }
      else {
        int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[3]]);
        stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
        stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
        depth_multiplier = getScalarData<int32_t>(mOperands[ins[6]]);
        activation       = getScalarData<int32_t>(mOperands[ins[7]]);



        int32_t input_width  = getSizeOfDimension(inputShape, 2);
        int32_t input_height = getSizeOfDimension(inputShape, 1);
        int32_t filter_width  = getSizeOfDimension(filterShape, 2);
        int32_t filter_height = getSizeOfDimension(filterShape, 1);

        calculateExplicitPadding(input_width, stride_width,
                             filter_width, padding_implicit,
                             &padding_left, &padding_right);
        calculateExplicitPadding(input_height, stride_height,
                             filter_height, padding_implicit,
                             &padding_top, &padding_bottom);
     }
     if (input.type == OperandType::TENSOR_FLOAT32){
        success = depthwiseConvPrepare(input.shape(), filter.shape(), bias.shape(),
                                      padding_left, padding_right,
                                      padding_top, padding_bottom,
                                      stride_width, stride_height,
                                      &outShape);
        if(!success)
            nnAssert(false);
     }
     stage_info.main_operation = DEPTHWISE_CONV_2D;
     stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
     stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
     stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
     stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

     stage_info.kernel_shape[0] = getSizeOfDimension(filterShape, 1);
     stage_info.kernel_shape[1] = getSizeOfDimension(filterShape, 2);
     stage_info.kernel_shape[2] = getSizeOfDimension(filterShape, 3);
     stage_info.kernel_shape[3] = getSizeOfDimension(filterShape, 0);

     stage_info.bias_shape[0] = getSizeOfDimension(biasShape, 0);
     stage_info.bias_shape[1] = getSizeOfDimension(biasShape, 1);
     stage_info.bias_shape[2] = getSizeOfDimension(biasShape, 2);
     stage_info.bias_shape[3] = getSizeOfDimension(biasShape, 3);

     stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
     stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
     stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
     stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);

     stage_info.kernel_buffer = reinterpret_cast<float*>(filter.buffer);
     stage_info.bias_buffer = reinterpret_cast<float*>(bias.buffer);
     stage_info.depth_multiplier = depth_multiplier;

     stage_info.stride_width = stride_width;
     stage_info.stride_height = stride_height;

     stage_info.padding_left = padding_left;
     stage_info.padding_right = padding_right;
     stage_info.padding_top = padding_top;
     stage_info.padding_bottom = padding_bottom;

     switch (activation) {
       case 0: stage_info.post_operation = NONE; break;
       case 1: stage_info.post_operation = RELU; break;
       case 2: stage_info.post_operation = RELU1; break;
       case 3: stage_info.post_operation = RELU6; break;
       default: stage_info.post_operation = NONE; break;
     }
     stage_info.kernel_data = true;
     stage_info.bias_data = true;
     stage_info.op_params_data = false;

#if file_dump
  //TODO temporary fix
  stage_info.kernel_shape[0] = (stage_info.kernel_shape[0] == 0 ) ? 1: stage_info.kernel_shape[0];
  stage_info.kernel_shape[1] = (stage_info.kernel_shape[1] == 0 ) ? 1: stage_info.kernel_shape[1];
  stage_info.kernel_shape[2] = (stage_info.kernel_shape[2] == 0 ) ? 1: stage_info.kernel_shape[2];
  stage_info.kernel_shape[3] = (stage_info.kernel_shape[3] == 0 ) ? 1: stage_info.kernel_shape[3];
  uint32_t nk_ele = stage_info.kernel_shape[0] * stage_info.kernel_shape[1] * stage_info.kernel_shape[2] * stage_info.kernel_shape[3];

  stage_info.bias_shape[0] = (stage_info.bias_shape[0] == 0 ) ? 1: stage_info.bias_shape[0];
  stage_info.bias_shape[1] = (stage_info.bias_shape[1] == 0 ) ? 1: stage_info.bias_shape[1];
  stage_info.bias_shape[2] = (stage_info.bias_shape[2] == 0 ) ? 1: stage_info.bias_shape[2];
  stage_info.bias_shape[3] = (stage_info.bias_shape[3] == 0 ) ? 1: stage_info.bias_shape[3];
  uint32_t nb_ele = stage_info.bias_shape[0] * stage_info.bias_shape[1] * stage_info.bias_shape[2] * stage_info.bias_shape[3];

  FILE *fp;

  if(stage_info.kernel_data){
    fp = fopen("/data/ncs_graph_data","ab+");
    if(!fp) ALOGE("unable to open the file /data/ncs_graph_data ");
    VLOG(VPUEXE) << "DEPTHWISE_CONV_2D nk_ele: " << nk_ele;
    fseek(fp, 0, SEEK_END);
    fwrite(reinterpret_cast<const float*>(filter.buffer),sizeof(float),nk_ele,fp);
    fclose(fp);
  }

  if(stage_info.bias_data){
    fp = fopen("/data/ncs_graph_data","ab+");
    if(!fp) ALOGE("unable to open the file /data/ncs_graph_data ");
    VLOG(VPUEXE) << "DEPTHWISE_CONV_2D nb_ele: " << nb_ele;
    fseek(fp, 0, SEEK_END);
    fwrite(reinterpret_cast<const float*>(bias.buffer),sizeof(float),nb_ele,fp);
    fclose(fp);
  }
#endif
   } break; //DEPTHWISE_CONV_2D end

  case OperationType::AVERAGE_POOL_2D: {  //AVERAGE_POOL_2D begin
      VLOG(VPUEXE) << toString(operation);
      const size_t inCount = ins.size();
      /*
      if ((inCount != 10 && inCount != 7) ||
                    !allParametersPresent(inCount, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }*/
      const RunTimeOperandInfo& input  = mOperands[ins[0]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      Shape inputShape = input.shape();
      Shape outShape = output.shape();

      int32_t padding_left, padding_right;
      int32_t padding_top, padding_bottom;
      int32_t stride_width, stride_height;
      int32_t filter_width, filter_height;
      int32_t activation;
      if (inCount == 10) {
          padding_left     = getScalarData<int32_t>(mOperands[ins[1]]);
          padding_right    = getScalarData<int32_t>(mOperands[ins[2]]);
          padding_top      = getScalarData<int32_t>(mOperands[ins[3]]);
          padding_bottom   = getScalarData<int32_t>(mOperands[ins[4]]);
          stride_width     = getScalarData<int32_t>(mOperands[ins[5]]);
          stride_height    = getScalarData<int32_t>(mOperands[ins[6]]);
          filter_width     = getScalarData<int32_t>(mOperands[ins[7]]);
          filter_height    = getScalarData<int32_t>(mOperands[ins[8]]);
          activation       = getScalarData<int32_t>(mOperands[ins[9]]);
      } else {
        int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[1]]);
        stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
        stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
        filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
        filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
        activation       = getScalarData<int32_t>(mOperands[ins[6]]);

        int32_t input_width  = getSizeOfDimension(inputShape, 2);
        int32_t input_height = getSizeOfDimension(inputShape, 1);
        calculateExplicitPadding(input_width, stride_width,
                                 filter_width, padding_implicit,
                                 &padding_left, &padding_right);
        calculateExplicitPadding(input_height, stride_height,
                                 filter_height, padding_implicit,
                                 &padding_top, &padding_bottom);
      }

      if (input.type == OperandType::TENSOR_FLOAT32){
       success = genericPoolingPrepare(input.shape(),
                             padding_left, padding_right,
                             padding_top, padding_bottom,
                             stride_width, stride_height,
                             filter_width, filter_height,
                             &outShape);
        if(!success)
            nnAssert(false);
      }

  stage_info.main_operation = AVERAGE_POOL_2D;
  stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
  stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
  stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
  stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

  stage_info.kernel_shape[0] = filter_width;
  stage_info.kernel_shape[1] = filter_height;
  stage_info.kernel_shape[2] = 1;
  stage_info.kernel_shape[3] = 1;

  stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
  stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
  stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
  stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);

  stage_info.kernel_buffer = nullptr;
  stage_info.bias_buffer = nullptr;
  stage_info.depth_multiplier = 0;

  stage_info.stride_width = stride_width;
  stage_info.stride_height = stride_height;

  stage_info.padding_left = padding_left;
  stage_info.padding_right = padding_right;
  stage_info.padding_top = padding_top;
  stage_info.padding_bottom = padding_bottom;

  switch (activation) {
    case 0: stage_info.post_operation = NONE; break;
    case 1: stage_info.post_operation = RELU; break;
    case 2: stage_info.post_operation = RELU1; break;
    case 3: stage_info.post_operation = RELU6; break;
    default: stage_info.post_operation = NONE; break;
  }

  stage_info.kernel_data = false;
  stage_info.bias_data = false;
  stage_info.op_params_data = false;

  bool DEBUG_AVERAGE_POOL_2D = false;
  //DEBUG_AVERAGE_POOL_2D = true;  //un comment this line to get AVERAGE_POOL_2D layer debug data
  if(DEBUG_AVERAGE_POOL_2D){
    VLOG(VPUEXE) << " AVERAGE_POOL_2D filter_width: " << filter_width;
    VLOG(VPUEXE) << "AVERAGE_POOL_2D filter_height: " << filter_height;

    VLOG(VPUEXE) << " AVERAGE_POOL_2D padding_left: " << padding_left;
    VLOG(VPUEXE) << "AVERAGE_POOL_2D padding_right: " << padding_right;
    VLOG(VPUEXE) << " AVERAGE_POOL_2D padding_top: " << padding_top;
    VLOG(VPUEXE) << "AVERAGE_POOL_2D padding_bottom: " << padding_bottom;

    VLOG(VPUEXE) << "AVERAGE_POOL_2D input_shape[0]: " << stage_info.input_shape[0];
    VLOG(VPUEXE) << "AVERAGE_POOL_2D input_shape[1]: " << stage_info.input_shape[1];
    VLOG(VPUEXE) << "AVERAGE_POOL_2D input_shape[2]: " << stage_info.input_shape[2];
    VLOG(VPUEXE) << "AVERAGE_POOL_2D input_shape[3]: " << stage_info.input_shape[3];

    VLOG(VPUEXE) << "AVERAGE_POOL_2D output_shape[0]: " << stage_info.output_shape[0];
    VLOG(VPUEXE) << "AVERAGE_POOL_2D output_shape[1]: " << stage_info.output_shape[1];
    VLOG(VPUEXE) << "AVERAGE_POOL_2D output_shape[2]: " << stage_info.output_shape[2];
    VLOG(VPUEXE) << "AVERAGE_POOL_2D output_shape[3]: " << stage_info.output_shape[3];
  }

  } break; //AVERAGE_POOL_2D end
  case OperationType::SOFTMAX: {  //SOFTMAX begin
    VLOG(VPUEXE) << toString(operation);
    /*
      if (!allParametersPresent(2, 1)) {
          return ANEURALNETWORKS_BAD_DATA;
      }*/
      const RunTimeOperandInfo& input  = mOperands[ins[0]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      Shape inputShape = input.shape();
      Shape outShape = output.shape();

      float beta = getScalarData<float>(mOperands[ins[1]]);

      if (beta <= 0.0f) {
          LOG(ERROR) << "beta must be positive for softmax";
          nnAssert(false);
      }

      if (input.type == OperandType::TENSOR_FLOAT32) {
          success = genericActivationPrepare(input.shape(), &outShape);
          if(!success)
              nnAssert(false);
      }
      stage_info.main_operation = SOFTMAX;
      stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
      stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
      stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
      stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

      stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
      stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
      stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
      stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);
      stage_info.beta = beta;
      stage_info.kernel_data = false;
      stage_info.bias_data = false;
      stage_info.op_params_data = true;

      stage_info.post_operation = NONE;

      VLOG(VPUEXE) << " SOFTMAX input_shape[0]: " << stage_info.input_shape[0];
      VLOG(VPUEXE) << " SOFTMAX input_shape[1]: " << stage_info.input_shape[1];
      VLOG(VPUEXE) << " SOFTMAX input_shape[2]: " << stage_info.input_shape[2];
      VLOG(VPUEXE) << " SOFTMAX input_shape[3]: " << stage_info.input_shape[3];

      VLOG(VPUEXE) << " SOFTMAX output_shape[0]: " << stage_info.output_shape[0];
      VLOG(VPUEXE) << " SOFTMAX output_shape[1]: " << stage_info.output_shape[1];
      VLOG(VPUEXE) << " SOFTMAX output_shape[2]: " << stage_info.output_shape[2];
      VLOG(VPUEXE) << " SOFTMAX output_shape[3]: " << stage_info.output_shape[3];
      ALOGD("BETA: %f",beta);
  } break; //SOFTMAX end
  case OperationType::RESHAPE: { //RESHAPE begin
      /*if (!allParametersPresent(2, 1)) {
          return ANEURALNETWORKS_BAD_DATA;
      }*/
      const RunTimeOperandInfo& input = mOperands[ins[0]];
      const RunTimeOperandInfo& targetShape = mOperands[ins[1]];
      RunTimeOperandInfo& output = mOperands[outs[0]];

      Shape inputShape = input.shape();
      Shape outShape = output.shape();

      success = reshapePrepare(input.shape(),
                               reinterpret_cast<const int32_t*>(targetShape.buffer),
                               getNumberOfElements(targetShape.shape()),
                               &outShape);
      stage_info.main_operation = RESHAPE;
      stage_info.input_shape[0] = getSizeOfDimension(inputShape, 0);
      stage_info.input_shape[1] = getSizeOfDimension(inputShape, 1);
      stage_info.input_shape[2] = getSizeOfDimension(inputShape, 2);
      stage_info.input_shape[3] = getSizeOfDimension(inputShape, 3);

      stage_info.output_shape[0] = getSizeOfDimension(outShape, 0);
      stage_info.output_shape[1] = getSizeOfDimension(outShape, 1);
      stage_info.output_shape[2] = getSizeOfDimension(outShape, 2);
      stage_info.output_shape[3] = getSizeOfDimension(outShape, 3);
      stage_info.post_operation = NONE;

      VLOG(VPUEXE) << " RESHAPE input_shape[0]: " << stage_info.input_shape[0];
      VLOG(VPUEXE) << " RESHAPE input_shape[1]: " << stage_info.input_shape[1];
      VLOG(VPUEXE) << " RESHAPE input_shape[2]: " << stage_info.input_shape[2];
      VLOG(VPUEXE) << " RESHAPE input_shape[3]: " << stage_info.input_shape[3];

      VLOG(VPUEXE) << " RESHAPE output_shape[0]: " << stage_info.output_shape[0];
      VLOG(VPUEXE) << " RESHAPE output_shape[1]: " << stage_info.output_shape[1];
      VLOG(VPUEXE) << " RESHAPE output_shape[2]: " << stage_info.output_shape[2];
      VLOG(VPUEXE) << " RESHAPE output_shape[3]: " << stage_info.output_shape[3];


  } break;    //RESHAPE end
    default:
        nnAssert(false);
        break;
  }
  return stage_info;
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
    // The model has serialized the operation in execution order.
#if file_dump
    FILE *fp;
    fp = fopen("/data/ncs_graph_data","rb");
    if(fp){
      remove("/data/ncs_graph_data");
    }
#endif
    VLOG(VPUEXE) << "Model Compiling for VPU Driver begin ";

    Oertaion_vector nn_ops_vectors;
    network_operations_vector nn_ncs_network;
    NCSoperations nn_ncs_operation;

    for (const auto& operation : model.operations) {

      switch (operation.type) {
        case OperationType::RELU: nn_ncs_operation = RELU;break;
        case OperationType::RELU1: nn_ncs_operation = RELU1;break;
        case OperationType::RELU6: nn_ncs_operation = RELU6;break;
        case OperationType::TANH: nn_ncs_operation = TANH;break;
        case OperationType::LOGISTIC: nn_ncs_operation = LOGISTIC;break;
        case OperationType::CONV_2D: nn_ncs_operation = CONV_2D;break;
        case OperationType::DEPTHWISE_CONV_2D: nn_ncs_operation = DEPTHWISE_CONV_2D;break;
        case OperationType::AVERAGE_POOL_2D: nn_ncs_operation = AVERAGE_POOL_2D;break;
        case OperationType::L2_POOL_2D: nn_ncs_operation = L2_POOL_2D;break;
        case OperationType::MAX_POOL_2D: nn_ncs_operation = MAX_POOL_2D;break;
        case OperationType::SOFTMAX: nn_ncs_operation = SOFTMAX;break;
        case OperationType::FULLY_CONNECTED: nn_ncs_operation = FULLY_CONNECTED;break;
        case OperationType::L2_NORMALIZATION: nn_ncs_operation = L2_NORMALIZATION;break;
        case OperationType::RESHAPE: nn_ncs_operation = RESHAPE;break;
        default: nn_ncs_operation = NONE;break;
      }
      nn_ncs_network.push_back(nn_ncs_operation);
      nn_ops_vectors.push_back(operation.type);
    }

    for(int i=0;i<nn_ops_vectors.size();i++){
      VLOG(VPUEXE) << "NN MODEL Operation Vector: " << getOperationName(nn_ops_vectors.at(i));
    }
    /*
    const hidl_vec<uint32_t>& network_inputs = model.operations[0].inputs;
    const RunTimeOperandInfo& network_input = mOperands[network_inputs[0]];
    Shape nw_input_shape = network_input.shape();
    uint32_t input_num_elements = getNumberOfElements(nw_input_shape);
    VLOG(VPUEXE) << "Before setInfoAndAllocateIfNeeded Input Num of Elements: " << input_num_elements;


    for(uint32_t i=0;i<input_num_elements;i++){
      ALOGD("Model Input is buffer[%d]:%f",i,*(reinterpret_cast<float*>(network_input.buffer)+i));
    }

    const hidl_vec<uint32_t>& network_outputs = model.operations[nn_ops_vectors.size()-1].outputs;
    RunTimeOperandInfo& network_output = mOperands[network_outputs[0]];
    Shape nw_output_shape = network_output.shape();
    uint32_t output_num_elements = getNumberOfElements(nw_output_shape);
    VLOG(VPUEXE) << "Before setInfoAndAllocateIfNeeded Output Num of Elements: " << output_num_elements;

    for(uint32_t i=0;i<output_num_elements;i++){
      *(reinterpret_cast<float*>(network_output.buffer)+i) = 10.0 * i;
      ALOGD("Model Output is buffer[%d]:%f",i,*(reinterpret_cast<float*>(network_output.buffer)+i));
    }*/

    bool status1;
    //status1 = setInfoAndAllocateIfNeeded(&network_output, nw_output_shape);
    /*
    for(int i=0;i<nn_ncs_network.size();i++)
    VLOG(VPUEXE) << " Network Operation is: " << nn_ncs_network.at(i) ;*/

    bool status;
    status = get_nn_network_from_android(nn_ncs_network);
    if(!status)
      return ANEURALNETWORKS_INCOMPLETE;

    Operation_inputs_info operation_operand_info;

    for (const auto& operation : model.operations){
      operation_operand_info = get_operation_operands_info(operation);
      bool status = parse_stage_from_android(operation_operand_info);
      VLOG(VPUEXE) << "Status " << status;
      if(!status){
        nnAssert(false);
        break;
      }
    }

    status = prepare_blob();
    if(!status){
      VLOG(VPUEXE) << "Unable to prepare NCS graph";
      nnAssert(false);
    }

    VLOG(VPUEXE) << "Model Compiling for VPU Driver: completed";


    const hidl_vec<uint32_t>& network_inputs = model.operations[0].inputs;
    const RunTimeOperandInfo& network_input = mOperands[network_inputs[0]];
    Shape nw_input_shape = network_input.shape();
    uint32_t input_num_elements = getNumberOfElements(nw_input_shape);
    VLOG(VPUEXE) << "Before setInfoAndAllocateIfNeeded Input Num of Elements: " << input_num_elements;
    const float *network_input_buffer;
    network_input_buffer = (float *)malloc(sizeof(float) * input_num_elements);
    network_input_buffer = reinterpret_cast<float*>(network_input.buffer);
    /*
    for(uint32_t i=0;i<input_num_elements;i++){
      ALOGD("Model Input is buffer[%d]:%f",i,*(reinterpret_cast<float*>(network_input.buffer)+i));
    }

    for(uint32_t i=0;i<input_num_elements;i++){
      ALOGD("Model Network Input buffer[%d]:%f",i,*(network_input_buffer+i));
    }*/

    const hidl_vec<uint32_t>& network_outputs = model.operations[nn_ops_vectors.size()-1].outputs;
    RunTimeOperandInfo& network_output = mOperands[network_outputs[0]];
    Shape nw_output_shape = network_output.shape();
    uint32_t output_num_elements = getNumberOfElements(nw_output_shape);
    float *network_output_buffer;
    network_output_buffer = (float *)malloc(sizeof(float) * output_num_elements);
    network_output_buffer = reinterpret_cast<float*>(network_output.buffer);

    VLOG(VPUEXE) << "Before setInfoAndAllocateIfNeeded Output Num of Elements: " << output_num_elements;

    int val = ncs_execute((float*)network_input_buffer,input_num_elements,network_output_buffer, output_num_elements);

    /*
    for(uint32_t i=0;i<output_num_elements;i++){
      ALOGD("NCS Output is buffer[%d]:%f",i,*(network_output_buffer+i));
      ALOGD("Model Output is buffer[%d]:%f",i,*(reinterpret_cast<float*>(network_output.buffer)+i));
    }*/

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

int VpuExecutor::executeOperation(const Operation& operation) {
    // VLOG(VPUEXE) << "VpuExecutor::executeOperation(" << toString(operation) << ")";
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    bool success = false;

    // Function to verify that the number of input and output parameters
    // matches what is expected.  Also checks that all the parameters have
    // values. This function is to be used only for operations that do not
    // accept optional arguments.
    // TODO Have a version that works for optional arguments.
    auto allParametersPresent = [&operation, &ins, &outs, this](size_t requiredIns,
                                                                size_t requiredOuts) -> bool {
        auto verify = [&operation, this](size_t requiredCount, const hidl_vec<uint32_t>& indexes,
                          const char* type) -> bool {
            size_t actualCount = indexes.size();
            if (actualCount != requiredCount) {
                LOG(ERROR) << getOperationName(operation.type)
                           << ": Invalid number of " << type << " operands. Got " << actualCount
                           << " of " << requiredCount;
                return false;
            }
            for (size_t i = 0; i < actualCount; i++) {
                if (mOperands[indexes[i]].lifetime == OperandLifeTime::NO_VALUE) {
                    LOG(ERROR) << getOperationName(operation.type) << " " << type
                               << " operand " << i << " is required but missing.";
                    return false;
                }
            }
            return true;
        };
        return verify(requiredIns, ins, "in") && verify(requiredOuts, outs, "out");
    };

    VLOG(VPUEXE) << "Operation on VPU driver!" << toString(operation);



/////////////////////////////////////////////////////////////////////////////////////////////////////////////

    switch (operation.type) { //switch start
      case OperationType::OEM_OPERATION: {
        LOG(ERROR) << "OEM operation not supported";
        success = false;
    } break;

    case OperationType::RELU: {
        if (!allParametersPresent(1, 1)) {

            return ANEURALNETWORKS_BAD_DATA;
        }
        VLOG(VPUEXE) << "running RELU operation on VPU driver!";
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
          success = true;
          success = genericActivationPrepare(input.shape(), &outShape) &&
                    setInfoAndAllocateIfNeeded(&output, outShape) &&
                    tanhFloat32(reinterpret_cast<const float*>(input.buffer),
                                input.shape(),
                                reinterpret_cast<float*>(output.buffer),
                                outShape);
        }
    } break;
    case OperationType::RELU1: {
        if (!allParametersPresent(1, 1)) {

            return ANEURALNETWORKS_BAD_DATA;
        }
        VLOG(VPUEXE) << "running RELU operation on VPU driver!";
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
          success = true;
          success = genericActivationPrepare(input.shape(), &outShape) &&
                    setInfoAndAllocateIfNeeded(&output, outShape) &&
                    tanhFloat32(reinterpret_cast<const float*>(input.buffer),
                                input.shape(),
                                reinterpret_cast<float*>(output.buffer),
                                outShape);
        }
    } break;
    case OperationType::RELU6: {
        if (!allParametersPresent(1, 1)) {

            return ANEURALNETWORKS_BAD_DATA;
        }
        VLOG(VPUEXE) << "running RELU operation on VPU driver!";
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();
        uint32_t input_element_count = getNumberOfElements(input.shape());

        VLOG(VPUEXE) << "RELU6 operation on VPU driver Shape Size:" << getNumberOfDimensions(input.shape());
        VLOG(VPUEXE) << "RELU6 operation on VPU driver Number pf Elements: " << getNumberOfElements(input.shape());
        VLOG(VPUEXE) << "RELU6 operation on VPU driver Shape index 0: " << getSizeOfDimension(input.shape(), 0);
        VLOG(VPUEXE) << "RELU6 operation on VPU driver Shape index 1: " << getSizeOfDimension(input.shape(), 1);
        VLOG(VPUEXE) << "RELU6 operation on VPU driver Shape index 2: " << getSizeOfDimension(input.shape(), 2);
        VLOG(VPUEXE) << "RELU6 operation on VPU driver Shape index 3: " << getSizeOfDimension(input.shape(), 3);

        for(uint32_t i=0; i<input_element_count;i++)
        VLOG(VPUEXE) << "RELU6 operation on VPU driver Elements : " << toString(*(input.buffer+i));


        if (input.type == OperandType::TENSOR_FLOAT32) {
          success = genericActivationPrepare(input.shape(), &outShape) &&
                    setInfoAndAllocateIfNeeded(&output, outShape) &&
                    tanhFloat32(reinterpret_cast<const float*>(input.buffer),
                                input.shape(),
                                reinterpret_cast<float*>(output.buffer),
                                outShape);
        }
    } break;
    case OperationType::TANH: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      tanhFloat32(reinterpret_cast<const float*>(input.buffer),
                                  input.shape(),
                                  reinterpret_cast<float*>(output.buffer),
                                  outShape);
        }
    } break;
    case OperationType::LOGISTIC: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        VLOG(VPUEXE) << "LOGISTIC operation on VPU driver Shape Size: " << getNumberOfDimensions(input.shape());
        VLOG(VPUEXE) << "LOGISTIC operation on VPU driver Number pf Elements: " << getNumberOfElements(input.shape());
        VLOG(VPUEXE) << "LOGISTIC operation on VPU driver Shape index 0: " << getSizeOfDimension(input.shape(), 0);
        VLOG(VPUEXE) << "LOGISTIC operation on VPU driver Shape index 1: " << getSizeOfDimension(input.shape(), 1);
        VLOG(VPUEXE) << "LOGISTIC operation on VPU driver Shape index 2: " << getSizeOfDimension(input.shape(), 2);
        VLOG(VPUEXE) << "LOGISTIC operation on VPU driver Shape index 3: " << getSizeOfDimension(input.shape(), 3);

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      logisticFloat32(reinterpret_cast<const float*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<float*>(output.buffer),
                                      outShape);
        }
      } break;
    default:
        nnAssert(false);
        break;
  }
    success = true;
    if (!success) {
        LOG(ERROR) << getOperationName(operation.type) << " failed.";
        return ANEURALNETWORKS_OP_FAILED;
    }

    //freeNoLongerUsedOperands(ins);
    return ANEURALNETWORKS_NO_ERROR;
}


}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android



/*

switch (operation.type) { //switch start
    case OperationType::OEM_OPERATION: {
        LOG(ERROR) << "OEM operation not supported for CPU execution";
        success = false;
    } break;
    case OperationType::ADD: {
        if (!allParametersPresent(3, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& in1 = mOperands[ins[0]];
        const RunTimeOperandInfo& in2 = mOperands[ins[1]];
        int32_t activation = getScalarData<int32_t>(mOperands[ins[2]]);

        RunTimeOperandInfo& out = mOperands[outs[0]];
        Shape outShape = out.shape();

        if (in1.type == OperandType::TENSOR_FLOAT32) {
            success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&out, outShape) &&
                      addFloat32(reinterpret_cast<const float*>(in1.buffer),
                                 in1.shape(),
                                 reinterpret_cast<const float*>(in2.buffer),
                                 in2.shape(),
                                 activation,
                                 reinterpret_cast<float*>(out.buffer),
                                 outShape);
        } else if (in1.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&out, outShape) &&
                      addQuant8(reinterpret_cast<const uint8_t*>(in1.buffer),
                                in1.shape(),
                                reinterpret_cast<const uint8_t*>(in2.buffer),
                                in2.shape(),
                                activation,
                                reinterpret_cast<uint8_t*>(out.buffer),
                                outShape);
        }
    } break;
    case OperationType::MUL: {
        if (!allParametersPresent(3, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& in1 = mOperands[ins[0]];
        const RunTimeOperandInfo& in2 = mOperands[ins[1]];
        int32_t activation = getScalarData<int32_t>(mOperands[ins[2]]);

        RunTimeOperandInfo& out = mOperands[outs[0]];
        Shape outShape = out.shape();

        if (in1.type == OperandType::TENSOR_FLOAT32) {
            success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&out, outShape) &&
                      mulFloat32(reinterpret_cast<const float*>(in1.buffer),
                                 in1.shape(),
                                 reinterpret_cast<const float*>(in2.buffer),
                                 in2.shape(),
                                 activation,
                                 reinterpret_cast<float*>(out.buffer),
                                 outShape);
        } else if (in1.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&out, outShape) &&
                      mulQuant8(reinterpret_cast<const uint8_t*>(in1.buffer),
                                in1.shape(),
                                reinterpret_cast<const uint8_t*>(in2.buffer),
                                in2.shape(),
                                activation,
                                reinterpret_cast<uint8_t*>(out.buffer),
                                outShape);
        }
    } break;
    case OperationType::FLOOR: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = floorPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      floorFloat32(reinterpret_cast<const float*>(input.buffer),
                                   reinterpret_cast<float*>(output.buffer),
                                   outShape);
        }
    } break;
    case OperationType::DEQUANTIZE: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = dequantizePrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      dequantizeQuant8ToFloat32(
                              reinterpret_cast<const uint8_t*>(input.buffer),
                              reinterpret_cast<float*>(output.buffer),
                              input.shape());
        }
    } break;
    case OperationType::DEPTHWISE_CONV_2D: {
        const size_t inCount = ins.size();
        if ((inCount != 11 && inCount != 8) ||
                !allParametersPresent(inCount, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input  = mOperands[ins[0]];
        const RunTimeOperandInfo& filter = mOperands[ins[1]];
        const RunTimeOperandInfo& bias   = mOperands[ins[2]];

        int32_t padding_left, padding_right;
        int32_t padding_top, padding_bottom;
        int32_t stride_width, stride_height;
        int32_t depth_multiplier;
        int32_t activation;

        if (inCount == 11) {
            padding_left     = getScalarData<int32_t>(mOperands[ins[3]]);
            padding_right    = getScalarData<int32_t>(mOperands[ins[4]]);
            padding_top      = getScalarData<int32_t>(mOperands[ins[5]]);
            padding_bottom   = getScalarData<int32_t>(mOperands[ins[6]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[7]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[8]]);
            depth_multiplier = getScalarData<int32_t>(mOperands[ins[9]]);
            activation       = getScalarData<int32_t>(mOperands[ins[10]]);
        } else {
            int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[3]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            depth_multiplier = getScalarData<int32_t>(mOperands[ins[6]]);
            activation       = getScalarData<int32_t>(mOperands[ins[7]]);

            Shape inputShape = input.shape();
            Shape filterShape = filter.shape();
            int32_t input_width  = getSizeOfDimension(inputShape, 2);
            int32_t input_height = getSizeOfDimension(inputShape, 1);
            int32_t filter_width  = getSizeOfDimension(filterShape, 2);
            int32_t filter_height = getSizeOfDimension(filterShape, 1);
            calculateExplicitPadding(input_width, stride_width,
                                     filter_width, padding_implicit,
                                     &padding_left, &padding_right);
            calculateExplicitPadding(input_height, stride_height,
                                     filter_height, padding_implicit,
                                     &padding_top, &padding_bottom);
        }

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = depthwiseConvPrepare(input.shape(), filter.shape(), bias.shape(),
                                           padding_left, padding_right,
                                           padding_top, padding_bottom,
                                           stride_width, stride_height,
                                           &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      depthwiseConvFloat32(reinterpret_cast<const float*>(input.buffer),
                                           input.shape(),
                                           reinterpret_cast<const float*>(filter.buffer),
                                           filter.shape(),
                                           reinterpret_cast<const float*>(bias.buffer),
                                           bias.shape(),
                                           padding_left, padding_right,
                                           padding_top, padding_bottom,
                                           stride_width, stride_height,
                                           depth_multiplier, activation,
                                           reinterpret_cast<float*>(output.buffer),
                                           outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = depthwiseConvPrepare(input.shape(), filter.shape(), bias.shape(),
                                           padding_left, padding_right,
                                           padding_top, padding_bottom,
                                           stride_width, stride_height,
                                           &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      depthwiseConvQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                          input.shape(),
                                          reinterpret_cast<const uint8_t*>(filter.buffer),
                                          filter.shape(),
                                          reinterpret_cast<const int32_t*>(bias.buffer),
                                          bias.shape(),
                                          padding_left, padding_right,
                                          padding_top, padding_bottom,
                                          stride_width, stride_height,
                                          depth_multiplier, activation,
                                          reinterpret_cast<uint8_t*>(output.buffer),
                                          outShape);
        }

    } break;
    case OperationType::CONV_2D: {
        const size_t inCount = ins.size();
        if ((inCount != 10 && inCount != 7) ||
                !allParametersPresent(inCount, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input  = mOperands[ins[0]];
        const RunTimeOperandInfo& filter = mOperands[ins[1]];
        const RunTimeOperandInfo& bias   = mOperands[ins[2]];

        int32_t padding_left, padding_right;
        int32_t padding_top, padding_bottom;
        int32_t stride_width, stride_height;
        int32_t activation;

        if (inCount == 10) {
            padding_left     = getScalarData<int32_t>(mOperands[ins[3]]);
            padding_right    = getScalarData<int32_t>(mOperands[ins[4]]);
            padding_top      = getScalarData<int32_t>(mOperands[ins[5]]);
            padding_bottom   = getScalarData<int32_t>(mOperands[ins[6]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[7]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[8]]);
            activation       = getScalarData<int32_t>(mOperands[ins[9]]);
        } else {
            int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[3]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            activation       = getScalarData<int32_t>(mOperands[ins[6]]);

            Shape inputShape = input.shape();
            Shape filterShape = filter.shape();
            int32_t input_width  = getSizeOfDimension(inputShape, 2);
            int32_t input_height = getSizeOfDimension(inputShape, 1);
            int32_t filter_width  = getSizeOfDimension(filterShape, 2);
            int32_t filter_height = getSizeOfDimension(filterShape, 1);
            calculateExplicitPadding(input_width, stride_width,
                                     filter_width, padding_implicit,
                                     &padding_left, &padding_right);
            calculateExplicitPadding(input_height, stride_height,
                                     filter_height, padding_implicit,
                                     &padding_top, &padding_bottom);
        }

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = convPrepare(input.shape(), filter.shape(), bias.shape(),
                                  padding_left, padding_right,
                                  padding_top, padding_bottom,
                                  stride_width, stride_height,
                                  &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      convFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                  reinterpret_cast<const float*>(filter.buffer), filter.shape(),
                                  reinterpret_cast<const float*>(bias.buffer), bias.shape(),
                                  padding_left, padding_right,
                                  padding_top, padding_bottom,
                                  stride_width, stride_height, activation,
                                  reinterpret_cast<float*>(output.buffer), outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = convPrepare(input.shape(), filter.shape(), bias.shape(),
                                  padding_left, padding_right,
                                  padding_top, padding_bottom,
                                  stride_width, stride_height,
                                  &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      convQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                 input.shape(),
                                 reinterpret_cast<const uint8_t*>(filter.buffer),
                                 filter.shape(),
                                 reinterpret_cast<const int32_t*>(bias.buffer),
                                 bias.shape(),
                                 padding_left, padding_right,
                                 padding_top, padding_bottom,
                                 stride_width, stride_height, activation,
                                 reinterpret_cast<uint8_t*>(output.buffer),
                                 outShape);
        }
    } break;
    case OperationType::AVERAGE_POOL_2D: {
        const size_t inCount = ins.size();
        if ((inCount != 10 && inCount != 7) ||
                !allParametersPresent(inCount, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];

        int32_t padding_left, padding_right;
        int32_t padding_top, padding_bottom;
        int32_t stride_width, stride_height;
        int32_t filter_width, filter_height;
        int32_t activation;

        if (inCount == 10) {
            padding_left     = getScalarData<int32_t>(mOperands[ins[1]]);
            padding_right    = getScalarData<int32_t>(mOperands[ins[2]]);
            padding_top      = getScalarData<int32_t>(mOperands[ins[3]]);
            padding_bottom   = getScalarData<int32_t>(mOperands[ins[4]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[5]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[6]]);
            filter_width     = getScalarData<int32_t>(mOperands[ins[7]]);
            filter_height    = getScalarData<int32_t>(mOperands[ins[8]]);
            activation       = getScalarData<int32_t>(mOperands[ins[9]]);
        } else {
            int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[1]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
            filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            activation       = getScalarData<int32_t>(mOperands[ins[6]]);

            Shape inputShape = input.shape();
            int32_t input_width  = getSizeOfDimension(inputShape, 2);
            int32_t input_height = getSizeOfDimension(inputShape, 1);
            calculateExplicitPadding(input_width, stride_width,
                                     filter_width, padding_implicit,
                                     &padding_left, &padding_right);
            calculateExplicitPadding(input_height, stride_height,
                                     filter_height, padding_implicit,
                                     &padding_top, &padding_bottom);
        }

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericPoolingPrepare(input.shape(),
                                            padding_left, padding_right,
                                            padding_top, padding_bottom,
                                            stride_width, stride_height,
                                            filter_width, filter_height,
                                            &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      averagePoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                         input.shape(),
                                         padding_left, padding_right,
                                         padding_top, padding_bottom,
                                         stride_width, stride_height,
                                         filter_width, filter_height, activation,
                                         reinterpret_cast<float*>(output.buffer),
                                         outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = genericPoolingPrepare(input.shape(),
                                            padding_left, padding_right,
                                            padding_top, padding_bottom,
                                            stride_width, stride_height,
                                            filter_width, filter_height,
                                            &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      averagePoolQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                        input.shape(),
                                        padding_left, padding_right,
                                        padding_top, padding_bottom,
                                        stride_width, stride_height,
                                        filter_width, filter_height, activation,
                                        reinterpret_cast<uint8_t*>(output.buffer),
                                        outShape);
        }
    } break;
    case OperationType::L2_POOL_2D: {
        const size_t inCount = ins.size();
        if ((inCount != 10 && inCount != 7) ||
                !allParametersPresent(inCount, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];

        int32_t padding_left, padding_right;
        int32_t padding_top, padding_bottom;
        int32_t stride_width, stride_height;
        int32_t filter_width, filter_height;
        int32_t activation;

        if (inCount == 10) {
            padding_left     = getScalarData<int32_t>(mOperands[ins[1]]);
            padding_right    = getScalarData<int32_t>(mOperands[ins[2]]);
            padding_top      = getScalarData<int32_t>(mOperands[ins[3]]);
            padding_bottom   = getScalarData<int32_t>(mOperands[ins[4]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[5]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[6]]);
            filter_width     = getScalarData<int32_t>(mOperands[ins[7]]);
            filter_height    = getScalarData<int32_t>(mOperands[ins[8]]);
            activation       = getScalarData<int32_t>(mOperands[ins[9]]);
        } else {
            int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[1]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
            filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            activation       = getScalarData<int32_t>(mOperands[ins[6]]);

            Shape inputShape = input.shape();
            int32_t input_width  = getSizeOfDimension(inputShape, 2);
            int32_t input_height = getSizeOfDimension(inputShape, 1);
            calculateExplicitPadding(input_width, stride_width,
                                     filter_width, padding_implicit,
                                     &padding_left, &padding_right);
            calculateExplicitPadding(input_height, stride_height,
                                     filter_height, padding_implicit,
                                     &padding_top, &padding_bottom);
        }

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericPoolingPrepare(input.shape(),
                                            padding_left, padding_right,
                                            padding_top, padding_bottom,
                                            stride_width, stride_height,
                                            filter_width, filter_height,
                                            &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      l2PoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                    input.shape(),
                                    padding_left, padding_right,
                                    padding_top, padding_bottom,
                                    stride_width, stride_height,
                                    filter_width, filter_height, activation,
                                    reinterpret_cast<float*>(output.buffer),
                                    outShape);
        }
    } break;
    case OperationType::MAX_POOL_2D: {
        const size_t inCount = ins.size();
        if ((inCount != 10 && inCount != 7) ||
                !allParametersPresent(inCount, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];

        int32_t padding_left, padding_right;
        int32_t padding_top, padding_bottom;
        int32_t stride_width, stride_height;
        int32_t filter_width, filter_height;
        int32_t activation;

        if (inCount == 10) {
            padding_left     = getScalarData<int32_t>(mOperands[ins[1]]);
            padding_right    = getScalarData<int32_t>(mOperands[ins[2]]);
            padding_top      = getScalarData<int32_t>(mOperands[ins[3]]);
            padding_bottom   = getScalarData<int32_t>(mOperands[ins[4]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[5]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[6]]);
            filter_width     = getScalarData<int32_t>(mOperands[ins[7]]);
            filter_height    = getScalarData<int32_t>(mOperands[ins[8]]);
            activation       = getScalarData<int32_t>(mOperands[ins[9]]);
        } else {
            int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[1]]);
            stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
            stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
            filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            activation       = getScalarData<int32_t>(mOperands[ins[6]]);

            Shape inputShape = input.shape();
            int32_t input_width  = getSizeOfDimension(inputShape, 2);
            int32_t input_height = getSizeOfDimension(inputShape, 1);
            calculateExplicitPadding(input_width, stride_width,
                                     filter_width, padding_implicit,
                                     &padding_left, &padding_right);
            calculateExplicitPadding(input_height, stride_height,
                                     filter_height, padding_implicit,
                                     &padding_top, &padding_bottom);
        }

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericPoolingPrepare(input.shape(),
                                            padding_left, padding_right,
                                            padding_top, padding_bottom,
                                            stride_width, stride_height,
                                            filter_width, filter_height,
                                            &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      maxPoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                     input.shape(),
                                     padding_left, padding_right,
                                     padding_top, padding_bottom,
                                     stride_width, stride_height,
                                     filter_width, filter_height, activation,
                                     reinterpret_cast<float*>(output.buffer),
                                     outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = genericPoolingPrepare(input.shape(),
                                            padding_left, padding_right,
                                            padding_top, padding_bottom,
                                            stride_width, stride_height,
                                            filter_width, filter_height,
                                            &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      maxPoolQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                    input.shape(),
                                    padding_left, padding_right,
                                    padding_top, padding_bottom,
                                    stride_width, stride_height,
                                    filter_width, filter_height, activation,
                                    reinterpret_cast<uint8_t*>(output.buffer),
                                    outShape);
        }

    } break;
    case OperationType::RELU: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      reluFloat32(reinterpret_cast<const float*>(input.buffer),
                                  input.shape(),
                                  reinterpret_cast<float*>(output.buffer),
                                  outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      reluQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                 input.shape(),
                                 reinterpret_cast<uint8_t*>(output.buffer),
                                 outShape);
        }
    } break;
    case OperationType::RELU1: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      relu1Float32(reinterpret_cast<const float*>(input.buffer),
                                   input.shape(),
                                   reinterpret_cast<float*>(output.buffer),
                                   outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      relu1Quant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                  input.shape(),
                                  reinterpret_cast<uint8_t*>(output.buffer),
                                  outShape);
        }
    } break;
    case OperationType::RELU6: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      relu6Float32(reinterpret_cast<const float*>(input.buffer),
                                   input.shape(),
                                   reinterpret_cast<float*>(output.buffer),
                                   outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      relu6Quant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                  input.shape(),
                                  reinterpret_cast<uint8_t*>(output.buffer),
                                  outShape);
        }
    } break;
    case OperationType::TANH: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      tanhFloat32(reinterpret_cast<const float*>(input.buffer),
                                  input.shape(),
                                  reinterpret_cast<float*>(output.buffer),
                                  outShape);
        }
    } break;
    case OperationType::LOGISTIC: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      logisticFloat32(reinterpret_cast<const float*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<float*>(output.buffer),
                                      outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      logisticQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                     input.shape(),
                                     reinterpret_cast<uint8_t*>(output.buffer),
                                     outShape);
        }
    } break;
    case OperationType::SOFTMAX: {
        if (!allParametersPresent(2, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        RunTimeOperandInfo& input = mOperands[ins[0]];
        float beta = getScalarData<float>(mOperands[ins[1]]);
        if (beta <= 0.0f) {
            LOG(ERROR) << "beta must be positive for softmax";
            return ANEURALNETWORKS_BAD_DATA;
        }

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      softmaxFloat32(reinterpret_cast<const float*>(input.buffer),
                                     input.shape(),
                                     beta,
                                     reinterpret_cast<float*>(output.buffer),
                                     output.shape());
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = genericActivationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      softmaxQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                    input.shape(),
                                    beta,
                                    reinterpret_cast<uint8_t*>(output.buffer),
                                    output.shape());
        }
    } break;
    case OperationType::FULLY_CONNECTED: {
        if (!allParametersPresent(4, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        RunTimeOperandInfo& input   = mOperands[ins[0]];
        RunTimeOperandInfo& weights = mOperands[ins[1]];
        RunTimeOperandInfo& bias    = mOperands[ins[2]];

        int32_t activation = getScalarData<int32_t>(mOperands[ins[3]]);

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = fullyConnectedPrepare(input.shape(), weights.shape(), bias.shape(),
                                            &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      fullyConnectedFloat32(reinterpret_cast<const float*>(input.buffer),
                                            input.shape(),
                                            reinterpret_cast<const float*>(weights.buffer),
                                            weights.shape(),
                                            reinterpret_cast<const float*>(bias.buffer),
                                            bias.shape(),
                                            activation,
                                            reinterpret_cast<float*>(output.buffer),
                                            outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = fullyConnectedPrepare(input.shape(), weights.shape(), bias.shape(),
                                            &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      fullyConnectedQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                           input.shape(),
                                           reinterpret_cast<const uint8_t*>(weights.buffer),
                                           weights.shape(),
                                           reinterpret_cast<const int32_t*>(bias.buffer),
                                           bias.shape(),
                                           activation,
                                           reinterpret_cast<uint8_t*>(output.buffer),
                                           outShape);
        }
    } break;
    case OperationType::CONCATENATION: {
        if (outs.size() != 1 || ins.size() < 2) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        int numInputTensors = ins.size() - 1;
        int32_t axis = getScalarData<int32_t>(mOperands[ins[numInputTensors]]);

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        const RunTimeOperandInfo& firstInput = mOperands[ins[0]];
        if (firstInput.type == OperandType::TENSOR_FLOAT32) {
            std::vector<Shape> inputShapes(numInputTensors);
            std::vector<const float*> inputDataPtrs(numInputTensors);

            for (int i=0; i<numInputTensors; i++) {
                RunTimeOperandInfo& input = mOperands[ins[i]];
                inputShapes[i] = input.shape();
                inputDataPtrs[i] = reinterpret_cast<const float*>(input.buffer);
            }
            success = concatenationPrepare(inputShapes, axis, &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      concatenationFloat32(inputDataPtrs, inputShapes, axis,
                                           reinterpret_cast<float*>(output.buffer), outShape);
        } else if (firstInput.type == OperandType::TENSOR_QUANT8_ASYMM) {
            std::vector<Shape> inputShapes(numInputTensors);
            std::vector<const uint8_t*> inputDataPtrs(numInputTensors);

            for (int i=0; i<numInputTensors; i++) {
                RunTimeOperandInfo& input = mOperands[ins[i]];
                inputShapes[i] = input.shape();
                inputDataPtrs[i] = reinterpret_cast<const uint8_t*>(input.buffer);
            }
            success = concatenationPrepare(inputShapes, axis, &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      concatenationQuant8(inputDataPtrs, inputShapes, axis,
                                          reinterpret_cast<uint8_t*>(output.buffer),
                                          outShape);
        }
    } break;
    case OperationType::L2_NORMALIZATION: {
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericNormalizationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      l2normFloat32(reinterpret_cast<const float*>(input.buffer),
                                    input.shape(),
                                    reinterpret_cast<float*>(output.buffer),
                                    outShape);
        } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            success = genericNormalizationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      l2normQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                   input.shape(),
                                   reinterpret_cast<uint8_t*>(output.buffer),
                                   outShape);
        }
    } break;
    case OperationType::LOCAL_RESPONSE_NORMALIZATION: {
        if (!allParametersPresent(5, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        int32_t radius = getScalarData<int32_t>(mOperands[ins[1]]);
        float bias = getScalarData<float>(mOperands[ins[2]]);
        float alpha = getScalarData<float>(mOperands[ins[3]]);
        float beta = getScalarData<float>(mOperands[ins[4]]);

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = genericNormalizationPrepare(input.shape(), &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      localResponseNormFloat32(reinterpret_cast<const float*>(input.buffer),
                                               input.shape(),
                                               radius, bias, alpha, beta,
                                               reinterpret_cast<float*>(output.buffer),
                                               outShape);
        }
    } break;
    case OperationType::RESHAPE: {
        if (!allParametersPresent(2, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        const RunTimeOperandInfo& targetShape = mOperands[ins[1]];

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        success = reshapePrepare(input.shape(),
                                 reinterpret_cast<const int32_t*>(targetShape.buffer),
                                 getNumberOfElements(targetShape.shape()),
                                 &outShape) &&
                  setInfoAndAllocateIfNeeded(&output, outShape) &&
                  reshapeGeneric(reinterpret_cast<const void*>(input.buffer),
                                 input.shape(),
                                 reinterpret_cast<void*>(output.buffer),
                                 outShape);
    } break;
    case OperationType::RESIZE_BILINEAR: {
        if (!allParametersPresent(3, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        int32_t width = getScalarData<int32_t>(mOperands[ins[1]]);
        int32_t height = getScalarData<int32_t>(mOperands[ins[2]]);

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        if (input.type == OperandType::TENSOR_FLOAT32) {
            success = resizeBilinearPrepare(input.shape(),
                                            width, height,
                                            &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape) &&
                      resizeBilinearFloat32(reinterpret_cast<const float*>(input.buffer),
                                            input.shape(),
                                            reinterpret_cast<float*>(output.buffer),
                                            outShape);
        }
    } break;
    case OperationType::DEPTH_TO_SPACE: {
        if (!allParametersPresent(2, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        int32_t blockSize = getScalarData<int32_t>(mOperands[ins[1]]);

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        success = depthToSpacePrepare(input.shape(),
                                      blockSize,
                                      &outShape) &&
                  setInfoAndAllocateIfNeeded(&output, outShape) &&
                  depthToSpaceGeneric(input.buffer,
                                      input.shape(),
                                      blockSize,
                                      output.buffer,
                                      outShape);
    } break;
    case OperationType::SPACE_TO_DEPTH: {
        if (!allParametersPresent(2, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }
        const RunTimeOperandInfo& input = mOperands[ins[0]];
        int32_t blockSize = getScalarData<int32_t>(mOperands[ins[1]]);

        RunTimeOperandInfo& output = mOperands[outs[0]];
        Shape outShape = output.shape();

        success = spaceToDepthPrepare(input.shape(),
                                      blockSize,
                                      &outShape) &&
                  setInfoAndAllocateIfNeeded(&output, outShape) &&
                  spaceToDepthGeneric(input.buffer,
                                      input.shape(),
                                      blockSize,
                                      output.buffer,
                                      outShape);
    } break;
    case OperationType::EMBEDDING_LOOKUP: {
        const RunTimeOperandInfo &values =
            mOperands[ins[EmbeddingLookup::kValueTensor]];
        const RunTimeOperandInfo &lookups =
            mOperands[ins[EmbeddingLookup::kLookupTensor]];
        RunTimeOperandInfo &output =
            mOperands[outs[EmbeddingLookup::kOutputTensor]];

        Shape outputShape;
        EmbeddingLookup lookup(operation, mOperands);

        success = embeddingLookupPrepare(values.shape(), lookups.shape(), &outputShape) &&
            setInfoAndAllocateIfNeeded(&output, outputShape) &&
            lookup.Eval();
    } break;
    case OperationType::HASHTABLE_LOOKUP: {
        const RunTimeOperandInfo &lookups =
            mOperands[ins[HashtableLookup::kLookupTensor]];
        const RunTimeOperandInfo &keys =
            mOperands[ins[HashtableLookup::kKeyTensor]];
        const RunTimeOperandInfo &values =
            mOperands[ins[HashtableLookup::kValueTensor]];

        RunTimeOperandInfo &output =
            mOperands[outs[HashtableLookup::kOutputTensor]];
        RunTimeOperandInfo &hits =
            mOperands[outs[HashtableLookup::kHitsTensor]];

        Shape outputShape, hitShape;
        HashtableLookup lookup(operation, mOperands);

        success = hashtableLookupPrepare(lookups.shape(), keys.shape(), values.shape(),
                                         &outputShape, &hitShape) &&
            setInfoAndAllocateIfNeeded(&output, outputShape) &&
            setInfoAndAllocateIfNeeded(&hits, hitShape) &&
            lookup.Eval();
    } break;
    case OperationType::LSH_PROJECTION: {
        RunTimeOperandInfo &output =
            mOperands[outs[LSHProjection::kOutputTensor]];

        Shape outputShape;
        LSHProjection lsh(operation, mOperands);

        success = LSHProjection::Prepare(operation, mOperands,
                                         &outputShape) &&
            setInfoAndAllocateIfNeeded(&output, outputShape) &&
            lsh.Eval();
    } break;
    case OperationType::LSTM: {
        RunTimeOperandInfo &scratch =
            mOperands[outs[LSTMCell::kScratchBufferTensor]];
        RunTimeOperandInfo &outputStateOut =
            mOperands[outs[LSTMCell::kOutputStateOutTensor]];
        RunTimeOperandInfo &cellStateOut =
            mOperands[outs[LSTMCell::kCellStateOutTensor]];
        RunTimeOperandInfo &output =
            mOperands[outs[LSTMCell::kOutputTensor]];

        Shape scratchShape, outputStateShape, cellStateShape, outputShape;
        LSTMCell lstm_cell(operation, mOperands);

        success = LSTMCell::Prepare(operation, mOperands,
                                    &scratchShape, &outputStateShape,
                                    &cellStateShape, &outputShape) &&
            setInfoAndAllocateIfNeeded(&scratch, scratchShape) &&
            setInfoAndAllocateIfNeeded(&outputStateOut, outputStateShape) &&
            setInfoAndAllocateIfNeeded(&cellStateOut, cellStateShape) &&
            setInfoAndAllocateIfNeeded(&output, outputShape) &&
            lstm_cell.Eval();
    } break;
    case OperationType::RNN: {
        RunTimeOperandInfo &hiddenStateOut =
            mOperands[outs[RNN::kHiddenStateOutTensor]];
        RunTimeOperandInfo &output =
            mOperands[outs[RNN::kOutputTensor]];

        Shape hiddenStateShape, outputShape;
        RNN rnn_cell(operation, mOperands);

        success = RNN::Prepare(operation, mOperands,
                               &hiddenStateShape, &outputShape) &&
            setInfoAndAllocateIfNeeded(&hiddenStateOut, hiddenStateShape) &&
            setInfoAndAllocateIfNeeded(&output, outputShape) &&
            rnn_cell.Eval();
    } break;
    case OperationType::SVDF: {
        RunTimeOperandInfo &stateOut =
            mOperands[outs[SVDF::kStateOutTensor]];
        RunTimeOperandInfo &output =
            mOperands[outs[SVDF::kOutputTensor]];

        Shape stateShape, outputShape;
        SVDF svdf(operation, mOperands);

        success = SVDF::Prepare(operation, mOperands,
                                &stateShape, &outputShape) &&
            setInfoAndAllocateIfNeeded(&stateOut, stateShape) &&
            setInfoAndAllocateIfNeeded(&output, outputShape) &&
            svdf.Eval();
    } break;
    default:
        nnAssert(false);
        break;
} //switch end

*/
