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

#define LOG_TAG "VpuPreparedModel"

#include "VpuPreparedModel.h"
#include "VpuUtils.h"
#include <string>
#include <ctime>
#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>
#include <iostream>
#include <stdio.h>

#include "ncs_lib.h"

#define DISABLE_ALL_QUANT 1
#define file_dump false
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


template <typename T>
T getOperandConstVal(const Model& model, const Operand& operand)
{
    const T* data = reinterpret_cast<const T *>(&model.operandValues[operand.location.offset]);
    return data[0];
}

/*
bool VpuPreparedModel::update_network_count(){
  VpuPreparedModel::network_count = VpuPreparedModel::network_count +1;
  return true;
}

int VpuPreparedModel::get_network_count(){
  return VpuPreparedModel::network_count;
}

*/
// initialize() function
int VpuPreparedModel::network_count_ex =0;

bool VpuPreparedModel::initialize(const Model& model) {
    VLOG(MODEL)<<"VpuPreparedModel::initialize()";
    bool success = false;


    if(VpuPreparedModel::network_count_ex>1){
      VLOG(MODEL) << "More than one graph is required to generate for given model, Model count is " << VpuPreparedModel::network_count_ex;
      VpuPreparedModel::network_count_ex =0;
      return false;
    }


    success = setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);

    if (!success) {
      LOG(ERROR)<<"setRunTimePoolInfosFromHidlMemories failed.";
      return false;
    }

    // code begin for understand model
    VLOG(MODEL) << "Model Compiling for VPU Driver begin ";
    Oertaion_vector nn_ops_vectors;
    network_operations_vector nn_ncs_network;
    NCSoperations nn_ncs_operation;

    if(!nn_ops_vectors.empty())
    nn_ops_vectors.clear();

    if(!nn_ncs_network.empty())
    nn_ncs_network.clear();


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
        case OperationType::MAX_POOL_2D: nn_ncs_operation = MAX_POOL_2D;break;
        case OperationType::SOFTMAX: nn_ncs_operation = SOFTMAX;break;
        case OperationType::FULLY_CONNECTED: nn_ncs_operation = FULLY_CONNECTED;break;
        case OperationType::RESHAPE: nn_ncs_operation = RESHAPE;break;
        default: nn_ncs_operation = NONE;break;
      }
      nn_ncs_network.push_back(nn_ncs_operation);
      nn_ops_vectors.push_back(operation.type);
    }

    bool status;
    status = get_nn_network_from_android(nn_ncs_network);
    if(!status)
      return false;

    Operation_inputs_info operation_operand_info;

    int count = model.operations.size();
    for(int m=0;m<count;m++){
      const auto operation = model.operations[m];
      VLOG(MODEL)<<"Operation: "<<toString(operation);
      operation_operand_info = get_operation_operands_info_model(model, operation);
      bool status = parse_stage_from_android(operation_operand_info);
      VLOG(MODEL) << "Status " << status;
      if(!status){
        return false;
      }
    }

    //VpuPreparedModel::network_count = VpuPreparedModel::network_count + 1;
    std::string network_name = "android-nn-model-";
    std::string network_name_final;
    network_name_final = network_name + std::to_string(network_count_ex);
    VLOG(MODEL) << "Current Network Count is " << network_count_ex << "Model Name is " << network_name_final;

    status = prepare_blob(network_name_final,network_count_ex);
    if(!status){
      VLOG(MODEL) << "Unable to prepare NCS graph";
      return false;
    }

    VLOG(MODEL) << "Model Compiling for VPU Driver: completed";
    //code end for understand model
    nn_ops_vectors.clear();
    nn_ncs_network.clear();


    int val;
    val = ncs_init();
    if (val!=0){
      LOG(ERROR) << "unable to initialize NCS device";
      return false;
    }

    val = ncs_load_graph();
    if (val!=0){
      LOG(ERROR) << "unable to Load graph into NCS device";
      return false;
    }

    return true;
  }


Operation_inputs_info VpuPreparedModel::get_operation_operands_info_model(const Model& model, const Operation& operation){
  Operation_inputs_info stage_info;
  const hidl_vec<uint32_t>& ins = operation.inputs;
  const hidl_vec<uint32_t>& outs = operation.outputs;
  bool success = false;

  /*
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
    };*/

    switch (operation.type) {
        case OperationType::RELU6:{//RELU6 begin
        VLOG(MODEL) << toString(operation);
        /*
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }*/

        const auto input = model.operands[operation.inputs[0]];
        auto output = model.operands[operation.outputs[0]];
		Shape inputShape, outputShape;

		inputShape.type = input.type;
        inputShape.dimensions = input.dimensions;
        inputShape.scale = input.scale;
        inputShape.offset = input.location.offset;

        outputShape.type = output.type;
        outputShape.dimensions = output.dimensions;
        outputShape.scale = output.scale;
        outputShape.offset = output.location.offset;

		if (input.type == OperandType::TENSOR_FLOAT32){
          success = genericActivationPrepare(inputShape, &outputShape);
          if(!success)
              nnAssert(false);
        }
        stage_info.main_operation = RELU6;
        stage_info.input_shape[0] = input.dimensions[0];
        stage_info.input_shape[1] = input.dimensions[1];
        stage_info.input_shape[2] = input.dimensions[2];
        stage_info.input_shape[3] = input.dimensions[3];

        stage_info.output_shape[0] = output.dimensions[0];
        stage_info.output_shape[1] = output.dimensions[1];
        stage_info.output_shape[2] = output.dimensions[2];
        stage_info.output_shape[3] = output.dimensions[3];

        stage_info.post_operation = NONE; break;

		bool DEBUG_RELU6 = false;
        //DEBUG_RELU6 = true;  /*un comment this line to get RELU6 layer debug data*/
        if(DEBUG_RELU6){
			VLOG(MODEL) << " RELU6 input_shape[0]: " << stage_info.input_shape[0];
            VLOG(MODEL) << " RELU6 input_shape[1]: " << stage_info.input_shape[1];
            VLOG(MODEL) << " RELU6 input_shape[2]: " << stage_info.input_shape[2];
            VLOG(MODEL) << " RELU6 input_shape[3]: " << stage_info.input_shape[3];

            VLOG(MODEL) << " RELU6 output_shape[0]: " << stage_info.output_shape[0];
            VLOG(MODEL) << " RELU6 output_shape[1]: " << stage_info.output_shape[1];
            VLOG(MODEL) << " RELU6 output_shape[2]: " << stage_info.output_shape[2];
            VLOG(MODEL) << " RELU6 output_shape[3]: " << stage_info.output_shape[3];
           }
        } break; //RELU6_END
	    case OperationType::RELU1:{//RELU1 begin
        VLOG(MODEL) << toString(operation);
        /*
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }*/

        const auto input = model.operands[operation.inputs[0]];
        auto output = model.operands[operation.outputs[0]];
		Shape inputShape, outputShape;

		inputShape.type = input.type;
        inputShape.dimensions = input.dimensions;
        inputShape.scale = input.scale;
        inputShape.offset = input.location.offset;

        outputShape.type = output.type;
        outputShape.dimensions = output.dimensions;
        outputShape.scale = output.scale;
        outputShape.offset = output.location.offset;

		if (input.type == OperandType::TENSOR_FLOAT32){
          success = genericActivationPrepare(inputShape, &outputShape);
          if(!success)
              nnAssert(false);
        }
        stage_info.main_operation = RELU1;
        stage_info.input_shape[0] = input.dimensions[0];
        stage_info.input_shape[1] = input.dimensions[1];
        stage_info.input_shape[2] = input.dimensions[2];
        stage_info.input_shape[3] = input.dimensions[3];

        stage_info.output_shape[0] = output.dimensions[0];
        stage_info.output_shape[1] = output.dimensions[1];
        stage_info.output_shape[2] = output.dimensions[2];
        stage_info.output_shape[3] = output.dimensions[3];

        stage_info.post_operation = NONE; break;

		bool DEBUG_RELU1 = false;
        //DEBUG_RELU1 = true;  /*un comment this line to get RELU1 layer debug data*/
        if(DEBUG_RELU1){
			VLOG(MODEL) << " RELU1 input_shape[0]: " << stage_info.input_shape[0];
            VLOG(MODEL) << " RELU1 input_shape[1]: " << stage_info.input_shape[1];
            VLOG(MODEL) << " RELU1 input_shape[2]: " << stage_info.input_shape[2];
            VLOG(MODEL) << " RELU1 input_shape[3]: " << stage_info.input_shape[3];

            VLOG(MODEL) << " RELU1 output_shape[0]: " << stage_info.output_shape[0];
            VLOG(MODEL) << " RELU1 output_shape[1]: " << stage_info.output_shape[1];
            VLOG(MODEL) << " RELU1 output_shape[2]: " << stage_info.output_shape[2];
            VLOG(MODEL) << " RELU1 output_shape[3]: " << stage_info.output_shape[3];
          }
        } break; //RELU1_END
	    case OperationType::RELU:{//RELU begin
        VLOG(MODEL) << toString(operation);
        /*
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }*/

        const auto input = model.operands[operation.inputs[0]];
        auto output = model.operands[operation.outputs[0]];
		Shape inputShape, outputShape;

		inputShape.type = input.type;
        inputShape.dimensions = input.dimensions;
        inputShape.scale = input.scale;
        inputShape.offset = input.location.offset;

        outputShape.type = output.type;
        outputShape.dimensions = output.dimensions;
        outputShape.scale = output.scale;
        outputShape.offset = output.location.offset;

		if (input.type == OperandType::TENSOR_FLOAT32){
          success = genericActivationPrepare(inputShape, &outputShape);
          if(!success)
              nnAssert(false);
        }
        stage_info.main_operation = RELU;
        stage_info.input_shape[0] = input.dimensions[0];
        stage_info.input_shape[1] = input.dimensions[1];
        stage_info.input_shape[2] = input.dimensions[2];
        stage_info.input_shape[3] = input.dimensions[3];

        stage_info.output_shape[0] = output.dimensions[0];
        stage_info.output_shape[1] = output.dimensions[1];
        stage_info.output_shape[2] = output.dimensions[2];
        stage_info.output_shape[3] = output.dimensions[3];

        stage_info.post_operation = NONE; break;

		bool DEBUG_RELU = false;
        //DEBUG_RELU = true;  /*un comment this line to get RELU layer debug data*/
        if(DEBUG_RELU){
			VLOG(MODEL) << " RELU input_shape[0]: " << stage_info.input_shape[0];
            VLOG(MODEL) << " RELU input_shape[1]: " << stage_info.input_shape[1];
            VLOG(MODEL) << " RELU input_shape[2]: " << stage_info.input_shape[2];
            VLOG(MODEL) << " RELU input_shape[3]: " << stage_info.input_shape[3];

            VLOG(MODEL) << " RELU output_shape[0]: " << stage_info.output_shape[0];
            VLOG(MODEL) << " RELU output_shape[1]: " << stage_info.output_shape[1];
            VLOG(MODEL) << " RELU output_shape[2]: " << stage_info.output_shape[2];
            VLOG(MODEL) << " RELU output_shape[3]: " << stage_info.output_shape[3];
           }
        } break; //RELU_END
	    case OperationType::TANH:{//TANH begin
        VLOG(MODEL) << toString(operation);
        /*
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }*/

        const auto input = model.operands[operation.inputs[0]];
        auto output = model.operands[operation.outputs[0]];
		Shape inputShape, outputShape;

		inputShape.type = input.type;
        inputShape.dimensions = input.dimensions;
        inputShape.scale = input.scale;
        inputShape.offset = input.location.offset;

        outputShape.type = output.type;
        outputShape.dimensions = output.dimensions;
        outputShape.scale = output.scale;
        outputShape.offset = output.location.offset;

		if (input.type == OperandType::TENSOR_FLOAT32){
          success = genericActivationPrepare(inputShape, &outputShape);
          if(!success)
              nnAssert(false);
        }
        stage_info.main_operation = TANH;
        stage_info.input_shape[0] = input.dimensions[0];
        stage_info.input_shape[1] = input.dimensions[1];
        stage_info.input_shape[2] = input.dimensions[2];
        stage_info.input_shape[3] = input.dimensions[3];

        stage_info.output_shape[0] = output.dimensions[0];
        stage_info.output_shape[1] = output.dimensions[1];
        stage_info.output_shape[2] = output.dimensions[2];
        stage_info.output_shape[3] = output.dimensions[3];

        stage_info.post_operation = NONE; break;

		bool DEBUG_TANH = false;
        //DEBUG_TANH = true;  /*un comment this line to get TANH layer debug data*/
        if(DEBUG_TANH){
			VLOG(MODEL) << " TANH input_shape[0]: " << stage_info.input_shape[0];
            VLOG(MODEL) << " TANH input_shape[1]: " << stage_info.input_shape[1];
            VLOG(MODEL) << " TANH input_shape[2]: " << stage_info.input_shape[2];
            VLOG(MODEL) << " TANH input_shape[3]: " << stage_info.input_shape[3];

            VLOG(MODEL) << " TANH output_shape[0]: " << stage_info.output_shape[0];
            VLOG(MODEL) << " TANH output_shape[1]: " << stage_info.output_shape[1];
            VLOG(MODEL) << " TANH output_shape[2]: " << stage_info.output_shape[2];
            VLOG(MODEL) << " TANH output_shape[3]: " << stage_info.output_shape[3];
           }
        } break; //TANH_END
		case OperationType::LOGISTIC:{//LOGISTIC begin
        VLOG(MODEL) << toString(operation);
        /*
        if (!allParametersPresent(1, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }*/

        const auto input = model.operands[operation.inputs[0]];
        auto output = model.operands[operation.outputs[0]];
		Shape inputShape, outputShape;

		inputShape.type = input.type;
        inputShape.dimensions = input.dimensions;
        inputShape.scale = input.scale;
        inputShape.offset = input.location.offset;

        outputShape.type = output.type;
        outputShape.dimensions = output.dimensions;
        outputShape.scale = output.scale;
        outputShape.offset = output.location.offset;

		if (input.type == OperandType::TENSOR_FLOAT32){
          success = genericActivationPrepare(inputShape, &outputShape);
          if(!success)
              nnAssert(false);
        }
        stage_info.main_operation = LOGISTIC;
        stage_info.input_shape[0] = input.dimensions[0];
        stage_info.input_shape[1] = input.dimensions[1];
        stage_info.input_shape[2] = input.dimensions[2];
        stage_info.input_shape[3] = input.dimensions[3];

        stage_info.output_shape[0] = output.dimensions[0];
        stage_info.output_shape[1] = output.dimensions[1];
        stage_info.output_shape[2] = output.dimensions[2];
        stage_info.output_shape[3] = output.dimensions[3];

        stage_info.post_operation = NONE; break;

		bool DEBUG_LOGISTIC = false;
        //DEBUG_LOGISTIC = true;  /*un comment this line to get LOGISTIC layer debug data*/
        if(DEBUG_LOGISTIC){
			VLOG(MODEL) << " LOGISTIC input_shape[0]: " << stage_info.input_shape[0];
            VLOG(MODEL) << " LOGISTIC input_shape[1]: " << stage_info.input_shape[1];
            VLOG(MODEL) << " LOGISTIC input_shape[2]: " << stage_info.input_shape[2];
            VLOG(MODEL) << " LOGISTIC input_shape[3]: " << stage_info.input_shape[3];

            VLOG(MODEL) << " LOGISTIC output_shape[0]: " << stage_info.output_shape[0];
            VLOG(MODEL) << " LOGISTIC output_shape[1]: " << stage_info.output_shape[1];
            VLOG(MODEL) << " LOGISTIC output_shape[2]: " << stage_info.output_shape[2];
            VLOG(MODEL) << " LOGISTIC output_shape[3]: " << stage_info.output_shape[3];
        }
        } break; //LOGISTIC_END
        case OperationType::CONV_2D: {
			VLOG(MODEL) << toString(operation);
			const size_t inCount = operation.inputs.size();
			const auto input = model.operands[operation.inputs[0]];
			const auto filter = model.operands[operation.inputs[1]];
			const auto bias = model.operands[operation.inputs[2]];
			auto output = model.operands[operation.outputs[0]];

			Shape inputShape,filterShape,biasShape,outputShape;

			inputShape.type = input.type;
			inputShape.dimensions = input.dimensions;
			inputShape.scale = input.scale;
			inputShape.offset = input.location.offset;

			filterShape.type = filter.type;
			filterShape.dimensions = filter.dimensions;
			filterShape.scale = filter.scale;
			filterShape.offset = filter.location.offset;

			biasShape.type = bias.type;
			biasShape.dimensions = bias.dimensions;
			biasShape.scale = bias.scale;
			biasShape.offset = bias.location.offset;

			outputShape.type = output.type;
			outputShape.dimensions = output.dimensions;
			outputShape.scale = output.scale;
			outputShape.offset = output.location.offset;

			int32_t padding_left, padding_right;
			int32_t padding_top, padding_bottom;
			int32_t stride_width, stride_height;
			int32_t activation;

			if (inCount == 10) {
				padding_left     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[3]]);
				padding_right    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[4]]);
				padding_top      = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[5]]);
				padding_bottom   = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);
				stride_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[7]]);
				stride_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[8]]);
				activation       = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[9]]);
            }
            else {
				int32_t padding_implicit = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[3]]);
				stride_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[4]]);
				stride_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[5]]);
				activation       = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);

				int32_t input_width  = input.dimensions[2];
				int32_t input_height = input.dimensions[1];
                int32_t filter_width  = filter.dimensions[2];
                int32_t filter_height = filter.dimensions[1];

                calculateExplicitPadding(input_width, stride_width,
                                   filter_width, padding_implicit,
                                   &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height,
                                   filter_height, padding_implicit,
                                   &padding_top, &padding_bottom);
            }

            if (input.type == OperandType::TENSOR_FLOAT32){
                success = convPrepare(inputShape, filterShape, biasShape,
                                   padding_left, padding_right,
                                   padding_top, padding_bottom,
                                   stride_width, stride_height,
                                   &outputShape);
                if(!success)
                    nnAssert(false);
            }
            stage_info.main_operation = CONV_2D;
            stage_info.input_shape[0] = input.dimensions[0];
            stage_info.input_shape[1] = input.dimensions[1];
            stage_info.input_shape[2] = input.dimensions[2];
            stage_info.input_shape[3] = input.dimensions[3];

            stage_info.kernel_shape[0] = filter.dimensions[1];
            stage_info.kernel_shape[1] = filter.dimensions[2];
            stage_info.kernel_shape[2] = filter.dimensions[3];
            stage_info.kernel_shape[3] = filter.dimensions[0];

            stage_info.bias_shape[0] = bias.dimensions[0];
            stage_info.bias_shape[1] = 1;
            stage_info.bias_shape[2] = 1;
            stage_info.bias_shape[3] = 1;

            stage_info.output_shape[0] = output.dimensions[0];
            stage_info.output_shape[1] = output.dimensions[1];
            stage_info.output_shape[2] = output.dimensions[2];
            stage_info.output_shape[3] = output.dimensions[3];

            //stage_info.kernel_buffer = getOperandbuffer(model,model.operands[operation.inputs[1]]);
            if(filter.type == OperandType::TENSOR_FLOAT32 && filter.lifetime == OperandLifeTime::CONSTANT_COPY){
                stage_info.kernel_buffer = reinterpret_cast<const float *>(&model.operandValues[filter.location.offset]);
            }

            if(bias.type == OperandType::TENSOR_FLOAT32 && bias.lifetime == OperandLifeTime::CONSTANT_COPY){
                stage_info.bias_buffer = reinterpret_cast<const float*>(&model.operandValues[bias.location.offset]);
            }

            if(filter.type == OperandType::TENSOR_FLOAT32 && filter.lifetime == OperandLifeTime::CONSTANT_REFERENCE){
                auto poolIndex = filter.location.poolIndex;
                auto& r = mPoolInfos[poolIndex];
                stage_info.kernel_buffer = reinterpret_cast<const float *>(r.buffer+filter.location.offset);
            }

            if(bias.type == OperandType::TENSOR_FLOAT32 && bias.lifetime == OperandLifeTime::CONSTANT_REFERENCE){
                auto poolIndex = bias.location.poolIndex;
                auto& r = mPoolInfos[poolIndex];
                stage_info.bias_buffer = reinterpret_cast<const float *>(r.buffer+bias.location.offset);
            }
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

        bool DEBUG_CONV_2D = false;
        //DEBUG_CONV_2D = true;  //un comment this line to get CONV_2D layer debug data
        if(DEBUG_CONV_2D){
            VLOG(MODEL) << " CONV_2D padding_left: " << padding_left;
            VLOG(MODEL) << " CONV_2D padding_right: " << padding_right;
            VLOG(MODEL) << " CONV_2D padding_top: " << padding_top;
            VLOG(MODEL) << " CONV_2D padding_bottom: " << padding_bottom;
		        VLOG(MODEL) << " CONV_2D stride_width: " << stage_info.stride_width;
            VLOG(MODEL) << " CONV_2D stride_height: " << stage_info.stride_height;

            VLOG(MODEL) << " CONV_2D input_shape[0]: " << stage_info.input_shape[0];
            VLOG(MODEL) << " CONV_2D input_shape[1]: " << stage_info.input_shape[1];
            VLOG(MODEL) << " CONV_2D input_shape[2]: " << stage_info.input_shape[2];
            VLOG(MODEL) << " CONV_2D input_shape[3]: " << stage_info.input_shape[3];

            VLOG(MODEL) << " CONV_2D kernel_shape[0]: " << stage_info.kernel_shape[0];
            VLOG(MODEL) << " CONV_2D kernel_shape[1]: " << stage_info.kernel_shape[1];
            VLOG(MODEL) << " CONV_2D kernel_shape[2]: " << stage_info.kernel_shape[2];
            VLOG(MODEL) << " CONV_2D kernel_shape[3]: " << stage_info.kernel_shape[3];

            VLOG(MODEL) << " CONV_2D bias_shape[0]: " << stage_info.bias_shape[0];
            VLOG(MODEL) << " CONV_2D bias_shape[1]: " << stage_info.bias_shape[1];
            VLOG(MODEL) << " CONV_2D bias_shape[2]: " << stage_info.bias_shape[2];
            VLOG(MODEL) << " CONV_2D bias_shape[3]: " << stage_info.bias_shape[3];

            VLOG(MODEL) << " CONV_2D output_shape[0]: " << stage_info.output_shape[0];
            VLOG(MODEL) << " CONV_2D output_shape[1]: " << stage_info.output_shape[1];
            VLOG(MODEL) << " CONV_2D output_shape[2]: " << stage_info.output_shape[2];
            VLOG(MODEL) << " CONV_2D output_shape[3]: " << stage_info.output_shape[3];
            }
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
              VLOG(MODEL) << "CONV_2D nk_ele: " << nk_ele;
              fseek(fp, 0, SEEK_END);
              fwrite(reinterpret_cast<const float*>(stage_info.kernel_buffer),sizeof(float),nk_ele,fp);
              fclose(fp);
            }

            if(stage_info.bias_data){
              fp = fopen("/data/ncs_graph_data","ab+");
              if(!fp) ALOGE("unable to open the file /data/ncs_graph_data ");
              VLOG(MODEL) << "CONV_2D nb_ele: " << nb_ele;
              fseek(fp, 0, SEEK_END);
              fwrite(reinterpret_cast<const float*>(stage_info.bias_buffer),sizeof(float),nb_ele,fp);
              fclose(fp);
            }
#endif
           /*for(int i=0;i<nk_ele;i++){
             ALOGD("kernel_data[%d]: %f",i,*(stage_info.kernel_buffer+i));
           }*/
        } break;//CONV_2D end

	    case OperationType::DEPTHWISE_CONV_2D: {
			VLOG(MODEL) << toString(operation);
            const size_t inCount = operation.inputs.size();
            const auto input = model.operands[operation.inputs[0]];
            const auto filter = model.operands[operation.inputs[1]];
            const auto bias = model.operands[operation.inputs[2]];

            auto output = model.operands[operation.outputs[0]];

			Shape inputShape,filterShape,biasShape,outputShape;

			inputShape.type = input.type;
			inputShape.dimensions = input.dimensions;
			inputShape.scale = input.scale;
			inputShape.offset = input.location.offset;

			filterShape.type = filter.type;
			filterShape.dimensions = filter.dimensions;
			filterShape.scale = filter.scale;
			filterShape.offset = filter.location.offset;

			biasShape.type = bias.type;
			biasShape.dimensions = bias.dimensions;
			biasShape.scale = bias.scale;
			biasShape.offset = bias.location.offset;

			outputShape.type = output.type;
			outputShape.dimensions = output.dimensions;
			outputShape.scale = output.scale;
			outputShape.offset = output.location.offset;

			int32_t padding_left, padding_right;
			int32_t padding_top, padding_bottom;
			int32_t stride_width, stride_height;
			int32_t depth_multiplier;
			int32_t activation;

			if (inCount == 11) {
				  padding_left     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[3]]);
				  padding_right    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[4]]);
				  padding_top      = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[5]]);
				  padding_bottom   = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);
				  stride_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[7]]);
				  stride_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[8]]);
				  depth_multiplier = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[9]]);
				  activation       = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[10]]);
				}
				else {
				  int32_t padding_implicit = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[3]]);
				  stride_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[4]]);
				  stride_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[5]]);
				  depth_multiplier = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);
				  activation       = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[7]]);

				  int32_t input_width  = input.dimensions[2];
				  int32_t input_height = input.dimensions[1];
				  int32_t filter_width  = filter.dimensions[2];
				  int32_t filter_height = filter.dimensions[1];

				  calculateExplicitPadding(input_width, stride_width,
									   filter_width, padding_implicit,
									   &padding_left, &padding_right);
				  calculateExplicitPadding(input_height, stride_height,
									   filter_height, padding_implicit,
									   &padding_top, &padding_bottom);
			   }
			   if (input.type == OperandType::TENSOR_FLOAT32){
				 success = depthwiseConvPrepare(inputShape, filterShape, biasShape,
									   padding_left, padding_right,
									   padding_top, padding_bottom,
									   stride_width, stride_height,
									   &outputShape);
				  if(!success)
					  nnAssert(false);
			   }
			   stage_info.main_operation = DEPTHWISE_CONV_2D;
			   stage_info.input_shape[0] = input.dimensions[0];
			   stage_info.input_shape[1] = input.dimensions[1];
			   stage_info.input_shape[2] = input.dimensions[2];
			   stage_info.input_shape[3] = input.dimensions[3];

			   stage_info.kernel_shape[0] = filter.dimensions[1];
			   stage_info.kernel_shape[1] = filter.dimensions[2];
			   stage_info.kernel_shape[2] = filter.dimensions[3];
			   stage_info.kernel_shape[3] = filter.dimensions[0];

			   stage_info.bias_shape[0] = bias.dimensions[0];
			   stage_info.bias_shape[1] = 1;
			   stage_info.bias_shape[2] = 1;
			   stage_info.bias_shape[3] = 1;

			   stage_info.output_shape[0] = output.dimensions[0];
			   stage_info.output_shape[1] = output.dimensions[1];
			   stage_info.output_shape[2] = output.dimensions[2];
			   stage_info.output_shape[3] = output.dimensions[3];

			   //stage_info.kernel_buffer = getOperandbuffer(model,model.operands[operation.inputs[1]]);
			   if(filter.type == OperandType::TENSOR_FLOAT32 && filter.lifetime == OperandLifeTime::CONSTANT_COPY){
				 stage_info.kernel_buffer = reinterpret_cast<const float *>(&model.operandValues[filter.location.offset]);
			   }

			   if(bias.type == OperandType::TENSOR_FLOAT32 && bias.lifetime == OperandLifeTime::CONSTANT_COPY){
				 stage_info.bias_buffer = reinterpret_cast<const float*>(&model.operandValues[bias.location.offset]);
			   }

			   if(filter.type == OperandType::TENSOR_FLOAT32 && filter.lifetime == OperandLifeTime::CONSTANT_REFERENCE){
				 auto poolIndex = filter.location.poolIndex;
				 auto& r = mPoolInfos[poolIndex];
				 stage_info.kernel_buffer = reinterpret_cast<const float *>(r.buffer+filter.location.offset);
			   }

			   if(bias.type == OperandType::TENSOR_FLOAT32 && bias.lifetime == OperandLifeTime::CONSTANT_REFERENCE){
				 auto poolIndex = bias.location.poolIndex;
				 auto& r = mPoolInfos[poolIndex];
				 stage_info.bias_buffer = reinterpret_cast<const float *>(r.buffer+bias.location.offset);
			   }

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

		    bool DEBUG_DEPTHWISE_CONV_2D = false;
		    //DEBUG_DEPTHWISE_CONV_2D = true;  //un comment this line to get DEPTHWISE_CONV_2D layer debug data
		    if(DEBUG_DEPTHWISE_CONV_2D){
				VLOG(MODEL) << " DEPTHWISE_CONV_2D padding_left: " << padding_left;
				VLOG(MODEL) << " DEPTHWISE_CONV_2D padding_right: " << padding_right;
				VLOG(MODEL) << " DEPTHWISE_CONV_2D padding_top: " << padding_top;
				VLOG(MODEL) << " DEPTHWISE_CONV_2D padding_bottom: " << padding_bottom;
				VLOG(MODEL) << " DEPTHWISE_CONV_2D stride_width: " << stage_info.stride_width;
				VLOG(MODEL) << " DEPTHWISE_CONV_2D stride_height: " << stage_info.stride_height;

				VLOG(MODEL) << " DEPTHWISE_CONV_2D input_shape[0]: " << stage_info.input_shape[0];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D input_shape[1]: " << stage_info.input_shape[1];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D input_shape[2]: " << stage_info.input_shape[2];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D input_shape[3]: " << stage_info.input_shape[3];

				VLOG(MODEL) << " DEPTHWISE_CONV_2D kernel_shape[0]: " << stage_info.kernel_shape[0];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D kernel_shape[1]: " << stage_info.kernel_shape[1];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D kernel_shape[2]: " << stage_info.kernel_shape[2];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D kernel_shape[3]: " << stage_info.kernel_shape[3];

				VLOG(MODEL) << " DEPTHWISE_CONV_2D bias_shape[0]: " << stage_info.bias_shape[0];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D bias_shape[1]: " << stage_info.bias_shape[1];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D bias_shape[2]: " << stage_info.bias_shape[2];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D bias_shape[3]: " << stage_info.bias_shape[3];

				VLOG(MODEL) << " DEPTHWISE_CONV_2D output_shape[0]: " << stage_info.output_shape[0];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D output_shape[1]: " << stage_info.output_shape[1];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D output_shape[2]: " << stage_info.output_shape[2];
				VLOG(MODEL) << " DEPTHWISE_CONV_2D output_shape[3]: " << stage_info.output_shape[3];
		    }
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
                      fwrite(reinterpret_cast<const float*>(stage_info.kernel_buffer),sizeof(float),nk_ele,fp);
                      fclose(fp);
                    }

                    if(stage_info.bias_data){
                      fp = fopen("/data/ncs_graph_data","ab+");
                      if(!fp) ALOGE("unable to open the file /data/ncs_graph_data ");
                      VLOG(VPUEXE) << "DEPTHWISE_CONV_2D nb_ele: " << nb_ele;
                      fseek(fp, 0, SEEK_END);
                      fwrite(reinterpret_cast<const float*>(stage_info.bias_buffer),sizeof(float),nb_ele,fp);
                      fclose(fp);
                    }
#endif
		} break;//DEPTHWISE_CONV_2D end

      case OperationType::AVERAGE_POOL_2D: {
			VLOG(MODEL) << toString(operation);
			const size_t inCount = operation.inputs.size();
			const auto input = model.operands[operation.inputs[0]];
			auto output = model.operands[operation.outputs[0]];

			Shape inputShape, outputShape;

			inputShape.type = input.type;
			inputShape.dimensions = input.dimensions;
			inputShape.scale = input.scale;
			inputShape.offset = input.location.offset;

			outputShape.type = output.type;
			outputShape.dimensions = output.dimensions;
			outputShape.scale = output.scale;
			outputShape.offset = output.location.offset;

			int32_t padding_left, padding_right;
			int32_t padding_top, padding_bottom;
			int32_t stride_width, stride_height;
			int32_t filter_width, filter_height;
			int32_t activation;

			if (inCount == 10) {
				padding_left     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[1]]);
				padding_right    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[2]]);
				padding_top      = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[3]]);
				padding_bottom   = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[4]]);
				stride_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[5]]);
				stride_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);
				filter_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[7]]);
				filter_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[8]]);
				activation       = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[9]]);
            }
            else {
				int32_t padding_implicit = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[1]]);
				stride_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[2]]);
				stride_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[3]]);
				filter_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[4]]);
				filter_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[5]]);
				activation       = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);

				int32_t input_width  = input.dimensions[2];
				int32_t input_height = input.dimensions[1];

                calculateExplicitPadding(input_width, stride_width,
                                   filter_width, padding_implicit,
                                   &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height,
                                   filter_height, padding_implicit,
                                   &padding_top, &padding_bottom);
            }

			if (input.type == OperandType::TENSOR_FLOAT32){
                success = genericPoolingPrepare(inputShape,
                             padding_left, padding_right,
                             padding_top, padding_bottom,
                             stride_width, stride_height,
                             filter_width, filter_height,
                             &outputShape);
                if(!success)
                    nnAssert(false);
            }

            stage_info.main_operation = AVERAGE_POOL_2D;
            stage_info.input_shape[0] = input.dimensions[0];
            stage_info.input_shape[1] = input.dimensions[1];
            stage_info.input_shape[2] = input.dimensions[2];
            stage_info.input_shape[3] = input.dimensions[3];

            stage_info.kernel_shape[0] = filter_width;
            stage_info.kernel_shape[1] = filter_height;
            stage_info.kernel_shape[2] = 1;
            stage_info.kernel_shape[3] = 1;

            stage_info.output_shape[0] = output.dimensions[0];
            stage_info.output_shape[1] = output.dimensions[1];
            stage_info.output_shape[2] = output.dimensions[2];
            stage_info.output_shape[3] = output.dimensions[3];

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
            VLOG(MODEL) << " AVERAGE_POOL_2D padding_left: " << padding_left;
            VLOG(MODEL) << " AVERAGE_POOL_2D padding_right: " << padding_right;
            VLOG(MODEL) << " AVERAGE_POOL_2D padding_top: " << padding_top;
            VLOG(MODEL) << " AVERAGE_POOL_2D padding_bottom: " << padding_bottom;
		        VLOG(MODEL) << " AVERAGE_POOL_2D stride_width: " << stage_info.stride_width;
            VLOG(MODEL) << " AVERAGE_POOL_2D stride_height: " << stage_info.stride_height;

            VLOG(MODEL) << " AVERAGE_POOL_2D input_shape[0]: " << stage_info.input_shape[0];
            VLOG(MODEL) << " AVERAGE_POOL_2D input_shape[1]: " << stage_info.input_shape[1];
            VLOG(MODEL) << " AVERAGE_POOL_2D input_shape[2]: " << stage_info.input_shape[2];
            VLOG(MODEL) << " AVERAGE_POOL_2D input_shape[3]: " << stage_info.input_shape[3];

            VLOG(MODEL) << " AVERAGE_POOL_2D kernel_shape[0]: " << stage_info.kernel_shape[0];
            VLOG(MODEL) << " AVERAGE_POOL_2D kernel_shape[1]: " << stage_info.kernel_shape[1];
            VLOG(MODEL) << " AVERAGE_POOL_2D kernel_shape[2]: " << stage_info.kernel_shape[2];
            VLOG(MODEL) << " AVERAGE_POOL_2D kernel_shape[3]: " << stage_info.kernel_shape[3];

            VLOG(MODEL) << " AVERAGE_POOL_2D output_shape[0]: " << stage_info.output_shape[0];
            VLOG(MODEL) << " AVERAGE_POOL_2D output_shape[1]: " << stage_info.output_shape[1];
            VLOG(MODEL) << " AVERAGE_POOL_2D output_shape[2]: " << stage_info.output_shape[2];
            VLOG(MODEL) << " AVERAGE_POOL_2D output_shape[3]: " << stage_info.output_shape[3];
            }
        } break;//AVERAGE_POOL_2D end

        case OperationType::MAX_POOL_2D: {
        VLOG(MODEL) << toString(operation);
        const size_t inCount = operation.inputs.size();
        const auto input = model.operands[operation.inputs[0]];
        auto output = model.operands[operation.outputs[0]];

        Shape inputShape, outputShape;

        inputShape.type = input.type;
        inputShape.dimensions = input.dimensions;
        inputShape.scale = input.scale;
        inputShape.offset = input.location.offset;

        outputShape.type = output.type;
        outputShape.dimensions = output.dimensions;
        outputShape.scale = output.scale;
        outputShape.offset = output.location.offset;

        int32_t padding_left, padding_right;
        int32_t padding_top, padding_bottom;
        int32_t stride_width, stride_height;
        int32_t filter_width, filter_height;
        int32_t activation;

        if (inCount == 10) {
          padding_left     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[1]]);
          padding_right    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[2]]);
          padding_top      = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[3]]);
          padding_bottom   = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[4]]);
          stride_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[5]]);
          stride_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);
          filter_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[7]]);
          filter_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[8]]);
          activation       = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[9]]);
              }
              else {
          int32_t padding_implicit = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[1]]);
          stride_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[2]]);
          stride_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[3]]);
          filter_width     = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[4]]);
          filter_height    = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[5]]);
          activation       = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);

          int32_t input_width  = input.dimensions[2];
          int32_t input_height = input.dimensions[1];

                  calculateExplicitPadding(input_width, stride_width,
                                     filter_width, padding_implicit,
                                     &padding_left, &padding_right);
                  calculateExplicitPadding(input_height, stride_height,
                                     filter_height, padding_implicit,
                                     &padding_top, &padding_bottom);
              }

        if (input.type == OperandType::TENSOR_FLOAT32){
                  success = genericPoolingPrepare(inputShape,
                               padding_left, padding_right,
                               padding_top, padding_bottom,
                               stride_width, stride_height,
                               filter_width, filter_height,
                               &outputShape);
                  if(!success)
                      nnAssert(false);
              }

              stage_info.main_operation = MAX_POOL_2D;
              stage_info.input_shape[0] = input.dimensions[0];
              stage_info.input_shape[1] = input.dimensions[1];
              stage_info.input_shape[2] = input.dimensions[2];
              stage_info.input_shape[3] = input.dimensions[3];

              stage_info.kernel_shape[0] = filter_width;
              stage_info.kernel_shape[1] = filter_height;
              stage_info.kernel_shape[2] = 1;
              stage_info.kernel_shape[3] = 1;

              stage_info.output_shape[0] = output.dimensions[0];
              stage_info.output_shape[1] = output.dimensions[1];
              stage_info.output_shape[2] = output.dimensions[2];
              stage_info.output_shape[3] = output.dimensions[3];

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

          bool DEBUG_MAX_POOL_2D = false;
          //DEBUG_MAX_POOL_2D = true;  //un comment this line to get MAX_POOL_2D layer debug data
          if(DEBUG_MAX_POOL_2D){
              VLOG(MODEL) << " MAX_POOL_2D padding_left: " << padding_left;
              VLOG(MODEL) << " MAX_POOL_2D padding_right: " << padding_right;
              VLOG(MODEL) << " MAX_POOL_2D padding_top: " << padding_top;
              VLOG(MODEL) << " MAX_POOL_2D padding_bottom: " << padding_bottom;
              VLOG(MODEL) << " MAX_POOL_2D stride_width: " << stage_info.stride_width;
              VLOG(MODEL) << " MAX_POOL_2D stride_height: " << stage_info.stride_height;

              VLOG(MODEL) << " MAX_POOL_2D input_shape[0]: " << stage_info.input_shape[0];
              VLOG(MODEL) << " MAX_POOL_2D input_shape[1]: " << stage_info.input_shape[1];
              VLOG(MODEL) << " MAX_POOL_2D input_shape[2]: " << stage_info.input_shape[2];
              VLOG(MODEL) << " MAX_POOL_2D input_shape[3]: " << stage_info.input_shape[3];

              VLOG(MODEL) << " MAX_POOL_2D kernel_shape[0]: " << stage_info.kernel_shape[0];
              VLOG(MODEL) << " MAX_POOL_2D kernel_shape[1]: " << stage_info.kernel_shape[1];
              VLOG(MODEL) << " MAX_POOL_2D kernel_shape[2]: " << stage_info.kernel_shape[2];
              VLOG(MODEL) << " MAX_POOL_2D kernel_shape[3]: " << stage_info.kernel_shape[3];

              VLOG(MODEL) << " MAX_POOL_2D output_shape[0]: " << stage_info.output_shape[0];
              VLOG(MODEL) << " MAX_POOL_2D output_shape[1]: " << stage_info.output_shape[1];
              VLOG(MODEL) << " MAX_POOL_2D output_shape[2]: " << stage_info.output_shape[2];
              VLOG(MODEL) << " MAX_POOL_2D output_shape[3]: " << stage_info.output_shape[3];
              }
          } break;//MAX_POOL_2D end

        case OperationType::SOFTMAX: {//SOFTMAX begin
        VLOG(MODEL) << toString(operation);
        /*
        if (!allParametersPresent(2, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }*/

        const auto input = model.operands[operation.inputs[0]];
        auto output = model.operands[operation.outputs[0]];
		    Shape inputShape, outputShape;
		    float beta = getOperandConstVal<float>(model,model.operands[operation.inputs[1]]);

		    inputShape.type = input.type;
        inputShape.dimensions = input.dimensions;
        inputShape.scale = input.scale;
        inputShape.offset = input.location.offset;

        outputShape.type = output.type;
        outputShape.dimensions = output.dimensions;
        outputShape.scale = output.scale;
        outputShape.offset = output.location.offset;


    		if (input.type == OperandType::TENSOR_FLOAT32){
              success = genericActivationPrepare(inputShape, &outputShape);
              if(!success)
                  nnAssert(false);
            }


        stage_info.main_operation = SOFTMAX;

        if(input.dimensions.size()==2){
          stage_info.input_shape[0] = 1;
          stage_info.input_shape[1] = 1;
          stage_info.input_shape[2] = 1;
          stage_info.input_shape[3] = input.dimensions[0] * input.dimensions[1];
        }
        else if(input.dimensions.size()==4){
          stage_info.input_shape[0] = 1;
          stage_info.input_shape[1] = 1;
          stage_info.input_shape[2] = 1;
          stage_info.input_shape[3] = input.dimensions[0] * input.dimensions[1] * input.dimensions[2] * input.dimensions[3];
        }

        if(output.dimensions.size()==2){
          stage_info.output_shape[0] = 1;
          stage_info.output_shape[1] = 1;
          stage_info.output_shape[2] = output.dimensions[0];
          stage_info.output_shape[3] = output.dimensions[1];
        }
        else if(output.dimensions.size()==4){
          stage_info.output_shape[0] = output.dimensions[0];
          stage_info.output_shape[1] = output.dimensions[1];
          stage_info.output_shape[2] = output.dimensions[2];
          stage_info.output_shape[3] = output.dimensions[3];
        }

		    stage_info.beta = beta;
        stage_info.kernel_data = false;
        stage_info.bias_data = false;
        stage_info.op_params_data = true;

        stage_info.post_operation = NONE;

		    bool DEBUG_SOFTMAX = false;
        DEBUG_SOFTMAX = true;  /*un comment this line to get SOFTMAX layer debug data*/
        if(DEBUG_SOFTMAX){
          VLOG(MODEL) << " SOFTMAX beta:"  << stage_info.beta;
          for(int i=0; i<4;i++)
          VLOG(MODEL) << " SOFTMAX input_shape[]:"  << stage_info.input_shape[i];

          for(int i=0; i< 4;i++)
          VLOG(MODEL) << " SOFTMAX output_shape[]:" << stage_info.output_shape[i];
        }
        } break; //SOFTMAX_END
        case OperationType::RESHAPE: {//RESHAPE begin
        VLOG(MODEL) << toString(operation);
        /*
        if (!allParametersPresent(2, 1)) {
            return ANEURALNETWORKS_BAD_DATA;
        }*/

        const auto input = model.operands[operation.inputs[0]];
		    const auto target = model.operands[operation.inputs[1]];
        auto output = model.operands[operation.outputs[0]];
		    Shape inputShape, targetShape, outputShape;

		    inputShape.type = input.type;
        inputShape.dimensions = input.dimensions;
        inputShape.scale = input.scale;
        inputShape.offset = input.location.offset;

		    targetShape.type = target.type;
        targetShape.dimensions = target.dimensions;
        targetShape.scale = target.scale;
        targetShape.offset = target.location.offset;

        outputShape.type = output.type;
        outputShape.dimensions = output.dimensions;
        outputShape.scale = output.scale;
        outputShape.offset = output.location.offset;

        const int32_t *buffer;
        buffer = reinterpret_cast<const int32_t *>(&model.operandValues[target.location.offset]);

		    success = reshapePrepare(inputShape,
                               buffer,
                               getNumberOfElements(targetShape),
                               &outputShape);

        stage_info.main_operation = RESHAPE;
        for(int i=0; i<input.dimensions.size();i++)
        stage_info.input_shape[i] = input.dimensions[i];

        /*stage_info.input_shape[1] = input.dimensions[1];
        stage_info.input_shape[2] = input.dimensions[2];
        stage_info.input_shape[3] = input.dimensions[3];*/

        for(int i=0; i<output.dimensions.size();i++)
        stage_info.output_shape[i] = output.dimensions[i];
        /*
        stage_info.output_shape[1] = output.dimensions[1];
        stage_info.output_shape[2] = output.dimensions[2];
        stage_info.output_shape[3] = output.dimensions[3];*/

        stage_info.kernel_data = false;
        stage_info.bias_data = false;
        stage_info.op_params_data = false;

        stage_info.post_operation = NONE;

		    bool DEBUG_RESHAPE = false;
        DEBUG_RESHAPE = true;  /*un comment this line to get RESHAPE layer debug data*/
        if(DEBUG_RESHAPE){
            for(int i=0; i<input.dimensions.size();i++)
			      VLOG(MODEL) << " RESHAPE input_shape[" << i << "]: " << stage_info.input_shape[i];

            for(int i=0; i< output.dimensions.size();i++)
			      VLOG(MODEL) << " RESHAPE output_shape[" << i << "]: " << stage_info.output_shape[i];
        }
      } break; //RESHAPE_END
    }
  return stage_info;
}

//isOperationSupported() function

bool VpuPreparedModel::isOperationSupported(const Operation& operation, const Model& model)
{

  VLOG(MODEL) << "Check for Operation support on VPU:  " << getOperationName(operation.type);

  VLOG(MODEL) << toString(operation);
  const auto input = model.operands[operation.inputs[0]];
  auto output = model.operands[operation.outputs[0]];
  VLOG(MODEL) << "SRISTI Input dimensions: ( " << input.dimensions[0] << ", "<< input.dimensions[1] << ", "<< input.dimensions[2] << ", "<< input.dimensions[3] << ")";
  VLOG(MODEL) << "SRISTI Output dimensions: ( " << output.dimensions[0] << ", "<< output.dimensions[1] << ", "<< output.dimensions[2] << ", "<< output.dimensions[3] << ")";

  if(operation.type == OperationType::CONV_2D){
        const auto input1 = model.operands[operation.inputs[1]];
        VLOG(MODEL) << "SRISTI Filter dimensions: ( " << input1.dimensions[0] << ", "<< input1.dimensions[1] << ", "<< input1.dimensions[2] << ", "<< input1.dimensions[3] << ")";
  }

#define VLOG_CHECKFAIL(fail) ALOGD("Check failed:ANEURALNETWORKS_TENSOR_QUANT8_ASYMM Operand is not supported by VPU %s", fail)

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

    switch(operation.type) {

        case OperationType::RELU:
        {
          VLOG(MODEL) << "RELU is supported operation ";
          break;
        }
        case OperationType::RELU1:
        {
          VLOG(MODEL) << "RELU1 is supported operation ";
          break;
        }
        case OperationType::RELU6:
        {
          VLOG(MODEL) << "RELU6 is supported operation ";
          break;
        }
        case OperationType::TANH:
        {
          VLOG(MODEL) << "TANH is supported operation ";
          break;
        }
        case OperationType::LOGISTIC:
        {
          VLOG(MODEL) << "LOGISTIC is supported operation ";
          break;
        }
        case OperationType::CONV_2D:
        {
          VLOG(MODEL) << "CONV_2D is supported operation ";
          break;
        }
        case OperationType::DEPTHWISE_CONV_2D:
        {
          const size_t inCount = operation.inputs.size();
          int32_t depth_multiplier;
          if(inCount==11)
          depth_multiplier = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[9]]);
          else
          depth_multiplier = getOperandConstVal<int32_t>(model,model.operands[operation.inputs[6]]);

          if(depth_multiplier!=1){
            VLOG(MODEL) << "depth_multiplier: "<< depth_multiplier;
            VLOG(MODEL) << "DEPTHWISE_CONV_2D is not supported operation ";
            return false;
          }
          VLOG(MODEL) << "DEPTHWISE_CONV_2D is supported operation ";
          break;
        }
        case OperationType::AVERAGE_POOL_2D:
        {
          VLOG(MODEL) << "AVERAGE_POOL_2D is supported operation ";
          break;
        }/*
        case OperationType::L2_POOL_2D:
        {
          VLOG(MODEL) << "L2_POOL_2D is supported operation ";
          break;
        }*/
        case OperationType::MAX_POOL_2D:
        {
          VLOG(MODEL) << "MAX_POOL_2D is supported operation ";
          break;
        }
        case OperationType::SOFTMAX:
        {
          float beta = getOperandConstVal<float>(model,model.operands[operation.inputs[1]]);


          if(beta!=1.0){
            VLOG(MODEL) << "SOFTMAX beta : " << beta;
            VLOG(MODEL) << "SOFTMAX is not supported operation ";
            return false;
          }
          VLOG(MODEL) << "SOFTMAX is supported operation ";
          break;
        }/*
        case OperationType::FULLY_CONNECTED:
        {
          VLOG(MODEL) << "FULLY_CONNECTED is supported operation ";
          break;
        }
        case OperationType::L2_NORMALIZATION:
        {
          VLOG(MODEL) << "L2_NORMALIZATION is supported operation ";
          break;
        }*/
        case OperationType::RESHAPE:
        {
          VLOG(MODEL) << "RESHAPE is supported operation "; //ANEURALNETWOKRS_RESHAPE
          break;
        }

        default:
           VLOG(MODEL) << getOperationName(operation.type) << " Operation not supported on VPU";
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
    VLOG(MODEL) << "deinitialize";
    int val;
    val = ncs_unload_graph();
    if (val != 0)
    VLOG(MODEL) << "unable to unload graph from NCS";

    val = ncs_deinit();
    if (val != 0)
    VLOG(MODEL) << "unable to deinitialize NCS device";
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
    ErrorStatus executionStatus =
            n == ANEURALNETWORKS_NO_ERROR ? ErrorStatus::NONE : ErrorStatus::GENERAL_FAILURE;
    Return<void> returned = callback->notify(executionStatus);
    if (!returned.isOk()) {
        LOG(ERROR) << " hidl callback failed to return properly: " << returned.description();
    }
    VpuPreparedModel::network_count_ex = 0;
}

}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
