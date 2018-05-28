/*
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
#include<stdio.h>
#include<string.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<stdint.h>
#include <log/log.h>
#include "Blob.h"

Blob_Stage_data get_CONV_1D_stage_data(Operation_inputs_info curr_stage_info);

Blob_Stage_data get_CONV_2D_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_conv2d;
  Operation_inputs_info conv2d_stage_info;
  uint32_t nk_ele, nb_ele;
  float *kernel_buf, *bias_buf;

  conv2d_stage_info = curr_stage_info;

  conv2d_stage_info.kernel_shape[0] = (conv2d_stage_info.kernel_shape[0] == 0) ? 1: conv2d_stage_info.kernel_shape[0];
  conv2d_stage_info.kernel_shape[1] = (conv2d_stage_info.kernel_shape[1] == 0) ? 1: conv2d_stage_info.kernel_shape[1];
  conv2d_stage_info.kernel_shape[2] = (conv2d_stage_info.kernel_shape[2] == 0) ? 1: conv2d_stage_info.kernel_shape[2];
  conv2d_stage_info.kernel_shape[3] = (conv2d_stage_info.kernel_shape[3] == 0) ? 1: conv2d_stage_info.kernel_shape[3];

  conv2d_stage_info.bias_shape[0] = (conv2d_stage_info.bias_shape[0] == 0) ? 1: conv2d_stage_info.bias_shape[0];
  conv2d_stage_info.bias_shape[1] = (conv2d_stage_info.bias_shape[1] == 0) ? 1: conv2d_stage_info.bias_shape[1];
  conv2d_stage_info.bias_shape[2] = (conv2d_stage_info.bias_shape[2] == 0) ? 1: conv2d_stage_info.bias_shape[2];
  conv2d_stage_info.bias_shape[3] = (conv2d_stage_info.bias_shape[3] == 0) ? 1: conv2d_stage_info.bias_shape[3];

  nk_ele = (conv2d_stage_info.kernel_shape[0] * conv2d_stage_info.kernel_shape[1] *
                                    conv2d_stage_info.kernel_shape[2] * conv2d_stage_info.kernel_shape[3]);

  nb_ele = (conv2d_stage_info.bias_shape[0] * conv2d_stage_info.bias_shape[1] *
            conv2d_stage_info.bias_shape[2] * conv2d_stage_info.bias_shape[3]);



  //initialize stage variables
  stage_conv2d.stage_name = "Convolution 2D";
  stage_conv2d.op_val = 0;

  if(conv2d_stage_info.input_shape[1] == 1 && conv2d_stage_info.input_shape[2] == 1 &&
    conv2d_stage_info.kernel_shape[0] == 1 && conv2d_stage_info.kernel_shape[1] == 1){
      stage_conv2d = get_CONV_1D_stage_data(curr_stage_info);
      return stage_conv2d;
    }

  stage_conv2d.opt_mask = 0x80000000;

  stage_conv2d.radixX = conv2d_stage_info.kernel_shape[1]; //op_y
  stage_conv2d.radixY = conv2d_stage_info.kernel_shape[0]; //op_x

  stage_conv2d.strideX = conv2d_stage_info.stride_width;
  stage_conv2d.strideY = conv2d_stage_info.stride_height;

  stage_conv2d.padX =  0;
  stage_conv2d.padY =  0;

  if(conv2d_stage_info.padding_left == 0 && conv2d_stage_info.padding_right == 0 &&
    conv2d_stage_info.padding_top == 0 && conv2d_stage_info.padding_bottom == 0 )
    stage_conv2d.padStyle_value = 1; //padding is tfvalid
  else
    stage_conv2d.padStyle_value = 3; //padding is tfsame

  if(conv2d_stage_info.input_shape[1]!=0)
     stage_conv2d.inputDimX = conv2d_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_conv2d.inputDimX = 1; //TODO update from Android

  if(conv2d_stage_info.input_shape[2]!=0)
    stage_conv2d.inputDimY = conv2d_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_conv2d.inputDimY = 1; //TODO update from Android

  if(conv2d_stage_info.input_shape[3]!=0)
     stage_conv2d.inputDimZ = conv2d_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_conv2d.inputDimZ = 1; //TODO update from Android


  stage_conv2d.tapDimX = conv2d_stage_info.kernel_shape[1] * conv2d_stage_info.kernel_shape[0];
  stage_conv2d.tapDimY = conv2d_stage_info.input_shape[3];
  stage_conv2d.tapDimZ = conv2d_stage_info.kernel_shape[3];

  //Output DIMENSIONS calculation
  if(stage_conv2d.padStyle_value == 1){ //padding is tfvalid
    stage_conv2d.outputDimX = (uint32_t) std::ceil((double)(stage_conv2d.inputDimX - stage_conv2d.radixX +1)/(double)stage_conv2d.strideX);
    stage_conv2d.outputDimY = (uint32_t) std::ceil((double)(stage_conv2d.inputDimY - stage_conv2d.radixY +1)/(double)stage_conv2d.strideY);
  }
  else if(stage_conv2d.padStyle_value == 3){
    stage_conv2d.outputDimX = (uint32_t) std::ceil((double)stage_conv2d.inputDimX/(double)stage_conv2d.strideX);
    stage_conv2d.outputDimY = (uint32_t) std::ceil((double)stage_conv2d.inputDimY/(double)stage_conv2d.strideY);
  }
  else{//TODO update the outputDimX & outputDimY for padStyle_value = 2
    stage_conv2d.outputDimX = (uint32_t) std::ceil((double)stage_conv2d.inputDimX/(double)stage_conv2d.strideX);
    stage_conv2d.outputDimY = (uint32_t) std::ceil((double)stage_conv2d.inputDimY/(double)stage_conv2d.strideY);
  }
  stage_conv2d.outputDimZ = stage_conv2d.tapDimZ;

  stage_conv2d.inputStrideX = 2 * stage_conv2d.inputDimZ;
  stage_conv2d.inputStrideY = 2* stage_conv2d.inputDimX * stage_conv2d.inputDimZ;
  stage_conv2d.inputStrideZ = 2;

  stage_conv2d.tapStrideX = 2 * stage_conv2d.tapDimZ;
  stage_conv2d.tapStrideY = 2 * stage_conv2d.tapDimZ;
  stage_conv2d.tapStrideZ = 2;

  stage_conv2d.outputStrideX = 2 * stage_conv2d.outputDimZ;
  stage_conv2d.outputStrideY = 2 * stage_conv2d.outputDimX * stage_conv2d.outputDimZ;
  stage_conv2d.outputStrideZ = 2;

  stage_conv2d.datatype_value = 2;
  stage_conv2d.precision_value = 2;
  stage_conv2d.storageOrder_value = 2;

  stage_conv2d.data_Pointer = get_output_Pointer_global();
  stage_conv2d.data_Index = get_output_Index_global();

  stage_conv2d.taps_Pointer = get_taps_Pointer_global();
  stage_conv2d.taps_Index = get_taps_Index_global();

  uint32_t new_taps_Pointer= 0;
  new_taps_Pointer = calculate_taps_pointer(conv2d_stage_info.kernel_shape[0],conv2d_stage_info.kernel_shape[1],conv2d_stage_info.kernel_shape[2],conv2d_stage_info.kernel_shape[3]);

  stage_conv2d.bias_Pointer = stage_conv2d.taps_Pointer + new_taps_Pointer;

  uint32_t new_bias_Pointer =0;
  new_bias_Pointer = stage_conv2d.bias_Pointer + calculate_bias_Pointer(conv2d_stage_info.bias_shape[0]);
  stage_conv2d.bias_Index = get_bias_Index_global();

  stage_conv2d.opPrarams_Pointer = 0;
  stage_conv2d.opPrarams_Index = 0;

  stage_conv2d.output_Pointer = calculate_output_pointer(stage_conv2d.outputDimX,stage_conv2d.outputDimY,stage_conv2d.outputDimZ);
  stage_conv2d.output_Index = get_output_Index_global()+1;

  stage_conv2d.preOp_value = 5;


  conv2d_stage_info.post_operation;
  switch (conv2d_stage_info.post_operation) {
    case RELU:{stage_conv2d.postOp_value = 6; stage_conv2d.post_param1[0] = 0x00; stage_conv2d.post_param1[1] = 0x00;stage_conv2d.post_param1[2] = 0x00;stage_conv2d.post_param1[3] = 0x00;}break;
    case RELU1:{stage_conv2d.postOp_value = 7; stage_conv2d.post_param1[0] = 0x00; stage_conv2d.post_param1[1] = 0x00;stage_conv2d.post_param1[2] = 0x80;stage_conv2d.post_param1[3] = 0x3F;}break;
    case RELU6:{stage_conv2d.postOp_value = 7; stage_conv2d.post_param1[0] = 0x00; stage_conv2d.post_param1[1] = 0x00;stage_conv2d.post_param1[2] = 0xC0;stage_conv2d.post_param1[3] = 0x40;}break;
    default: {stage_conv2d.postOp_value = 5; stage_conv2d.post_param1[0] = 0x00; stage_conv2d.post_param1[1] = 0x00;stage_conv2d.post_param1[2] = 0x00;stage_conv2d.post_param1[3] = 0x00;}break;
  }

  stage_conv2d.post_strideX = 0;
  stage_conv2d.post_strideY = 0;

  if(update_taps_Pointer_g(new_bias_Pointer)!=true)
    ALOGE("unable to update taps_Pointer global");

  if(update_output_Pointer_g(stage_conv2d.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_conv2d.output_Index)!=true)
    ALOGE("unable to update output_Index global");


  return stage_conv2d;
}


Blob_Stage_data get_CONV_1D_stage_data(Operation_inputs_info curr_stage_info){

    Blob_Stage_data stage_conv1d;
    Operation_inputs_info conv1d_stage_info;


    conv1d_stage_info = curr_stage_info;

    //TODO update is needed get the inut & output buffer sizes from Android

    //initialize stage variables
    stage_conv1d.stage_name = "1D CONV using FCL";
    stage_conv1d.op_val = 4; //FCL Fully_Connected_Layer

    stage_conv1d.opt_mask = 0x80000000;

    stage_conv1d.radixX = conv1d_stage_info.kernel_shape[1]; //op_y
    stage_conv1d.radixY = conv1d_stage_info.kernel_shape[0]; //op_x

    stage_conv1d.strideX = conv1d_stage_info.stride_width;
    stage_conv1d.strideY = conv1d_stage_info.stride_height;

    stage_conv1d.padX =  0;
    stage_conv1d.padY =  0;

    if(conv1d_stage_info.padding_left == 0 && conv1d_stage_info.padding_right == 0 &&
      conv1d_stage_info.padding_top == 0 && conv1d_stage_info.padding_bottom == 0 )
      stage_conv1d.padStyle_value = 1; //padding is tfvalid
    else
      stage_conv1d.padStyle_value = 3; //padding is tfsame

    stage_conv1d.inputDimX = 1; //TODO update from Android
    stage_conv1d.inputDimY = 1; //TODO update from Android
    if(conv1d_stage_info.input_shape[3]!=0)
       stage_conv1d.inputDimZ = conv1d_stage_info.input_shape[3]; //TODO update from Android
    else
      stage_conv1d.inputDimZ = 1; //TODO update from Android

    stage_conv1d.tapDimX = 1;
    stage_conv1d.tapDimY = conv1d_stage_info.input_shape[3];
    stage_conv1d.tapDimZ = conv1d_stage_info.kernel_shape[3];

    stage_conv1d.outputDimX = 1;
    stage_conv1d.outputDimY = 1;
    stage_conv1d.outputDimZ = stage_conv1d.tapDimZ;

    stage_conv1d.inputStrideX = 2 * stage_conv1d.inputDimZ;
    stage_conv1d.inputStrideY = 2* stage_conv1d.inputDimX * stage_conv1d.inputDimZ;
    stage_conv1d.inputStrideZ = 2;

    stage_conv1d.tapStrideX = 2 * stage_conv1d.tapDimZ;
    stage_conv1d.tapStrideY = 2 * stage_conv1d.tapDimZ;
    stage_conv1d.tapStrideZ = 2;

    stage_conv1d.outputStrideX = 2 * stage_conv1d.outputDimZ;
    stage_conv1d.outputStrideY = 2 * stage_conv1d.outputDimX * stage_conv1d.outputDimZ;
    stage_conv1d.outputStrideZ = 2;

    stage_conv1d.datatype_value = 2;
    stage_conv1d.precision_value = 2;
    stage_conv1d.storageOrder_value = 2;

    stage_conv1d.data_Pointer = get_output_Pointer_global();
    stage_conv1d.data_Index = get_output_Index_global();

    stage_conv1d.taps_Pointer = get_taps_Pointer_global();
    stage_conv1d.taps_Index = get_taps_Index_global();

    uint32_t new_taps_Pointer= 0;
    new_taps_Pointer = calculate_taps_pointer(conv1d_stage_info.kernel_shape[0],conv1d_stage_info.kernel_shape[1],conv1d_stage_info.kernel_shape[2],conv1d_stage_info.kernel_shape[3]);

    stage_conv1d.bias_Pointer = stage_conv1d.taps_Pointer + new_taps_Pointer;

    uint32_t new_bias_Pointer =0;
    new_bias_Pointer = stage_conv1d.bias_Pointer + calculate_bias_Pointer(conv1d_stage_info.bias_shape[0]);
    stage_conv1d.bias_Index = get_bias_Index_global();

    stage_conv1d.opPrarams_Pointer = 0;
    stage_conv1d.opPrarams_Index = 0;

    stage_conv1d.output_Pointer = calculate_output_pointer(stage_conv1d.outputDimX,stage_conv1d.outputDimY,stage_conv1d.outputDimZ);
    stage_conv1d.output_Index = get_output_Index_global()+1;

    stage_conv1d.preOp_value = 5;
    stage_conv1d.postOp_value = 5;

    stage_conv1d.post_param1[0] = 0x00;
    stage_conv1d.post_param1[1] = 0x00;
    stage_conv1d.post_param1[2] = 0x00;
    stage_conv1d.post_param1[3] = 0x00;

    stage_conv1d.post_strideX = 0;
    stage_conv1d.post_strideY = 0;


    if(update_taps_Pointer_g(new_bias_Pointer)!=true)
      ALOGE("unable to update taps_Pointer global");

    if(update_output_Pointer_g(stage_conv1d.output_Pointer)!=true)
      ALOGE("unable to update output_Pointer global");

    if(update_output_Index_g(stage_conv1d.output_Index)!=true)
      ALOGE("unable to update output_Index global");


    return stage_conv1d;
}
