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

Blob_Stage_data get_DEPTHWISE_CONV_2D_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_depth_conv2d;
  Operation_inputs_info depth_conv2d_stage_info;
  uint32_t nk_ele, nb_ele;
  float *kernel_buf, *bias_buf;


  depth_conv2d_stage_info = curr_stage_info;

  depth_conv2d_stage_info.kernel_shape[0] = (depth_conv2d_stage_info.kernel_shape[0] == 0) ? 1: depth_conv2d_stage_info.kernel_shape[0];
  depth_conv2d_stage_info.kernel_shape[1] = (depth_conv2d_stage_info.kernel_shape[1] == 0) ? 1: depth_conv2d_stage_info.kernel_shape[1];
  depth_conv2d_stage_info.kernel_shape[2] = (depth_conv2d_stage_info.kernel_shape[2] == 0) ? 1: depth_conv2d_stage_info.kernel_shape[2];
  depth_conv2d_stage_info.kernel_shape[3] = (depth_conv2d_stage_info.kernel_shape[3] == 0) ? 1: depth_conv2d_stage_info.kernel_shape[3];

  depth_conv2d_stage_info.bias_shape[0] = (depth_conv2d_stage_info.bias_shape[0] == 0) ? 1: depth_conv2d_stage_info.bias_shape[0];
  depth_conv2d_stage_info.bias_shape[1] = (depth_conv2d_stage_info.bias_shape[1] == 0) ? 1: depth_conv2d_stage_info.bias_shape[1];
  depth_conv2d_stage_info.bias_shape[2] = (depth_conv2d_stage_info.bias_shape[2] == 0) ? 1: depth_conv2d_stage_info.bias_shape[2];
  depth_conv2d_stage_info.bias_shape[3] = (depth_conv2d_stage_info.bias_shape[3] == 0) ? 1: depth_conv2d_stage_info.bias_shape[3];

  nk_ele = (depth_conv2d_stage_info.kernel_shape[0] * depth_conv2d_stage_info.kernel_shape[1] *
            depth_conv2d_stage_info.kernel_shape[2] * depth_conv2d_stage_info.kernel_shape[3]);

  nb_ele = (depth_conv2d_stage_info.bias_shape[0] * depth_conv2d_stage_info.bias_shape[1] *
            depth_conv2d_stage_info.bias_shape[2] * depth_conv2d_stage_info.bias_shape[3]);

  //TODO update is needed get the inut & output buffer sizes from Android

  //initialize stage variables
  stage_depth_conv2d.stage_name = "Depth Convolution 2D";
  stage_depth_conv2d.op_val = 8;

  stage_depth_conv2d.opt_mask = 0x80000000;

  stage_depth_conv2d.radixX = depth_conv2d_stage_info.kernel_shape[1]; //op_y
  stage_depth_conv2d.radixY = depth_conv2d_stage_info.kernel_shape[0]; //op_x

  stage_depth_conv2d.strideX = depth_conv2d_stage_info.stride_width;
  stage_depth_conv2d.strideY = depth_conv2d_stage_info.stride_height;

  stage_depth_conv2d.padX =  0;
  stage_depth_conv2d.padY =  0;
  if(depth_conv2d_stage_info.padding_left == 0 && depth_conv2d_stage_info.padding_right == 0 &&
    depth_conv2d_stage_info.padding_top == 0 && depth_conv2d_stage_info.padding_bottom == 0 )
    stage_depth_conv2d.padStyle_value = 1; //padding is tfvalid
  else
    stage_depth_conv2d.padStyle_value = 3; //padding is tfsame

  if(depth_conv2d_stage_info.input_shape[1]!=0)
     stage_depth_conv2d.inputDimX = depth_conv2d_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_depth_conv2d.inputDimX = 1; //TODO update from Android

  if(depth_conv2d_stage_info.input_shape[2]!=0)
    stage_depth_conv2d.inputDimY = depth_conv2d_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_depth_conv2d.inputDimY = 1; //TODO update from Android

  if(depth_conv2d_stage_info.input_shape[3]!=0)
     stage_depth_conv2d.inputDimZ = depth_conv2d_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_depth_conv2d.inputDimZ = 1; //TODO update from Android


  stage_depth_conv2d.tapDimX = depth_conv2d_stage_info.kernel_shape[1] * depth_conv2d_stage_info.kernel_shape[0];
  stage_depth_conv2d.tapDimY = depth_conv2d_stage_info.input_shape[3];
  stage_depth_conv2d.tapDimZ = depth_conv2d_stage_info.kernel_shape[3] * depth_conv2d_stage_info.kernel_shape[2];

  if(stage_depth_conv2d.padStyle_value == 1){ //padding is tfvalid
    stage_depth_conv2d.outputDimX = (uint32_t) std::ceil((double)(stage_depth_conv2d.inputDimX - stage_depth_conv2d.radixX +1)/(double)stage_depth_conv2d.strideX);
    stage_depth_conv2d.outputDimY = (uint32_t) std::ceil((double)(stage_depth_conv2d.inputDimY - stage_depth_conv2d.radixY +1)/(double)stage_depth_conv2d.strideY);
  }
  else if(stage_depth_conv2d.padStyle_value == 3){
    stage_depth_conv2d.outputDimX = (uint32_t) std::ceil((double)stage_depth_conv2d.inputDimX/(double)stage_depth_conv2d.strideX);
    stage_depth_conv2d.outputDimY = (uint32_t) std::ceil((double)stage_depth_conv2d.inputDimY/(double)stage_depth_conv2d.strideY);
  }
  else{//TODO update the outputDimX & outputDimY for padStyle_value = 2
    stage_depth_conv2d.outputDimX = (uint32_t) std::ceil((double)stage_depth_conv2d.inputDimX/(double)stage_depth_conv2d.strideX);
    stage_depth_conv2d.outputDimY = (uint32_t) std::ceil((double)stage_depth_conv2d.inputDimY/(double)stage_depth_conv2d.strideY);
  }
  stage_depth_conv2d.outputDimZ = stage_depth_conv2d.tapDimZ;

  stage_depth_conv2d.inputStrideX = 2 * stage_depth_conv2d.inputDimZ;
  stage_depth_conv2d.inputStrideY = 2* stage_depth_conv2d.inputDimX * stage_depth_conv2d.inputDimZ;
  stage_depth_conv2d.inputStrideZ = 2;

  stage_depth_conv2d.tapStrideX = 2 * stage_depth_conv2d.tapDimZ;
  stage_depth_conv2d.tapStrideY = 2 * stage_depth_conv2d.tapDimZ;
  stage_depth_conv2d.tapStrideZ = 2;

  stage_depth_conv2d.outputStrideX = 2 * stage_depth_conv2d.outputDimZ;
  stage_depth_conv2d.outputStrideY = 2 * stage_depth_conv2d.outputDimX * stage_depth_conv2d.outputDimZ;
  stage_depth_conv2d.outputStrideZ = 2;

  stage_depth_conv2d.datatype_value = 2;
  stage_depth_conv2d.precision_value = 2;
  stage_depth_conv2d.storageOrder_value = 2;

  stage_depth_conv2d.data_Pointer = get_output_Pointer_global();
  stage_depth_conv2d.data_Index = get_output_Index_global();

  stage_depth_conv2d.taps_Pointer = get_taps_Pointer_global();
  stage_depth_conv2d.taps_Index = get_taps_Index_global();

  uint32_t new_taps_Pointer= 0;
  new_taps_Pointer = calculate_taps_pointer(depth_conv2d_stage_info.kernel_shape[0],depth_conv2d_stage_info.kernel_shape[1],depth_conv2d_stage_info.kernel_shape[2],depth_conv2d_stage_info.kernel_shape[3]);
  //ALOGD("stage_depth_conv2d.taps_Pointer: %d, new_taps_Pointer: %d ",stage_depth_conv2d.taps_Pointer,new_taps_Pointer);
  stage_depth_conv2d.bias_Pointer = stage_depth_conv2d.taps_Pointer + new_taps_Pointer;

  uint32_t new_bias_Pointer =0;
  new_bias_Pointer = stage_depth_conv2d.bias_Pointer + calculate_bias_Pointer(depth_conv2d_stage_info.bias_shape[0]);
  stage_depth_conv2d.bias_Index = get_bias_Index_global();

  stage_depth_conv2d.opPrarams_Pointer = 0;
  stage_depth_conv2d.opPrarams_Index = 0;

  stage_depth_conv2d.output_Pointer = calculate_output_pointer(stage_depth_conv2d.outputDimX,stage_depth_conv2d.outputDimY,stage_depth_conv2d.outputDimZ);
  stage_depth_conv2d.output_Index = get_output_Index_global()+1;

  stage_depth_conv2d.preOp_value = 5;


  depth_conv2d_stage_info.post_operation;
  switch (depth_conv2d_stage_info.post_operation) {
    case RELU:{stage_depth_conv2d.postOp_value = 6; stage_depth_conv2d.post_param1[0] = 0x00; stage_depth_conv2d.post_param1[1] = 0x00;stage_depth_conv2d.post_param1[2] = 0x00;stage_depth_conv2d.post_param1[3] = 0x00;}break;
    case RELU1:{stage_depth_conv2d.postOp_value = 7; stage_depth_conv2d.post_param1[0] = 0x00; stage_depth_conv2d.post_param1[1] = 0x00;stage_depth_conv2d.post_param1[2] = 0x80;stage_depth_conv2d.post_param1[3] = 0x3F;}break;
    case RELU6:{stage_depth_conv2d.postOp_value = 7; stage_depth_conv2d.post_param1[0] = 0x00; stage_depth_conv2d.post_param1[1] = 0x00;stage_depth_conv2d.post_param1[2] = 0xC0;stage_depth_conv2d.post_param1[3] = 0x40;}break;
    default: {stage_depth_conv2d.postOp_value = 5; stage_depth_conv2d.post_param1[0] = 0x00; stage_depth_conv2d.post_param1[1] = 0x00;stage_depth_conv2d.post_param1[2] = 0x00;stage_depth_conv2d.post_param1[3] = 0x00;}break;
  }

  stage_depth_conv2d.post_strideX = 0;
  stage_depth_conv2d.post_strideY = 0;

  if(update_taps_Pointer_g(new_bias_Pointer)!=true)
    ALOGE("unable to update taps_Pointer global");

  if(update_output_Pointer_g(stage_depth_conv2d.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_depth_conv2d.output_Index)!=true)
    ALOGE("unable to update output_Index global");


  return stage_depth_conv2d;
}
