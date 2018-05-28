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
#include<stdint.h>
#include <log/log.h>
#include "Blob.h"


Blob_Stage_data get_RELU_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_relu;
  Operation_inputs_info relu_stage_info;

  relu_stage_info = curr_stage_info;

  //TODO update is needed get the inut & output buffer sizes from Android

  //initialize stage variables
  stage_relu.stage_name = "RELU";
  stage_relu.op_val = 19; //please see stage_relu.postOp_value

  stage_relu.opt_mask = 0x80000000;

  stage_relu.radixX = 1;
  stage_relu.radixY = 1;

  stage_relu.strideX = 1;
  stage_relu.strideY = 1;

  stage_relu.padX =  0;
  stage_relu.padY =  0;
  stage_relu.padStyle_value = 2; //TODO update with zero

  if(relu_stage_info.input_shape[1]!=0)
     stage_relu.inputDimX = relu_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_relu.inputDimX = 1; //TODO update from Android

  if(relu_stage_info.input_shape[2]!=0)
    stage_relu.inputDimY = relu_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_relu.inputDimY = 1; //TODO update from Android

  if(relu_stage_info.input_shape[3]!=0)
     stage_relu.inputDimZ = relu_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_relu.inputDimZ = 1; //TODO update from Android

  stage_relu.tapDimX = 0;
  stage_relu.tapDimY = 1;
  stage_relu.tapDimZ = 1;

  stage_relu.outputDimX = stage_relu.inputDimX;
  stage_relu.outputDimY = stage_relu.inputDimY;
  stage_relu.outputDimZ = stage_relu.inputDimZ;

  stage_relu.inputStrideX = 2 * stage_relu.inputDimZ;
  stage_relu.inputStrideY = 2 * stage_relu.inputDimX * stage_relu.inputDimZ;
  stage_relu.inputStrideZ = 2;

  stage_relu.tapStrideX = 2 * stage_relu.tapDimZ;
  stage_relu.tapStrideY = 2 * stage_relu.tapDimZ;
  stage_relu.tapStrideZ = 2;

  stage_relu.outputStrideX = 2 * stage_relu.outputDimZ;
  stage_relu.outputStrideY = 2 * stage_relu.outputDimX * stage_relu.outputDimZ;
  stage_relu.outputStrideZ = 2;

  stage_relu.datatype_value = 2;
  stage_relu.precision_value = 2;
  stage_relu.storageOrder_value = 2;

  stage_relu.data_Pointer = get_output_Pointer_global();
  stage_relu.data_Index = get_output_Index_global();

  stage_relu.taps_Pointer = 0;
  stage_relu.taps_Index = 0;

  stage_relu.bias_Pointer = 0;
  stage_relu.bias_Index = 0;

  stage_relu.opPrarams_Pointer = 0;
  stage_relu.opPrarams_Index = 0;

  stage_relu.output_Pointer = calculate_output_pointer(stage_relu.outputDimX, stage_relu.outputDimY, stage_relu.outputDimZ);
  stage_relu.output_Index = get_output_Index_global()+1;

  stage_relu.preOp_value = 5;
  stage_relu.postOp_value = 6;

  //stage_relu.post_param1 = 0;
  stage_relu.post_param1[0] = 0x00;
  stage_relu.post_param1[1] = 0x00;
  stage_relu.post_param1[2] = 0x00;
  stage_relu.post_param1[3] = 0x00;

  stage_relu.post_strideX = 0;
  stage_relu.post_strideY = 0;


  if(update_output_Pointer_g(stage_relu.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_relu.output_Index)!=true)
    ALOGE("unable to update output_Index global");

  return stage_relu;
}


Blob_Stage_data get_RELU1_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_relu1;
  Operation_inputs_info relu1_stage_info;


  relu1_stage_info = curr_stage_info;

  //TODO update is needed get the inut & output buffer sizes from Android

  //initialize stage variables
  stage_relu1.stage_name = "RELU1";
  stage_relu1.op_val = 19; //please see stage_relu1.postOp_value

  stage_relu1.opt_mask = 0x80000000;

  stage_relu1.radixX = 1;
  stage_relu1.radixY = 1;

  stage_relu1.strideX = 1;
  stage_relu1.strideY = 1;

  stage_relu1.padX =  0;
  stage_relu1.padY =  0;
  stage_relu1.padStyle_value = 2; //TODO update with zero

  //DIM-X
  if(relu1_stage_info.input_shape[1]!=0)
     stage_relu1.inputDimX = relu1_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_relu1.inputDimX = 1; //TODO update from Android
  //DIM-Y
  if(relu1_stage_info.input_shape[2]!=0)
    stage_relu1.inputDimY = relu1_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_relu1.inputDimY = 1; //TODO update from Android
  //DIM-Z
  if(relu1_stage_info.input_shape[3]!=0)
     stage_relu1.inputDimZ = relu1_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_relu1.inputDimZ = 1; //TODO update from Android

  stage_relu1.tapDimX = 0;
  stage_relu1.tapDimY = 1;
  stage_relu1.tapDimZ = 1;

  stage_relu1.outputDimX = stage_relu1.inputDimX;
  stage_relu1.outputDimY = stage_relu1.inputDimY;
  stage_relu1.outputDimZ = stage_relu1.inputDimZ;

  stage_relu1.inputStrideX = 2 * stage_relu1.inputDimZ;
  stage_relu1.inputStrideY = 2 * stage_relu1.inputDimX * stage_relu1.inputDimZ;
  stage_relu1.inputStrideZ = 2;

  stage_relu1.tapStrideX = 2 * stage_relu1.tapDimZ;
  stage_relu1.tapStrideY = 2 * stage_relu1.tapDimZ;
  stage_relu1.tapStrideZ = 2;

  stage_relu1.outputStrideX = 2 * stage_relu1.outputDimZ;
  stage_relu1.outputStrideY = 2 * stage_relu1.outputDimX * stage_relu1.outputDimZ;
  stage_relu1.outputStrideZ = 2;

  stage_relu1.datatype_value = 2;
  stage_relu1.precision_value = 2;
  stage_relu1.storageOrder_value = 4;

  stage_relu1.data_Pointer = get_output_Pointer_global();
  stage_relu1.data_Index = get_output_Index_global();

  stage_relu1.taps_Pointer = 0;
  stage_relu1.taps_Index = 0;

  stage_relu1.bias_Pointer = 0;
  stage_relu1.bias_Index = 0;

  stage_relu1.opPrarams_Pointer = 0;
  stage_relu1.opPrarams_Index = 0;

  stage_relu1.output_Pointer = calculate_output_pointer(stage_relu1.outputDimX, stage_relu1.outputDimY, stage_relu1.outputDimZ);
  stage_relu1.output_Index = get_output_Index_global()+1;

  stage_relu1.preOp_value = 5;
  stage_relu1.postOp_value = 7;

  stage_relu1.post_param1[0] = 0x00;
  stage_relu1.post_param1[1] = 0x00;
  stage_relu1.post_param1[2] = 0x80;
  stage_relu1.post_param1[3] = 0x3F;

  stage_relu1.post_strideX = 0;
  stage_relu1.post_strideY = 0;


  if(update_output_Pointer_g(stage_relu1.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_relu1.output_Index)!=true)
    ALOGE("unable to update output_Index global");

  return stage_relu1;
}


Blob_Stage_data get_RELU6_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_relu6;
  Operation_inputs_info relu6_stage_info;


  relu6_stage_info = curr_stage_info;

  //TODO update is needed get the inut & output buffer sizes from Android

  //initialize stage variables
  stage_relu6.stage_name = "RELU6";
  stage_relu6.op_val = 19; //please see stage_relu6.postOp_value

  stage_relu6.opt_mask = 0x80000000;

  stage_relu6.radixX = 1;
  stage_relu6.radixY = 1;

  stage_relu6.strideX = 1;
  stage_relu6.strideY = 1;

  stage_relu6.padX =  0;
  stage_relu6.padY =  0;
  stage_relu6.padStyle_value = 2; //TODO update with zero

  //DIM-X
  if(relu6_stage_info.input_shape[1]!=0)
     stage_relu6.inputDimX = relu6_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_relu6.inputDimX = 1; //TODO update from Android
  //DIM-Y
  if(relu6_stage_info.input_shape[2]!=0)
    stage_relu6.inputDimY = relu6_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_relu6.inputDimY = 1; //TODO update from Android
  //DIM-Z
  if(relu6_stage_info.input_shape[3]!=0)
     stage_relu6.inputDimZ = relu6_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_relu6.inputDimZ = 1; //TODO update from Android

  stage_relu6.tapDimX = 0;
  stage_relu6.tapDimY = 1;
  stage_relu6.tapDimZ = 1;

  stage_relu6.outputDimX = stage_relu6.inputDimX; //relu6_stage_info.output_shape[0]; //TODO update from Android
  stage_relu6.outputDimY = stage_relu6.inputDimY; //relu6_stage_info.output_shape[1]; //TODO update from Android
  stage_relu6.outputDimZ = stage_relu6.inputDimZ; //relu6_stage_info.output_shape[2]; //TODO update from Android

  stage_relu6.inputStrideX = 2 * stage_relu6.inputDimZ;
  stage_relu6.inputStrideY = 2 * stage_relu6.inputDimX * stage_relu6.inputDimZ;
  stage_relu6.inputStrideZ = 2;

  stage_relu6.tapStrideX = 2 * stage_relu6.tapDimZ;
  stage_relu6.tapStrideY = 2 * stage_relu6.tapDimZ;
  stage_relu6.tapStrideZ = 2;

  stage_relu6.outputStrideX = 2 * stage_relu6.outputDimZ;
  stage_relu6.outputStrideY = 2 * stage_relu6.outputDimX * stage_relu6.outputDimZ;
  stage_relu6.outputStrideZ = 2;

  stage_relu6.datatype_value = 2;
  stage_relu6.precision_value = 2;
  stage_relu6.storageOrder_value = 4;

  stage_relu6.data_Pointer = get_output_Pointer_global();
  stage_relu6.data_Index = get_output_Index_global()-1;

  stage_relu6.taps_Pointer = 0;
  stage_relu6.taps_Index = 0;

  stage_relu6.bias_Pointer = 0;
  stage_relu6.bias_Index = 0;

  stage_relu6.opPrarams_Pointer = 0;
  stage_relu6.opPrarams_Index = 0;

  stage_relu6.output_Pointer = calculate_output_pointer(stage_relu6.outputDimX, stage_relu6.outputDimY, stage_relu6.outputDimZ);
  stage_relu6.output_Index = get_output_Index_global();

  stage_relu6.preOp_value = 5;
  stage_relu6.postOp_value = 7;

  stage_relu6.post_param1[0] = 0x00;
  stage_relu6.post_param1[1] = 0x00;
  stage_relu6.post_param1[2] = 0xC0;
  stage_relu6.post_param1[3] = 0x40;

  stage_relu6.post_strideX = 0;
  stage_relu6.post_strideY = 0;



  if(update_output_Pointer_g(stage_relu6.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_relu6.output_Index)!=true)
    ALOGE("unable to update output_Index global");

  return stage_relu6;
}
