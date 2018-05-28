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

Blob_Stage_data get_LOGISTIC_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_sigmoid;
  Operation_inputs_info sigmoid_stage_info;


  sigmoid_stage_info = curr_stage_info; //parse_logistic_stage_info();

  //TODO update is needed get the inut & output buffer sizes from Android

  //initialize stage variables
  stage_sigmoid.stage_name = "Sigmoid";
  stage_sigmoid.op_val = 20;

  stage_sigmoid.opt_mask = 0x80000000;

  stage_sigmoid.radixX = 1;
  stage_sigmoid.radixY = 1;

  stage_sigmoid.strideX = 1;
  stage_sigmoid.strideY = 1;

  stage_sigmoid.padX =  0;
  stage_sigmoid.padY =  0;
  stage_sigmoid.padStyle_value = 2; //TODO update with zero

  if(sigmoid_stage_info.input_shape[1]!=0)
     stage_sigmoid.inputDimX = sigmoid_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_sigmoid.inputDimX = 1; //TODO update from Android

  if(sigmoid_stage_info.input_shape[2]!=0)
    stage_sigmoid.inputDimY = sigmoid_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_sigmoid.inputDimY = 1; //TODO update from Android

  if(sigmoid_stage_info.input_shape[3]!=0)
     stage_sigmoid.inputDimZ = sigmoid_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_sigmoid.inputDimZ = 1; //TODO update from Android

  stage_sigmoid.tapDimX = 1;
  stage_sigmoid.tapDimY = 1;
  stage_sigmoid.tapDimZ = 1;

  stage_sigmoid.outputDimX = stage_sigmoid.inputDimX;
  stage_sigmoid.outputDimY = stage_sigmoid.inputDimY;
  stage_sigmoid.outputDimZ = stage_sigmoid.inputDimZ;

  stage_sigmoid.inputStrideX = 2 * stage_sigmoid.inputDimZ;
  stage_sigmoid.inputStrideY = 2* stage_sigmoid.inputDimX * stage_sigmoid.inputDimZ;
  stage_sigmoid.inputStrideZ = 2;

  stage_sigmoid.tapStrideX = 2 * stage_sigmoid.tapDimZ;
  stage_sigmoid.tapStrideY = 2 * stage_sigmoid.tapDimZ;
  stage_sigmoid.tapStrideZ = 2;

  stage_sigmoid.outputStrideX = 2 * stage_sigmoid.outputDimZ;
  stage_sigmoid.outputStrideY = 2 * stage_sigmoid.outputDimX * stage_sigmoid.outputDimZ;
  stage_sigmoid.outputStrideZ = 2;

  stage_sigmoid.datatype_value = 2;
  stage_sigmoid.precision_value = 2;
  stage_sigmoid.storageOrder_value = 4;

  stage_sigmoid.data_Pointer = get_output_Pointer_global();
  stage_sigmoid.data_Index = get_output_Index_global();

  stage_sigmoid.taps_Pointer = 0;
  stage_sigmoid.taps_Index = 0;

  stage_sigmoid.bias_Pointer = 0;
  stage_sigmoid.bias_Index = 0;

  stage_sigmoid.opPrarams_Pointer = 0;
  stage_sigmoid.opPrarams_Index = 0;

  stage_sigmoid.output_Pointer = calculate_output_pointer(stage_sigmoid.outputDimX,stage_sigmoid.outputDimY,stage_sigmoid.outputDimZ);
  stage_sigmoid.output_Index = get_output_Index_global()+1;

  stage_sigmoid.preOp_value = 5;
  stage_sigmoid.postOp_value = 5;

  stage_sigmoid.post_param1[0] = 0x00;
  stage_sigmoid.post_param1[1] = 0x00;
  stage_sigmoid.post_param1[2] = 0x00;
  stage_sigmoid.post_param1[3] = 0x00;

  stage_sigmoid.post_strideX = 0;
  stage_sigmoid.post_strideY = 0;

  if(update_output_Pointer_g(stage_sigmoid.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_sigmoid.output_Index)!=true)
    ALOGE("unable to update output_Index global");

  return stage_sigmoid;
}
