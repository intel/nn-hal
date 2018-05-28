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

Blob_Stage_data get_TANH_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_tanh;
  Operation_inputs_info tanh_stage_info;


  tanh_stage_info = curr_stage_info;

  //TODO update is needed get the inut & output buffer sizes from Android

  //initialize stage variables
  stage_tanh.stage_name = "TanH";
  stage_tanh.op_val = 21;

  stage_tanh.opt_mask = 0x80000000;

  stage_tanh.radixX = 1;
  stage_tanh.radixY = 1;

  stage_tanh.strideX = 1;
  stage_tanh.strideY = 1;

  stage_tanh.padX =  0;
  stage_tanh.padY =  0;
  stage_tanh.padStyle_value = 2; //TODO update with zero

  if(tanh_stage_info.input_shape[1]!=0)
     stage_tanh.inputDimX = tanh_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_tanh.inputDimX = 1; //TODO update from Android

  if(tanh_stage_info.input_shape[2]!=0)
    stage_tanh.inputDimY = tanh_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_tanh.inputDimY = 1; //TODO update from Android

  if(tanh_stage_info.input_shape[3]!=0)
     stage_tanh.inputDimZ = tanh_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_tanh.inputDimZ = 1; //TODO update from Android


  stage_tanh.tapDimX = 0; //tanh_stage_info.kernel_shape[0];
  stage_tanh.tapDimY = 1; //tanh_stage_info.kernel_shape[1];
  stage_tanh.tapDimZ = 1; //tanh_stage_info.kernel_shape[2];

  stage_tanh.outputDimX = stage_tanh.inputDimX;
  stage_tanh.outputDimY = stage_tanh.inputDimY;
  stage_tanh.outputDimZ = stage_tanh.inputDimZ;

  stage_tanh.inputStrideX = 2 * stage_tanh.inputDimZ;
  stage_tanh.inputStrideY = 2* stage_tanh.inputDimX * stage_tanh.inputDimZ;
  stage_tanh.inputStrideZ = 2;

  stage_tanh.tapStrideX = 2 * stage_tanh.tapDimZ;
  stage_tanh.tapStrideY = 2 * stage_tanh.tapDimZ;
  stage_tanh.tapStrideZ = 2;

  stage_tanh.outputStrideX = 2 * stage_tanh.outputDimZ;
  stage_tanh.outputStrideY = 2 * stage_tanh.outputDimX * stage_tanh.outputDimZ;
  stage_tanh.outputStrideZ = 2;

  stage_tanh.datatype_value = 2;
  stage_tanh.precision_value = 2;
  stage_tanh.storageOrder_value = 2;

  stage_tanh.data_Pointer = get_output_Pointer_global();
  stage_tanh.data_Index = get_output_Index_global();

  stage_tanh.taps_Pointer = 0;
  stage_tanh.taps_Index = 0;

  stage_tanh.bias_Pointer = 0;
  stage_tanh.bias_Index = 0;

  stage_tanh.opPrarams_Pointer = 0;
  stage_tanh.opPrarams_Index = 0;

  stage_tanh.output_Pointer = calculate_output_pointer(stage_tanh.outputDimX,stage_tanh.outputDimY,stage_tanh.outputDimZ);
  stage_tanh.output_Index = get_output_Index_global()+1;

  stage_tanh.preOp_value = 5;
  stage_tanh.postOp_value = 5;

  stage_tanh.post_param1[0] = 0x00;
  stage_tanh.post_param1[1] = 0x00;
  stage_tanh.post_param1[2] = 0x00;
  stage_tanh.post_param1[3] = 0x00;

  stage_tanh.post_strideX = 0;
  stage_tanh.post_strideY = 0;


  if(update_output_Pointer_g(stage_tanh.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_tanh.output_Index)!=true)
    ALOGE("unable to update output_Index global");


  return stage_tanh;
}
