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

Blob_Stage_data get_Reshape_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_reshape;
  Operation_inputs_info reshape_stage_info;

  reshape_stage_info = curr_stage_info;

  //TODO update is needed get the inut & output buffer sizes from Android

  //initialize stage variables
  stage_reshape.stage_name = "Reshape Layer";
  stage_reshape.op_val = 24;

  stage_reshape.opt_mask = 0x80000000;

  stage_reshape.radixX = 1;
  stage_reshape.radixY = 1;

  stage_reshape.strideX = 1;
  stage_reshape.strideY = 1;

  stage_reshape.padX =  0;
  stage_reshape.padY =  0;
  stage_reshape.padStyle_value = 0; //No padding support

  if(reshape_stage_info.input_shape[1]!=0)
     stage_reshape.inputDimX = reshape_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_reshape.inputDimX = 1; //TODO update from Android

  if(reshape_stage_info.input_shape[2]!=0)
    stage_reshape.inputDimY = reshape_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_reshape.inputDimY = 1; //TODO update from Android

  if(reshape_stage_info.input_shape[3]!=0)
     stage_reshape.inputDimZ = reshape_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_reshape.inputDimZ = 1; //TODO update from Android


  stage_reshape.tapDimX = 1; //reshape_stage_info.kernel_shape[0];
  stage_reshape.tapDimY = 1; //reshape_stage_info.kernel_shape[1];
  stage_reshape.tapDimZ = 1; //reshape_stage_info.kernel_shape[2];

  if(reshape_stage_info.output_shape[1]!=0)
     stage_reshape.outputDimX = reshape_stage_info.output_shape[1];
  else
     stage_reshape.outputDimX = 1; //TODO update from Android

  if(reshape_stage_info.output_shape[2]!=0)
    stage_reshape.outputDimY = reshape_stage_info.output_shape[2];
  else
     stage_reshape.outputDimY = 1; //TODO update from Android

  if(reshape_stage_info.output_shape[3]!=0)
     stage_reshape.outputDimZ = reshape_stage_info.output_shape[3];
  else
    stage_reshape.outputDimZ = 1; //TODO update from Android


  stage_reshape.inputStrideX = 2 * stage_reshape.inputDimZ;
  stage_reshape.inputStrideY = 2 * stage_reshape.inputDimX * stage_reshape.inputDimZ;
  stage_reshape.inputStrideZ = 2;

  stage_reshape.tapStrideX = 2 * stage_reshape.tapDimZ;
  stage_reshape.tapStrideY = 2 * stage_reshape.tapDimZ;
  stage_reshape.tapStrideZ = 2;

  stage_reshape.outputStrideX = 2 * stage_reshape.outputDimZ;
  stage_reshape.outputStrideY = 2 * stage_reshape.outputDimX * stage_reshape.outputDimZ;
  stage_reshape.outputStrideZ = 2;

  stage_reshape.datatype_value = 2;
  stage_reshape.precision_value = 2;
  stage_reshape.storageOrder_value = 2;

  stage_reshape.data_Pointer = get_output_Pointer_global();
  stage_reshape.data_Index = get_output_Index_global();

  stage_reshape.taps_Pointer = 0;
  stage_reshape.taps_Index = 0;

  stage_reshape.bias_Pointer = 0;
  stage_reshape.bias_Index = 0;

  stage_reshape.opPrarams_Pointer = 0;
  stage_reshape.opPrarams_Index = 0;

  stage_reshape.output_Pointer = calculate_output_pointer(stage_reshape.outputDimX,stage_reshape.outputDimY,stage_reshape.outputDimZ);
  stage_reshape.output_Index = get_output_Index_global()+1;

  stage_reshape.preOp_value = 5;
  stage_reshape.postOp_value = 5;

  stage_reshape.post_param1[0] = 0x00;
  stage_reshape.post_param1[1] = 0x00;
  stage_reshape.post_param1[2] = 0x00;
  stage_reshape.post_param1[3] = 0x00;

  stage_reshape.post_strideX = 0;
  stage_reshape.post_strideY = 0;


  if(update_output_Pointer_g(stage_reshape.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_reshape.output_Index)!=true)
    ALOGE("unable to update output_Index global");


  return stage_reshape;
}
