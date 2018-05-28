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
#include<stdint.h>
#include<cstdlib>
#include <log/log.h>
#include "Blob.h"

Blob_Stage_data get_input_stage_layer(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_input;
  Operation_inputs_info input_stage_info;

  input_stage_info = curr_stage_info;

  //TODO update is needed get the network input and output buffer dimensions

  //initialize stage variables
  stage_input.stage_name ="Input Layer";
  stage_input.op_val = 5;

  stage_input.opt_mask = 0x80000000;

  stage_input.radixX = 1;
  stage_input.radixY = 1;

  stage_input.strideX = 1;
  stage_input.strideY = 1;

  stage_input.padX =  0;
  stage_input.padY =  0;
  stage_input.padStyle_value = 2; //TODO update with zero

  if(input_stage_info.input_shape[1]!=0)
       stage_input.inputDimX = input_stage_info.input_shape[1];
  else
       stage_input.inputDimX = 1;

  if(input_stage_info.input_shape[2]!=0)
      stage_input.inputDimY = input_stage_info.input_shape[2];
  else
       stage_input.inputDimY = 1;

  if(input_stage_info.input_shape[3]!=0)
       stage_input.inputDimZ = input_stage_info.input_shape[3];
  else
      stage_input.inputDimZ = 1;

  stage_input.tapDimX = 1;
  stage_input.tapDimY = 1;
  stage_input.tapDimZ = 1;

  stage_input.outputDimX = stage_input.inputDimX;
  stage_input.outputDimY = stage_input.inputDimY;
  stage_input.outputDimZ = stage_input.inputDimZ;

  stage_input.inputStrideX = 2 * stage_input.inputDimZ;
  stage_input.inputStrideY = 2* stage_input.inputDimX * stage_input.inputDimZ;
  stage_input.inputStrideZ = 2;

  stage_input.tapStrideX = 2 * stage_input.tapDimZ;
  stage_input.tapStrideY = 2 * stage_input.tapDimZ;
  stage_input.tapStrideZ = 2;

  stage_input.outputStrideX = 2 * stage_input.outputDimX;
  stage_input.outputStrideY = 2 * stage_input.outputDimX * stage_input.outputDimZ;
  stage_input.outputStrideZ = 2;

  stage_input.datatype_value = 2;
  stage_input.precision_value = 2;
  stage_input.storageOrder_value = 4;

  stage_input.data_Pointer = 0; //get_data_Pointer_global();
  stage_input.data_Index = 1; //get_data_Index_global();

  stage_input.taps_Pointer = 0;
  stage_input.taps_Index = 0;

  stage_input.bias_Pointer = 0;
  stage_input.bias_Index = 0;

  stage_input.opPrarams_Pointer = 0;
  stage_input.opPrarams_Index = 0;

  //stage_input.output_Pointer = calculate_output_pointer(stage_input.outputDimX, stage_input.outputDimY, stage_input.outputDimZ);
  //stage_input.output_Index = get_output_Index_global();

  stage_input.output_Pointer = 0; //TODO update later
  stage_input.output_Index = 2;  //TODO update later

  stage_input.preOp_value = 5;
  stage_input.postOp_value = 5;

  stage_input.post_param1[0] = 0x00;
  stage_input.post_param1[1] = 0x00;
  stage_input.post_param1[2] = 0x00;
  stage_input.post_param1[3] = 0x00;

  stage_input.post_strideX = 0;
  stage_input.post_strideY = 0;

  return stage_input;
}
