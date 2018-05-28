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

Blob_Stage_data get_Softmax_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_softmax;
  Operation_inputs_info softmax_stage_info;
  softmax_stage_info = curr_stage_info;
  uint32_t size;
  //initialize stage variables
  stage_softmax.stage_name = "SoftMax Layer";
  stage_softmax.op_val = 3;

  stage_softmax.opt_mask = 0x80000000;

  stage_softmax.radixX = 1;
  stage_softmax.radixY = 1;

  stage_softmax.strideX = 1;
  stage_softmax.strideY = 1;

  stage_softmax.padX =  0;
  stage_softmax.padY =  0;
  stage_softmax.padStyle_value = 0; //No Padding Support

  if(softmax_stage_info.input_shape[0] ==0)
     softmax_stage_info.input_shape[0] = 1;
  if(softmax_stage_info.input_shape[1] ==0)
     softmax_stage_info.input_shape[1] = 1;
  if(softmax_stage_info.input_shape[2] ==0)
     softmax_stage_info.input_shape[2] = 1;
  if(softmax_stage_info.input_shape[3] ==0)
     softmax_stage_info.input_shape[3] = 1;
  else
     stage_softmax.inputDimX = 1; //TODO update from Android

  if(softmax_stage_info.input_shape[1]!=0)
     stage_softmax.inputDimX = 1; //TODO update from Android
  else
     stage_softmax.inputDimX = 1; //TODO update from Android

  if(softmax_stage_info.input_shape[2]!=0)
    stage_softmax.inputDimY = 1; //TODO update from Android
  else
     stage_softmax.inputDimY = 1; //TODO update from Android

  if(softmax_stage_info.input_shape[3]!=0)
     stage_softmax.inputDimZ = softmax_stage_info.input_shape[0] * softmax_stage_info.input_shape[1] * softmax_stage_info.input_shape[2] * softmax_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_softmax.inputDimZ = 1; //TODO update from Android

  size = softmax_stage_info.input_shape[0] * softmax_stage_info.input_shape[1] * softmax_stage_info.input_shape[2] * softmax_stage_info.input_shape[3]; //TODO update from Android
  stage_softmax.tapDimX = 0;
  stage_softmax.tapDimY = size;
  stage_softmax.tapDimZ = size;

  stage_softmax.outputDimX = stage_softmax.inputDimX;
  stage_softmax.outputDimY = stage_softmax.inputDimY;
  stage_softmax.outputDimZ = stage_softmax.inputDimZ;

  stage_softmax.inputStrideX = 2 * stage_softmax.inputDimZ;
  stage_softmax.inputStrideY = 2* stage_softmax.inputDimX * stage_softmax.inputDimZ;
  stage_softmax.inputStrideZ = 2;

  stage_softmax.tapStrideX = 2 * stage_softmax.tapDimZ;
  stage_softmax.tapStrideY = 2 * stage_softmax.tapDimZ;
  stage_softmax.tapStrideZ = 2;

  stage_softmax.outputStrideX = 2 * stage_softmax.outputDimZ;
  stage_softmax.outputStrideY = 2 * stage_softmax.outputDimX * stage_softmax.outputDimZ;
  stage_softmax.outputStrideZ = 2;

  stage_softmax.datatype_value = 2;
  stage_softmax.precision_value = 2;
  stage_softmax.storageOrder_value = 2;

  stage_softmax.data_Pointer = get_output_Pointer_global();
  stage_softmax.data_Index = get_output_Index_global();

  stage_softmax.taps_Pointer = 0;
  stage_softmax.taps_Index = 0;

  stage_softmax.bias_Pointer = 0;
  stage_softmax.bias_Index = 0;

  stage_softmax.opPrarams_Pointer = get_taps_Pointer_global();
  stage_softmax.opPrarams_Index = get_taps_Index_global();

  stage_softmax.output_Pointer = calculate_output_pointer(stage_softmax.outputDimX,stage_softmax.outputDimY,stage_softmax.outputDimZ);
  stage_softmax.output_Index = get_output_Index_global()+1;

  stage_softmax.preOp_value = 5;
  stage_softmax.postOp_value = 5;

  stage_softmax.post_param1[0] = 0x00;
  stage_softmax.post_param1[1] = 0x00;
  stage_softmax.post_param1[2] = 0x00;
  stage_softmax.post_param1[3] = 0x00;

  stage_softmax.post_strideX = 0;
  stage_softmax.post_strideY = 0;

  uint32_t new_bias_Pointer =0;
  new_bias_Pointer = stage_softmax.opPrarams_Pointer + 64; //TODO FIX the had code later

  if(update_taps_Pointer_g(new_bias_Pointer)!=true)
    ALOGE("unable to update taps_Pointer global");


  if(update_output_Pointer_g(stage_softmax.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_softmax.output_Index)!=true)
    ALOGE("unable to update output_Index global");


  return stage_softmax;
}
