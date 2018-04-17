#include<stdio.h>
#include<string.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<stdint.h>
#include <log/log.h>
#include "Blob.h"

Blob_Stage_data get_AVG_POOL_stage_data(Operation_inputs_info curr_stage_info){

  Blob_Stage_data stage_avg_pool;
  Operation_inputs_info avgpool_stage_info;


  avgpool_stage_info = curr_stage_info;

  //TODO update is needed get the inut & output buffer sizes from Android

  //initialize stage variables
  stage_avg_pool.stage_name = "Average Pooling";
  stage_avg_pool.op_val = 2;

  stage_avg_pool.opt_mask = 0x80000000;

  stage_avg_pool.radixX = avgpool_stage_info.kernel_shape[1]; //op_y
  stage_avg_pool.radixY = avgpool_stage_info.kernel_shape[0]; //op_x

  stage_avg_pool.strideX = avgpool_stage_info.stride_width;
  stage_avg_pool.strideY = avgpool_stage_info.stride_height;

  stage_avg_pool.padX =  0;
  stage_avg_pool.padY =  0;

  if(avgpool_stage_info.padding_left == 0 && avgpool_stage_info.padding_right == 0 &&
    avgpool_stage_info.padding_top == 0 && avgpool_stage_info.padding_bottom == 0 )
    stage_avg_pool.padStyle_value = 1; //padding is tfvalid
  else
    stage_avg_pool.padStyle_value = 3; //padding is tfsame

  if(avgpool_stage_info.input_shape[1]!=0)
     stage_avg_pool.inputDimX = avgpool_stage_info.input_shape[1]; //TODO update from Android
  else
     stage_avg_pool.inputDimX = 1; //TODO update from Android

  if(avgpool_stage_info.input_shape[2]!=0)
    stage_avg_pool.inputDimY = avgpool_stage_info.input_shape[2]; //TODO update from Android
  else
     stage_avg_pool.inputDimY = 1; //TODO update from Android

  if(avgpool_stage_info.input_shape[3]!=0)
     stage_avg_pool.inputDimZ = avgpool_stage_info.input_shape[3]; //TODO update from Android
  else
    stage_avg_pool.inputDimZ = 1; //TODO update from Android


  stage_avg_pool.tapDimX = avgpool_stage_info.kernel_shape[1] * avgpool_stage_info.kernel_shape[0];
  stage_avg_pool.tapDimY = avgpool_stage_info.input_shape[3];
  stage_avg_pool.tapDimZ = avgpool_stage_info.output_shape[3];

  //Output DIMENSIONS calculation
  if(stage_avg_pool.padStyle_value == 1){ //padding is tfvalid
    stage_avg_pool.outputDimX = (uint32_t) std::ceil((double)(stage_avg_pool.inputDimX - stage_avg_pool.radixX +1)/(double)stage_avg_pool.strideX);
    stage_avg_pool.outputDimY = (uint32_t) std::ceil((double)(stage_avg_pool.inputDimY - stage_avg_pool.radixY +1)/(double)stage_avg_pool.strideY);
  }
  else if(stage_avg_pool.padStyle_value == 3){
    stage_avg_pool.outputDimX = (uint32_t) std::ceil((double)stage_avg_pool.inputDimX/(double)stage_avg_pool.strideX);
    stage_avg_pool.outputDimY = (uint32_t) std::ceil((double)stage_avg_pool.inputDimY/(double)stage_avg_pool.strideY);
  }
  else{//TODO update the outputDimX & outputDimY for padStyle_value = 2
    stage_avg_pool.outputDimX = (uint32_t) std::ceil((double)stage_avg_pool.inputDimX/(double)stage_avg_pool.strideX);
    stage_avg_pool.outputDimY = (uint32_t) std::ceil((double)stage_avg_pool.inputDimY/(double)stage_avg_pool.strideY);
  }
  stage_avg_pool.outputDimZ = stage_avg_pool.tapDimZ;

  stage_avg_pool.inputStrideX = 2 * stage_avg_pool.inputDimZ;
  stage_avg_pool.inputStrideY = 2* stage_avg_pool.inputDimX * stage_avg_pool.inputDimZ;
  stage_avg_pool.inputStrideZ = 2;

  stage_avg_pool.tapStrideX = 2 * stage_avg_pool.tapDimZ;
  stage_avg_pool.tapStrideY = 2 * stage_avg_pool.tapDimZ;
  stage_avg_pool.tapStrideZ = 2;

  stage_avg_pool.outputStrideX = 2 * stage_avg_pool.outputDimZ;
  stage_avg_pool.outputStrideY = 2 * stage_avg_pool.outputDimX * stage_avg_pool.outputDimZ;
  stage_avg_pool.outputStrideZ = 2;

  stage_avg_pool.datatype_value = 2;
  stage_avg_pool.precision_value = 2;
  stage_avg_pool.storageOrder_value = 2;

  stage_avg_pool.data_Pointer = get_output_Pointer_global();
  stage_avg_pool.data_Index = get_output_Index_global();

  stage_avg_pool.taps_Pointer = 0;
  stage_avg_pool.taps_Index = 0;

  stage_avg_pool.bias_Pointer = 0;
  stage_avg_pool.bias_Index = 0;

  stage_avg_pool.opPrarams_Pointer = 0;
  stage_avg_pool.opPrarams_Index = 0;

  stage_avg_pool.output_Pointer = calculate_output_pointer(stage_avg_pool.outputDimX,stage_avg_pool.outputDimY,stage_avg_pool.outputDimZ);
  stage_avg_pool.output_Index = get_output_Index_global()+1;

  stage_avg_pool.preOp_value = 5;


  avgpool_stage_info.post_operation;
  switch (avgpool_stage_info.post_operation) {
    case RELU:{stage_avg_pool.postOp_value = 6; stage_avg_pool.post_param1[0] = 0x00; stage_avg_pool.post_param1[1] = 0x00;stage_avg_pool.post_param1[2] = 0x00;stage_avg_pool.post_param1[3] = 0x00;}break;
    case RELU1:{stage_avg_pool.postOp_value = 7; stage_avg_pool.post_param1[0] = 0x00; stage_avg_pool.post_param1[1] = 0x00;stage_avg_pool.post_param1[2] = 0x80;stage_avg_pool.post_param1[3] = 0x3F;}break;
    case RELU6:{stage_avg_pool.postOp_value = 7; stage_avg_pool.post_param1[0] = 0x00; stage_avg_pool.post_param1[1] = 0x00;stage_avg_pool.post_param1[2] = 0xC0;stage_avg_pool.post_param1[3] = 0x40;}break;
    default: {stage_avg_pool.postOp_value = 5; stage_avg_pool.post_param1[0] = 0x00; stage_avg_pool.post_param1[1] = 0x00;stage_avg_pool.post_param1[2] = 0x00;stage_avg_pool.post_param1[3] = 0x00;}break;
  }

  stage_avg_pool.post_strideX = 0;
  stage_avg_pool.post_strideY = 0;


  if(update_output_Pointer_g(stage_avg_pool.output_Pointer)!=true)
    ALOGE("unable to update output_Pointer global");

  if(update_output_Index_g(stage_avg_pool.output_Index)!=true)
    ALOGE("unable to update output_Index global");


  return stage_avg_pool;
}
