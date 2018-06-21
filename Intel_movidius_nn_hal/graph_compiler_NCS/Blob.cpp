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
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<stdint.h>
#include <log/log.h>
#include <string>
#include "fp.h"

//#include "VpuPreparemodel.h" //TODO add it later
#include "Blob.h"
//#include "stage_header.h"
#define LOG_TAG "BLOB"

std::vector<std::string> graph_file_names_vector;
std::string graph_filename;
//#define dump_blob_to_file true
/*

count_network_stages_count()

prepare_blob()

estimate_file_size()

align_size()

generate_graph()

get_header_buffer()

*/
network_operations_vector nw_vector_global;
Blob_Stage_data input_stage_data;

Network_Vector_Stageinfo nwk_vector_stages_info;
unsigned int stage_count=1;


uint32_t zero_data_offset_g = 0;
uint16_t buffer_index_g = 0;

uint32_t data_Pointer_g = 0;
uint16_t data_Index_g = 1;

uint32_t taps_Pointer_g = 0;
uint16_t taps_Index_g = 3;

uint32_t bias_Pointer_g = 0;
uint16_t bias_Index_g = 3;

uint32_t opPrarams_Pointer_g = 0;
uint16_t opPrarams_Index_g = 0;

uint32_t output_Pointer_g = 0;
uint16_t output_Index_g = 3;

uint32_t global_buffer_index =0;

#define RADIX_MAX 5

float *post_data_buffer;



bool update_post_data_buffer(uint32_t size, float *buf){
  if(buf == NULL)
  ALOGD("buf is null inside update_post_data_buffer");
  if(post_data_buffer+get_global_buffer_index() == NULL)
  ALOGE("post_data_buffer+get_global_buffer_index() buffer is null");

  memcpy(post_data_buffer+get_global_buffer_index(),buf,size);
  ALOGD("Copied %lu bytes data",size);
  return true;
}

bool update_global_buffer_index(uint32_t value){
  global_buffer_index += value;
  ALOGD("updated global_buffer_index is : %lu",global_buffer_index);
  return true;
}

uint32_t get_global_buffer_index(){
  return global_buffer_index;
}

bool update_zero_data_offset_g(uint32_t value){
  zero_data_offset_g = value;
  return true;
}

uint32_t get_zero_data_offset_global(){
  return zero_data_offset_g;
}

bool update_buffer_index_g(uint16_t value){
  buffer_index_g = value;
  return true;
}

uint16_t get_buffer_index_global(){
  return buffer_index_g;
}


bool update_data_Pointer_g(uint32_t value){
  data_Pointer_g = value;
  return true;
}

uint32_t get_data_Pointer_global(){
  return data_Pointer_g;
}

bool update_data_Index_g(uint16_t value){
  data_Index_g = value;
  return true;
}

uint16_t get_data_Index_global(){
  return data_Index_g;
}

bool update_taps_Pointer_g(uint32_t value){
  taps_Pointer_g = value;
  return true;
}

uint32_t get_taps_Pointer_global(){
  return taps_Pointer_g;
}

bool update_taps_Index_g(uint16_t value){
  taps_Index_g = value;
  return true;
}

uint16_t get_taps_Index_global(){
  return taps_Index_g;
}

bool update_bias_Pointer_g(uint32_t value){
  bias_Pointer_g = value;
  return true;
}

uint32_t get_bias_Pointer_global(){
  return bias_Pointer_g;
}

bool update_bias_Index_g(uint16_t value){
  bias_Index_g = value;
  return true;
}

uint16_t get_bias_Index_global(){
  return bias_Index_g;
}

bool update_opPrarams_Pointer_g(uint32_t value){
  opPrarams_Pointer_g = value;
  return true;
}

uint32_t get_opPrarams_Pointer_global(){
  return opPrarams_Pointer_g;
}

bool update_opPrarams_Index_g(uint16_t value){
  opPrarams_Index_g = value;
  return true;
}

uint16_t get_opPrarams_Index_global(){
  return opPrarams_Index_g;
}

bool update_output_Pointer_g(uint32_t value){
  output_Pointer_g = value;
  return true;
}

uint32_t get_output_Pointer_global(){
  return output_Pointer_g;
}

bool update_output_Index_g(uint16_t value){
  output_Index_g = value;
  return true;
}

uint16_t get_output_Index_global(){
  return output_Index_g;
}


uint32_t calculate_output_pointer(uint32_t X, uint32_t Y, uint32_t Z){
  uint32_t output_pointer,buffer_size,zero_data_offset,pad;
  uint16_t buffer_index;
  uint8_t dtype = 2; //TODO fix later with proper code (dype is fp16)
  pad = ((int)(RADIX_MAX/2)) * (X+1)* (Z) * dtype;
  if(DEBUG_get_input_stage_buffer) ALOGD("pad : %u",pad);
  buffer_size = X * Y * Z * dtype + 2 * pad;
  if(DEBUG_get_input_stage_buffer) ALOGD("buffer_size : %u",buffer_size);
  //align buffer size to 64
  buffer_size += align_size(buffer_size,64);
  if(DEBUG_get_input_stage_buffer) ALOGD("align_buffer_size : %u",buffer_size);
  zero_data_offset = buffer_size+get_zero_data_offset_global();
  if(DEBUG_get_input_stage_buffer) ALOGD("zero_data_offset : %u",zero_data_offset);
  if(update_zero_data_offset_g(zero_data_offset)!=true)
    ALOGE("unable to update zero_data_offset_g");

  buffer_index= get_buffer_index_global()+1;

  if(update_buffer_index_g(buffer_index)!=true)
    ALOGE("unable to update buffer_index_g");

  output_pointer = zero_data_offset - buffer_size + pad;

  return output_pointer;
}

uint32_t calculate_taps_pointer(uint32_t X, uint32_t Y, uint32_t Z, uint32_t W){
  uint32_t taps_Pointer = 0;
  uint8_t dtype = 2; //TODO fix later with proper code (dype is fp16)
  taps_Pointer = dtype * X * Y * Z * W;
  taps_Pointer += align_size(taps_Pointer,64);
  return taps_Pointer;
}

uint32_t calculate_bias_Pointer(uint32_t X){
  uint32_t bias_Pointer = 0;
  uint8_t dtype = 2; //TODO fix later with proper code (dype is fp16)
  bias_Pointer = dtype * X;
  bias_Pointer += align_size(bias_Pointer,64);
  return bias_Pointer;
}

// Network section begin
bool get_nn_network_from_android(network_operations_vector nw_vector1){
  nw_vector_global = nw_vector1;
  return true;
}

std::vector<NCSoperations> get_network_operations_details(){
  //update the global variable for network_operations_vector(nw_vector_global)
  return nw_vector_global;
}

bool display(Operation_inputs_info cur_stage_android, int count){

  //ALOGD("Stage Count  : %d",count);
  ALOGD("cur_stage_android.main_operation : %d",cur_stage_android.main_operation);
  ALOGD("cur_stage_android.num_inputs : %d",cur_stage_android.num_inputs);
  ALOGD("cur_stage_android.input_shape : (%d, %d, %d, %d)",cur_stage_android.input_shape[0],cur_stage_android.input_shape[1],cur_stage_android.input_shape[2],cur_stage_android.input_shape[3]);

  //if(cur_stage_android.main_operation == CONV_2D || cur_stage_android.main_operation ==  DEPTHWISE_CONV_2D || cur_stage_android.main_operation == AVERAGE_POOL_2D || cur_stage_android.main_operation == MAX_POOL_2D){
  if(cur_stage_android.main_operation == CONV_2D || DEPTHWISE_CONV_2D ){
    ALOGD("cur_stage_android.kernel_shape : (%d, %d, %d, %d) ",cur_stage_android.kernel_shape[0],cur_stage_android.kernel_shape[1],cur_stage_android.kernel_shape[2],cur_stage_android.kernel_shape[3]);

    uint32_t num_of_kernel_elements = cur_stage_android.kernel_shape[0] * cur_stage_android.kernel_shape[1] *
                                      cur_stage_android.kernel_shape[2] * cur_stage_android.kernel_shape[3];
    ALOGD("cur_stage_android.kernel_num of elements : %d",num_of_kernel_elements);

    if(cur_stage_android.main_operation == CONV_2D || cur_stage_android.main_operation ==  DEPTHWISE_CONV_2D){
      ALOGD("cur_stage_android.bias_shape : (%d, %d, %d, %d) ",cur_stage_android.bias_shape[0],cur_stage_android.bias_shape[1],cur_stage_android.bias_shape[2],cur_stage_android.bias_shape[3]);
    }
    if(cur_stage_android.main_operation == DEPTHWISE_CONV_2D )
    ALOGD("cur_stage_android.depth_multiplier : %d",cur_stage_android.depth_multiplier);

    ALOGD("cur_stage_android.output_shape : (%d, %d, %d, %d) ",cur_stage_android.output_shape[0],cur_stage_android.output_shape[1],cur_stage_android.output_shape[2],cur_stage_android.output_shape[3]);


    ALOGD("cur_stage_android.padding_left : %d",cur_stage_android.padding_left);
    ALOGD("cur_stage_android.padding_right : %d",cur_stage_android.padding_right);
    ALOGD("cur_stage_android.padding_top : %d",cur_stage_android.padding_top);
    ALOGD("cur_stage_android.padding_bottom : %d",cur_stage_android.padding_bottom);

    ALOGD("cur_stage_android.stride_width : %d",cur_stage_android.stride_width);
    ALOGD("cur_stage_android.stride_height : %d",cur_stage_android.stride_height);
  }
  ALOGD("cur_stage_android.post_operation : %d",cur_stage_android.post_operation);
  return true;
}

// Network Stage section begin
bool parse_stage_from_android(Operation_inputs_info cur_stage_android){
  bool success;
  nwk_vector_stages_info.push_back(cur_stage_android);
  //TODO Fix me comment the debug

  bool enable_debug = false;
  if(nwk_vector_stages_info.size() == nw_vector_global.size() && enable_debug){
    ALOGD("Stage Count  : %d",stage_count);
    for(int i=0;i<nwk_vector_stages_info.size();i++){
    success = display(nwk_vector_stages_info.at(i), stage_count);
    }

  }
  stage_count = stage_count+1;
  return true;
}


//Handles only Input Stage

void get_input_stage_buffer(char *stage_buffer, unsigned int stage_size, Operation_inputs_info curr_stage_info){

  unsigned int index = 0;

  memset(stage_buffer,0,STAGE_SIZE);
  Blob_Stage_data current_stage_data;


  current_stage_data = get_input_stage_layer(curr_stage_info);

  //TODO create the stage_buffer from current_stage_data variable;
  //copy the stagename;
  memset((stage_buffer+index),0,SIZE_OF_STAGE_NAME);
  memcpy((stage_buffer+index),current_stage_data.stage_name.c_str(),current_stage_data.stage_name.length());
  index += SIZE_OF_STAGE_NAME; // move the index pointer by SIZE_OF_STAGE_NAME

  *(stage_buffer+index) = current_stage_data.op_val;
  index += sizeof(current_stage_data.op_val);

  //  uint32_t opt_mask
  *(stage_buffer+index++) = current_stage_data.opt_mask;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 8;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 16;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 24;

  *(stage_buffer+index) = current_stage_data.radixX;
  index += sizeof(current_stage_data.radixX);
  *(stage_buffer+index) = current_stage_data.radixY;
  index += sizeof(current_stage_data.radixY);

  *(stage_buffer+index) = current_stage_data.strideX;
  index += sizeof(current_stage_data.strideX);
  *(stage_buffer+index) = current_stage_data.strideY;
  index += sizeof(current_stage_data.strideY);

  *(stage_buffer+index) = current_stage_data.padX;
  index += sizeof(current_stage_data.padX);
  *(stage_buffer+index) = current_stage_data.padY;
  index += sizeof(current_stage_data.padY);
  *(stage_buffer+index) = current_stage_data.padStyle_value;
  index += sizeof(current_stage_data.padStyle_value);

  //  uint32_t inputDimX
  *(stage_buffer+index++) = current_stage_data.inputDimX;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 24;

  //  uint32_t inputDimY
  *(stage_buffer+index++) = current_stage_data.inputDimY;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 24;

  // uint32_t inputDimZ
  *(stage_buffer+index++) = current_stage_data.inputDimZ;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 24;

  //  uint32_t tapDimX
  *(stage_buffer+index++) = current_stage_data.tapDimX;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 24;

  //  uint32_t tapDimY
  *(stage_buffer+index++) = current_stage_data.tapDimY;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 24;

  //  uint32_t tapDimZ
  *(stage_buffer+index++) = current_stage_data.tapDimZ;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 24;

  //  uint32_t outputDimX
  *(stage_buffer+index++) = current_stage_data.outputDimX;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 24;

  //  uint32_t outputDimY
  *(stage_buffer+index++) = current_stage_data.outputDimY;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 24;

  //  uint32_t outputDimZ
  *(stage_buffer+index++) = current_stage_data.outputDimZ;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 24;

  //  uint32_t inputStrideX
  *(stage_buffer+index++) = current_stage_data.inputStrideX;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 24;

  //  uint32_t inputStrideY
  *(stage_buffer+index++) = current_stage_data.inputStrideY;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 24;

  //  uint32_t inputStrideZ
  *(stage_buffer+index++) = current_stage_data.inputStrideZ;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 24;

  //  uint32_t tapStrideX
  *(stage_buffer+index++) = current_stage_data.tapStrideX;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 24;

  //  uint32_t tapStrideY
  *(stage_buffer+index++) = current_stage_data.tapStrideY;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 24;

  //  uint32_t tapStrideZ
  *(stage_buffer+index++) = current_stage_data.tapStrideZ;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 24;

  //  uint32_t outputStrideX
  *(stage_buffer+index++) = current_stage_data.outputStrideX;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 24;

  //  uint32_t outputStrideY
  *(stage_buffer+index++) = current_stage_data.outputStrideY;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 24;

  //  uint32_t outputStrideZ
  *(stage_buffer+index++) = current_stage_data.outputStrideZ;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 24;

  *(stage_buffer+index) = current_stage_data.datatype_value;
  index += sizeof(current_stage_data.datatype_value);
  *(stage_buffer+index) = current_stage_data.precision_value;
  index += sizeof(current_stage_data.precision_value);
  *(stage_buffer+index) = current_stage_data.storageOrder_value;
  index += sizeof(current_stage_data.storageOrder_value);

  *(stage_buffer+index++) = current_stage_data.data_Pointer;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.data_Index;
  *(stage_buffer+index++) = current_stage_data.data_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.taps_Pointer;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.taps_Index;
  *(stage_buffer+index++) = current_stage_data.taps_Index >> 8;


  *(stage_buffer+index++) = current_stage_data.bias_Pointer;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.bias_Index;
  *(stage_buffer+index++) = current_stage_data.bias_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Index;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.output_Pointer;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.output_Index;
  *(stage_buffer+index++) = current_stage_data.output_Index >> 8;

  *(stage_buffer+index) = current_stage_data.preOp_value;
  index += sizeof(current_stage_data.preOp_value);
  *(stage_buffer+index) = current_stage_data.postOp_value;
  index += sizeof(current_stage_data.postOp_value);

  *(stage_buffer+index++) = current_stage_data.post_param1[0];
  *(stage_buffer+index++) = current_stage_data.post_param1[1];
  *(stage_buffer+index++) = current_stage_data.post_param1[2];
  *(stage_buffer+index++) = current_stage_data.post_param1[3];


  *(stage_buffer+index) = current_stage_data.post_strideX;
  index += sizeof(current_stage_data.post_strideX);
  *(stage_buffer+index) = current_stage_data.post_strideY;
  index += sizeof(current_stage_data.post_strideY);

  if(DEBUG_get_input_stage_buffer){
    ALOGD("current_stage_data.stage_name: %s",current_stage_data.stage_name.c_str());
    ALOGD("current_stage_data.op_val: %d",current_stage_data.op_val);
    ALOGD("current_stage_data.opt_mask: %u",current_stage_data.opt_mask);
    ALOGD("current_stage_data.radixX: %d",current_stage_data.radixX);
    ALOGD("current_stage_data.radixY: %d",current_stage_data.radixY);
    ALOGD("current_stage_data.strideX: %d",current_stage_data.strideX);
    ALOGD("current_stage_data.strideY: %d",current_stage_data.strideY);
    ALOGD("current_stage_data.padX: %d",current_stage_data.padX);
    ALOGD("current_stage_data.padY: %d",current_stage_data.padY);
    ALOGD("current_stage_data.padStyle_value: %d",current_stage_data.padStyle_value);

    ALOGD("current_stage_data.inputDimX: %u",current_stage_data.inputDimX);
    ALOGD("current_stage_data.inputDimY: %u",current_stage_data.inputDimY);
    ALOGD("current_stage_data.inputDimZ: %u",current_stage_data.inputDimZ);
    ALOGD("current_stage_data.tapDimX: %u",current_stage_data.tapDimX);
    ALOGD("current_stage_data.tapDimY: %u",current_stage_data.tapDimY);
    ALOGD("current_stage_data.tapDimZ: %u",current_stage_data.tapDimZ);
    ALOGD("current_stage_data.outputDimX: %u",current_stage_data.outputDimX);
    ALOGD("current_stage_data.outputDimY: %u",current_stage_data.outputDimY);
    ALOGD("current_stage_data.outputDimZ: %u",current_stage_data.outputDimZ);

    ALOGD("current_stage_data.inputStrideX: %u",current_stage_data.inputStrideX);
    ALOGD("current_stage_data.inputStrideY: %u",current_stage_data.inputStrideY);
    ALOGD("current_stage_data.inputStrideZ: %u",current_stage_data.inputStrideZ);
    ALOGD("current_stage_data.tapStrideX: %u",current_stage_data.tapStrideX);
    ALOGD("current_stage_data.tapStrideY: %u",current_stage_data.tapStrideY);
    ALOGD("current_stage_data.tapStrideZ: %u",current_stage_data.tapStrideZ);
    ALOGD("current_stage_data.outputStrideX: %u",current_stage_data.outputStrideX);
    ALOGD("current_stage_data.outputStrideY: %u",current_stage_data.outputStrideY);
    ALOGD("current_stage_data.outputStrideZ: %u",current_stage_data.outputStrideZ);

    ALOGD("current_stage_data.datatype_value: %d",current_stage_data.datatype_value);
    ALOGD("current_stage_data.precision_value: %d",current_stage_data.precision_value);
    ALOGD("current_stage_data.storageOrder_value: %d",current_stage_data.storageOrder_value);

    ALOGD("current_stage_data.data_Pointer: %u",current_stage_data.data_Pointer);
    ALOGD("current_stage_data.data_Index: %u",current_stage_data.data_Index);

    ALOGD("current_stage_data.taps_Pointer: %u",current_stage_data.taps_Pointer);
    ALOGD("current_stage_data.taps_Index: %u",current_stage_data.taps_Index);

    ALOGD("current_stage_data.bias_Pointer: %u",current_stage_data.bias_Pointer);
    ALOGD("current_stage_data.bias_Index: %u",current_stage_data.bias_Index);

    ALOGD("current_stage_data.opPrarams_Pointer: %u",current_stage_data.opPrarams_Pointer);
    ALOGD("current_stage_data.opPrarams_Index: %u",current_stage_data.opPrarams_Index);

    ALOGD("current_stage_data.output_Pointer: %u",current_stage_data.output_Pointer);
    ALOGD("current_stage_data.output_Index: %u",current_stage_data.output_Index);

    ALOGD("current_stage_data.preOp_value: %d",current_stage_data.preOp_value);
    ALOGD("current_stage_data.postOp_value: %d",current_stage_data.postOp_value);

    ALOGD("current_stage_data.post_param1[0]: %d",current_stage_data.post_param1[0]);
    ALOGD("current_stage_data.post_param1[1]: %d",current_stage_data.post_param1[1]);
    ALOGD("current_stage_data.post_param1[2]: %d",current_stage_data.post_param1[2]);
    ALOGD("current_stage_data.post_param1[3]: %d",current_stage_data.post_param1[3]);

    ALOGD("current_stage_data.post_strideX: %d",current_stage_data.post_strideX);
    ALOGD("current_stage_data.post_strideY: %d",current_stage_data.post_strideY);
  }
}



void get_stage_buffer(char *stage_buffer, NCSoperations curr_operation, unsigned int stage_size, Operation_inputs_info curr_stage_info){

  unsigned int index = 0;
  Blob_Stage_data current_stage_data;

  ALOGD("Opertion Index is : %d", curr_operation);

  switch (curr_operation) {
    case LOGISTIC:  current_stage_data = get_LOGISTIC_stage_data(curr_stage_info); break;
    case TANH : current_stage_data = get_TANH_stage_data(curr_stage_info); break;
    case RELU : current_stage_data = get_RELU_stage_data(curr_stage_info); break;
    case RELU1 : current_stage_data = get_RELU1_stage_data(curr_stage_info); break;
    case RELU6 : current_stage_data = get_RELU6_stage_data(curr_stage_info); break;
    case CONV_2D : current_stage_data = get_CONV_2D_stage_data(curr_stage_info); break;
    case DEPTHWISE_CONV_2D : current_stage_data = get_DEPTHWISE_CONV_2D_stage_data(curr_stage_info); break;
    case AVERAGE_POOL_2D : current_stage_data = get_AVG_POOL_stage_data(curr_stage_info); break;
	case MAX_POOL_2D : current_stage_data = get_MAX_POOL_stage_data(curr_stage_info); break;
    case RESHAPE : current_stage_data = get_Reshape_stage_data(curr_stage_info); break;
    case SOFTMAX : current_stage_data = get_Softmax_stage_data(curr_stage_info); break;
    default: break;
  }
  //TODO create the stage_buffer from current_stage_data variable;
  //copy the stagename;
  memset((stage_buffer+index),0,SIZE_OF_STAGE_NAME);
  memcpy((stage_buffer+index),current_stage_data.stage_name.c_str(),current_stage_data.stage_name.length());
  index += SIZE_OF_STAGE_NAME; // move the index pointer by SIZE_OF_STAGE_NAME

  *(stage_buffer+index) = current_stage_data.op_val;
  index += sizeof(current_stage_data.op_val);

  //  uint32_t opt_mask
  *(stage_buffer+index++) = current_stage_data.opt_mask;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 8;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 16;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 24;

  *(stage_buffer+index) = current_stage_data.radixX;
  index += sizeof(current_stage_data.radixX);
  *(stage_buffer+index) = current_stage_data.radixY;
  index += sizeof(current_stage_data.radixY);

  *(stage_buffer+index) = current_stage_data.strideX;
  index += sizeof(current_stage_data.strideX);
  *(stage_buffer+index) = current_stage_data.strideY;
  index += sizeof(current_stage_data.strideY);

  *(stage_buffer+index) = current_stage_data.padX;
  index += sizeof(current_stage_data.padX);
  *(stage_buffer+index) = current_stage_data.padY;
  index += sizeof(current_stage_data.padY);
  *(stage_buffer+index) = current_stage_data.padStyle_value;
  index += sizeof(current_stage_data.padStyle_value);

  //  uint32_t inputDimX
  *(stage_buffer+index++) = current_stage_data.inputDimX;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 24;

  //  uint32_t inputDimY
  *(stage_buffer+index++) = current_stage_data.inputDimY;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 24;

  // uint32_t inputDimZ
  *(stage_buffer+index++) = current_stage_data.inputDimZ;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 24;

  //  uint32_t tapDimX
  *(stage_buffer+index++) = current_stage_data.tapDimX;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 24;

  //  uint32_t tapDimY
  *(stage_buffer+index++) = current_stage_data.tapDimY;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 24;

  //  uint32_t tapDimZ
  *(stage_buffer+index++) = current_stage_data.tapDimZ;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 24;

  //  uint32_t outputDimX
  *(stage_buffer+index++) = current_stage_data.outputDimX;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 24;

  //  uint32_t outputDimY
  *(stage_buffer+index++) = current_stage_data.outputDimY;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 24;

  //  uint32_t outputDimZ
  *(stage_buffer+index++) = current_stage_data.outputDimZ;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 24;

  //  uint32_t inputStrideX
  *(stage_buffer+index++) = current_stage_data.inputStrideX;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 24;

  //  uint32_t inputStrideY
  *(stage_buffer+index++) = current_stage_data.inputStrideY;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 24;

  //  uint32_t inputStrideZ
  *(stage_buffer+index++) = current_stage_data.inputStrideZ;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 24;

  //  uint32_t tapStrideX
  *(stage_buffer+index++) = current_stage_data.tapStrideX;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 24;

  //  uint32_t tapStrideY
  *(stage_buffer+index++) = current_stage_data.tapStrideY;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 24;

  //  uint32_t tapStrideZ
  *(stage_buffer+index++) = current_stage_data.tapStrideZ;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 24;

  //  uint32_t outputStrideX
  *(stage_buffer+index++) = current_stage_data.outputStrideX;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 24;

  //  uint32_t outputStrideY
  *(stage_buffer+index++) = current_stage_data.outputStrideY;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 24;

  //  uint32_t outputStrideZ
  *(stage_buffer+index++) = current_stage_data.outputStrideZ;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 24;

  *(stage_buffer+index) = current_stage_data.datatype_value;
  index += sizeof(current_stage_data.datatype_value);
  *(stage_buffer+index) = current_stage_data.precision_value;
  index += sizeof(current_stage_data.precision_value);
  *(stage_buffer+index) = current_stage_data.storageOrder_value;
  index += sizeof(current_stage_data.storageOrder_value);

  *(stage_buffer+index++) = current_stage_data.data_Pointer;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.data_Index;
  *(stage_buffer+index++) = current_stage_data.data_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.taps_Pointer;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.taps_Index;
  *(stage_buffer+index++) = current_stage_data.taps_Index >> 8;


  *(stage_buffer+index++) = current_stage_data.bias_Pointer;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.bias_Index;
  *(stage_buffer+index++) = current_stage_data.bias_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Index;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.output_Pointer;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.output_Index;
  *(stage_buffer+index++) = current_stage_data.output_Index >> 8;

  *(stage_buffer+index) = current_stage_data.preOp_value;
  index += sizeof(current_stage_data.preOp_value);
  *(stage_buffer+index) = current_stage_data.postOp_value;
  index += sizeof(current_stage_data.postOp_value);

  *(stage_buffer+index++) = current_stage_data.post_param1[0];
  *(stage_buffer+index++) = current_stage_data.post_param1[1];
  *(stage_buffer+index++) = current_stage_data.post_param1[2];
  *(stage_buffer+index++) = current_stage_data.post_param1[3];


  *(stage_buffer+index) = current_stage_data.post_strideX;
  index += sizeof(current_stage_data.post_strideX);
  *(stage_buffer+index) = current_stage_data.post_strideY;
  index += sizeof(current_stage_data.post_strideY);

  if(DEBUG_get_stage_buffer){
    ALOGD("current_stage_data.stage_name: %s",current_stage_data.stage_name.c_str());
    ALOGD("current_stage_data.op_val: %d",current_stage_data.op_val);
    ALOGD("current_stage_data.opt_mask: %lu",current_stage_data.opt_mask);
    ALOGD("current_stage_data.radixX: %d",current_stage_data.radixX);
    ALOGD("current_stage_data.radixY: %d",current_stage_data.radixY);
    ALOGD("current_stage_data.strideX: %d",current_stage_data.strideX);
    ALOGD("current_stage_data.strideY: %d",current_stage_data.strideY);
    ALOGD("current_stage_data.padX: %d",current_stage_data.padX);
    ALOGD("current_stage_data.padY: %d",current_stage_data.padY);
    ALOGD("current_stage_data.padStyle_value: %d",current_stage_data.padStyle_value);

    ALOGD("current_stage_data.inputDimX: %lu",current_stage_data.inputDimX);
    ALOGD("current_stage_data.inputDimY: %lu",current_stage_data.inputDimY);
    ALOGD("current_stage_data.inputDimZ: %lu",current_stage_data.inputDimZ);
    ALOGD("current_stage_data.tapDimX: %lu",current_stage_data.tapDimX);
    ALOGD("current_stage_data.tapDimY: %lu",current_stage_data.tapDimY);
    ALOGD("current_stage_data.tapDimZ: %lu",current_stage_data.tapDimZ);
    ALOGD("current_stage_data.outputDimX: %lu",current_stage_data.outputDimX);
    ALOGD("current_stage_data.outputDimY: %lu",current_stage_data.outputDimY);
    ALOGD("current_stage_data.outputDimZ: %lu",current_stage_data.outputDimZ);

    ALOGD("current_stage_data.inputStrideX: %lu",current_stage_data.inputStrideX);
    ALOGD("current_stage_data.inputStrideY: %lu",current_stage_data.inputStrideY);
    ALOGD("current_stage_data.inputStrideZ: %lu",current_stage_data.inputStrideZ);
    ALOGD("current_stage_data.tapStrideX: %lu",current_stage_data.tapStrideX);
    ALOGD("current_stage_data.tapStrideY: %lu",current_stage_data.tapStrideY);
    ALOGD("current_stage_data.tapStrideZ: %lu",current_stage_data.tapStrideZ);
    ALOGD("current_stage_data.outputStrideX: %lu",current_stage_data.outputStrideX);
    ALOGD("current_stage_data.outputStrideY: %lu",current_stage_data.outputStrideY);
    ALOGD("current_stage_data.outputStrideZ: %lu",current_stage_data.outputStrideZ);

    ALOGD("current_stage_data.datatype_value: %d",current_stage_data.datatype_value);
    ALOGD("current_stage_data.precision_value: %d",current_stage_data.precision_value);
    ALOGD("current_stage_data.storageOrder_value: %d",current_stage_data.storageOrder_value);

    ALOGD("current_stage_data.data_Pointer: %u",current_stage_data.data_Pointer);
    ALOGD("current_stage_data.data_Index: %u",current_stage_data.data_Index);

    ALOGD("current_stage_data.taps_Pointer: %u",current_stage_data.taps_Pointer);
    ALOGD("current_stage_data.taps_Index: %u",current_stage_data.taps_Index);

    ALOGD("current_stage_data.bias_Pointer: %u",current_stage_data.bias_Pointer);
    ALOGD("current_stage_data.bias_Index: %u",current_stage_data.bias_Index);

    ALOGD("current_stage_data.opPrarams_Pointer: %u",current_stage_data.opPrarams_Pointer);
    ALOGD("current_stage_data.opPrarams_Index: %u",current_stage_data.opPrarams_Index);

    ALOGD("current_stage_data.output_Pointer: %u",current_stage_data.output_Pointer);
    ALOGD("current_stage_data.output_Index: %u",current_stage_data.output_Index);

    ALOGD("current_stage_data.preOp_value: %d",current_stage_data.preOp_value);
    ALOGD("current_stage_data.postOp_value: %d",current_stage_data.postOp_value);

    ALOGD("current_stage_data.post_param1[0]: %d",current_stage_data.post_param1[0]);
    ALOGD("current_stage_data.post_param1[1]: %d",current_stage_data.post_param1[1]);
    ALOGD("current_stage_data.post_param1[2]: %d",current_stage_data.post_param1[2]);
    ALOGD("current_stage_data.post_param1[3]: %d",current_stage_data.post_param1[3]);

    ALOGD("current_stage_data.post_strideX: %d",current_stage_data.post_strideX);
    ALOGD("current_stage_data.post_strideY: %d",current_stage_data.post_strideY);
  }

}

void get_first_stage_buffer(char *stage_buffer, NCSoperations curr_operation, unsigned int stage_size, Operation_inputs_info curr_stage_info){
  unsigned int index = 0;
  Blob_Stage_data current_stage_data;
  ALOGD("Opertion Index is : %d", curr_operation);

  switch (curr_operation) {
    case LOGISTIC:  current_stage_data = get_LOGISTIC_stage_data(curr_stage_info); break;
    case TANH : current_stage_data = get_TANH_stage_data(curr_stage_info); break;
    case RELU : current_stage_data = get_RELU_stage_data(curr_stage_info); break;
    case RELU1 : current_stage_data = get_RELU1_stage_data(curr_stage_info); break;
    case RELU6 : current_stage_data = get_RELU6_stage_data(curr_stage_info); break;
    case CONV_2D : current_stage_data = get_CONV_2D_stage_data(curr_stage_info); break;
    case DEPTHWISE_CONV_2D : current_stage_data = get_DEPTHWISE_CONV_2D_stage_data(curr_stage_info); break;
    case AVERAGE_POOL_2D : current_stage_data = get_AVG_POOL_stage_data(curr_stage_info); break;
	case MAX_POOL_2D : current_stage_data = get_MAX_POOL_stage_data(curr_stage_info); break;
    case RESHAPE : current_stage_data = get_Reshape_stage_data(curr_stage_info); break;
    case SOFTMAX : current_stage_data = get_Softmax_stage_data(curr_stage_info); break;
    default: break;
  }

  //TODO create the stage_buffer from current_stage_data variable;
  //copy the stagename;
  memset((stage_buffer+index),0,SIZE_OF_STAGE_NAME);
  memcpy((stage_buffer+index),current_stage_data.stage_name.c_str(),current_stage_data.stage_name.length());
  index += SIZE_OF_STAGE_NAME; // move the index pointer by SIZE_OF_STAGE_NAME

  *(stage_buffer+index) = current_stage_data.op_val;
  index += sizeof(current_stage_data.op_val);

  //  uint32_t opt_mask
  *(stage_buffer+index++) = current_stage_data.opt_mask;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 8;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 16;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 24;

  *(stage_buffer+index) = current_stage_data.radixX;
  index += sizeof(current_stage_data.radixX);
  *(stage_buffer+index) = current_stage_data.radixY;
  index += sizeof(current_stage_data.radixY);

  *(stage_buffer+index) = current_stage_data.strideX;
  index += sizeof(current_stage_data.strideX);
  *(stage_buffer+index) = current_stage_data.strideY;
  index += sizeof(current_stage_data.strideY);

  *(stage_buffer+index) = current_stage_data.padX;
  index += sizeof(current_stage_data.padX);
  *(stage_buffer+index) = current_stage_data.padY;
  index += sizeof(current_stage_data.padY);
  *(stage_buffer+index) = current_stage_data.padStyle_value;
  index += sizeof(current_stage_data.padStyle_value);

  //  uint32_t inputDimX
  *(stage_buffer+index++) = current_stage_data.inputDimX;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 24;

  //  uint32_t inputDimY
  *(stage_buffer+index++) = current_stage_data.inputDimY;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 24;

  // uint32_t inputDimZ
  *(stage_buffer+index++) = current_stage_data.inputDimZ;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 24;

  //  uint32_t tapDimX
  *(stage_buffer+index++) = current_stage_data.tapDimX;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 24;

  //  uint32_t tapDimY
  *(stage_buffer+index++) = current_stage_data.tapDimY;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 24;

  //  uint32_t tapDimZ
  *(stage_buffer+index++) = current_stage_data.tapDimZ;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 24;

  //  uint32_t outputDimX
  *(stage_buffer+index++) = current_stage_data.outputDimX;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 24;

  //  uint32_t outputDimY
  *(stage_buffer+index++) = current_stage_data.outputDimY;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 24;

  //  uint32_t outputDimZ
  *(stage_buffer+index++) = current_stage_data.outputDimZ;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 24;

  //  uint32_t inputStrideX
  *(stage_buffer+index++) = current_stage_data.inputStrideX;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 24;

  //  uint32_t inputStrideY
  *(stage_buffer+index++) = current_stage_data.inputStrideY;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 24;

  //  uint32_t inputStrideZ
  *(stage_buffer+index++) = current_stage_data.inputStrideZ;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 24;

  //  uint32_t tapStrideX
  *(stage_buffer+index++) = current_stage_data.tapStrideX;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 24;

  //  uint32_t tapStrideY
  *(stage_buffer+index++) = current_stage_data.tapStrideY;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 24;

  //  uint32_t tapStrideZ
  *(stage_buffer+index++) = current_stage_data.tapStrideZ;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 24;

  //  uint32_t outputStrideX
  *(stage_buffer+index++) = current_stage_data.outputStrideX;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 24;

  //  uint32_t outputStrideY
  *(stage_buffer+index++) = current_stage_data.outputStrideY;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 24;

  //  uint32_t outputStrideZ
  *(stage_buffer+index++) = current_stage_data.outputStrideZ;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 24;

  *(stage_buffer+index) = current_stage_data.datatype_value;
  index += sizeof(current_stage_data.datatype_value);
  *(stage_buffer+index) = current_stage_data.precision_value;
  index += sizeof(current_stage_data.precision_value);
  *(stage_buffer+index) = current_stage_data.storageOrder_value;
  index += sizeof(current_stage_data.storageOrder_value);

  current_stage_data.data_Pointer = (uint32_t) FIRST_STAGE_DATA_POINTER;

  *(stage_buffer+index++) = current_stage_data.data_Pointer;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 24;

  current_stage_data.data_Index =(uint16_t) FIRST_STAGE_DATA_INDEX;

  *(stage_buffer+index++) = current_stage_data.data_Index;
  *(stage_buffer+index++) = current_stage_data.data_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.taps_Pointer;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.taps_Index;
  *(stage_buffer+index++) = current_stage_data.taps_Index >> 8;


  *(stage_buffer+index++) = current_stage_data.bias_Pointer;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.bias_Index;
  *(stage_buffer+index++) = current_stage_data.bias_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Index;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.output_Pointer;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.output_Index;
  *(stage_buffer+index++) = current_stage_data.output_Index >> 8;

  *(stage_buffer+index) = current_stage_data.preOp_value;
  index += sizeof(current_stage_data.preOp_value);
  *(stage_buffer+index) = current_stage_data.postOp_value;
  index += sizeof(current_stage_data.postOp_value);

  *(stage_buffer+index++) = current_stage_data.post_param1[0];
  *(stage_buffer+index++) = current_stage_data.post_param1[1];
  *(stage_buffer+index++) = current_stage_data.post_param1[2];
  *(stage_buffer+index++) = current_stage_data.post_param1[3];


  *(stage_buffer+index) = current_stage_data.post_strideX;
  index += sizeof(current_stage_data.post_strideX);
  *(stage_buffer+index) = current_stage_data.post_strideY;
  index += sizeof(current_stage_data.post_strideY);

  if(DEBUG_get_first_stage_buffer){
    ALOGD("current_stage_data.stage_name: %s",current_stage_data.stage_name.c_str());
    ALOGD("current_stage_data.op_val: %d",current_stage_data.op_val);
    ALOGD("current_stage_data.opt_mask: %lu",current_stage_data.opt_mask);
    ALOGD("current_stage_data.radixX: %d",current_stage_data.radixX);
    ALOGD("current_stage_data.radixY: %d",current_stage_data.radixY);
    ALOGD("current_stage_data.strideX: %d",current_stage_data.strideX);
    ALOGD("current_stage_data.strideY: %d",current_stage_data.strideY);
    ALOGD("current_stage_data.padX: %d",current_stage_data.padX);
    ALOGD("current_stage_data.padY: %d",current_stage_data.padY);
    ALOGD("current_stage_data.padStyle_value: %d",current_stage_data.padStyle_value);

    ALOGD("current_stage_data.inputDimX: %lu",current_stage_data.inputDimX);
    ALOGD("current_stage_data.inputDimY: %lu",current_stage_data.inputDimY);
    ALOGD("current_stage_data.inputDimZ: %lu",current_stage_data.inputDimZ);
    ALOGD("current_stage_data.tapDimX: %lu",current_stage_data.tapDimX);
    ALOGD("current_stage_data.tapDimY: %lu",current_stage_data.tapDimY);
    ALOGD("current_stage_data.tapDimZ: %lu",current_stage_data.tapDimZ);
    ALOGD("current_stage_data.outputDimX: %lu",current_stage_data.outputDimX);
    ALOGD("current_stage_data.outputDimY: %lu",current_stage_data.outputDimY);
    ALOGD("current_stage_data.outputDimZ: %lu",current_stage_data.outputDimZ);

    ALOGD("current_stage_data.inputStrideX: %lu",current_stage_data.inputStrideX);
    ALOGD("current_stage_data.inputStrideY: %lu",current_stage_data.inputStrideY);
    ALOGD("current_stage_data.inputStrideZ: %lu",current_stage_data.inputStrideZ);
    ALOGD("current_stage_data.tapStrideX: %lu",current_stage_data.tapStrideX);
    ALOGD("current_stage_data.tapStrideY: %lu",current_stage_data.tapStrideY);
    ALOGD("current_stage_data.tapStrideZ: %lu",current_stage_data.tapStrideZ);
    ALOGD("current_stage_data.outputStrideX: %lu",current_stage_data.outputStrideX);
    ALOGD("current_stage_data.outputStrideY: %lu",current_stage_data.outputStrideY);
    ALOGD("current_stage_data.outputStrideZ: %lu",current_stage_data.outputStrideZ);

    ALOGD("current_stage_data.datatype_value: %d",current_stage_data.datatype_value);
    ALOGD("current_stage_data.precision_value: %d",current_stage_data.precision_value);
    ALOGD("current_stage_data.storageOrder_value: %d",current_stage_data.storageOrder_value);

    ALOGD("current_stage_data.data_Pointer: %lu",current_stage_data.data_Pointer);
    ALOGD("current_stage_data.data_Index: %u",current_stage_data.data_Index);

    ALOGD("current_stage_data.taps_Pointer: %lu",current_stage_data.taps_Pointer);
    ALOGD("current_stage_data.taps_Index: %u",current_stage_data.taps_Index);

    ALOGD("current_stage_data.bias_Pointer: %lu",current_stage_data.bias_Pointer);
    ALOGD("current_stage_data.bias_Index: %u",current_stage_data.bias_Index);

    ALOGD("current_stage_data.opPrarams_Pointer: %lu",current_stage_data.opPrarams_Pointer);
    ALOGD("current_stage_data.opPrarams_Index: %u",current_stage_data.opPrarams_Index);

    ALOGD("current_stage_data.output_Pointer: %lu",current_stage_data.output_Pointer);
    ALOGD("current_stage_data.output_Index: %u",current_stage_data.output_Index);

    ALOGD("current_stage_data.preOp_value: %d",current_stage_data.preOp_value);
    ALOGD("current_stage_data.postOp_value: %d",current_stage_data.postOp_value);

    ALOGD("current_stage_data.post_param1[0]: %d",current_stage_data.post_param1[0]);
    ALOGD("current_stage_data.post_param1[1]: %d",current_stage_data.post_param1[1]);
    ALOGD("current_stage_data.post_param1[2]: %d",current_stage_data.post_param1[2]);
    ALOGD("current_stage_data.post_param1[3]: %d",current_stage_data.post_param1[3]);

    ALOGD("current_stage_data.post_strideX: %d",current_stage_data.post_strideX);
    ALOGD("current_stage_data.post_strideY: %d",current_stage_data.post_strideY);
  }

}

void get_last_stage_buffer(char *stage_buffer, NCSoperations curr_operation, unsigned int stage_size, Operation_inputs_info curr_stage_info){

  unsigned int index = 0;
  Blob_Stage_data current_stage_data;

  ALOGD("Opertion Index is : %d", curr_operation);

  switch (curr_operation) {
    case LOGISTIC:  current_stage_data = get_LOGISTIC_stage_data(curr_stage_info); break;
    case TANH : current_stage_data = get_TANH_stage_data(curr_stage_info); break;
    case RELU : current_stage_data = get_RELU_stage_data(curr_stage_info); break;
    case RELU1 : current_stage_data = get_RELU1_stage_data(curr_stage_info); break;
    case RELU6 : current_stage_data = get_RELU6_stage_data(curr_stage_info); break;
    case CONV_2D : current_stage_data = get_CONV_2D_stage_data(curr_stage_info); break;
    case DEPTHWISE_CONV_2D : current_stage_data = get_DEPTHWISE_CONV_2D_stage_data(curr_stage_info); break;
    case AVERAGE_POOL_2D : current_stage_data = get_AVG_POOL_stage_data(curr_stage_info); break;
    case MAX_POOL_2D : current_stage_data = get_MAX_POOL_stage_data(curr_stage_info); break;
	case RESHAPE : current_stage_data = get_Reshape_stage_data(curr_stage_info); break;
    case SOFTMAX : current_stage_data = get_Softmax_stage_data(curr_stage_info); break;
    default: break;
  }
  //TODO create the stage_buffer from current_stage_data variable;
  //copy the stagename;
  memset((stage_buffer+index),0,SIZE_OF_STAGE_NAME);
  memcpy((stage_buffer+index),current_stage_data.stage_name.c_str(),current_stage_data.stage_name.length());
  index += SIZE_OF_STAGE_NAME; // move the index pointer by SIZE_OF_STAGE_NAME

  *(stage_buffer+index) = current_stage_data.op_val;
  index += sizeof(current_stage_data.op_val);

  //  uint32_t opt_mask
  *(stage_buffer+index++) = current_stage_data.opt_mask;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 8;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 16;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 24;

  *(stage_buffer+index) = current_stage_data.radixX;
  index += sizeof(current_stage_data.radixX);
  *(stage_buffer+index) = current_stage_data.radixY;
  index += sizeof(current_stage_data.radixY);

  *(stage_buffer+index) = current_stage_data.strideX;
  index += sizeof(current_stage_data.strideX);
  *(stage_buffer+index) = current_stage_data.strideY;
  index += sizeof(current_stage_data.strideY);

  *(stage_buffer+index) = current_stage_data.padX;
  index += sizeof(current_stage_data.padX);
  *(stage_buffer+index) = current_stage_data.padY;
  index += sizeof(current_stage_data.padY);
  *(stage_buffer+index) = current_stage_data.padStyle_value;
  index += sizeof(current_stage_data.padStyle_value);

  //  uint32_t inputDimX
  *(stage_buffer+index++) = current_stage_data.inputDimX;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 24;

  //  uint32_t inputDimY
  *(stage_buffer+index++) = current_stage_data.inputDimY;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 24;

  // uint32_t inputDimZ
  *(stage_buffer+index++) = current_stage_data.inputDimZ;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 24;

  //  uint32_t tapDimX
  *(stage_buffer+index++) = current_stage_data.tapDimX;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 24;

  //  uint32_t tapDimY
  *(stage_buffer+index++) = current_stage_data.tapDimY;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 24;

  //  uint32_t tapDimZ
  *(stage_buffer+index++) = current_stage_data.tapDimZ;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 24;

  //  uint32_t outputDimX
  *(stage_buffer+index++) = current_stage_data.outputDimX;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 24;

  //  uint32_t outputDimY
  *(stage_buffer+index++) = current_stage_data.outputDimY;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 24;

  //  uint32_t outputDimZ
  *(stage_buffer+index++) = current_stage_data.outputDimZ;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 24;

  //  uint32_t inputStrideX
  *(stage_buffer+index++) = current_stage_data.inputStrideX;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 24;

  //  uint32_t inputStrideY
  *(stage_buffer+index++) = current_stage_data.inputStrideY;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 24;

  //  uint32_t inputStrideZ
  *(stage_buffer+index++) = current_stage_data.inputStrideZ;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 24;

  //  uint32_t tapStrideX
  *(stage_buffer+index++) = current_stage_data.tapStrideX;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 24;

  //  uint32_t tapStrideY
  *(stage_buffer+index++) = current_stage_data.tapStrideY;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 24;

  //  uint32_t tapStrideZ
  *(stage_buffer+index++) = current_stage_data.tapStrideZ;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 24;

  //  uint32_t outputStrideX
  *(stage_buffer+index++) = current_stage_data.outputStrideX;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 24;

  //  uint32_t outputStrideY
  *(stage_buffer+index++) = current_stage_data.outputStrideY;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 24;

  //  uint32_t outputStrideZ
  *(stage_buffer+index++) = current_stage_data.outputStrideZ;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 24;

  *(stage_buffer+index) = current_stage_data.datatype_value;
  index += sizeof(current_stage_data.datatype_value);
  *(stage_buffer+index) = current_stage_data.precision_value;
  index += sizeof(current_stage_data.precision_value);
  *(stage_buffer+index) = current_stage_data.storageOrder_value;
  index += sizeof(current_stage_data.storageOrder_value);

  *(stage_buffer+index++) = current_stage_data.data_Pointer;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.data_Index;
  *(stage_buffer+index++) = current_stage_data.data_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.taps_Pointer;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.taps_Index;
  *(stage_buffer+index++) = current_stage_data.taps_Index >> 8;


  *(stage_buffer+index++) = current_stage_data.bias_Pointer;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.bias_Index;
  *(stage_buffer+index++) = current_stage_data.bias_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Index;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Index >> 8;

  current_stage_data.output_Pointer = (uint32_t) LAST_STAGE_OUTPUT_POINTER;


  *(stage_buffer+index++) = current_stage_data.output_Pointer;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 24;

  current_stage_data.output_Index =(uint16_t) LAST_STAGE_OUTPUT_INDEX;
  *(stage_buffer+index++) = current_stage_data.output_Index;
  *(stage_buffer+index++) = current_stage_data.output_Index >> 8;

  *(stage_buffer+index) = current_stage_data.preOp_value;
  index += sizeof(current_stage_data.preOp_value);
  *(stage_buffer+index) = current_stage_data.postOp_value;
  index += sizeof(current_stage_data.postOp_value);

  *(stage_buffer+index++) = current_stage_data.post_param1[0];
  *(stage_buffer+index++) = current_stage_data.post_param1[1];
  *(stage_buffer+index++) = current_stage_data.post_param1[2];
  *(stage_buffer+index++) = current_stage_data.post_param1[3];


  *(stage_buffer+index) = current_stage_data.post_strideX;
  index += sizeof(current_stage_data.post_strideX);
  *(stage_buffer+index) = current_stage_data.post_strideY;
  index += sizeof(current_stage_data.post_strideY);

  if(DEBUG_get_last_stage_buffer){
    ALOGD("current_stage_data.stage_name: %s",current_stage_data.stage_name.c_str());
    ALOGD("current_stage_data.op_val: %d",current_stage_data.op_val);
    ALOGD("current_stage_data.opt_mask: %u",current_stage_data.opt_mask);
    ALOGD("current_stage_data.radixX: %d",current_stage_data.radixX);
    ALOGD("current_stage_data.radixY: %d",current_stage_data.radixY);
    ALOGD("current_stage_data.strideX: %d",current_stage_data.strideX);
    ALOGD("current_stage_data.strideY: %d",current_stage_data.strideY);
    ALOGD("current_stage_data.padX: %d",current_stage_data.padX);
    ALOGD("current_stage_data.padY: %d",current_stage_data.padY);
    ALOGD("current_stage_data.padStyle_value: %d",current_stage_data.padStyle_value);

    ALOGD("current_stage_data.inputDimX: %u",current_stage_data.inputDimX);
    ALOGD("current_stage_data.inputDimY: %u",current_stage_data.inputDimY);
    ALOGD("current_stage_data.inputDimZ: %u",current_stage_data.inputDimZ);
    ALOGD("current_stage_data.tapDimX: %u",current_stage_data.tapDimX);
    ALOGD("current_stage_data.tapDimY: %u",current_stage_data.tapDimY);
    ALOGD("current_stage_data.tapDimZ: %u",current_stage_data.tapDimZ);
    ALOGD("current_stage_data.outputDimX: %u",current_stage_data.outputDimX);
    ALOGD("current_stage_data.outputDimY: %u",current_stage_data.outputDimY);
    ALOGD("current_stage_data.outputDimZ: %u",current_stage_data.outputDimZ);

    ALOGD("current_stage_data.inputStrideX: %u",current_stage_data.inputStrideX);
    ALOGD("current_stage_data.inputStrideY: %u",current_stage_data.inputStrideY);
    ALOGD("current_stage_data.inputStrideZ: %u",current_stage_data.inputStrideZ);
    ALOGD("current_stage_data.tapStrideX: %u",current_stage_data.tapStrideX);
    ALOGD("current_stage_data.tapStrideY: %u",current_stage_data.tapStrideY);
    ALOGD("current_stage_data.tapStrideZ: %u",current_stage_data.tapStrideZ);
    ALOGD("current_stage_data.outputStrideX: %u",current_stage_data.outputStrideX);
    ALOGD("current_stage_data.outputStrideY: %u",current_stage_data.outputStrideY);
    ALOGD("current_stage_data.outputStrideZ: %u",current_stage_data.outputStrideZ);

    ALOGD("current_stage_data.datatype_value: %d",current_stage_data.datatype_value);
    ALOGD("current_stage_data.precision_value: %d",current_stage_data.precision_value);
    ALOGD("current_stage_data.storageOrder_value: %d",current_stage_data.storageOrder_value);

    ALOGD("current_stage_data.data_Pointer: %u",current_stage_data.data_Pointer);
    ALOGD("current_stage_data.data_Index: %u",current_stage_data.data_Index);

    ALOGD("current_stage_data.taps_Pointer: %u",current_stage_data.taps_Pointer);
    ALOGD("current_stage_data.taps_Index: %u",current_stage_data.taps_Index);

    ALOGD("current_stage_data.bias_Pointer: %u",current_stage_data.bias_Pointer);
    ALOGD("current_stage_data.bias_Index: %u",current_stage_data.bias_Index);

    ALOGD("current_stage_data.opPrarams_Pointer: %u",current_stage_data.opPrarams_Pointer);
    ALOGD("current_stage_data.opPrarams_Index: %u",current_stage_data.opPrarams_Index);

    ALOGD("current_stage_data.output_Pointer: %u",current_stage_data.output_Pointer);
    ALOGD("current_stage_data.output_Index: %u",current_stage_data.output_Index);

    ALOGD("current_stage_data.preOp_value: %d",current_stage_data.preOp_value);
    ALOGD("current_stage_data.postOp_value: %d",current_stage_data.postOp_value);

    ALOGD("current_stage_data.post_param1[0]: %d",current_stage_data.post_param1[0]);
    ALOGD("current_stage_data.post_param1[1]: %d",current_stage_data.post_param1[1]);
    ALOGD("current_stage_data.post_param1[2]: %d",current_stage_data.post_param1[2]);
    ALOGD("current_stage_data.post_param1[3]: %d",current_stage_data.post_param1[3]);

    ALOGD("current_stage_data.post_strideX: %d",current_stage_data.post_strideX);
    ALOGD("current_stage_data.post_strideY: %d",current_stage_data.post_strideY);
  }
}


void get_one_stage_buffer(char *stage_buffer, NCSoperations curr_operation, unsigned int stage_size, Operation_inputs_info curr_stage_info){

  unsigned int index = 0;
  Blob_Stage_data current_stage_data;

  ALOGD("Opertion Index is : %d", curr_operation);

  switch (curr_operation) {
    case LOGISTIC:  current_stage_data = get_LOGISTIC_stage_data(curr_stage_info); break;
    case TANH : current_stage_data = get_TANH_stage_data(curr_stage_info); break;
    case RELU : current_stage_data = get_RELU_stage_data(curr_stage_info); break;
    case RELU1 : current_stage_data = get_RELU1_stage_data(curr_stage_info); break;
    case RELU6 : current_stage_data = get_RELU6_stage_data(curr_stage_info); break;
    case CONV_2D : current_stage_data = get_CONV_2D_stage_data(curr_stage_info); break;
    case DEPTHWISE_CONV_2D : current_stage_data = get_DEPTHWISE_CONV_2D_stage_data(curr_stage_info); break;
    case AVERAGE_POOL_2D : current_stage_data = get_AVG_POOL_stage_data(curr_stage_info); break;
	case MAX_POOL_2D : current_stage_data = get_MAX_POOL_stage_data(curr_stage_info); break;
    case RESHAPE : current_stage_data = get_Reshape_stage_data(curr_stage_info); break;
    case SOFTMAX : current_stage_data = get_Softmax_stage_data(curr_stage_info); break;
    default: break;
  }
  //TODO create the stage_buffer from current_stage_data variable;
  //copy the stagename;
  memset((stage_buffer+index),0,SIZE_OF_STAGE_NAME);
  memcpy((stage_buffer+index),current_stage_data.stage_name.c_str(),current_stage_data.stage_name.length());
  index += SIZE_OF_STAGE_NAME; // move the index pointer by SIZE_OF_STAGE_NAME

  *(stage_buffer+index) = current_stage_data.op_val;
  index += sizeof(current_stage_data.op_val);

  //  uint32_t opt_mask
  *(stage_buffer+index++) = current_stage_data.opt_mask;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 8;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 16;
  *(stage_buffer+index++) = current_stage_data.opt_mask >> 24;

  *(stage_buffer+index) = current_stage_data.radixX;
  index += sizeof(current_stage_data.radixX);
  *(stage_buffer+index) = current_stage_data.radixY;
  index += sizeof(current_stage_data.radixY);

  *(stage_buffer+index) = current_stage_data.strideX;
  index += sizeof(current_stage_data.strideX);
  *(stage_buffer+index) = current_stage_data.strideY;
  index += sizeof(current_stage_data.strideY);

  *(stage_buffer+index) = current_stage_data.padX;
  index += sizeof(current_stage_data.padX);
  *(stage_buffer+index) = current_stage_data.padY;
  index += sizeof(current_stage_data.padY);
  *(stage_buffer+index) = current_stage_data.padStyle_value;
  index += sizeof(current_stage_data.padStyle_value);

  //  uint32_t inputDimX
  *(stage_buffer+index++) = current_stage_data.inputDimX;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimX >> 24;

  //  uint32_t inputDimY
  *(stage_buffer+index++) = current_stage_data.inputDimY;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimY >> 24;

  // uint32_t inputDimZ
  *(stage_buffer+index++) = current_stage_data.inputDimZ;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputDimZ >> 24;

  //  uint32_t tapDimX
  *(stage_buffer+index++) = current_stage_data.tapDimX;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimX >> 24;

  //  uint32_t tapDimY
  *(stage_buffer+index++) = current_stage_data.tapDimY;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimY >> 24;

  //  uint32_t tapDimZ
  *(stage_buffer+index++) = current_stage_data.tapDimZ;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapDimZ >> 24;

  //  uint32_t outputDimX
  *(stage_buffer+index++) = current_stage_data.outputDimX;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimX >> 24;

  //  uint32_t outputDimY
  *(stage_buffer+index++) = current_stage_data.outputDimY;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimY >> 24;

  //  uint32_t outputDimZ
  *(stage_buffer+index++) = current_stage_data.outputDimZ;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputDimZ >> 24;

  //  uint32_t inputStrideX
  *(stage_buffer+index++) = current_stage_data.inputStrideX;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideX >> 24;

  //  uint32_t inputStrideY
  *(stage_buffer+index++) = current_stage_data.inputStrideY;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideY >> 24;

  //  uint32_t inputStrideZ
  *(stage_buffer+index++) = current_stage_data.inputStrideZ;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.inputStrideZ >> 24;

  //  uint32_t tapStrideX
  *(stage_buffer+index++) = current_stage_data.tapStrideX;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideX >> 24;

  //  uint32_t tapStrideY
  *(stage_buffer+index++) = current_stage_data.tapStrideY;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideY >> 24;

  //  uint32_t tapStrideZ
  *(stage_buffer+index++) = current_stage_data.tapStrideZ;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.tapStrideZ >> 24;

  //  uint32_t outputStrideX
  *(stage_buffer+index++) = current_stage_data.outputStrideX;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideX >> 24;

  //  uint32_t outputStrideY
  *(stage_buffer+index++) = current_stage_data.outputStrideY;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideY >> 24;

  //  uint32_t outputStrideZ
  *(stage_buffer+index++) = current_stage_data.outputStrideZ;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 8;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 16;
  *(stage_buffer+index++) = current_stage_data.outputStrideZ >> 24;

  *(stage_buffer+index) = current_stage_data.datatype_value;
  index += sizeof(current_stage_data.datatype_value);
  *(stage_buffer+index) = current_stage_data.precision_value;
  index += sizeof(current_stage_data.precision_value);
  *(stage_buffer+index) = current_stage_data.storageOrder_value;
  index += sizeof(current_stage_data.storageOrder_value);

  current_stage_data.data_Pointer = (uint32_t) FIRST_STAGE_DATA_POINTER;

  *(stage_buffer+index++) = current_stage_data.data_Pointer;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.data_Pointer >> 24;

  current_stage_data.data_Index = (uint16_t) FIRST_STAGE_DATA_INDEX;

  *(stage_buffer+index++) = current_stage_data.data_Index;
  *(stage_buffer+index++) = current_stage_data.data_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.taps_Pointer;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.taps_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.taps_Index;
  *(stage_buffer+index++) = current_stage_data.taps_Index >> 8;


  *(stage_buffer+index++) = current_stage_data.bias_Pointer;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.bias_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.bias_Index;
  *(stage_buffer+index++) = current_stage_data.bias_Index >> 8;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Pointer >> 24;

  *(stage_buffer+index++) = current_stage_data.opPrarams_Index;
  *(stage_buffer+index++) = current_stage_data.opPrarams_Index >> 8;

  current_stage_data.output_Pointer = (uint32_t) LAST_STAGE_OUTPUT_POINTER;


  *(stage_buffer+index++) = current_stage_data.output_Pointer;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 8;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 16;
  *(stage_buffer+index++) = current_stage_data.output_Pointer >> 24;

  current_stage_data.output_Index =(uint16_t) LAST_STAGE_OUTPUT_INDEX;
  *(stage_buffer+index++) = current_stage_data.output_Index;
  *(stage_buffer+index++) = current_stage_data.output_Index >> 8;

  *(stage_buffer+index) = current_stage_data.preOp_value;
  index += sizeof(current_stage_data.preOp_value);
  *(stage_buffer+index) = current_stage_data.postOp_value;
  index += sizeof(current_stage_data.postOp_value);

  *(stage_buffer+index++) = current_stage_data.post_param1[0];
  *(stage_buffer+index++) = current_stage_data.post_param1[1];
  *(stage_buffer+index++) = current_stage_data.post_param1[2];
  *(stage_buffer+index++) = current_stage_data.post_param1[3];


  *(stage_buffer+index) = current_stage_data.post_strideX;
  index += sizeof(current_stage_data.post_strideX);
  *(stage_buffer+index) = current_stage_data.post_strideY;
  index += sizeof(current_stage_data.post_strideY);

  if(DEBUG_get_one_stage_buffer){
    ALOGD("current_stage_data.stage_name: %s",current_stage_data.stage_name.c_str());
    ALOGD("current_stage_data.op_val: %d",current_stage_data.op_val);
    ALOGD("current_stage_data.opt_mask: %u",current_stage_data.opt_mask);
    ALOGD("current_stage_data.radixX: %d",current_stage_data.radixX);
    ALOGD("current_stage_data.radixY: %d",current_stage_data.radixY);
    ALOGD("current_stage_data.strideX: %d",current_stage_data.strideX);
    ALOGD("current_stage_data.strideY: %d",current_stage_data.strideY);
    ALOGD("current_stage_data.padX: %d",current_stage_data.padX);
    ALOGD("current_stage_data.padY: %d",current_stage_data.padY);
    ALOGD("current_stage_data.padStyle_value: %d",current_stage_data.padStyle_value);

    ALOGD("current_stage_data.inputDimX: %u",current_stage_data.inputDimX);
    ALOGD("current_stage_data.inputDimY: %u",current_stage_data.inputDimY);
    ALOGD("current_stage_data.inputDimZ: %u",current_stage_data.inputDimZ);
    ALOGD("current_stage_data.tapDimX: %u",current_stage_data.tapDimX);
    ALOGD("current_stage_data.tapDimY: %u",current_stage_data.tapDimY);
    ALOGD("current_stage_data.tapDimZ: %u",current_stage_data.tapDimZ);
    ALOGD("current_stage_data.outputDimX: %u",current_stage_data.outputDimX);
    ALOGD("current_stage_data.outputDimY: %u",current_stage_data.outputDimY);
    ALOGD("current_stage_data.outputDimZ: %u",current_stage_data.outputDimZ);

    ALOGD("current_stage_data.inputStrideX: %u",current_stage_data.inputStrideX);
    ALOGD("current_stage_data.inputStrideY: %u",current_stage_data.inputStrideY);
    ALOGD("current_stage_data.inputStrideZ: %u",current_stage_data.inputStrideZ);
    ALOGD("current_stage_data.tapStrideX: %u",current_stage_data.tapStrideX);
    ALOGD("current_stage_data.tapStrideY: %u",current_stage_data.tapStrideY);
    ALOGD("current_stage_data.tapStrideZ: %u",current_stage_data.tapStrideZ);
    ALOGD("current_stage_data.outputStrideX: %u",current_stage_data.outputStrideX);
    ALOGD("current_stage_data.outputStrideY: %u",current_stage_data.outputStrideY);
    ALOGD("current_stage_data.outputStrideZ: %u",current_stage_data.outputStrideZ);

    ALOGD("current_stage_data.datatype_value: %d",current_stage_data.datatype_value);
    ALOGD("current_stage_data.precision_value: %d",current_stage_data.precision_value);
    ALOGD("current_stage_data.storageOrder_value: %d",current_stage_data.storageOrder_value);

    ALOGD("current_stage_data.data_Pointer: %u",current_stage_data.data_Pointer);
    ALOGD("current_stage_data.data_Index: %u",current_stage_data.data_Index);

    ALOGD("current_stage_data.taps_Pointer: %u",current_stage_data.taps_Pointer);
    ALOGD("current_stage_data.taps_Index: %u",current_stage_data.taps_Index);

    ALOGD("current_stage_data.bias_Pointer: %u",current_stage_data.bias_Pointer);
    ALOGD("current_stage_data.bias_Index: %u",current_stage_data.bias_Index);

    ALOGD("current_stage_data.opPrarams_Pointer: %u",current_stage_data.opPrarams_Pointer);
    ALOGD("current_stage_data.opPrarams_Index: %u",current_stage_data.opPrarams_Index);

    ALOGD("current_stage_data.output_Pointer: %u",current_stage_data.output_Pointer);
    ALOGD("current_stage_data.output_Index: %u",current_stage_data.output_Index);

    ALOGD("current_stage_data.preOp_value: %d",current_stage_data.preOp_value);
    ALOGD("current_stage_data.postOp_value: %d",current_stage_data.postOp_value);

    ALOGD("current_stage_data.post_param1[0]: %d",current_stage_data.post_param1[0]);
    ALOGD("current_stage_data.post_param1[1]: %d",current_stage_data.post_param1[1]);
    ALOGD("current_stage_data.post_param1[2]: %d",current_stage_data.post_param1[2]);
    ALOGD("current_stage_data.post_param1[3]: %d",current_stage_data.post_param1[3]);

    ALOGD("current_stage_data.post_strideX: %d",current_stage_data.post_strideX);
    ALOGD("current_stage_data.post_strideY: %d",current_stage_data.post_strideY);
  }
}


bool prepare_blob(std::string str,int graph_count){

  Blobconfig blob1;
  Myriadconfig mconfig;
  network_operations_vector network_operations;

  network_operations = get_network_operations_details();

  blob1.version = 2;
  blob1.network_name = str;
  blob1.blob_report_dir = "";
  blob1.stage_count = network_operations.size()+1;
  blob1.filesize = estimate_file_size(true, blob1.stage_count);
  blob1.filesize_without_data = estimate_file_size(false, blob1.stage_count);

  mconfig.firstShave = 0;
  mconfig.lastShave = 11;
  mconfig.leonMemLocation = 0;
  mconfig.leonMemSize = 0;
  mconfig.dmaAgent = 0;

  ALOGD("network_name: %s",blob1.network_name.c_str());

  char* graph_buffer;

  graph_buffer = (char*)malloc(blob1.filesize_without_data);
  if(graph_buffer == NULL)
  ALOGE("unable to allocate graph_buffer");
  memset(graph_buffer,0,blob1.filesize_without_data);


  graph_filename = "/data/ncs_graph"; //+std::to_string(graph_count);
  graph_buffer = generate_graph(graph_buffer, blob1, mconfig);

  FILE *fp;
  fp=fopen(graph_filename.c_str(),"wb");

  if(fp == NULL)
    {
        ALOGE("unable to open file %s",graph_filename.c_str());
        return false;
    }
  graph_file_names_vector.push_back(graph_filename);
  fwrite(graph_buffer,blob1.filesize_without_data,1,fp);
  fclose(fp);

  bool status;

  status = wrtie_post_stage_data(blob1, mconfig);

  //free(stage_buffer);
  //free(post_data_buffer);
  free(graph_buffer);
  nwk_vector_stages_info.clear();
  memset(&input_stage_data,0,sizeof(input_stage_data));
  nwk_vector_stages_info.clear();
  stage_count = 0;

  uint32_t zero_data_offset_g = 0;
  uint16_t buffer_index_g = 0;

  uint32_t data_Pointer_g = 0;
  uint16_t data_Index_g = 1;

  uint32_t taps_Pointer_g = 0;
  uint16_t taps_Index_g = 3;

  uint32_t bias_Pointer_g = 0;
  uint16_t bias_Index_g = 3;

  uint32_t opPrarams_Pointer_g = 0;
  uint16_t opPrarams_Index_g = 0;

  uint32_t output_Pointer_g = 0;
  uint16_t output_Index_g = 3;

  uint32_t global_buffer_index =0;

  if(update_taps_Pointer_g(taps_Pointer_g)!=true)
    ALOGE("unable to update taps_Pointer global");

  return true;
}

bool wrtie_post_stage_data(Blobconfig blob_config, Myriadconfig mconfig){
  bool status = false;
  for(int i=0;i<nwk_vector_stages_info.size();i++){
    if(nwk_vector_stages_info.at(i).kernel_data == true || nwk_vector_stages_info.at(i).bias_data == true || nwk_vector_stages_info.at(i).op_params_data == true){
      ALOGD("nwk_vector_stages_info.at(i).main_operation %d", nwk_vector_stages_info.at(i).main_operation);
      status = write_kernel_bias_data_buffer_to_file(nwk_vector_stages_info.at(i));
    }
  //ALOGD("wrtie_post_stage_data status %d", status);
  }
  return status;
}

bool write_kernel_bias_data_buffer_to_file(Operation_inputs_info curr_stage_info){
  float *kernel_data_buffer, *bias_data_buffer, *op_params_buffer,*final_float_data_buffer;
  float *kernel_data_buffer_android;
  uint32_t buf_index = 0, kenrel_data_size = 0, bias_data_size = 0;
  uint32_t op_params_size = 0, final_data_size = 0;

  uint32_t kenrel_data_size_align = 0, bias_data_size_align = 0;
  uint32_t op_params_size_align = 0, final_data_size_align = 0;

  uint8_t dtype = 2; //FP16 only supported data size in Bytes
  uint8_t dtype_android = 4; //FP32 from Android data size in Bytes
  FILE *fp;
  half *buffer_fp16;

  if(curr_stage_info.kernel_data == true){
    curr_stage_info.kernel_shape[0] = (curr_stage_info.kernel_shape[0] == 0) ? 1: curr_stage_info.kernel_shape[0];
    curr_stage_info.kernel_shape[1] = (curr_stage_info.kernel_shape[1] == 0) ? 1: curr_stage_info.kernel_shape[1];
    curr_stage_info.kernel_shape[2] = (curr_stage_info.kernel_shape[2] == 0) ? 1: curr_stage_info.kernel_shape[2];
    curr_stage_info.kernel_shape[3] = (curr_stage_info.kernel_shape[3] == 0) ? 1: curr_stage_info.kernel_shape[3];

    kenrel_data_size = dtype_android * curr_stage_info.kernel_shape[0] * curr_stage_info.kernel_shape[1] *
                                      curr_stage_info.kernel_shape[2] * curr_stage_info.kernel_shape[3];
#if 1

    kernel_data_buffer_android = (float *)malloc(kenrel_data_size);
    if(kernel_data_buffer_android == NULL){
       ALOGE("unable to allocate buffer");
       kernel_data_buffer_android = nullptr;
    }else{
      memset(kernel_data_buffer_android,0, kenrel_data_size);
      /*
      ALOGD("curr_stage_info.kernel_shape[0]:%u",curr_stage_info.kernel_shape[0]);
      ALOGD("curr_stage_info.kernel_shape[1]:%u",curr_stage_info.kernel_shape[1]);
      ALOGD("curr_stage_info.kernel_shape[2]:%u",curr_stage_info.kernel_shape[2]);
      ALOGD("curr_stage_info.kernel_shape[3]:%u",curr_stage_info.kernel_shape[3]);*/

      uint32_t INCH = curr_stage_info.kernel_shape[2];
      uint32_t OUTCH = curr_stage_info.kernel_shape[3];
      uint32_t FILTER_HEIGHT = curr_stage_info.kernel_shape[0];
      uint32_t FILTER_WIDTH = curr_stage_info.kernel_shape[1];
      uint32_t index =0;
      for(int i =0;i<(INCH*FILTER_HEIGHT*FILTER_WIDTH);i++){
        for(int j=0;j<OUTCH;j++){
          *(kernel_data_buffer_android+index)= *(curr_stage_info.kernel_buffer+j*(INCH*FILTER_HEIGHT*FILTER_WIDTH)+i);
          index = index+1;
        }
      }
    }

#endif
    kenrel_data_size_align = kenrel_data_size + align_size(kenrel_data_size,128);

    kernel_data_buffer = (float *)malloc(kenrel_data_size_align);
    if(kernel_data_buffer == NULL)
    ALOGE("unable to allocate kernel_data_buffer exit");
    if(curr_stage_info.kernel_buffer == NULL)
    ALOGE(" curr_stage_info.kernel_buffer is null ");
    memset(kernel_data_buffer,0,kenrel_data_size_align);
    //memcpy(kernel_data_buffer,curr_stage_info.kernel_buffer,kenrel_data_size);
    memcpy(kernel_data_buffer,kernel_data_buffer_android,kenrel_data_size);
    buffer_fp16 = (half*) malloc(kenrel_data_size_align/2);
    memset(buffer_fp16,0,kenrel_data_size_align/2);
    if( buffer_fp16 == NULL)
    ALOGE("unable to allocate buffer_fp16 exit");
    else
    ALOGD("buffer_fp16 allocation success");
    floattofp16((unsigned char *)buffer_fp16, kernel_data_buffer, kenrel_data_size_align/4);

    fp = fopen(graph_filename.c_str(),"ab+");
    if(fp == NULL)
    ALOGE("unable to open ncs_graph file for kernel_data writing");
    else
    ALOGD("ncs_graph file is open for kernel_data writing");
    fseek(fp,0,SEEK_END);
    fwrite((unsigned char *)buffer_fp16,kenrel_data_size_align/2,1,fp);
    fclose(fp);
  ALOGD("copied kernel_data_buffer %u bytes....",kenrel_data_size_align/2);
  free(buffer_fp16);
  }

  if(curr_stage_info.bias_data == true){
    curr_stage_info.bias_shape[0] = (curr_stage_info.bias_shape[0] == 0) ? 1: curr_stage_info.bias_shape[0];
    curr_stage_info.bias_shape[1] = (curr_stage_info.bias_shape[1] == 0) ? 1: curr_stage_info.bias_shape[1];
    curr_stage_info.bias_shape[2] = (curr_stage_info.bias_shape[2] == 0) ? 1: curr_stage_info.bias_shape[2];
    curr_stage_info.bias_shape[3] = (curr_stage_info.bias_shape[3] == 0) ? 1: curr_stage_info.bias_shape[3];

    bias_data_size = dtype_android * curr_stage_info.bias_shape[0] * curr_stage_info.bias_shape[1] *
                                      curr_stage_info.bias_shape[2] * curr_stage_info.bias_shape[3];


    bias_data_size_align = bias_data_size + align_size(bias_data_size,128);

    bias_data_buffer = (float *)malloc(bias_data_size_align);

    if(bias_data_buffer == NULL)
    ALOGE("unable to allocate bias_data_buffer exit");
    if(curr_stage_info.bias_buffer == NULL)
    ALOGE(" curr_stage_info.bias_buffer is null ");


    memset(bias_data_buffer,0,bias_data_size_align);
    memcpy(bias_data_buffer,curr_stage_info.bias_buffer,bias_data_size);

    buffer_fp16 = (half*) malloc(bias_data_size_align/2);
    memset(buffer_fp16,0,bias_data_size_align/2);
    if( buffer_fp16 == NULL)
    ALOGE("unable to allocate buffer_fp16 exit");
    else
    ALOGD("buffer_fp16 allocation success");
    floattofp16((unsigned char *)buffer_fp16, bias_data_buffer, bias_data_size_align/4);

    fp = fopen(graph_filename.c_str(),"ab+");
    if(fp == NULL)
    ALOGE("unable to open ncs_graph file for bias_data writing");
    else
    ALOGD("ncs_graph file is open for bias_data writing");
    fseek(fp,0,SEEK_END);
    fwrite((unsigned char *)buffer_fp16,bias_data_size_align/2,1,fp);
    fclose(fp);

    ALOGD("copied bias_data_buffer %u bytes....",bias_data_size_align/2);
    free(buffer_fp16);
  }

  if(curr_stage_info.op_params_data == true){
    op_params_size_align = 128;
    op_params_buffer = (float *)malloc(op_params_size_align);
    uint32_t index = 0;

    if(op_params_buffer == NULL)
    ALOGE("unable to allocate bias_data_buffer exit");
    if(curr_stage_info.beta == NULL)
    ALOGE(" curr_stage_info.beta is null ");
    float Beta = curr_stage_info.beta;
    memset(op_params_buffer,0,op_params_size_align);

    *(op_params_buffer) = Beta;
    ALOGD("op_params_buffer[0]: %f",*(op_params_buffer));

    buffer_fp16 = (half*) malloc(op_params_size_align/2);
    if( buffer_fp16 == NULL)
    ALOGE("unable to allocate buffer_fp16 exit");
    else
    ALOGD("buffer_fp16 allocation success");
    memset(buffer_fp16,0,op_params_size_align/2);
    *buffer_fp16 = 1;

    fp = fopen(graph_filename.c_str(),"ab+");
    if(fp == NULL)
    ALOGE("unable to open ncs_graph file for op_params_data writing");
    else
    ALOGD("ncs_graph file is open for op_params_data writing");
    fseek(fp,0,SEEK_END);
    fwrite((unsigned char *)buffer_fp16,op_params_size_align/2,1,fp);
    fclose(fp);

    ALOGD("copied op_params_buffer %u bytes....",op_params_size_align/2);
    free(buffer_fp16);
  }

  if(curr_stage_info.kernel_data == true) free(kernel_data_buffer);
  if(curr_stage_info.kernel_data == true) free(kernel_data_buffer_android);
  if(curr_stage_info.bias_data == true) free(bias_data_buffer);
  if(curr_stage_info.op_params_data == true) free(op_params_buffer);

  return true;
}

uint32_t estimate_file_size(bool with_buf_size,uint32_t stage_count){
  Blobconfig blob1;
  Myriadconfig mconfig;
  Blob_Stage_data stage_data;

  //initialize the filesize to "0"
  uint32_t filesize =0;
  //VCS_FIX bug, hence add 32 byes of zeros
  if(VCS_FIX)
   filesize += sizeof(uint64_t) * 4; //VCS_FIX
  //Blob elements size
  filesize += sizeof(blob1.filesize);
  filesize += sizeof(blob1.version);
  filesize += SIZE_OF_NETOWRK_NAME; //TODO fix me later sizeof(blob1.network_name);
  filesize += SIZE_OF_DIR_NAME; //TODO fix me later sizeof(blob1.blob_report_dir);
  filesize += sizeof(blob1.stage_count);
  filesize += sizeof(blob1.filesize);
  //Myriad2 Config elements size
  filesize += sizeof(mconfig.firstShave);
  filesize += sizeof(mconfig.lastShave);
  filesize += sizeof(mconfig.leonMemLocation);
  filesize += sizeof(mconfig.leonMemSize);
  filesize += sizeof(mconfig.dmaAgent);

  //size from each network stage and must be multiplied by total network stages
  //TODO fix the structure size issue
  //filesize += sizeof(stage_data) * blob1.stage_count;
  filesize += STAGE_SIZE * stage_count;
  filesize += align_size(filesize,8); //Should be 8 bytes aligned

  if(with_buf_size){
    filesize+= calculate_data_buffer_size(); //TODO calculate how to find databuf size
  }else
  filesize = filesize;
  return filesize;
}

uint32_t calculate_data_buffer_size(){

  uint32_t data_buf_size=0;
  uint8_t dtype = 2; //FP16 only supported data size in Bytes
  uint8_t dtype_android = 4; //FP32 from Android data size in Bytes

  for(int i=0;i<nwk_vector_stages_info.size();i++){
    Operation_inputs_info curr_stage_info = nwk_vector_stages_info.at(i);
    uint32_t buf_index = 0, kenrel_data_size = 0, bias_data_size = 0;
    uint32_t op_params_size = 0, final_data_size = 0;
    uint32_t kenrel_data_size_align = 0, bias_data_size_align = 0;
    uint32_t op_params_size_align = 0, final_data_size_align = 0;

    if(curr_stage_info.kernel_data == true){
      curr_stage_info.kernel_shape[0] = (curr_stage_info.kernel_shape[0] == 0) ? 1: curr_stage_info.kernel_shape[0];
      curr_stage_info.kernel_shape[1] = (curr_stage_info.kernel_shape[1] == 0) ? 1: curr_stage_info.kernel_shape[1];
      curr_stage_info.kernel_shape[2] = (curr_stage_info.kernel_shape[2] == 0) ? 1: curr_stage_info.kernel_shape[2];
      curr_stage_info.kernel_shape[3] = (curr_stage_info.kernel_shape[3] == 0) ? 1: curr_stage_info.kernel_shape[3];

      kenrel_data_size = dtype * curr_stage_info.kernel_shape[0] * curr_stage_info.kernel_shape[1] *
                                        curr_stage_info.kernel_shape[2] * curr_stage_info.kernel_shape[3];
      kenrel_data_size_align = kenrel_data_size + align_size(kenrel_data_size,64);
    }else{
      kenrel_data_size_align =0;
    }


    if(curr_stage_info.bias_data == true){
      curr_stage_info.bias_shape[0] = (curr_stage_info.bias_shape[0] == 0) ? 1: curr_stage_info.bias_shape[0];
      curr_stage_info.bias_shape[1] = (curr_stage_info.bias_shape[1] == 0) ? 1: curr_stage_info.bias_shape[1];
      curr_stage_info.bias_shape[2] = (curr_stage_info.bias_shape[2] == 0) ? 1: curr_stage_info.bias_shape[2];
      curr_stage_info.bias_shape[3] = (curr_stage_info.bias_shape[3] == 0) ? 1: curr_stage_info.bias_shape[3];

      bias_data_size = dtype * curr_stage_info.bias_shape[0] * curr_stage_info.bias_shape[1] *
                                        curr_stage_info.bias_shape[2] * curr_stage_info.bias_shape[3];
      bias_data_size_align = bias_data_size + align_size(bias_data_size,64);
  }else{
    bias_data_size_align = 0;
  }

    if(curr_stage_info.op_params_data == true){
      op_params_size_align = 64;
    }else{
      op_params_size_align = 0;
    }

    final_data_size_align = kenrel_data_size_align + bias_data_size_align + op_params_size_align;
    data_buf_size += final_data_size_align;
  }
  return data_buf_size;
}


uint32_t align_size(uint32_t fsize, unsigned int align_to){
  if((fsize%align_to)!=0)
    return (align_to - fsize%align_to);
  else
    return 0;
}

void get_header_buffer(char *buf_Herader, Blobconfig blob_config, Myriadconfig mconfig){


  unsigned int index = 0;
  if(buf_Herader == NULL){
    ALOGE("unable to create graph buffer");
  }

  //VCS_FIX bug writes 32 bytes values with zeors
  if(VCS_FIX) {
    for(int i=0;i<32;i++){
      *(buf_Herader+index++) = 0;
    }
  }

  *(buf_Herader+index++) = blob_config.filesize; //blob_config.filesize
  *(buf_Herader+index++) = blob_config.filesize >> 8;
  *(buf_Herader+index++) = blob_config.filesize >> 16;
  *(buf_Herader+index++) = blob_config.filesize >> 24;
  *(buf_Herader+index) = blob_config.version;
  index += sizeof(blob_config.version);
  memset((buf_Herader+index),0,SIZE_OF_NETOWRK_NAME);
  memcpy((buf_Herader+index),blob_config.network_name.c_str(),SIZE_OF_NETOWRK_NAME); //(buf_Herader+index++) = blob_config.network_name;
  index += SIZE_OF_NETOWRK_NAME; // move the index pointer by SIZE_OF_NETOWRK_NAME
  memset((buf_Herader+index),0,SIZE_OF_DIR_NAME);

  index += SIZE_OF_DIR_NAME; // move the index pointer by SIZE_OF_NETOWRK_NAME
  *(buf_Herader+index) = blob_config.stage_count;
  index += sizeof(blob_config.stage_count);
  *(buf_Herader+index++) = blob_config.filesize_without_data;
  *(buf_Herader+index++) = blob_config.filesize_without_data >> 8;
  *(buf_Herader+index++) = blob_config.filesize_without_data >> 16;
  *(buf_Herader+index++) = blob_config.filesize_without_data >> 24;
  //copy Myriad params to buffer
  *(buf_Herader+index) = mconfig.firstShave;
  index += sizeof(mconfig.firstShave);
  *(buf_Herader+index) = mconfig.lastShave;
  index += sizeof(mconfig.lastShave);
  *(buf_Herader+index) = mconfig.leonMemLocation;
  index += sizeof(mconfig.leonMemLocation);
  *(buf_Herader+index) = mconfig.leonMemSize;
  index += sizeof(mconfig.leonMemSize);
  *(buf_Herader+index) = mconfig.dmaAgent;
  index += sizeof(mconfig.dmaAgent);

  if(DEBUG_get_header_buffer){
    ALOGD("blob_config.filesize: %lu", blob_config.filesize);
    ALOGD("blob_config.version: %lu", blob_config.version);
    ALOGD("blob_config.network_name: %s", blob_config.network_name.c_str());
    ALOGD("blob_config.blob_report_dir: %s", blob_config.blob_report_dir.c_str());
    ALOGD("blob_config.stage_count: %lu", blob_config.stage_count);
    ALOGD("blob_config.filesize_without_data: %lu", blob_config.filesize_without_data);

    ALOGD("mconfig.firstShave: %u", mconfig.firstShave);
    ALOGD("mconfig.lastShave: %u", mconfig.lastShave);
    ALOGD("mconfig.leonMemLocation: %lu", mconfig.leonMemLocation);
    ALOGD("mconfig.leonMemSize: %lu", mconfig.leonMemSize);
    ALOGD("mconfig.dmaAgent: %lu", mconfig.dmaAgent);
  }

}


char* generate_graph(char* graph_buf, Blobconfig blob_config, Myriadconfig mconfig){

  unsigned int buf_index = 0;

  network_operations_vector network_operations;
  uint32_t nw_stage_count = blob_config.stage_count;

  network_operations = get_network_operations_details();

  //ALOGD("Netowrk Size: %d",network_operations.size());
  //for(int i=0; i<network_operations.size();i++)
  //ALOGD("Operation number is %d",network_operations.at(i));

  //TODO copy the get_header_buffer data to
  char* buf_Herader ;
  buf_Herader = (char *)malloc(SIZE_HEADER);
  if(buf_Herader == NULL)
  ALOGE("Unable to allocate memory buffer for buf_Herader");

  get_header_buffer(buf_Herader, blob_config,mconfig);
  memcpy(graph_buf,buf_Herader,SIZE_HEADER);
  buf_index += SIZE_HEADER;
  free(buf_Herader);

  //copy the input get_stage_buffer
  //TODO add the code
  char *stage_buffer;

  stage_buffer = (char *)malloc(STAGE_SIZE);
  if(stage_buffer == NULL)
  ALOGE("Unable to allocate memory buffer for stage_buffer");

  memset(stage_buffer,0,STAGE_SIZE);
  get_input_stage_buffer(stage_buffer, STAGE_SIZE,nwk_vector_stages_info.at(0));
  memcpy(graph_buf+buf_index,stage_buffer,STAGE_SIZE);
  buf_index += STAGE_SIZE;
  free(stage_buffer);

  if(network_operations.size() > 1){
    //copy each network stage data
    stage_buffer = (char *)malloc(STAGE_SIZE);
    if(stage_buffer == NULL)
    ALOGE("Unable to allocate memory buffer for stage_buffer");
    memset(stage_buffer,0,STAGE_SIZE);
    get_first_stage_buffer(stage_buffer,network_operations.at(0),STAGE_SIZE,nwk_vector_stages_info.at(0));
    memcpy(graph_buf+buf_index,stage_buffer,STAGE_SIZE);
    buf_index += STAGE_SIZE;
    free(stage_buffer);

    for(int i=1;i<network_operations.size()-1;i++){
      stage_buffer = (char *)malloc(STAGE_SIZE);
      if(stage_buffer == NULL)
      ALOGE("Unable to allocate memory buffer for stage_buffer");
      memset(stage_buffer,0,STAGE_SIZE);
      get_stage_buffer(stage_buffer,network_operations.at(i),STAGE_SIZE,nwk_vector_stages_info.at(i));
      memcpy(graph_buf+buf_index,stage_buffer,STAGE_SIZE);
      buf_index += STAGE_SIZE;
      free(stage_buffer);
    }

    stage_buffer = (char *)malloc(STAGE_SIZE);
    if(stage_buffer == NULL)
    ALOGE("Unable to allocate memory buffer for stage_buffer");
    memset(stage_buffer,0,STAGE_SIZE);
    get_last_stage_buffer(stage_buffer,network_operations.at(network_operations.size()-1),STAGE_SIZE,nwk_vector_stages_info.at(network_operations.size()-1));
    memcpy(graph_buf+buf_index,stage_buffer,STAGE_SIZE);
    buf_index += STAGE_SIZE;
    free(stage_buffer);
  }
  else{
    stage_buffer = (char *)malloc(STAGE_SIZE);
    if(stage_buffer == NULL)
    ALOGE("Unable to allocate memory buffer for stage_buffer");
    memset(stage_buffer,0,STAGE_SIZE);
    get_one_stage_buffer(stage_buffer,network_operations.at(network_operations.size()-1),STAGE_SIZE,nwk_vector_stages_info.at(network_operations.size()-1));
    memcpy(graph_buf+buf_index,stage_buffer,STAGE_SIZE);
    buf_index += STAGE_SIZE;
    free(stage_buffer);
  }


  return graph_buf;
}

bool delete_graphs(){
  int size = graph_file_names_vector.size();
  int perror;
  for(int i=0;i<size;i++){
    perror = std::remove(graph_file_names_vector.at(i).c_str());
    if(perror==0)
    return true;
    else
    return false;
  }
  return true;
}
