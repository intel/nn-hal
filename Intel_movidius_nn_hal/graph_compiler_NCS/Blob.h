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
#ifndef __BLOB_H_INCLUDED__
#define __BLOB_H_INCLUDED__

#include<stdio.h>
#include<string>
#include<iostream>
#include<stdint.h>

#include "myriad.h"

#define STAGE_SIZE 227
#define SIZE_HEADER 264
#define SIZE_OF_STAGE_NAME 100
#define SIZE_OF_NETOWRK_NAME 100
#define SIZE_OF_DIR_NAME 100

#define FIRST_STAGE_DATA_POINTER 0
#define FIRST_STAGE_DATA_INDEX 1

#define LAST_STAGE_OUTPUT_POINTER 0
#define LAST_STAGE_OUTPUT_INDEX 2

#define LOG_TAG "NCS_GRAPH_COMPILER"
#define VCS_FIX true
#define DEBUG_get_header_buffer false
#define DEBUG_get_stage_buffer false
#define DEBUG_generate_graph false
#define DEBUG_get_input_stage_buffer false
#define DEBUG_get_last_stage_buffer false
#define DEBUG_get_one_stage_buffer false
#define DEBUG_get_first_stage_buffer false

typedef unsigned short half;


bool update_post_data_buffer(uint32_t size, float *buf);
bool update_global_buffer_index(uint32_t value);
uint32_t get_global_buffer_index();

bool update_zero_data_offset_g(uint32_t value);
uint32_t get_zero_data_offset_global();

bool update_buffer_index_g(uint16_t value);
uint16_t get_buffer_index_global();

bool update_data_Pointer_g(uint32_t value);
uint32_t get_data_Pointer_global();


bool update_data_Index_g(uint16_t value);
uint16_t get_data_Index_global();

bool update_taps_Pointer_g(uint32_t value);

uint32_t get_taps_Pointer_global();

bool update_taps_Index_g(uint16_t value);

uint16_t get_taps_Index_global();

bool update_bias_Pointer_g(uint32_t value);

uint32_t get_bias_Pointer_global();

bool update_bias_Index_g(uint16_t value);

uint16_t get_bias_Index_global();

bool update_opPrarams_Pointer_g(uint32_t value);

uint32_t get_opPrarams_Pointer_global();

bool update_opPrarams_Index_g(uint16_t value);

uint16_t get_opPrarams_Index_global();

bool update_output_Pointer_g(uint32_t value);

uint32_t get_output_Pointer_global();

bool update_output_Index_g(uint16_t value);

uint16_t get_output_Index_global();

uint32_t estimate_file_size(bool with_buf_size,uint32_t stage_count);
uint32_t align_size(uint32_t fsize, unsigned int align_to);

bool prepare_blob(std::string str, int graph_count);//TODO update required

char* generate_graph(char *buf, Blobconfig blob_config, Myriadconfig mconfig);

bool wrtie_post_stage_data(Blobconfig blob_config, Myriadconfig mconfig);


void get_header_buffer(char *buf_Herader, Blobconfig blob_config, Myriadconfig mconfig);

std::vector<NCSoperations> get_network_operations_details();

void get_input_stage_buffer(char *stage_buffer, NCSoperations curr_operation, unsigned int stage_size, Operation_inputs_info curr_stage_info);
void get_stage_buffer(char *stage_buffer, NCSoperations curr_operation, unsigned int stage_size, Operation_inputs_info curr_stage_info);
void get_last_stage_buffer(char *stage_buffer, NCSoperations curr_operation, unsigned int stage_size, Operation_inputs_info curr_stage_info);
void get_one_stage_buffer(char *stage_buffer, NCSoperations curr_operation, unsigned int stage_size, Operation_inputs_info curr_stage_info);
void get_kernel_bias_data_buffer(half * buffer_fp16, Operation_inputs_info curr_stage_info,uint32_t *data_size_location);
bool write_kernel_bias_data_buffer_to_file(Operation_inputs_info curr_stage_info);

uint32_t calculate_output_pointer(uint32_t X, uint32_t Y, uint32_t Z);
uint32_t calculate_taps_pointer(uint32_t X, uint32_t Y, uint32_t Z, uint32_t W);
uint32_t calculate_bias_Pointer(uint32_t X);
uint32_t calculate_data_buffer_size();

Blob_Stage_data get_input_stage_layer(Operation_inputs_info curr_stage_info);

Blob_Stage_data get_LOGISTIC_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_TANH_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_RELU_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_RELU1_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_RELU6_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_CONV_2D_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_DEPTHWISE_CONV_2D_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_AVG_POOL_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_MAX_POOL_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_Softmax_stage_data(Operation_inputs_info curr_stage_info);
Blob_Stage_data get_Reshape_stage_data(Operation_inputs_info curr_stage_info);


bool parse_logistic_from_android(Operation_inputs_info sig_stage_android);
bool delete_graphs();

Operation_inputs_info parse_logistic_stage_info();
Operation_inputs_info parse_tanh_stage_info();
Operation_inputs_info parse_relu_stage_info();
Operation_inputs_info parse_relu1_stage_info();
Operation_inputs_info parse_relu6_stage_info();

Operation_inputs_info parse_input_stage_info();

bool get_nn_network_from_android(network_operations_vector nw_vector1);
bool parse_stage_from_android(Operation_inputs_info cur_stage_android);
#endif
