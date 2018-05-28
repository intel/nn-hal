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
#ifndef __MYRIAD_H_INCLUDED__
#define __MYRIAD_H_INCLUDED__

#include <stdio.h>
#include <string>
#include <iostream>
#include <stdint.h>
#include <vector>

#define SIZE 4

typedef unsigned int VpuShape[SIZE];

typedef enum ncsoperations {
                            NONE,
                            INPUT,
                            OEM_OPERATION,
                            ADD,
                            MUL,
                            FLOOR,
                            DEQUANTIZE,
                            DEPTHWISE_CONV_2D,
                            CONV_2D,
                            AVERAGE_POOL_2D,
                            L2_POOL_2D,
                            MAX_POOL_2D,
                            RELU,
                            RELU1,
                            RELU6,
                            TANH,
                            LOGISTIC,
                            SOFTMAX,
                            FULLY_CONNECTED,
                            CONCATENATION,
                            L2_NORMALIZATION,
                            LOCAL_RESPONSE_NORMALIZATION,
                            RESHAPE,
                            RESIZE_BILINEAR,
                            DEPTH_TO_SPACE,
                            SPACE_TO_DEPTH,
                            EMBEDDING_LOOKUP,
                            HASHTABLE_LOOKUP,
                            LSH_PROJECTION,
                            LSTM,
                            RNN,
                            SVDF,
                          } NCSoperations;

typedef struct operation_inputs_info {
  NCSoperations main_operation;
  unsigned int num_inputs;
  VpuShape input_shape;
  VpuShape kernel_shape;
  const float *kernel_buffer;
  VpuShape bias_shape;
  const float *bias_buffer;
  VpuShape output_shape;
  int32_t padding_left;
  int32_t padding_right;
  int32_t padding_top;
  int32_t padding_bottom;
  int32_t stride_width;
  int32_t stride_height;
  int32_t depth_multiplier;
  float beta;
  bool kernel_data = false;
  bool bias_data = false;
  bool op_params_data = false;
  NCSoperations post_operation; //it is used for activation functions
}Operation_inputs_info;

typedef std::vector<Operation_inputs_info> Network_Vector_Stageinfo;

typedef std::vector<NCSoperations> network_operations_vector;

typedef struct myriadconfig {
			uint16_t firstShave;
			uint16_t lastShave; //Total NCS Myriad2 Shave number is 12 (0-11)
			uint32_t leonMemLocation;
			uint32_t leonMemSize;
			uint32_t dmaAgent;
			std::string optimization_list;
	} Myriadconfig;

//typedef enum ncs_errors {NCS_GRAPH_ERROR,   };

enum operation_type {

  	Convolution = 0,
  	MaxPooling = 1,
  	Average_Pooling = 2,

  	Softmax = 3,
  	Fully_Connected_Layer = 4,
  	None = 5,

  	ReLU = 6,
  	ReLU_X = 7,
  	DepthWise_Convolution = 8,

  	Bias = 9,
  	PReLU = 10,
  	LRN = 11,

  	Elementwise_Sum = 12,
  	Elementwise_Product = 13,
  	Elementwise_Max = 14,

  	Scale = 15,
  	RelayOut = 16,
  	Square = 17,

  	Inner_LRN = 18,
  	Copy = 19,
  	Sigmoid = 20,

  	TanH = 21,
  	Deconvolution = 22,
  	ELU = 23,

  	Reshape = 24,
    //TODO update the operations list as per the Models/EnumDeclarations.py
};

enum VpuDataType {  //VPU Data type to handle the FP16, FP32, etc...
		U8,
		FP16,
		FP32
	};

enum DataStorageOrder{
			order_YXZ = 1,
			order_ZYX = 2,
			order_YZX = 3,
			order_XYZ = 4,
			order_XZY = 5
			};

enum tapsDataOrder{ //TODO correct the variable names

					kchw = 0,
					hwck = 1,
					};

enum padstyle{

				none = 0,
				tfvalid = 1,
				caffe = 2,
				tfsame = 3,
};

typedef struct blobconfig {
		uint32_t version;
		uint32_t filesize;
		uint32_t filesize_without_data;
		std::string network_name;
		std::string blob_report_dir;
		uint32_t stage_count;
		Myriadconfig myriad_params;
}Blobconfig;

typedef struct blob_stage_data {
    std::string stage_name;
	  unsigned char op_val;
	  uint32_t opt_mask;

	  uint8_t radixX;
	  uint8_t radixY;

	  uint8_t strideX;
	  uint8_t strideY;

	  uint8_t padX;
	  uint8_t padY;
	  uint8_t padStyle_value;

	  uint32_t inputDimX;
	  uint32_t inputDimY;
	  uint32_t inputDimZ;

	  uint32_t tapDimX;
	  uint32_t tapDimY;
	  uint32_t tapDimZ;

	  uint32_t outputDimX;
	  uint32_t outputDimY;
	  uint32_t outputDimZ;

	  uint32_t inputStrideX;
	  uint32_t inputStrideY;
	  uint32_t inputStrideZ;

	  uint32_t tapStrideX;
	  uint32_t tapStrideY;
	  uint32_t tapStrideZ;

	  uint32_t outputStrideX;
	  uint32_t outputStrideY;
	  uint32_t outputStrideZ;

	  uint8_t datatype_value;
	  uint8_t precision_value;
	  uint8_t storageOrder_value;

	  uint32_t data_Pointer;
	  uint16_t data_Index;

	  uint32_t taps_Pointer;
	  uint16_t taps_Index;

	  uint32_t bias_Pointer;
	  uint16_t bias_Index;

	  uint32_t opPrarams_Pointer;
	  uint16_t opPrarams_Index;

	  uint32_t output_Pointer;
	  uint16_t output_Index;

	  uint8_t preOp_value;
	  uint8_t postOp_value;

	  //float *post_param1;
    //unsigned int post_param1;
    char post_param1[4];

	  unsigned short post_strideX;
	  unsigned short post_strideY;

}Blob_Stage_data;

/*
typedef struct nw_stage_data {
	  string layer_name;
	  enum DataStorageOrder s_order; //s_order;
	  uint32_t pad_y;
	  uint32_t pad_x;
	  enum padstyle pad_type;
	  enum VpuDataType precision;
	  enum operation_type op_type;
	  uint32_t op_y;
	  uint32_t op_x;
	  uint32_t sy;
	  uint32_t sx;
	  uint32_t x;
	  uint32_t y;
	  uint32_t c;
	  uint32_t fh;
	  uint32_t fw;
	  uint32_t k;
	  uint32_t* taps;
	  enum tapsDataOrder taps_order;
	  uint32_t* bias;
	  enum operation_type pre_op_type; // TODO change the type from string to appropriate
	  enum operation_type post_op_type; // TODO change the type from string to appropriate
	  uint32_t post_1;
	  uint32_t post_sx;
	  uint32_t post_sy;
	  string slicing;
	  Myriadconfig ns_myriad_config;
	  int opParams;
	  uint32_t new_x;
	  uint32_t new_y;
	  uint32_t new_c;
} Nw_Stage_data;*/

#endif
