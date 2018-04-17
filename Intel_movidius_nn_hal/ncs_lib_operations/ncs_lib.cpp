// Copyright 2018 Intel Corporation.
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title to such
// Material remains with Intel Corporation or its suppliers or licensors.
// The Material contains proprietary information of Intel or its suppliers and
// licensors. The Material is protected by worldwide copyright laws and treaty
// provisions.
// No part of the Material may be used, copied, reproduced, modified, published,
// uploaded, posted, transmitted, distributed or disclosed in any way without
// Intel's prior express written permission. No license under any patent,
// copyright or other intellectual property rights in the Material is granted to
// or conferred upon you, either expressly, by implication, inducement, estoppel
// or otherwise.
// Any license under such intellectual property rights must be express and
// approved by Intel in writing.



#define LOG_TAG "NCS_LIB"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mvnc.h>
#include <log/log.h>
#include "fp.h"
#include "ncs_lib.h"

// Global Variables
#define NCS_NUM 1
#define NAME_SIZE 100

/*
This is temporary fix. TODO Modify later
graph buf address locations for input layer offset

oneD_ip_layer_size_ip_offset

oneD - 1D arrary as input
ip_layer - input layer in graph
size_ip_offset - input size offset location

*/
#define oneD_ip_layer_size_ip_offset 376
#define oneD_ip_layer_size_op_offset 400
#define oneD_op1_layer_size_ip_offset 603
#define oneD_op1_layer_size_op_offset 627

mvncStatus retCode;
void *deviceHandle;
bool device_online = false;
char devName[100];
unsigned int graphFileLen;
void* graphHandle;

void* resultData16;
void* userParam;
unsigned int lenResultData;

unsigned int lenip1_fp16, lenip2_fp16;

// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;

half *ip1_fp16;
half *ip2_fp16;
//tensor dimention of 4 only Supported
mvncStatus rungraph(NCSoperations op_ncs,
  float *input1_data, VpuShape input1_shape,
  float *input2_data, VpuShape input2_shape,
  float *output_data, VpuShape output_shape);
char *postprocess_graphbuf(char *buf,NCSoperations op_ncs,unsigned ip1_size){
  switch (op_ncs) {
    case LOGISTIC:{
      *(buf+oneD_ip_layer_size_ip_offset) = ip1_size;
      *(buf+oneD_ip_layer_size_op_offset) = ip1_size;
      *(buf+oneD_op1_layer_size_ip_offset) = ip1_size;
      *(buf+oneD_op1_layer_size_op_offset) = ip1_size;
    } break;
    case TANH:{
      *(buf+oneD_ip_layer_size_ip_offset) = ip1_size;
      *(buf+oneD_ip_layer_size_op_offset) = ip1_size;
      *(buf+oneD_op1_layer_size_ip_offset) = ip1_size;
      *(buf+oneD_op1_layer_size_op_offset) = ip1_size;
    } break;
    case RELU:{
      *(buf+oneD_ip_layer_size_ip_offset) = ip1_size;
      *(buf+oneD_ip_layer_size_op_offset) = ip1_size;
      *(buf+oneD_op1_layer_size_ip_offset) = ip1_size;
      *(buf+oneD_op1_layer_size_op_offset) = ip1_size;
    } break;
  }
  return buf;
}

//graph file loading
void *LoadgraphFile(const char *path, unsigned int *length,NCSoperations op_ncs,unsigned ip1_size){

  FILE *fp;
  char *buf;
  fp = fopen(path, "rb");
  if(fp == NULL)
  {
    ALOGE("unable to open graph file");
  }
  fseek(fp, 0, SEEK_END);
  *length = ftell(fp);
  rewind(fp);
  if(!(buf = (char*) malloc(*length)))
  {
    ALOGE("unable to create graph buffer");
  }
  if(fread(buf, 1, *length, fp) != *length)
  {
    fclose(fp);
    free(buf);
  }
  fclose(fp);
  buf = postprocess_graphbuf(buf, op_ncs,ip1_size);//TODO add arguements
  return buf;
}

//mvncStatus init(int ncs_num);

mvncStatus init(int ncs_num){

  if(!device_online){
    retCode = mvncGetDeviceName(ncs_num-1, devName, NAME_SIZE);

    if (retCode != MVNC_OK)
    {   // failed to get device name, maybe none plugged in.
        ALOGE("Error- No NCS Device found ErrorCode: %d",retCode); //printf("Error - No NCS devices found.\n");
        device_online = false;
        return retCode;
    }else{
      ALOGE("NCS Device found with ErrorCode: %d",retCode);
      device_online = true;
    }

    retCode = mvncOpenDevice(devName, &deviceHandle);

    if (retCode != MVNC_OK)
    {   // failed to open the device.
        ALOGE("Error - Could not open NCS device ErrorCode: %d",retCode); //printf("Error - Could not open NCS device.\n");
        device_online = false;
        return retCode;
    }
    device_online = true;
  }
  return retCode;
}

//mvncStatus deinit();

mvncStatus deinit(){
  retCode = mvncCloseDevice(deviceHandle);
  deviceHandle = NULL;
  if (retCode != MVNC_OK)
  {
      ALOGE("Error - Could not close NCS device ErrorCode: %d",retCode);
  }
  return retCode;
}//end of deinit


/*
mvncStatus rungraph(const char *path,
  float *input1_data, Shape input1_shape,
  float *input2_data, Shape input2_shape,
  float *output_data, Shape output_shape)

*/

mvncStatus rungraph(NCSoperations op_ncs,
  float *input1_data, VpuShape input1_shape,
  float *input2_data, VpuShape input2_shape,
  float *output_data, VpuShape output_shape){

    char *path;
    unsigned nele1 = input1_shape[0] * input1_shape[1] * input1_shape[2] * input1_shape[3];
    unsigned nele2 = input2_shape[0] * input2_shape[1] * input2_shape[2] * input2_shape[3];

    switch (op_ncs) {
      case RELU: path="/system/vendor/firmware/mvnc/graph_relu";break;
      case LOGISTIC: path="/system/vendor/firmware/mvnc/graph_sigm";break;
      case TANH: path="/system/vendor/firmware/mvnc/graph_tanh";break;
    }

    // Now read in a graph file


    void* graphFileBuf = LoadgraphFile(path, &graphFileLen,op_ncs,nele1);

    // allocate the graph
    retCode = mvncAllocateGraph(deviceHandle, &graphHandle, graphFileBuf, graphFileLen);
    if (retCode != MVNC_OK){
      ALOGE("Could not allocate graph for file: %d",retCode);
      return retCode;
    }

    //convert inputs from fp32 to fp16

    //allocate fp16 input1 with inpu1 shape
    ip1_fp16 = (half*) malloc(sizeof(*ip1_fp16) * nele1);

    //allocate fp16 input1 with inpu2 shape
    ip2_fp16 = (half*) malloc(sizeof(*ip2_fp16) * nele2);

    floattofp16((unsigned char *)ip1_fp16, input1_data, nele1);
    floattofp16((unsigned char *)ip2_fp16, input2_data, nele2);

    lenip1_fp16 = nele1 * sizeof(*ip1_fp16);
    lenip2_fp16 = nele2 * sizeof(*ip2_fp16);

    // start the inference with mvncLoadTensor()
    retCode = mvncLoadTensor(graphHandle, ip1_fp16, lenip1_fp16, NULL);
    if (retCode != MVNC_OK){
      ALOGE("Could not LoadTensor into NCS: %d",retCode);
      return retCode;
    }
    ALOGD("Input Tensor Loaded successfully!");
    retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
    if (retCode != MVNC_OK){
      ALOGE("NCS could not return result %d",retCode);
      return retCode;
    }

    int numResults = lenResultData / sizeof(half);

    fp16tofloat(output_data, (unsigned char*)resultData16, numResults);


    free(graphFileBuf);
    free(ip1_fp16);
    free(ip2_fp16);
    ALOGD("Error code end of the rungraph is : %d",retCode);
    return retCode;
  }//end of rungraph


int run_vpu_op_relu_float32(const float* inputData,int input_SIZE, float* outputData){
  mvncStatus status;
  VpuShape ip1_s;
  VpuShape ip2_s;
  VpuShape op_s;
  NCSoperations cur_op = RELU; //TODO change to RELU later

  ip1_s[0]= 1;
  ip1_s[1]= 1;
  ip1_s[2]= 1;
  ip1_s[3]= input_SIZE;

  ip2_s[0]= 1;
  ip2_s[1]= 1;
  ip2_s[2]= 1;
  ip2_s[3]= 1;

  op_s[0]= 1;
  op_s[1]= 1;
  op_s[2]= 1;
  op_s[3]= input_SIZE;


  if (init(NCS_NUM) != MVNC_OK){
    ALOGE("Device Unavilable");
  }
  ALOGD("Device avilable");

  status = rungraph(cur_op, (float*) inputData, ip1_s, (float*)inputData, ip2_s, outputData, op_s);

  if(status != MVNC_OK)
  {
  ALOGE("Error- rungraph ErrorCode: %d",status);
  exit(-1);
  }

  if (deinit() != MVNC_OK){
    ALOGE("Device not Closed properly");
    exit(-1);
  }
  ALOGD("Device Closed properly");
  return 0;
}

int run_vpu_op_tanh_float32(const float* inputData,int input_SIZE, float* outputData){
  mvncStatus status;
  VpuShape ip1_s;
  VpuShape ip2_s;
  VpuShape op_s;
  NCSoperations cur_op = TANH; //TODO change to RELU later

  ip1_s[0]= 1;
  ip1_s[1]= 1;
  ip1_s[2]= 1;
  ip1_s[3]= input_SIZE;

  ip2_s[0]= 1;
  ip2_s[1]= 1;
  ip2_s[2]= 1;
  ip2_s[3]= 1;

  op_s[0]= 1;
  op_s[1]= 1;
  op_s[2]= 1;
  op_s[3]= input_SIZE;


  if (init(NCS_NUM) != MVNC_OK){
    ALOGE("Device Unavilable");
  }
  ALOGD("Device avilable");

  status = rungraph(cur_op, (float*) inputData, ip1_s, (float*)inputData, ip2_s, outputData, op_s);

  if(status != MVNC_OK)
  {
  ALOGE("Error- rungraph ErrorCode: %d",status);
  exit(-1);
  }

  if (deinit() != MVNC_OK){
    ALOGE("Device not Closed properly");
    exit(-1);
  }
  ALOGD("Device Closed properly");
  return 0;
}

int run_vpu_op_sigm_float32(const float* inputData,int input_SIZE, float* outputData){
  mvncStatus status;
  VpuShape ip1_s;
  VpuShape ip2_s;
  VpuShape op_s;
  NCSoperations cur_op = LOGISTIC; //TODO change to RELU later

  ip1_s[0]= 1;
  ip1_s[1]= 1;
  ip1_s[2]= 1;
  ip1_s[3]= input_SIZE;

  ip2_s[0]= 1;
  ip2_s[1]= 1;
  ip2_s[2]= 1;
  ip2_s[3]= 1;

  op_s[0]= 1;
  op_s[1]= 1;
  op_s[2]= 1;
  op_s[3]= input_SIZE;


  if (init(NCS_NUM) != MVNC_OK){
    ALOGE("Device Unavilable");
  }
  ALOGD("Device avilable");

  status = rungraph(cur_op, (float*) inputData, ip1_s, (float*)inputData, ip2_s, outputData, op_s);

  if(status != MVNC_OK)
  {
  ALOGE("Error- rungraph ErrorCode: %d",status);
  exit(-1);
  }

  if (deinit() != MVNC_OK){
    ALOGE("Device not Closed properly");
    exit(-1);
  }
  ALOGD("Device Closed properly");
  return 0;
}
