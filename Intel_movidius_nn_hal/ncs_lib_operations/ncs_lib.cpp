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
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mvnc.h>
#include <log/log.h>
#include "fp.h"
#include "ncs_lib.h"

// Global Variables
#define NCS_NUM 0
#define NAME_SIZE 100
#define NCS_CHECK_TIMES 5

//TODO Check what is needed

mvncStatus retCode;
void *deviceHandle;
void* graphHandle;
void* graphFileBuf;
bool device_online = false;
bool graph_load = false;
char devName[NAME_SIZE];
unsigned int graphFileLen;


void* resultData16;
void* userParam;
unsigned int lenResultData;

unsigned int lenip1_fp16;

// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;
half *ip1_fp16;
void *LoadgraphFile(const char *path, unsigned int *length);

//----------------------------------- Declaration is done

//graph file loading
void *LoadgraphFile(const char *path, unsigned int *length){

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
  memset(buf, 0, *length);
  if(fread(buf, 1, *length, fp) != *length)
  {
    fclose(fp);
    free(buf);
  }
  fclose(fp);
  return buf;
}

//ncs_init() begin
int ncs_init(){

  if(!device_online){
    retCode = mvncGetDeviceName(NCS_NUM, devName, NAME_SIZE);
    if (retCode != MVNC_OK)
    {   // failed to get device name, maybe none plugged in.
        ALOGE("Error- No NCS Device found ErrorCode: %d",retCode); //printf("Error - No NCS devices found.\n");
        device_online = false;
        return 6;
    }

    retCode = mvncOpenDevice(devName, &deviceHandle);

    if (retCode != MVNC_OK)
    {   // failed to open the device.
        ALOGE("Error - Could not open NCS device ErrorCode: %d",retCode); //printf("Error - Could not open NCS device.\n");
        device_online = false;
        return 6;
    }
    device_online = true;
  }
  return 0;
}
//ncs_init() end


int ncs_load_graph(){

  if(!graph_load){
    char *path;
    path="/data/ncs_graph";
    graphFileBuf = LoadgraphFile(path, &graphFileLen);

    // allocate the graph
    retCode = mvncAllocateGraph(deviceHandle, &graphHandle, graphFileBuf, graphFileLen);
    if (retCode != MVNC_OK){
      ALOGE("Could not allocate graph for file: %d",retCode);
      graph_load = false;
      return 6;
    }
    ALOGD("Graph Allocated successfully!");
    graph_load = true;
  }else{
    ALOGD("Graph already Allocated");
  }
  return 0;
}

mvncStatus ncs_rungraph(float *input_data, uint32_t input_num_of_elements,
                    float *output_data, uint32_t output_num_of_elements)
                    {

                      //convert inputs from fp32 to fp16
                      float *input_data_buffer = (float *)malloc(input_num_of_elements*sizeof(float));
                      if(input_data_buffer==NULL){
                        ALOGE("unable to allocate input_data_buffer");
                        return MVNC_ERROR;
                      }

                      memset(input_data_buffer,0,input_num_of_elements*sizeof(float));
                      memcpy(input_data_buffer,input_data,input_num_of_elements*sizeof(float));

                      //allocate fp16 input1 with inpu1 shape
                      ip1_fp16 = (half*) malloc(sizeof(*ip1_fp16) * input_num_of_elements);
                      ALOGD("Converting input from Float to FP16 Begin");
                      floattofp16((unsigned char *)ip1_fp16, input_data_buffer, input_num_of_elements);
                      ALOGD("Converting input from Float to FP16 end");
                      lenip1_fp16 = input_num_of_elements * sizeof(*ip1_fp16);

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
                      ALOGD("Got the Result");

                      float *output_data_buffer = (float *)malloc(output_num_of_elements*sizeof(float));
                      if(output_data_buffer==NULL){
                        ALOGE("unable to allocate output_data_buffer");
                        return MVNC_ERROR;
                      }

                      memset(output_data_buffer,0,output_num_of_elements*sizeof(float));
                      ALOGD("Converting output from FP16 to Float Begin");
                      fp16tofloat(output_data_buffer, (unsigned char*)resultData16, output_num_of_elements);
                      ALOGD("Converting output from FP16 to Float end");
                      memcpy(output_data,output_data_buffer,output_num_of_elements*sizeof(float));
                      ALOGD("Output data is copied");

                      free(input_data_buffer);
                      free(output_data_buffer);
                      free(ip1_fp16);
                      ALOGD("Error code end of the rungraph is : %d",retCode);

                      return retCode;
                    }


int ncs_execute(float *input_data, uint32_t input_num_of_elements,float *output_data, uint32_t output_num_of_elements){
  retCode = ncs_rungraph(input_data, input_num_of_elements, output_data, output_num_of_elements);
  if (retCode != MVNC_OK){
    ALOGE("NCS unable to executeGraph with ErrorCode: %d",retCode);
    return 6;
  }
  return 0;
}

int ncs_unload_graph(){
  retCode = mvncDeallocateGraph(graphHandle);
  if (retCode != MVNC_OK){
    ALOGE("NCS could not Deallocate Graph %d",retCode);
    return 6;
  }
  ALOGD("Graph Deallocated successfully!");
  free(graphFileBuf);
  graphHandle = NULL;
  graph_load = false;
  return 0;
}

int ncs_deinit(){
  retCode = mvncCloseDevice(deviceHandle);
  if (retCode != MVNC_OK)
  {
      ALOGE("Error - Could not close NCS device ErrorCode: %d",retCode);
      return 6;
  }
  deviceHandle = NULL;
  ALOGD("NCS device closed");
  device_online = false;
  return 0;
}
