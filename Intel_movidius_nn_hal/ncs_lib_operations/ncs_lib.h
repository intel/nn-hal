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
#ifndef __NCS_H_INCLUDED__
#define __NCS_H_INCLUDED__

#include "myriad.h"
#ifdef __cplusplus
extern "C"
{
#endif


#define LOG_TAG "VPU_LIB"

#include <mvnc.h>

/*
NO ERROR - 0
INCOMPLETE - 2
BAD STATE - 6
*/
int ncs_register();
int ncs_deregister();

int ncs_init();

int ncs_deinit();

int ncs_load_graph();

int ncs_unload_graph();

//void ncs_reset();

mvncStatus ncs_rungraph(float *input_data, uint32_t input_num_of_elements,
                    float *output_data, uint32_t output_num_of_elements);

int ncs_execute(float *input_data, uint32_t input_num_of_elements,float *output_data, uint32_t output_num_of_elements);

#ifdef __cplusplus
}
#endif

#endif
