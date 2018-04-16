#ifndef __NCS_H_INCLUDED__
#define __NCS_H_INCLUDED__

#include "myriad.h"
#ifdef __cplusplus
extern "C"
{
#endif


#define LOG_TAG "VPU_LIB"

#include <mvnc.h>

mvncStatus ncs_init(int ncs_num);

mvncStatus ncs_deinit();

mvncStatus ncs_rungraph(float *input_data, uint32_t input_num_of_elements,
                    float *output_data, uint32_t output_num_of_elements);

int ncs_execute(float *input_data, uint32_t input_num_of_elements,float *output_data, uint32_t output_num_of_elements);

#ifdef __cplusplus
}
#endif

#endif
