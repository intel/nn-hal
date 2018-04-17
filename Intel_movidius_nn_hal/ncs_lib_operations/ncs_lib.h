#ifndef __NCS_H_INCLUDED__
#define __NCS_H_INCLUDED__

#include "myriad.h"
#ifdef __cplusplus
extern "C"
{
#endif


#define LOG_TAG "NCS_LIB"

#include <mvnc.h>
/*
#define SIZE 4
typedef unsigned int VpuShape[SIZE];

typedef enum NCSoperations {
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
                          };
*/
mvncStatus init(int ncs_num);

mvncStatus deinit();

mvncStatus rungraph(NCSoperations op_ncs,
  float *input1_data, VpuShape input1_shape,
  float *input2_data, VpuShape input2_shape,
  float *output_data, VpuShape output_shape);

mvncStatus execute(float *input_data, uint32_t input_num_of_elements,float *output_data, uint32_t output_num_of_elements);


// VPU operations
int run_vpu_op_relu_float32(const float* inputData,int input_SIZE, float* outputData);
int run_vpu_op_tanh_float32(const float* inputData,int input_SIZE, float* outputData);
int run_vpu_op_sigm_float32(const float* inputData,int input_SIZE, float* outputData);

#ifdef __cplusplus
}
#endif

NCSoperations add_Operation_to_Network(NCSoperations nn_operation);
bool parse_ncs_network(network_operations_vector *nn_ncs_network);
#endif
