/*
 * Copyright (C) 2018 The Android Open Source Project
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
#define LOG_TAG "VpuActivation"
#include <log/log.h>
#include "VpuOperations.h"
#include "VpuOperationsUtils.h"
#include "ncs_lib.h"

//#include "internal/optimized/optimized_ops.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

  bool reluFloat32(const float* inputData, const Shape& inputShape,
                   float* outputData, const Shape& outputShape) {
      int numElements = getNumberOfElements(inputShape);
      //int val = run_vpu_op_relu_float32(inputData,getNumberOfElements(inputShape),outputData);
      /*
      for (int i=0; i<std::max(numElements,numElements_out); i++, inputData++, outputData++) {
          *outputData = std::max(0.f, *inputData);
      }
      */
      return true;
  }

  bool tanhFloat32(const float* inputData, const Shape& inputShape,
                   float* outputData, const Shape& outputShape) {
      int numElements = getNumberOfElements(inputShape);
      //int val = run_vpu_op_tanh_float32(inputData,getNumberOfElements(inputShape),outputData);
      return true;
  }

  bool logisticFloat32(const float* inputData, const Shape& inputShape,
                   float* outputData, const Shape& outputShape) {
      int numElements = getNumberOfElements(inputShape);
      //int val = run_vpu_op_sigm_float32(inputData,getNumberOfElements(inputShape),outputData);
      return true;
  }

/*
  #define ANDROID_NN_RELUX_QUANT8(activation)                           \
    int numElements = getNumberOfElements(inputShape);                  \
    int numElements_out = getNumberOfElements(outputShape);             \
    int32_t output_activation_min = 0;                                  \
    int32_t output_activation_max = 0;                                  \
                                                                        \
    CalculateActivationRangeUint8(activation, inputShape,               \
                                  &output_activation_min,               \
                                  &output_activation_max);              \
                                                                        \
    for (int i=0; i<std::max(numElements,numElements_out); i++, inputData++, outputData++) {      \
        *outputData = std::min((uint8_t)output_activation_max,          \
                std::max((uint8_t)output_activation_min, *inputData));  \
    }


/*bool reluQuant8(const uint8_t* inputData, const Shape& inputShape,
                uint8_t* outputData, const Shape& outputShape) {
    ANDROID_NN_RELUX_QUANT8(kActivationRelu)
    return true;
}*/

}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
