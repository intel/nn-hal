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

#ifndef ANDROID_ML_NN_VPU_OPERATIONS_H
#define ANDROID_ML_NN_VPU_OPERATIONS_H


#include <stddef.h>

#include <cstdint>
#include <vector>
#include "HalInterfaces.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

struct Shape;


bool reluFloat32(const float* inputData, const Shape& inputShape, float* outputData, const Shape& outputShape);

bool tanhFloat32(const float* inputData, const Shape& inputShape, float* outputData, const Shape& outputShape);

bool logisticFloat32(const float* inputData, const Shape& inputShape, float* outputData, const Shape& outputShape);

// VPU does not support Quant8
//bool reluQuant8(const uint8_t* inputData, const Shape& inputShape, uint8_t* outputData, const Shape& outputShape);

}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif // ANDROID_ML_NN_VPU_OPERATIONS_H
