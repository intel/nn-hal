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

#ifndef ANDROID_ML_NN_VPU_OPERATIONS_UTILS_H
#define ANDROID_ML_NN_VPU_OPERATIONS_UTILS_H

#include "ActivationFunctor.h"
#include "VpuUtils.h"

#include <cstdint>
#include <vector>

// Macro to check if the input parameters for operation are valid or not.
#define NN_CHECK(v)                                                     \
  do {                                                                  \
    if (!(v)) {                                                         \
      LOG(ERROR) << "NN_CHECK failed: "  << #v << "'\n";                \
      return false;                                                     \
    }                                                                   \
  } while(0);

#define NN_CHECK_EQ(actual, expected)           \
  NN_CHECK((actual) == (expected))

#define NN_OPS_CHECK NN_CHECK

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {


enum PaddingScheme {
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
};

// The type and dimensions of an operand.
struct Shape {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t offset;
};

// Verifies that the two shapes are the same.
bool SameShape(const Shape& in1, const Shape& in2);

// Sets out to the same shape as in.
bool SetShape(const Shape& in, Shape* out);

// Return the total number of elements, i.e. all the dimensions multiplied
// together. For a scalar, returns one.
uint32_t getNumberOfElements(const Shape& shape);

uint32_t getNumberOfDimensions(const Shape& shape);

uint32_t getSizeOfDimension(const Shape& shape, uint32_t dimensionIdx);

inline uint32_t computeOutSize(uint32_t imageSize, uint32_t filterSize, uint32_t stride,
                               uint32_t paddingHead, uint32_t paddingTail) {
    return (imageSize - filterSize + stride + paddingHead + paddingTail) / stride;
}

/*
__wur
bool QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int32_t* right_shift);

__wur
bool QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

__wur
bool GetQuantizedConvolutionMultipler(const Shape& inputShape,
                                      const Shape& filterShape,
                                      const Shape& biasShape,
                                      const Shape& outputShape,
                                      float* multiplier);
*/
void CalculateActivationRangeUint8(int32_t activation,
                                   const Shape& outputShape,
                                   int32_t* act_min,
                                   int32_t* act_max);

/*
int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift);
*/
inline void calculateExplicitPadding(int32_t in_size, int32_t stride,
                                     int32_t filter_size, int32_t padding_implicit,
                                     int32_t* padding_head, int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;

    if (padding_implicit == kPaddingSame) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

inline PaddingScheme getPaddingScheme(int32_t inWidth, int32_t inHeight,
                                      int32_t strideWidth, int32_t strideHeight,
                                      int32_t filterWidth, int32_t filterHeight,
                                      int32_t paddingLeft, int32_t paddingRight,
                                      int32_t paddingTop, int32_t paddingBottom) {
    if (paddingLeft == 0 && paddingRight == 0 && paddingTop == 0 && paddingBottom == 0) {
        return kPaddingValid;
    }

    int32_t expectedPaddingLeft, expectedPaddingRight;
    int32_t expectedPaddingTop, expectedPaddingBottom;

    calculateExplicitPadding(inWidth, strideWidth, filterWidth, kPaddingSame,
                             &expectedPaddingLeft, &expectedPaddingRight);
    calculateExplicitPadding(inHeight, strideHeight, filterHeight, kPaddingSame,
                             &expectedPaddingTop, &expectedPaddingBottom);
    if (expectedPaddingLeft == paddingLeft && expectedPaddingRight == paddingRight &&
        expectedPaddingTop == paddingTop && expectedPaddingBottom == paddingBottom) {
        return kPaddingSame;
    } else {
        return kPaddingUnknown;
    }
}

// Preparation functions for the corresponding ops
/*
bool addMulPrepare(const Shape& in1, const Shape& in2, Shape* out1);

bool floorPrepare(const Shape& input, Shape* output);

bool dequantizePrepare(const Shape& input, Shape* output);
*/
bool depthwiseConvPrepare(const Shape& input,
                          const Shape& filter,
                          const Shape& bias,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          Shape* output);

bool convPrepare(const Shape& input,
                 const Shape& filter,
                 const Shape& bias,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
                 Shape* output);

bool genericPoolingPrepare(const Shape& input,
                           int32_t padding_left, int32_t padding_right,
                           int32_t padding_top, int32_t padding_bottom,
                           int32_t stride_width, int32_t stride_height,
                           int32_t filter_width, int32_t filter_height,
                           Shape* output);

bool genericActivationPrepare(const Shape& input, Shape* output);
/*
bool fullyConnectedPrepare(const Shape& input,
                           const Shape& weights,
                           const Shape& bias,
                           Shape* output);

bool concatenationPrepare(const std::vector<Shape>& inputShapes,
                          int32_t axis,
                          Shape* output);

bool genericNormalizationPrepare(const Shape& input, Shape* output);
*/
bool reshapePrepare(const Shape& input,
                    const int32_t* targetDims,
                    const int32_t targetDimsSize,
                    Shape* output);
/*
bool resizeBilinearPrepare(const Shape& input,
                           int32_t height,
                           int32_t width,
                           Shape* output);

bool depthToSpacePrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output);

bool spaceToDepthPrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output);

bool embeddingLookupPrepare(const Shape &valueShape,
                            const Shape &lookupShape,
                            Shape *outputShape);

bool hashtableLookupPrepare(const Shape &lookupShape,
                            const Shape &keyShape,
                            const Shape &valueShape,
                            Shape *outputShape,
                            Shape *hitShape);
*/
#define ANDROID_NN_MACRO_DISPATCH(macro)                                    \
    switch (activation) {                                                   \
        case (int32_t) FusedActivationFunc::NONE:                           \
            macro(kNone);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU:                           \
            macro(kRelu);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU1:                          \
            macro(kRelu1);                                                  \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU6:                          \
            macro(kRelu6);                                                  \
            break;                                                          \
        default:                                                            \
            LOG(ERROR) << "Unsupported fused activation function type";     \
            return false;                                                   \
    }






}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif // ANDROID_ML_NN_VPU_OPERATIONS_UTILS_H
