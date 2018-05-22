//
// INTEL CONFIDENTIAL
// Copyright 2017-2018 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#pragma once

#include <cstdint>

#ifdef _MSC_VER
#  define PACKED(name) __pragma(pack(push, 1)) struct name __pragma(pack(pop))
#elif defined(__GNUC__)
#  define PACKED(name) struct __attribute__((packed)) name
#endif

#ifdef AKS
#define PACKED(name) struct __attribute__((packed)) name
#endif

namespace VPU {

const uint32_t DATA_ALIGNMENT = 64u;
const uint32_t WEIGHTS_ALIGNMENT = 16u;

const uint32_t MV_TENSOR_DEFAULT_OPT = 0x80000000u;

enum t_MvTensorOpType {
    kConv,
    kMaxPool,
    kAvgPool,
    kSoftMax,
    kFC,
    kNone0,
    kRelu,
    kReluX,
    kDepthConv,
    kBias,
    kPRelu,
    kLRN,
    kSum,
    kProd,
    kMax,
    kScale,
    kRelayout,
    kSquare,
    kInnerLRN,
    kCopy,
    kSigmoid,
    kTanh,
    kDeconvolution,
    kElu,
    kReshape,
    kToPlaneMajor,
    kPower,
    kCrop,
    kTile,
    kRegionYolo,
    kReorgYolo,
    kConvert_u8f16,
    kConvert_f32f16,
    kConvert_f16f32,
    kPermute,
    kNormalize,
    kPriorBox,
    kDetectionOutput,
    kMyriadXHwConvolution,
    kMyriadXHwPooling,
    kMyriadXHwFCL,
    kMyriadXHwPostOps,
    kConvertOrder,
    kCTCDecoder,
    kLeakyRelu,
    kBiasRelu,
    kBiasLeakyRelu,
    kScaleShift,
    kCopyMakeBorderCHW,
    kIm2ColConvolution,
    kCHWBiasRelu,
    kCHWBiasLeakyRelu,
    kCHWBias,
    kCHWScale,
    kCHWScaleShift,
    kCHWPower,
    kHwFcRelayout,

    OP_TYPE_COUNT_
};

enum t_MvTensorStorageOrder{
    orderYXZ,
    orderZYX,
    orderYZX,
    orderXYZ,
    orderXZY
};

enum t_MvTensorPaddStyle {
    paddStyleNone,
    paddStyleTFValid,
    paddStyleCaffe,
    paddStyleTFSame
};

enum t_MvTensorOptimization {
    opt_conv_3_3_1_1_specific,
    opt_conv_im2col,
    opt_conv_im2col_v2,
    opt_conv_3_3_2_2_specific,
    opt_conv_5_5_1_1_specific,
    opt_conv_5_5_2_2_specific,
    opt_conv_7_7_2_2_specific,
    opt_maxpool_2_2_2_2_specific,
    opt_maxpool_3_3_1_1_specific,
    opt_conv_7_7_2_2_spatial,
    opt_conv_generic_spatial,
    opt_deconv_3_3_1_1_same_specific,
    opt_deconv_5_5_1_1_same_specific,
    opt_deconv_M_N_1_1_same_spatial,
    opt_deconv_M_N_1_1_same_generic,
    opt_power_accurate,
    opt_deconv_general,
    opt_MAXIMUM_NAME_SIZE = 50,
    opt_MAXIMUM_OPTIMIZATIONS = 40,
};

enum IndexCodes {
    IndexNone = 0,
    IndexInput = 1,
    IndexOutput = 2,
    IndexBlob = 3,
    IndexBSS = 4,
    IndexCMX = 5,
};

}  // namespace VPU
