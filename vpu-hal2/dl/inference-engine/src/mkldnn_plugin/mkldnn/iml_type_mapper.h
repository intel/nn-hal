//
// INTEL CONFIDENTIAL
// Copyright 2017 Intel Corporation.
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

#include <string>

namespace MKLDNNPlugin {

enum impl_desc_type {
    unknown = 0x00000000,
    undef,
    // Optimization approach
    ref    = 1<<7,
    jit    = 1<<8,
    gemm   = 1<<9,
    // CPU version
    sse42  = 1<<10,
    avx2   = 1<<11,
    avx512 = 1<<12,
    blas   = 1<<13,
    any    = 1<<14,
    // Other specificator
    _1x1    = 1<<15,
    _dw     = 1<<16,
    // Other info
    reorder = 1<<17,
    // real types
    ref_any         = ref  | any,

    gemm_any        = gemm | any,
    gemm_blas       = gemm | blas,
    gemm_avx512     = gemm | avx512,
    gemm_avx2       = gemm | avx2,
    gemm_sse42      = gemm | sse42,

    jit_avx512      = jit  | avx512,
    jit_avx2        = jit  | avx2,
    jit_sse42       = jit  | sse42,
    jit_uni         = jit  | any,

    jit_avx512_1x1  = jit  | avx512 | _1x1,
    jit_avx2_1x1    = jit  | avx2   | _1x1,
    jit_sse42_1x1   = jit  | sse42  | _1x1,
    jit_uni_1x1     = jit  | any    | _1x1,

    jit_avx512_dw   = jit  | avx512 | _dw,
    jit_avx2_dw     = jit  | avx2   | _dw,
    jit_sse42_dw    = jit  | sse42  | _dw,
    jit_uni_dw      = jit  | any    | _dw,
};

impl_desc_type parse_impl_name(std::string impl_desc_name);

}  // namespace MKLDNNPlugin