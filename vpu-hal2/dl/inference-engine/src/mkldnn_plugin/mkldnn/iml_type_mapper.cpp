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

#include "iml_type_mapper.h"

using namespace MKLDNNPlugin;

impl_desc_type MKLDNNPlugin::parse_impl_name(std::string impl_desc_name) {
    impl_desc_type res = impl_desc_type::unknown;

#define SEARCH_WORD(_wrd) if (impl_desc_name.find(#_wrd) != std::string::npos) \
    res = static_cast<impl_desc_type>(res | impl_desc_type::_wrd);

    SEARCH_WORD(ref);
    SEARCH_WORD(jit);
    SEARCH_WORD(gemm);
    SEARCH_WORD(blas);
    SEARCH_WORD(sse42);
    SEARCH_WORD(avx2);
    SEARCH_WORD(avx512);
    SEARCH_WORD(any);
    SEARCH_WORD(_1x1);
    SEARCH_WORD(_dw);
    SEARCH_WORD(reorder);
#undef SEARCH_WORD

#define SEARCH_WORD_2(_wrd, _key) if (impl_desc_name.find(#_wrd) != std::string::npos) \
    res = static_cast<impl_desc_type>(res | impl_desc_type::_key);

    SEARCH_WORD_2(nchw, ref);
#undef SEARCH_WORD_2

    return res;
}