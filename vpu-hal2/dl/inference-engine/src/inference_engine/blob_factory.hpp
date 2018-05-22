//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
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

#include <utility>
#include "inference_engine.hpp"

template <InferenceEngine::Precision::ePrecision precision>
class BlobFactory {
 public:
    using BlobType = typename InferenceEngine::PrecisionTrait<precision>::value_type;
    static InferenceEngine::Blob::Ptr make(InferenceEngine::Layout l, InferenceEngine::SizeVector dims) {
        return InferenceEngine::make_shared_blob<BlobType>(precision, l, dims);
    }
    static InferenceEngine::Blob::Ptr make(InferenceEngine::Layout l, InferenceEngine::SizeVector dims, void* ptr) {
        return InferenceEngine::make_shared_blob<BlobType>(precision, l, dims, reinterpret_cast<BlobType*>(ptr));
    }
    static InferenceEngine::Blob::Ptr make(const InferenceEngine::TensorDesc& desc) {
        return InferenceEngine::make_shared_blob<BlobType>(desc);
    }
    static InferenceEngine::Blob::Ptr make(const InferenceEngine::TensorDesc& desc, void* ptr) {
        return InferenceEngine::make_shared_blob<BlobType>(desc, reinterpret_cast<BlobType*>(ptr));
    }
};

template <InferenceEngine::Precision::ePrecision precision, class ... Args> InferenceEngine::Blob::Ptr make_shared_blob2(Args && ... args) {
    return BlobFactory<precision>::make(std::forward<Args>(args) ...);
}

// TODO: customize make_shared_blob2
#define USE_FACTORY(precision)\
    case InferenceEngine::Precision::precision  : return make_shared_blob2<InferenceEngine::Precision::precision>(std::forward<Args>(args) ...);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr) make_blob_with_precision(const InferenceEngine::TensorDesc& desc);
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr) make_blob_with_precision(const InferenceEngine::TensorDesc& desc, void* ptr);

template <class ... Args>
InferenceEngine::Blob::Ptr make_blob_with_precision(InferenceEngine::Precision precision, Args &&... args) {
    switch (precision) {
        USE_FACTORY(FP32);
        USE_FACTORY(FP16);
        USE_FACTORY(Q78);
        USE_FACTORY(I16);
        USE_FACTORY(U8);
        USE_FACTORY(I8);
        USE_FACTORY(U16);
        USE_FACTORY(I32);
        default:
            THROW_IE_EXCEPTION << "cannot locate blob for precision: " << precision;
    }
}

#undef USE_FACTORY

/**
 * Create blob with custom precision
 * @tparam T - type off underlined elements
 * @tparam Args
 * @param args
 * @return
 */
template <class T, class ... Args>
InferenceEngine::Blob::Ptr make_custom_blob(Args &&... args) {
    return InferenceEngine::make_shared_blob<T>(InferenceEngine::Precision::fromType<T>(), std::forward<Args>(args) ...);
}
