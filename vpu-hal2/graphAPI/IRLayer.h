/*
 * INTEL CONFIDENTIAL
 * Copyright 2017 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include "file_utils.h"
#include "w_unistd.h"
#include "ie_common.h"
#include "ie_api.h"
#include "ie_layouts.h"
#include "ie_input_info.hpp"
#include "ie_blob.h"
#include "ie_layers.h"
#include "pugixml.hpp"

#ifdef _WIN32
#define strncasecmp _strnicmp
#endif

namespace FileUtils {
// Hack: should be in file utils, this is to avoid chaning master
inline std::string GetCWD()
{
	char cwd[4096];
	return getcwd(cwd, 4095);
}
};


namespace IRBuilder
{

template<typename T>
std::string &operator<<(std::string &&src, T i)
{
    std::stringstream oss;
    oss << i;
    return src.append(oss.str());
}

template<typename T>
std::string operator<<(const std::string &src, T i)
{
    std::stringstream oss;
    oss << src;
    oss << i;
    return oss.str();
}

#define THROW(x) THROW_IE_EXCEPTION << x
#define IR_ASSERT(x) if (!(x)) THROW_IE_EXCEPTION << "Assert failed for " #x << ": "

typedef InferenceEngine::CNNLayer::Ptr 	IRLayer;
typedef InferenceEngine::DataPtr 	OutputPort;
typedef InferenceEngine::Blob 		IRBlob;

struct Vector
{
    uint32_t length;
    IRBlob::Ptr data;
};

struct DelayObj
{
    IRLayer in_t; // x(t)
    IRLayer out_t_1; // x(t-1)
};

typedef InferenceEngine::SizeVector TensorDims;

inline size_t sizeOf(const TensorDims &dims)
{
    size_t ret = dims[0];
    for(int i = 1; i < dims.size(); ++i) ret *= dims[i];
    return ret;
}

void operator>>(const InferenceEngine::DataPtr &lhs, const InferenceEngine::CNNLayerPtr &rhs);

inline void operator>>(const InferenceEngine::CNNLayerPtr &lhs, const InferenceEngine::CNNLayerPtr &rhs)
{
    lhs->outData[0] >> rhs;
}

template <typename T>
IRBlob::Ptr readBlobFromFile(const std::string &file)
{
    auto fs = FileUtils::fileSize(file);
    if (fs <= 0) THROW("blob file ") << file << " not found or empty";
    InferenceEngine::Precision precision = sizeof(T) == sizeof(short)
        ? InferenceEngine::Precision::FP16 : InferenceEngine::Precision::FP32;
    auto ret = typename InferenceEngine::TBlob<T>::Ptr(
        new InferenceEngine::TBlob<T>(precision, InferenceEngine::C,
        { static_cast<size_t>(fs / sizeof(T)) }));
    ret->allocate();
    FileUtils::readAllFile(file, ret->data(), fs);
    return ret;
}

template <typename T>
IRBlob::Ptr readBlobFromFile(const std::string &file, const TensorDims &dims, InferenceEngine::Layout l)
{
    auto data = readBlobFromFile<T>(file);
    data->Reshape(dims, l);
    return data;
}

}  // namespace IRBuilder
