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

#include <memory>
#include <vector>

#include "inference_engine.hpp"
#include "mkldnn_dims.h"
#include <mkldnn.hpp>
#include <string>
#include <mkldnn_types.h>
#include <functional>

namespace MKLDNNPlugin {

class MKLDNNMemoryDesc {
public:
    MKLDNNMemoryDesc(): desc({}, mkldnn::memory::data_type::f32, mkldnn::memory::format::format_undef) {}
    explicit MKLDNNMemoryDesc(const InferenceEngine::TensorDesc& tDesc);
    explicit MKLDNNMemoryDesc(const mkldnn::memory::desc& desc): desc(desc) {}
    MKLDNNMemoryDesc(mkldnn::memory::dims dims, mkldnn::memory::data_type dataType, mkldnn::memory::format format);

    const mkldnn::memory::desc& getDesc() const {
        return desc;
    }

    mkldnn::memory::format getFormat() const {
        return static_cast<mkldnn::memory::format>(desc.data.format);
    }

    MKLDNNDims getDims() const {
        return MKLDNNDims(desc.data.dims, desc.data.ndims);
    }

    MKLDNNMemoryDesc& operator=(const mkldnn::memory::desc& desc) {
        this->desc = desc;
        return *this;
    }

    operator bool() const {
        return getFormat() != mkldnn::memory::format::any && getFormat() != mkldnn::memory::format::format_undef;
    }

    bool operator == (const MKLDNNMemoryDesc& rhs) const;
    bool operator != (const MKLDNNMemoryDesc& rhs) const;

    operator mkldnn::memory::desc() const;
    operator InferenceEngine::TensorDesc() const;

private:
    mkldnn::memory::desc desc;
};


class MKLDNNMemory;

using MKLDNNMemoryPtr = std::shared_ptr<MKLDNNMemory>;

class MKLDNNMemory {
public:
    explicit MKLDNNMemory(const mkldnn::engine& eng);

    const mkldnn::memory& GetPrimitive() const {
        return *prim;
    }

    const std::shared_ptr<mkldnn::memory>& GetPrimitivePtr() const {
        return prim;
    }

    mkldnn::memory::desc GetDescriptor() const {
        return prim->get_primitive_desc().desc();
    }

    mkldnn::memory::primitive_desc GetPrimitiveDescriptor() const {
        return prim->get_primitive_desc();
    }

    void* GetData() const {
        return prim->get_data_handle();
    }

    mkldnn::memory::data_type GetDataType() const {
        return static_cast<mkldnn::memory::data_type>(GetDescriptor().data.data_type);
    }

    size_t GetSize() const;

    mkldnn::memory::format GetFormat() const {
        return static_cast<mkldnn::memory::format>(prim->get_primitive_desc().desc().data.format);
    }

    mkldnn::memory::dims GetDims() const {
        auto data = GetDescriptor().data;

        return std::vector<int>(data.dims, data.dims + data.ndims);
    }

    void Create(mkldnn::memory::dims dims, mkldnn::memory::data_type data_type, mkldnn::memory::format format,
                const void* data = nullptr);

    void Create(const mkldnn::memory::desc& desc, const void* data = nullptr);
    void CreateFrom(mkldnn::memory::dims dims, const MKLDNNMemory& src);
    void CreateFrom(mkldnn::memory::primitive_desc &pdesc, const void* data = nullptr);

    void SetData(mkldnn::memory::data_type dataType, mkldnn::memory::format format, const void* data, size_t size, bool ftz = true) const;
    void SetData(mkldnn::memory::data_type dataType, mkldnn::memory::format format, const std::vector<void*>& data,
                 const std::vector<size_t>& size, bool ftz = true) const;

    void FillZero();

    InferenceEngine::Blob::Ptr GetBlob() const;

    static bool IsPlainFormat(mkldnn::memory::format format);
    static mkldnn::memory::format GetPlainFormat(mkldnn::memory::dims dims);
    static bool isConsistant(mkldnn::memory::dims dims, mkldnn::memory::format format);
    static bool formatEquals(const mkldnn::memory::format &lformat, const mkldnn::memory::format &rformat) noexcept;


    static std::string formatToString(mkldnn::memory::format fmt);

    static void CreateBlockingDesc(mkldnn::memory::desc& desc);

private:
    std::shared_ptr<mkldnn::memory> prim;
    mkldnn::engine eng;
};


}  // namespace MKLDNNPlugin
