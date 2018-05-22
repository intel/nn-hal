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

#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mkldnn_types.h>
#include <unordered_set>

#include "mkldnn_memory.h"
#include "mkldnn_node.h"
#include "mkldnn_extension_utils.h"

using namespace InferenceEngine;
using namespace mkldnn;

namespace MKLDNNPlugin {

MKLDNNMemory::MKLDNNMemory(const engine& eng) : eng(eng) {}

size_t MKLDNNMemory::GetSize() const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(GetDataType()));

    auto desc = GetDescriptor();
//        return GetPrimitiveDescriptor().get_size();
    std::vector<int> dims(desc.data.layout_desc.blocking.padding_dims,
                          desc.data.layout_desc.blocking.padding_dims + desc.data.ndims);
    return std::accumulate(std::begin(dims), std::end(dims), (size_t) 1, std::multiplies<size_t>()) * itemSize;
}

void MKLDNNMemory::Create(memory::dims dims, memory::data_type data_type, memory::format format, const void* data) {
    if (!isConsistant(dims, format)) {
        THROW_IE_EXCEPTION << "dims and format are inconsistent.";
    }

    if (format == memory::blocked) {
        format = memory::any;
    }

    memory::desc desc = mkldnn::memory::desc({dims}, data_type, format);

    if (format == memory::any) {
        CreateBlockingDesc(desc);
    }

    Create(desc, data);
}

void MKLDNNMemory::Create(const mkldnn::memory::desc& desc, const void *data) {
    auto primitive_desc = memory::primitive_desc(desc, eng);
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(desc.data.data_type));

    if (data == nullptr) {
        prim.reset(new memory(primitive_desc));

        size_t real_size = 0;
        if (prim->get_primitive_desc().desc().data.ndims > 0) {
            real_size = static_cast<size_t>(prim->get_primitive_desc().desc().data.layout_desc.blocking.padding_dims[0]);
            for (int i = 1; i < prim->get_primitive_desc().desc().data.ndims; i++) {
                real_size *= prim->get_primitive_desc().desc().data.layout_desc.blocking.padding_dims[i];
            }
        }
        uint8_t* dataPtr = static_cast<uint8_t*>(GetData());
        dataPtr += itemSize * prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;

        memset(dataPtr, 0, real_size * itemSize);
    } else {
        // MKLDNN accepts not a const data, probably need to remove some level of consteness in a call stack
        prim.reset(new memory(primitive_desc, const_cast<void*>(data)));
    }
}

void MKLDNNMemory::CreateFrom(memory::dims dims, const MKLDNNMemory& src) {
    auto data = src.GetDescriptor().data;

    auto dataType = static_cast<memory::data_type>(data.data_type);
    auto format = static_cast<memory::format>(data.format);

    Create(dims, dataType, format);
}

void MKLDNNMemory::CreateFrom(memory::primitive_desc &pdesc, const void* data) {
    if (data == nullptr) {
        prim = std::shared_ptr<memory>(new memory(pdesc));
    } else {
        prim = std::shared_ptr<memory>(new memory(pdesc, const_cast<void*>(data)));
    }
}

void MKLDNNMemory::SetData(memory::data_type dataType, memory::format format, const void* data, size_t size, bool ftz) const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dataType));

    if (static_cast<mkldnn_memory_format_t>(format) != GetDescriptor().data.format ||
            GetDataType() != dataType) {
        auto memData = GetDescriptor().data;

        std::vector<int> dims(memData.dims, memData.dims + memData.ndims);

        auto dataType = GetDataType();

        MKLDNNMemory src(eng);
        src.Create(dims, dataType, format, data);

        std::shared_ptr<mkldnn::reorder> pReorder =
                std::shared_ptr<mkldnn::reorder>(new mkldnn::reorder(src.GetPrimitive(), GetPrimitive()));

        mkldnn::stream(stream::kind::eager).submit({*pReorder});
    } else {
        uint8_t* dataPtr = static_cast<uint8_t*>(GetData());
        dataPtr += itemSize * prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;
        memcpy(dataPtr, data, size);
    }

    if (ftz) {
        assert(dataType == mkldnn_f32);  // TODO Remove FP32 assumption here
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;
        size_t realSize = GetSize() / sizeof(float);
        for (size_t i = 0; i < realSize; i++) {
            if (memData[i] != 0 && (fabsf(memData[i]) < std::numeric_limits<float>::min())) {
                memData[i] = 0.0f;
            }
        }
    }
}

void MKLDNNMemory::SetData(memory::data_type dataType, memory::format format, const std::vector<void*>& data,
                           const std::vector<size_t>& size, bool ftz) const {
    size_t totalSize = static_cast<size_t >(std::accumulate(size.begin(), size.end(), 0));

    char* buffer = new char[totalSize];
    char* bufferPtr = buffer;

    for (int i = 0; i < size.size(); i++) {
        memcpy(bufferPtr, data[i], size[i]);
        bufferPtr += size[i];
    }

    SetData(dataType, format, buffer, totalSize, ftz);

    delete [] buffer;
}

void MKLDNNMemory::FillZero() {
    void* dataPtr = GetData();
    memset(dataPtr, 0, GetSize());
}

bool MKLDNNMemory::isConsistant(memory::dims dims, memory::format format) {
    using f = mkldnn::memory::format;

    size_t ndims = 0;

    switch (format) {
        case f::x:
            ndims = 1; break;
        case f::nc:
        case f::oi:
        case f::io:
            ndims = 2; break;
        case f::nchw:
        case f::nhwc:
        case f::chwn:
        case f::nChw8c:
        case f::nChw16c:
        case f::oihw:
        case f::ihwo:
        case f::OIhw8i8o:
        case f::OIhw16i16o:
        case f::OIhw8o8i:
        case f::OIhw16o16i:
        case f::OIhw8i16o2i:
        case f::OIhw8o16i2o:
        case f::Ohwi8o:
        case f::Ohwi16o:
        case f::OhIw16o4i:
            ndims = 4; break;
        case f::goihw:
        case f::gOIhw8i8o:
        case f::gOIhw16i16o:
        case f::gOIhw8i16o2i:
        case f::gOIhw8o16i2o:
        case f::gOhwi8o:
        case f::gOhwi16o:
        case f::gOIhw8o8i:
        case f::gOIhw16o16i:
        case f::gOhIw16o4i:
        case f::Goihw8g:
        case f::Goihw16g:
            ndims = 5; break;
        case f::format_undef:
            ndims = 0; break;
        case f::any:
        case f::blocked:
            return true;
        default:
            return false;
    }

    return (dims.size() == ndims);
}

bool MKLDNNMemory::formatEquals(const memory::format &lformat, const memory::format &rformat) noexcept {
    return (lformat == rformat) || (lformat == memory::nc && rformat == memory::oi) ||
           (lformat == memory::oi && rformat == memory::nc) || (lformat == memory::nchw && rformat == memory::oihw) ||
           (lformat == memory::oihw && rformat == memory::nchw);
}

bool MKLDNNMemory::IsPlainFormat(memory::format format) {
    std::vector<memory::format> plains = {memory::nc, memory::nchw, memory::nhwc, memory::chwn,
        memory::oi, memory::io, memory::oihw, memory::ihwo,
        memory::goihw,
        memory::blocked};

    for (auto it : plains) {
        if (format == it) {
            return true;
        }
    }

    return false;
}

memory::format MKLDNNMemory::GetPlainFormat(memory::dims dims) {
    switch (dims.size()) {
        case 1:
            return memory::x;
        case 2:
            return memory::nc;
        case 4:
            return memory::nchw;
        default:
            return memory::blocked;
    }
}

void MKLDNNMemory::CreateBlockingDesc(memory::desc &desc) {
    auto dims = desc.data.dims;
    int ndims = desc.data.ndims;

    desc.data.format = mkldnn_blocked;

    auto& blk = desc.data.layout_desc.blocking;

    blk.offset_padding = 0;

    for (int i = 0; i < ndims; i++) {
        blk.block_dims[i] = 1;
        blk.strides[1][i] = 1;
        blk.padding_dims[i] = dims[i];
        blk.offset_padding_to_data[i] = 0;
    }

    int perm[TENSOR_MAX_DIMS] = {0};

    for (int i = 0; i < ndims; ++i) {
        perm[i] = i;
    }

    blk.strides[0][perm[ndims - 1]] = 1;

    for (int d = 1; d < ndims; ++d) {
        const int prev_idx = perm[ndims - d];
        const int curr_idx = perm[ndims - 1 - d];

        blk.strides[0][curr_idx] = dims[curr_idx] == 0 ? 1 : blk.strides[0][prev_idx] * (std::max)(1, dims[prev_idx]);
    }
}

std::string MKLDNNMemory::formatToString(memory::format fmt) {
    switch (fmt) {
        case memory::format_undef: return "undef";
        case memory::any: return "any";
        case memory::blocked: return "blocked";

        case memory::x: return "x";

        case memory::nc: return "nc";
        case memory::oi: return "oi";
        case memory::io: return "io";

        case memory::nchw: return "nchw";
        case memory::nhwc: return "nhwc";
        case memory::chwn: return "chwn";
        case memory::nChw8c: return "nChw8c";
        case memory::nChw16c: return "nChw16c";

        case memory::oihw: return "oihw";
        case memory::ihwo: return "ihwo";
        case memory::OIhw8i8o: return "OIhw8i8o";
        case memory::OIhw16i16o: return "OIhw16i16o";
        case memory::OIhw8o8i: return "OIhw8o8i";
        case memory::OIhw16o16i: return "OIhw16o16i";
        case memory::OIhw8i16o2i: return "OIhw8i16o2i";
        case memory::OIhw8o16i2o: return "OIhw8o16i2o";
        case memory::Ohwi8o: return "Ohwi8o";
        case memory::Ohwi16o: return "Ohwi16o";
        case memory::OhIw16o4i: return "OhIw16o4i";

        case memory::goihw: return "goihw";
        case memory::gOIhw8i8o: return "gOIhw8i8o";
        case memory::gOIhw16i16o: return "gOIhw16i16o";
        case memory::gOIhw8i16o2i: return "gOIhw8i16o2i";
        case memory::gOIhw8o16i2o: return "gOIhw8o16i2o";
        case memory::gOhwi8o: return "gOhwi8o";
        case memory::gOhwi16o: return "gOhwi16o";
        case memory::gOIhw8o8i: return "gOIhw8o8i";
        case memory::gOIhw16o16i: return "gOIhw16o16i";
        case memory::gOhIw16o4i: return "gOhIw16o4i";
        default: {
            THROW_IE_EXCEPTION << "Unsupported data type.";
        }
    }
}

InferenceEngine::Blob::Ptr MKLDNNMemory::GetBlob() const {
    MKLDNNMemoryDesc desc(prim->get_primitive_desc().desc());

    InferenceEngine::Blob::Ptr blob;

    switch (GetDataType()) {
    case mkldnn_f32: {
    auto * data = static_cast<float *>(GetData()) - prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;
            blob = InferenceEngine::make_shared_blob<float>(desc, data, GetSize());
        }
        break;
    case mkldnn_u8: {
            auto * data = static_cast<uint8_t *>(GetData()) - prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;
            blob = InferenceEngine::make_shared_blob<uint8_t>(desc, data, GetSize());
        }
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported dataType: " << MKLDNNExtensionUtils::DataTypeToIEPrecision(GetDataType()).name();
    }

    return blob;
}

bool MKLDNNMemoryDesc::operator==(const MKLDNNMemoryDesc &rhs) const {
    auto dims_equal = [] (mkldnn_memory_desc_t ldata, mkldnn_memory_desc_t rdata) {
        if (ldata.ndims != rdata.ndims)
            return false;
        for (int i = 0; i < ldata.ndims; i++) {
            if (ldata.dims[i] != rdata.dims[i])
                return false;
        }
        return true;
    };
    auto blocking_equal = [] (mkldnn_memory_desc_t ldata, mkldnn_memory_desc_t rdata) {
        if (ldata.ndims != rdata.ndims)
            return false;
        mkldnn_blocking_desc_t lblock = ldata.layout_desc.blocking;
        mkldnn_blocking_desc_t rblock = rdata.layout_desc.blocking;
        if (lblock.offset_padding != rblock.offset_padding)
            return false;
        for (int i = 0; i < ldata.ndims; i++) {
            if (lblock.block_dims[i] != rblock.block_dims[i] ||
                lblock.offset_padding_to_data[i] != rblock.offset_padding_to_data[i] ||
                lblock.padding_dims[i] != rblock.padding_dims[i] || lblock.strides[0][i] != rblock.strides[0][i] ||
                lblock.strides[1][i] != rblock.strides[1][i])
                return false;
        }
        return true;
    };
    return dims_equal(this->desc.data, rhs.desc.data) &&
           this->desc.data.data_type == rhs.desc.data.data_type &&
           this->desc.data.format == rhs.desc.data.format &&
           this->desc.data.primitive_kind == rhs.desc.data.primitive_kind &&
           blocking_equal(this->desc.data, rhs.desc.data);
}

bool MKLDNNMemoryDesc::operator!=(const MKLDNNMemoryDesc &rhs) const {
    return !(*this == rhs);
}

MKLDNNMemoryDesc::operator mkldnn::memory::desc() const {
    return desc;
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(mkldnn::memory::dims dims, mkldnn::memory::data_type dataType,
                                   mkldnn::memory::format format): desc(dims, dataType, mkldnn::memory::any) {
    if (format != memory::blocked) {
        desc = mkldnn::memory::desc({dims}, dataType, format);
        return;
    }
    MKLDNNMemory::CreateBlockingDesc(desc);
}

MKLDNNMemoryDesc::operator InferenceEngine::TensorDesc() const {
    Precision precision;
    switch (desc.data.data_type) {
        case mkldnn_f32:
            precision = Precision::FP32;
            break;
        case mkldnn_u8:
            precision = Precision::U8;
            break;
        default:
            THROW_IE_EXCEPTION << "Cannot cast to TensorDesc. Unsupported precision!";
    }
    Layout layout;
    switch (getFormat()) {
        case memory::format_undef:
            THROW_IE_EXCEPTION << "Cannot cast to tensor desc. Format is undefined!";
        case memory::any:
            layout = Layout::ANY;
            return TensorDesc(precision, getDims().ToSizeVector(), layout);
        case memory::x:
            layout = Layout::C;
            break;
        case memory::nc:
            layout = Layout::NC;
            break;
        case memory::nchw:
            layout = Layout::NCHW;
            break;
        case memory::nhwc:
            layout = Layout::NHWC;
            break;
        default:
            layout = Layout::BLOCKED;
    }

    auto blkInfo = desc.data.layout_desc.blocking;
    // recovery order
    SizeVector orders(desc.data.ndims);
    SizeVector strides, blkStrides;
    SizeVector sortedStrides;
    auto total_size = static_cast<size_t>(desc.data.ndims);
    for (size_t i = 0; i < desc.data.ndims; i++) {
        sortedStrides.push_back(static_cast<size_t>(blkInfo.strides[0][i]));
        sortedStrides.push_back(static_cast<size_t>(blkInfo.strides[1][i]));
        strides.push_back(static_cast<size_t>(blkInfo.strides[0][i]));
        blkStrides.push_back(static_cast<size_t>(blkInfo.strides[1][i]));
        orders[i]++;
        if (blkInfo.block_dims[i] != 1) {
            orders[i]++;
            total_size++;
        }
    }

    std::sort(sortedStrides.begin(), sortedStrides.end());
    while (sortedStrides.size() > 1 && sortedStrides[0] == 1 && sortedStrides[1] == 1 && sortedStrides.size() > total_size) {
        sortedStrides.erase(sortedStrides.begin());
    }
    while (sortedStrides.size() > total_size) {
        bool notFound = true;
        for (size_t i = 1; i < sortedStrides.size(); i++) {
            if (sortedStrides[i - 1] == sortedStrides[i]) {
                notFound = false;
                sortedStrides.erase(sortedStrides.begin() + i);
                break;
            }
        }
        if (notFound)
            break;
    }

    if (sortedStrides.size() != total_size) {
        THROW_IE_EXCEPTION << "Cannot detect the right dimensions size with blocked.";
    }

    SizeVector recoveredOrder;
    SizeVector blkDims;
    for (size_t i = 0; i < sortedStrides.size(); i++) {
        std::vector<bool> order(desc.data.ndims, false);
        for (size_t j = 0; j < desc.data.ndims; j++) {
            if ((strides[j] == sortedStrides[i] || blkStrides[j] == sortedStrides[i]) && orders[j] > 0) {
                order[j] = true;
            }
        }
        bool wasFind = false;
        for (size_t k = 1; k <= order.size(); k++) {
            size_t ord = order.size() - k;
            if (!order[ord])
                continue;
            if (blkInfo.block_dims[ord] != 1) {
                size_t new_find_number = sortedStrides[i] * blkInfo.block_dims[ord];
                if (i + 1 < sortedStrides.size() && new_find_number != sortedStrides[i + 1]) {
                    continue;
                }

                for (size_t j = 0; j < desc.data.ndims; j++) {
                    if (strides[j] == new_find_number || blkStrides[j] == new_find_number) {
                        wasFind = true;
                        recoveredOrder.insert(recoveredOrder.begin(), ord);
                        blkDims.insert(blkDims.begin(), blkInfo.block_dims[ord]);
                        orders[ord]--;
                        break;
                    }
                }
                if (wasFind) {
                    blkInfo.padding_dims[ord] /= blkInfo.block_dims[ord];
                    blkInfo.block_dims[ord] = 1;
                    break;
                }
            }
        }
        if (wasFind)
            continue;
        for (size_t k = 1; k <= order.size(); k++) {
            size_t ord = order.size() - k;
            if (!order[ord])
                continue;
            size_t new_find_number = sortedStrides[i] * blkInfo.padding_dims[ord];

            if (i + 1 < sortedStrides.size() && new_find_number != sortedStrides[i + 1]) {
                continue;
            }
            for (size_t j = 0; j < desc.data.ndims; j++) {
                if (strides[j] == new_find_number || blkStrides[j] == new_find_number) {
                    wasFind = true;
                    recoveredOrder.insert(recoveredOrder.begin(), ord);
                    blkDims.insert(blkDims.begin(), blkInfo.padding_dims[ord]);
                    orders[ord]--;
                    break;
                }
            }
            if (wasFind) {
                blkInfo.padding_dims[ord] /= blkInfo.block_dims[ord];
                blkInfo.block_dims[ord] = 1;
                break;
            }
        }
        if (!wasFind) {
            bool one = false;
            size_t ord = 0;
            for (size_t i = 0; i < order.size(); i++) {
                if (order[i]) {
                    if (one) {
                        one = false;
                        break;
                    }
                    one = true;
                    ord = i;
                }
            }
            if (one) {
                orders[ord]--;
                recoveredOrder.insert(recoveredOrder.begin(), ord);
                blkDims.insert(blkDims.begin(), blkInfo.padding_dims[ord]);
            }
        }
    }
    size_t offset = blkInfo.offset_padding;
    SizeVector offsetsForDims;

    for (size_t i = 0; i < blkDims.size() && i < TENSOR_MAX_DIMS; i++) {
        if (i < orders.size())
            offsetsForDims.push_back(blkInfo.offset_padding_to_data[i]);
        else
            offsetsForDims.push_back(0);
    }
    TensorDesc tensorDesc(precision, getDims().ToSizeVector(), {blkDims, recoveredOrder, offset, offsetsForDims});

    tensorDesc.setLayout(layout);
    return tensorDesc;
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(const TensorDesc& tDesc):
        desc({}, mkldnn::memory::data_type::f32, mkldnn::memory::format::format_undef) {
    mkldnn::memory::data_type data_type;
    switch (tDesc.getPrecision()) {
        case Precision::FP32:
            data_type = mkldnn::memory::data_type::f32;
            break;
        case Precision::U8:
            data_type = mkldnn::memory::data_type::u8;
            break;
        default:
            THROW_IE_EXCEPTION << "Cannot create MKLDNNMemoryDesc from TensorDesc. Unsupported precision!";
    }

    MKLDNNDims dims(tDesc.getDims());

    mkldnn::memory::format mkldnnFormat = memory::format::format_undef;
    SizeVector blkdDims = tDesc.getBlockingDesc().getBlockDims();
    SizeVector order = tDesc.getBlockingDesc().getOrder();
    SizeVector offsetsToData = tDesc.getBlockingDesc().getOffsetPaddingToData();
    SizeVector strides = tDesc.getBlockingDesc().getStrides();
    switch (tDesc.getLayout()) {
        case ANY:
            mkldnnFormat = memory::format::any;
            break;
        case NCHW:
            mkldnnFormat = memory::format::nchw;
            break;
        case NHWC:
            mkldnnFormat = memory::format::nhwc;
            break;
        case OIHW:
            mkldnnFormat = memory::format::oihw;
            break;
        case C:
            mkldnnFormat = memory::format::x;
            break;
        case CHW:
            mkldnnFormat = memory::format::blocked;
            break;
        case HW:
        case NC:
            mkldnnFormat = memory::format::nc;
            break;
        case BLOCKED:
            if (dims.ndims() == 1) {
                mkldnnFormat = memory::format::x;
                break;
            } else if (dims.ndims() == 2) {
                mkldnnFormat = memory::format::nc;
                break;
            } else if (dims.ndims() == 4) {
                if (order.size() == 5 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 1) {
                    if (blkdDims[4] == 8) {
                        mkldnnFormat = memory::format::nChw8c;
                        break;
                    } else if (blkdDims[4] == 16) {
                        mkldnnFormat = memory::format::nChw16c;
                        break;
                    }
                } else if (order.size() == 4) {
                    if (order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3) {
                        mkldnnFormat = memory::format::nchw;
                        break;
                    } else if (order[0] == 0 && order[1] == 2 && order[2] == 3 && order[3] == 1) {
                        mkldnnFormat = memory::format::nhwc;
                        break;
                    }
                }
            }
            mkldnnFormat = memory::format::blocked;
            break;
        case CN:
            mkldnnFormat = memory::format::blocked;
            break;
    }
    if (mkldnnFormat == memory::format_undef)
        THROW_IE_EXCEPTION << "Cannot detect the right memory format!";

    bool notDefault = false;
    size_t currentStride = 1;
    for (size_t i = 0; i < order.size(); i++) {
        if (offsetsToData[i] != 0) {
            notDefault = true;
            break;
        }
        if (strides[strides.size() - (1 +i)] != currentStride) {
            notDefault = true;
            break;
        }
        currentStride *= blkdDims[blkdDims.size() - (1 + i)];
    }

    if (notDefault)
        THROW_IE_EXCEPTION << "Currently MKLDNNPlugin supports only packaged memory";

    if (mkldnnFormat == memory::blocked) {
        desc = mkldnn::memory::desc(dims, data_type, memory::any);
        desc.data.format = mkldnn_blocked;

        auto& blk = desc.data.layout_desc.blocking;

        blk.offset_padding = tDesc.getBlockingDesc().getOffsetPadding();

        for (size_t i = 0; i < dims.ndims(); i++) {
            blk.block_dims[i] = 1;
            blk.strides[1][i] = 1;
            blk.padding_dims[i] = dims[i];
            blk.offset_padding_to_data[i] = offsetsToData[i];
        }

        int perm[TENSOR_MAX_DIMS] = {0};

        for (size_t i = 0; i < dims.ndims(); ++i) {
            perm[i] = i;
        }

        blk.strides[0][perm[dims.ndims() - 1]] = 1;

        for (int d = 1; d < dims.ndims(); ++d) {
            const int prev_idx = perm[dims.ndims() - d];
            const int curr_idx = perm[dims.ndims() - 1 - d];

            blk.strides[0][curr_idx] = dims[curr_idx] == 0 ? 1 : blk.strides[0][prev_idx] * (std::max)(1, dims[prev_idx]);
        }
    } else {
        desc = mkldnn::memory::desc(autoBlockingDims(dims, mkldnnFormat), data_type, mkldnnFormat);
    }

    desc.data.layout_desc.blocking.offset_padding = tDesc.getBlockingDesc().getOffsetPadding();
    for (size_t i = 0; i < tDesc.getBlockingDesc().getOffsetPaddingToData().size() && i < TENSOR_MAX_DIMS; i++) {
        desc.data.layout_desc.blocking.offset_padding_to_data[i] = static_cast<int>(offsetsToData[i]);
    }
}

}  // namespace MKLDNNPlugin
