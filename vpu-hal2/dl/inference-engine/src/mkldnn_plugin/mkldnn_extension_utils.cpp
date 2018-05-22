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

#include "mkldnn_extension_utils.h"
#include <limits>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine::MKLDNNPlugin;

uint8_t MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type dataType) {
    switch (dataType) {
    case mkldnn_f32:
        return 4;
    case mkldnn_s32:
        return 4;
    case mkldnn_s16:
        return 2;
    case mkldnn_s8:
        return 1;
    case mkldnn_u8:
        return 1;
    default:
        THROW_IE_EXCEPTION << "Unsupported data type.";
    }
}

memory::data_type MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision prec) {
    switch (prec) {
        case InferenceEngine::Precision::FP32:
            return memory::f32;
        case InferenceEngine::Precision::I16:
            return memory::s16;
        case InferenceEngine::Precision::I8:
            return memory::s8;
        case InferenceEngine::Precision::U8:
            return memory::u8;

        default: {
            THROW_IE_EXCEPTION << "The plugin does not support " << prec.name();
        }
    }
}

InferenceEngine::Precision MKLDNNExtensionUtils::DataTypeToIEPrecision(memory::data_type dataType) {
    switch (dataType) {
        case memory::f32:
            return InferenceEngine::Precision(InferenceEngine::Precision::FP32);

        default: {
            THROW_IE_EXCEPTION << "Unsupported data type.";
        }
    }
}

InferenceEngine::SizeVector MKLDNNExtensionUtils::MKLDimsToSizeVector(memory::dims dims) {
    InferenceEngine::SizeVector size;

    for (int i = 0; i < dims.size(); i++) {
        size.push_back(dims[i]);
    }

    return size;
}

MemoryFormat MKLDNNExtensionUtils::MKLFormatToMemoryFormat(memory::dims dims, memory::format fmt) {
    switch (fmt) {
        case memory::format_undef: return MemoryFormat::format_undef;
        case memory::any: return MemoryFormat::any;
        case memory::blocked: return MemoryFormat::blocked;
        case memory::x: return MemoryFormat::x;
        case memory::nc: return MemoryFormat::nc;
        case memory::nchw: return MemoryFormat::nchw;
        case memory::nhwc: return MemoryFormat::nhwc;
        case memory::chwn: return MemoryFormat::chwn;
        case memory::nChw8c: return MemoryFormat::nChw8c;
        case memory::nChw16c: return MemoryFormat::nChw16c;
        case memory::oi: return MemoryFormat::oi;
        case memory::io: return MemoryFormat::io;
        case memory::oihw: return MemoryFormat::oihw;
        case memory::ihwo: return MemoryFormat::ihwo;
        case memory::hwio: return MemoryFormat::hwio;
        case memory::OIhw8i8o: return MemoryFormat::OIhw8i8o;
        case memory::OIhw16i16o: return MemoryFormat::OIhw16i16o;
        case memory::OIhw8i16o2i: return MemoryFormat::OIhw8i16o2i;
        case memory::OIhw8o16i2o: return MemoryFormat::OIhw8o16i2o;
        case memory::OIhw8o8i: return MemoryFormat::OIhw8o8i;
        case memory::OIhw16o16i: return MemoryFormat::OIhw16o16i;
        case memory::Oihw8o: return MemoryFormat::Oihw8o;
        case memory::Oihw16o: return MemoryFormat::Oihw16o;
        case memory::Ohwi8o: return MemoryFormat::Ohwi8o;
        case memory::Ohwi16o: return MemoryFormat::Ohwi16o;
        case memory::OhIw16o4i: return MemoryFormat::OhIw16o4i;
        case memory::goihw: return MemoryFormat::goihw;
        case memory::gOIhw8i8o: return MemoryFormat::gOIhw8i8o;
        case memory::gOIhw16i16o: return MemoryFormat::gOIhw16i16o;
        case memory::gOIhw8i16o2i: return MemoryFormat::gOIhw8i16o2i;
        case memory::gOIhw8o16i2o: return MemoryFormat::gOIhw8o16i2o;
        case memory::gOIhw8o8i: return MemoryFormat::gOIhw8o8i;
        case memory::gOIhw16o16i: return MemoryFormat::gOIhw16o16i;
        case memory::gOihw8o: return MemoryFormat::gOihw8o;
        case memory::gOihw16o: return MemoryFormat::gOihw16o;
        case memory::gOhwi8o: return MemoryFormat::gOhwi8o;
        case memory::gOhwi16o: return MemoryFormat::gOhwi16o;
        case memory::gOhIw16o4i: return MemoryFormat::gOhIw16o4i;
        default: {
            THROW_IE_EXCEPTION << "Unsupported data type.";
        }
    }
}

memory::format MKLDNNExtensionUtils::MemoryFormatToMKLFormat(MemoryFormat fmt) {
    switch (fmt) {
        case MemoryFormat::format_undef: return memory::format_undef;
        case MemoryFormat::any: return memory::any;
        case MemoryFormat::blocked: return memory::blocked;
        case MemoryFormat::x: return memory::x;
        case MemoryFormat::nc: return memory::nc;
        case MemoryFormat::nchw: return memory::nchw;
        case MemoryFormat::nhwc: return memory::nhwc;
        case MemoryFormat::chwn: return memory::chwn;
        case MemoryFormat::nChw8c: return memory::nChw8c;
        case MemoryFormat::nChw16c: return memory::nChw16c;
        case MemoryFormat::oi: return memory::oi;
        case MemoryFormat::io: return memory::io;
        case MemoryFormat::oihw: return memory::oihw;
        case MemoryFormat::ihwo: return memory::ihwo;
        case MemoryFormat::hwio: return memory::hwio;
        case MemoryFormat::OIhw8i8o: return memory::OIhw8i8o;
        case MemoryFormat::OIhw16i16o: return memory::OIhw16i16o;
        case MemoryFormat::OIhw8i16o2i: return memory::OIhw8i16o2i;
        case MemoryFormat::OIhw8o16i2o: return memory::OIhw8o16i2o;
        case MemoryFormat::OIhw8o8i: return memory::OIhw8o8i;
        case MemoryFormat::OIhw16o16i: return memory::OIhw16o16i;
        case MemoryFormat::Oihw8o: return memory::Oihw8o;
        case MemoryFormat::Oihw16o: return memory::Oihw16o;
        case MemoryFormat::Ohwi8o: return memory::Ohwi8o;
        case MemoryFormat::Ohwi16o: return memory::Ohwi16o;
        case MemoryFormat::OhIw16o4i: return memory::OhIw16o4i;
        case MemoryFormat::goihw: return memory::goihw;
        case MemoryFormat::gOIhw8i8o: return memory::gOIhw8i8o;
        case MemoryFormat::gOIhw16i16o: return memory::gOIhw16i16o;
        case MemoryFormat::gOIhw8i16o2i: return memory::gOIhw8i16o2i;
        case MemoryFormat::gOIhw8o16i2o: return memory::gOIhw8o16i2o;
        case MemoryFormat::gOIhw8o8i: return memory::gOIhw8o8i;
        case MemoryFormat::gOIhw16o16i: return memory::gOIhw16o16i;
        case MemoryFormat::gOihw8o: return memory::gOihw8o;
        case MemoryFormat::gOihw16o: return memory::gOihw16o;
        case MemoryFormat::gOhwi8o: return memory::gOhwi8o;
        case MemoryFormat::gOhwi16o: return memory::gOhwi16o;
        case MemoryFormat::gOhIw16o4i: return memory::gOhIw16o4i;
        default: {
            THROW_IE_EXCEPTION << "Unsupported data type.";
        }
    }
}

MKLDNNPrimitiveMemory MKLDNNExtensionUtils::MKLMemoryToGenericMemory(const MKLDNNMemory& mem) {
    MKLDNNPrimitiveMemory memory;

    memory.dims = MKLDimsToSizeVector(mem.GetDims());
    memory.data = mem.GetData();
    memory.precision = DataTypeToIEPrecision(mem.GetDataType());
    memory.format = MKLFormatToMemoryFormat(mem.GetDims(), mem.GetFormat());

    return memory;
}
