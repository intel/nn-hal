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

#include <cassert>
#include <algorithm>
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <tuple>
#include <precision_utils.h>
#ifdef AKS
#include <array>
#endif

#include "graph_transformer.hpp"
#include "mv_common.h"
#include "mv_blob_format.h"

using namespace InferenceEngine;
using namespace VPU;

//
// Utility functions and types
//

const uint32_t CMX_BUFFER_SIZE_LIMIT = 401408;

template <typename T>
T alignVal(T val, T pow2) {
    return (val + (pow2 - 1)) & ~(pow2 - 1);
}

namespace Dim {
    enum {
        N = 3,
        C = 2,
        H = 1,
        W = 0,

        X = 0,
        Y = 1,
        Z = 2
    };
}  // namespace Dim

enum class VpuDataType {
    U8,
    FP16,
    FP32
};

uint32_t getDataTypeSize(VpuDataType type);
VpuDataType iePrecisionToVpu(const Precision& precision);

std::string mvTensorOpTypeToStr(t_MvTensorOpType type);
std::string mvTensorStorageOrderToStr(t_MvTensorStorageOrder order);
std::string mvDataIndexToStr(IndexCodes index);
std::string dataTypeToStr(VpuDataType type);

template <typename T, size_t MaxSize>
struct DynArr {
    std::array<T, MaxSize> data = {};
    size_t size = 0;

    DynArr() = default;
    DynArr(size_t size) : size(size) {}
    DynArr(std::initializer_list<T> l) {
        assert(l.size() <= MaxSize);
        std::copy(l.begin(), l.end(), data.begin());
        size = l.size();
    }

    DynArr(const DynArr&) = default;
    DynArr(DynArr&&) = default;
    DynArr& operator=(const DynArr&) = default;
    DynArr& operator=(DynArr&&) = default;

    T& operator[](size_t index) {
        assert(index < size);
        return data[index];
    }

    const T& operator[](size_t index) const {
        assert(index < size);
        return data[index];
    }

    size_t count() const { return size; }

    void resize(size_t newSize) {
        assert(newSize <= MaxSize);
        size = newSize;
    }
};

template <typename T, size_t MaxSize>
std::ostream& operator<<(std::ostream& os, const DynArr<T, MaxSize>& arr) {
    os << "(";
    if (arr.count() > 0) {
        os << arr[0];
        for (size_t i = 1; i < arr.count(); ++i) {
            os << ", " << arr[i];
        }
    }
    os << ")";
    return os;
}

const size_t MaxDimensions = 6;

struct VpuDims : DynArr<uint32_t, MaxDimensions> {
    using Base = DynArr<uint32_t, MaxDimensions>;

    VpuDims() = default;
    VpuDims(size_t size) : Base(size) {}
    VpuDims(std::initializer_list<uint32_t> l) : Base(l) {}
    VpuDims(const VpuDims&) = default;
    VpuDims(VpuDims&&) = default;
    VpuDims& operator=(const VpuDims&) = default;
    VpuDims& operator=(VpuDims&&) = default;

    uint32_t totalSize() const {
        uint32_t total = 1;
        for (size_t i = 0; i < size; ++i)
            total *= data[i];
        return total;
    }
};

VpuDims ieDimsToVpu(const SizeVector& ieDims);

VpuDims calcDataOffset(int axis, const VpuDims& dims);

struct VpuStrides : DynArr<uint32_t, MaxDimensions> {
    using Base = DynArr<uint32_t, MaxDimensions>;

    VpuStrides() = default;
    VpuStrides(size_t size) : Base(size) {}
    VpuStrides(std::initializer_list<uint32_t> l) : Base(l) {}
    VpuStrides(const VpuStrides&) = default;
    VpuStrides(VpuStrides&&) = default;
    VpuStrides& operator=(const VpuStrides&) = default;
    VpuStrides& operator=(VpuStrides&&) = default;
};

VpuStrides calcStrides(const VpuDims& dims, VpuDataType type, t_MvTensorStorageOrder order, uint32_t align2nd = 1);

template <typename T>
void kchw_to_hwck(const T* src, T* dst, const VpuDims& dims) {
    assert(dims.count() >= 3);
    for (uint32_t x = 0; x < dims[Dim::X]; ++x) {
        for (uint32_t y = 0; y < dims[Dim::Y]; ++y) {
            for (uint32_t z = 0; z < dims[Dim::Z]; ++z) {
                auto input  = x + dims[Dim::X] * y + dims[Dim::X] * dims[Dim::Y] * z;
                auto output = z + dims[Dim::Z] * y + dims[Dim::Z] * dims[Dim::Y] * x;
                dst[output] = src[input];
            }
        }
    }
}

template <typename T>
void kchw_to_khwc(const T* src, T* dst, const VpuDims& dims) {
    assert(dims.count() >= 3);
    for (uint32_t x = 0; x < dims[Dim::X]; ++x) {
        for (uint32_t y = 0; y < dims[Dim::Y]; ++y) {
            for (uint32_t z = 0; z < dims[Dim::Z]; ++z) {
                auto input  = x + dims[Dim::X] * y + dims[Dim::X] * dims[Dim::Y] * z;
                auto output = y + dims[Dim::Y] * x + dims[Dim::Y] * dims[Dim::X] * z;
                dst[output] = src[input];
            }
        }
    }
}

template <typename T>
void kchw_to_hwkc(const T* src, T* dst, const VpuDims& dims) {
    assert(dims.count() >= 3);
    for (uint32_t x = 0; x < dims[Dim::X]; ++x) {
        for (uint32_t y = 0; y < dims[Dim::Y]; ++y) {
            for (uint32_t z = 0; z < dims[Dim::Z]; ++z) {
                auto input  = x + dims[Dim::X] * y + dims[Dim::X] * dims[Dim::Y] * z;
                auto output = y + dims[Dim::Y] * z + dims[Dim::Z] * dims[Dim::Y] * x;
                dst[output] = src[input];
            }
        }
    }
}

//
// Non owning pointer (based on weak_ptr) with overloaded operators `*` and `->`.
// It assumes that object is alive and is used in single-threaded environment.
//

template <typename T>
class Handle {
public:
    Handle() = default;
    Handle(std::nullptr_t) {}
    Handle(const Handle&) = default;
    Handle(Handle&&) = default;
    Handle& operator=(const Handle&) = default;
    Handle& operator=(std::nullptr_t) { _ptr.reset(); return *this; }
    Handle& operator=(Handle&&) = default;

    T* get() const { return _ptr.lock().get(); }
    T& operator*() const { return *check(get()); }
    T* operator->() const { return check(get()); }

    template <typename U>
    Handle<U> staticCast() const {
        auto obj = _ptr.lock();
        return Handle<U>(std::static_pointer_cast<U>(obj));
    }

    template <typename U>
    Handle<U> dynamicCast() const {
        auto obj = _ptr.lock();
        return Handle<U>(std::dynamic_pointer_cast<U>(obj));
    }

public:
    Handle(const std::shared_ptr<T>& obj) : _ptr(obj) {
        assert(obj != nullptr);
    }

private:
    static T* check(T* val) {
        assert(val != nullptr);
        return val;
    }

private:
    std::weak_ptr<T> _ptr;
};

template <typename T>
bool operator==(const Handle<T>& first, const Handle<T>& second) {
    return first.get() == second.get();
}

template <typename T>
bool operator!=(const Handle<T>& first, const Handle<T>& second) {
    return first.get() != second.get();
}

template <typename T>
bool operator==(std::nullptr_t, const Handle<T>& h) {
    return h.get() == nullptr;
}

template <typename T>
bool operator==(const Handle<T>& h, std::nullptr_t) {
    return h.get() == nullptr;
}

template <typename T>
bool operator!=(std::nullptr_t, const Handle<T>& h) {
    return h.get() != nullptr;
}

template <typename T>
bool operator!=(const Handle<T>& h, std::nullptr_t) {
    return h.get() != nullptr;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Handle<T>& h) {
    return os << h.get();
}

template <typename T>
struct HandleHash {
    using BaseHash = std::hash<T*>;

    using result_type = typename BaseHash::result_type;
    using argument_type = Handle<T>;

    result_type operator()(const argument_type& handle) const {
        assert(handle != nullptr);
        return BaseHash()(handle.get());
    }
};

//
// Types pre-declaration
//

struct VpuData;
using VpuDataPtr = std::shared_ptr<VpuData>;
using VpuDataHandle = Handle<VpuData>;
using VpuDataHandleHash = HandleHash<VpuData>;

struct VpuStage;
using VpuStagePtr = std::shared_ptr<VpuStage>;
using VpuStageHandle = Handle<VpuStage>;
using VpuStageHandleHash = HandleHash<VpuStage>;

//
// Blob write helpers
//

struct BlobWriter {
    std::vector<char> stagesData;
    std::unordered_map<VpuData*, uint32_t> dataMap;

    template <typename T>
    void write(const T& params) {
        stagesData.insert(stagesData.end(),
                          reinterpret_cast<const char*>(&params),
                          reinterpret_cast<const char*>(&params) + sizeof(params));
    }
};

//
// VpuData
//

class DataWriter {
public:
    virtual ~DataWriter();

    // Returns required size in bytes in output buffer
    virtual size_t byteSize() const = 0;

    virtual void write(void* dst) const = 0;
};
using DataWriterPtr = std::shared_ptr<DataWriter>;

struct VpuData {
    std::string name;

    IndexCodes index = IndexNone;

    VpuDataType type = VpuDataType::FP16;
    t_MvTensorStorageOrder order = orderYXZ;
    VpuDims dims = VpuDims(3);
    VpuStrides strides = VpuStrides(3);

    uint32_t offset = 0;

    DataWriterPtr writer;

    VpuStageHandle producer;
    int producerOutInd = -1;
    std::unordered_set<VpuStageHandle, VpuStageHandleHash> consumers;

    VpuDims offsetFromParent = VpuDims(3);
    VpuDataHandle parent;
    std::unordered_set<VpuDataHandle, VpuDataHandleHash> subData;

    void dumpToDot(std::ostream& os);

    void dumpToBlob(BlobWriter& writer);

#ifdef AKS
    virtual ~VpuData();
#endif
};

VpuDataHandle getDataTopParent(const VpuDataHandle &data);

template <class Op>
void loopOverSubData(VpuDataHandle data, const Op& op) {
    for (auto& subData : data->subData) {
        assert(subData != nullptr);
        op(subData);
        loopOverSubData(subData, op);
    }
}

//
// VpuStage
//

struct VpuStage {
    std::string name;

    t_MvTensorOpType type = kNone0;
    CNNLayerPtr layer;

    uint32_t optMask = MV_TENSOR_DEFAULT_OPT;

    bool optimized = false;

    std::vector<VpuDataHandle> inputs;
    std::vector<VpuDataHandle> outputs;
    VpuDataHandle buffer;

    std::vector<t_MvTensorStorageOrder> requiredInputOrder;
    std::vector<size_t> requiredInputAlignment;
    std::vector<t_MvTensorStorageOrder> requiredOutputOrder;
    std::vector<size_t> requiredOutputAlignment;

    VpuStageHandle parentOp;
    VpuStageHandle postOp;

    virtual ~VpuStage();

    virtual void dumpToDot(std::ostream& os);

    virtual void dumpToBlob(BlobWriter& writer);
};

bool isHwStage(const VpuStageHandle& stage);

struct VpuConvertStage : VpuStage {
    float scale = 1.0f;
    float bias = 0.0f;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;

#ifdef AKS
    virtual ~VpuConvertStage();
#endif
};

struct VpuEltwiseStage : VpuStage {
    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
        virtual ~VpuEltwiseStage();
#endif
};

typedef VpuEltwiseStage VpuBiasStage;

struct VpuConvStage : VpuStage {
    uint32_t radixX = 0;
    uint32_t radixY = 0;
    uint32_t strideX = 0;
    uint32_t strideY = 0;
    uint32_t padX = 0;
    uint32_t padY = 0;
    uint32_t dilationX = 0;
    uint32_t dilationY = 0;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuConvStage();
#endif
};

struct VpuPowerStage : VpuStage {
    float offset = 0.0f;
    float scale = 0.0f;
    float power = 0.0f;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuPowerStage();
#endif
};

struct VpuCopyStage : VpuStage {
};

struct VpuPoolStage : VpuStage {
    uint32_t radixX = 0;
    uint32_t radixY = 0;
    uint32_t strideX = 0;
    uint32_t strideY = 0;
    uint32_t padX = 0;
    uint32_t padY = 0;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuPoolStage();
#endif
};

struct VpuReluStage : VpuStage {
    float negativeSlope;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuReluStage();
#endif
};

struct VpuLRNStage : VpuStage {
    uint32_t size;
    uint32_t k;
    float alpha;
    float beta;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuLRNStage();
#endif
};

struct VpuFullyConnectedStage : VpuStage {
    void dumpToBlob(BlobWriter& writer) override;
};

struct VpuSoftMaxStage : VpuStage {
    char axis;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuSoftMaxStage();
#endif
};

struct VpuScaleStage : VpuStage {
    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuScaleStage();
#endif
};

struct VpuScaleShiftStage : VpuStage {
    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuScaleShiftStage();
#endif
};

struct VpuPermuteStage : VpuStage {
    int32_t order0;
    int32_t order1;
    int32_t order2;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuPermuteStage();
#endif
};

struct VpuToPlaneMajorStage : VpuStage {
};

struct VpuReshapeStage : VpuStage {
};


PACKED(DetectionOutputParams {
    int32_t num_classes;
    int32_t share_location;
    int32_t background_label_id;
    float nms_threshold;
    int32_t top_k;
    int32_t code_type;
    int32_t keep_top_k;
    float confidence_threshold;
    int32_t variance_encoded_in_target;
    float eta;
    int32_t num_priors;
};)

struct VpuDetectionOutputStage : VpuStage {
    DetectionOutputParams params;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuDetectionOutputStage();
#endif
};

struct VpuSigmoidStage : VpuStage {
};

struct VpuTanhStage : VpuStage {
};

struct VpuPReluStage : VpuStage {
    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuPReluStage();
#endif
};

struct VpuEluStage : VpuStage {
    float alpha;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuEluStage();
#endif
};

struct VpuCropStage : VpuStage {
    int32_t offset[3] = {};

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuCropStage();
#endif
};

struct VpuTileStage : VpuStage {
    int32_t axis = 0;
    int32_t tiles = 0;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuTileStage();
#endif
};

struct VpuNormalizeStage : VpuStage {
    int32_t acrossSpatial = 0;
    int32_t channelShared = 0;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuNormalizeStage();
#endif
};

struct VpuRegionYoloStage : VpuStage {
    int32_t classes = 0;
    int32_t coords = 0;
    int32_t num = 0;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuRegionYoloStage();
#endif
};

struct VpuReorgYoloStage : VpuStage {
    int32_t stride = 0;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuReorgYoloStage();
#endif
};

struct VpuCTCDecoderStage : VpuStage {
    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuCTCDecoderStage();
#endif
};

struct VpuConvertHwSwStage : VpuStage {
};

struct HwPaddingInfo {
    bool enable = false;
    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t top = 0;
    uint32_t bottom = 0;
};

struct VpuMyriadXHwConvolutionStage : VpuStage {
    using Tiles = std::vector<std::tuple<uint32_t, cnnOperationMode>>;

    uint32_t radixX = 0;
    uint32_t radixY = 0;
    uint32_t stride = 0;

    HwPaddingInfo pad;

    uint32_t newInputDimZ = 0;
    uint32_t newOutputDimZ = 0;
    Tiles tiles;

    bool hasRelu = false;

    bool withPool = false;
    uint32_t poolRadX = 0;
    uint32_t poolRadY = 0;

    bool hasParallelCopy = false;

    std::vector<cnnDescriptor> descriptors;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuMyriadXHwConvolutionStage();
#endif
};

struct VpuMyriadXHwFullyConnectedStage : VpuStage {
    using SubTiles = std::vector<std::tuple<uint32_t, uint32_t, cnnOperationMode>>;
    using Tiles = std::vector<SubTiles>;

    uint32_t newInputDimZ = 0;
    uint32_t newOutputDimZ = 0;
    Tiles tiles;

    bool hasRelu = false;

    std::vector<cnnDescriptor> descriptors;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuMyriadXHwFullyConnectedStage();
#endif
};

struct VpuMyriadXHwPoolingStage : VpuStage {
    using Tiles = std::vector<std::tuple<uint32_t, cnnOperationMode>>;

    uint32_t radixX = 0;
    uint32_t radixY = 0;
    uint32_t stride = 0;

    cnnPoolType poolType = POOL_MAX;

    HwPaddingInfo pad;

    uint32_t newOutputDimZ = 0;
    Tiles tiles;

    bool hasRelu = false;

    bool hasParallelCopy = false;

    std::vector<cnnDescriptor> descriptors;

    void dumpToDot(std::ostream& os) override;

    void dumpToBlob(BlobWriter& writer) override;
#ifdef AKS
    virtual ~VpuMyriadXHwPoolingStage();
#endif
};

//
// GraphTransformerImpl
//

class GraphTransformerImpl : public IGraphTransformer {
public:
    GraphTransformerImpl(const BlobConfig& blobConfig,
                         const Common::LoggerPtr& log);

    void generate(ICNNNetwork& network,
                  std::vector<char>& blob,
                  std::vector<BlobMetaData>& metaData,
                  size_t& numStages) override;

public:
    void parseConvolution(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parsePooling(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseReLU(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseFullyConnected(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseSoftMax(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseNorm(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseConcat(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parsePower(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseScale(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parsePriorBoxClustered(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parsePermute(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseFlatten(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseReshape(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseDetectionOutput(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseEltwise(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseSplit(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseSigmoid(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseTanH(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parsePReLU(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseBatchNorm(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseDeconvolution(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseCopy(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseELU(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseCrop(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseTile(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseNormalize(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parsePriorBox(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseRegionYolo(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseReorgYolo(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseBias(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void parseCTCDecoder(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void checkBatchDefault(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void checkBatchPermute(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void checkBatchFC(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void checkBatchCTCDecoder(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);
    void checkBatchVPUDimensions(const CNNLayerPtr& layer, const std::vector<VpuDataHandle>& inputs, const std::vector<VpuDataHandle>& outputs);

public:
#ifndef NDEBUG
    void dumpInternalGraphToDot(const std::string& fileName) const;
#endif

private:
    void parseNetwork(ICNNNetwork& network);

    void parseInputAndOutputData();
    void addInputConvertStages();
    void addPreProcessStages();

    void generateStages();

    void addOutputConvertStages();

    void packPostOps();
    void addHWStages();
    void packHWConcat();
    void addConvertOrderStages();
    void eliminateCopyStages();
    void eliminateReshapeStages();
    void fillHWDescriptors();
    void packMemory();

    void finalize(std::vector<char>& blob);

    void getMetaData(std::vector<BlobMetaData>& metaData);

private:
    using DataId = const void*;

    VpuDataHandle getVpuData(const DataPtr& data);
    VpuDataHandle getVpuDataFP16(const DataPtr& data);

    void getInputAndOutputData(const CNNLayerPtr& layer,
                               std::vector<VpuDataHandle>& inputs,
                               std::vector<VpuDataHandle>& outputs);

    DataId dataId(const DataPtr& data);
    DataId dataId_FP16(const DataPtr& data);
    DataId newDataId();

    template <class Setter>
    VpuDataHandle addNewData(DataId dataId, const Setter& setter, const VpuDataHandle& parent = nullptr);

    template <class StageInfo, class Setter>
    VpuStageHandle addNewStage(const std::string& name,
                               t_MvTensorOpType type,
                               const CNNLayerPtr& layer,
                               const Setter& setter,
                               const std::vector<VpuDataHandle>& inputs,
                               const std::vector<VpuDataHandle>& outputs,
                               const VpuStageHandle& parentOp = nullptr,
                               const std::list<VpuStagePtr>::iterator* pos = nullptr);

    void addHWConv(const std::list<VpuStagePtr>::iterator& stageIt,
                   VpuDataHandle input,
                   VpuDataHandle output,
                   float scale,
                   Handle<VpuPoolStage> postPoolStage,
                   bool isLastTile = true,
                   const std::string& extraSuffix = "",
                   VpuDataHandle copyInput = nullptr,
                   VpuDataHandle copyOutput = nullptr);

    void processHWConv(const std::list<VpuStagePtr>::iterator& stageIt,
                       uint32_t cmxLimit,
                       bool isYoloNetwork,
                       bool isOriginalYolo);
    void processHWPool(const std::list<VpuStagePtr>::iterator& stageIt,
                       uint32_t cmxLimit);
    void processHWFC(const std::list<VpuStagePtr>::iterator& stageIt,
                     bool isYoloNetwork,
                     bool isOriginalYolo);

    Handle<VpuCopyStage> addCopyStage(const std::string& name,
                                      const CNNLayerPtr& layer,
                                      const VpuDataHandle& inputs,
                                      const VpuDataHandle& outputs,
                                      const std::list<VpuStagePtr>::iterator* pos = nullptr);

    Handle<VpuStage> addNoneStage(const std::string& name,
                                  const CNNLayerPtr& layer,
                                  const std::vector<VpuDataHandle>& inputs,
                                  const std::vector<VpuDataHandle>& outputs,
                                  const std::list<VpuStagePtr>::iterator* pos = nullptr);

    using PostOpInfo = std::tuple<VpuStageHandle, VpuDataHandle, std::string>;
    PostOpInfo getPostOpInfoForHW(const VpuStagePtr& mainStage);

    VpuDataHandle findOrCreateConvertedData(
            std::unordered_map<VpuDataHandle, std::list<VpuDataHandle>, VpuDataHandleHash>& convertedDataMap,
            const VpuDataHandle& orig,
            t_MvTensorStorageOrder reqOrder,
            size_t reqAlignment,
            const std::list<VpuStagePtr>::iterator& stageIt);
    VpuDataHandle addConvertedData(const VpuDataHandle& orig,
                                   t_MvTensorStorageOrder order,
                                   size_t alignment);
    VpuStageHandle addConvertStage(const std::list<VpuStagePtr>::iterator& stageIt,
                                   const VpuDataHandle& input,
                                   const VpuDataHandle& output);

    VpuDataHandle addAlignedData(const VpuDataHandle& orig,
                                 size_t alignment);
    VpuDataHandle reshapeZYXToYXZ(const VpuDataHandle& origData);
    VpuDataHandle reshapeYXZToZYX(const VpuDataHandle& origData);

private:
    BlobConfig _blobConfig;
    Common::LoggerPtr _log;

    std::string _networkName;
    InputsDataMap _networkInputs;
    OutputsDataMap _networkOutputs;
    std::list<CNNLayerPtr> _orderedLayers;

    std::list<VpuDataPtr> _datas;
    std::list<VpuStagePtr> _stages;

    std::unordered_map<DataId, VpuDataHandle> _vpuDatasById;
    std::unordered_map<DataPtr, DataId> _fp16Ids;
    std::list<std::unique_ptr<char>> _dataIds;

    uint32_t _blobTotalDataSize = 0;
    uint32_t _bssMemSize = 0;
};

typedef void (GraphTransformerImpl::*parser_t)(const CNNLayerPtr& layer,
                                               const std::vector<VpuDataHandle>& inputs,
                                               const std::vector<VpuDataHandle>& outputs);

typedef void (GraphTransformerImpl::*checkBatch_t)(const CNNLayerPtr& layer,
                                               const std::vector<VpuDataHandle>& inputs,
                                               const std::vector<VpuDataHandle>& outputs);
template <class Setter>
VpuDataHandle GraphTransformerImpl::addNewData(DataId dataId, const Setter& setter, const VpuDataHandle& parent) {
    auto data = std::make_shared<VpuData>();

    setter(data.get());

    if (parent != nullptr) {
        data->parent = parent;
        parent->subData.insert(data);
    }

    _datas.push_back(data);

    auto res = _vpuDatasById.insert({dataId, data});
    assert(res.second);

    return data;
}

template <class StageInfo, class Setter>
VpuStageHandle GraphTransformerImpl::addNewStage(const std::string& name,
                                                 t_MvTensorOpType type,
                                                 const CNNLayerPtr& layer,
                                                 const Setter& setter,
                                                 const std::vector<VpuDataHandle>& inputs,
                                                 const std::vector<VpuDataHandle>& outputs,
                                                 const VpuStageHandle& parentOp,
                                                 const std::list<VpuStagePtr>::iterator* pos) {
    auto stage = std::make_shared<StageInfo>();
    VpuStageHandle stageHandle = std::static_pointer_cast<VpuStage>(stage);

    stage->name = name;
    stage->type = type;
    stage->layer = layer;

    for (const auto& input : inputs) {
        stage->inputs.push_back(input);
        stage->requiredInputOrder.push_back(input->order);
        stage->requiredInputAlignment.push_back(1);
        input->consumers.insert(stageHandle);
    }

    int outInd = 0;
    for (const auto& output : outputs) {
        stage->outputs.push_back(output);
        stage->requiredOutputOrder.push_back(output->order);
        stage->requiredOutputAlignment.push_back(1);
        output->producer = stageHandle;
        output->producerOutInd = outInd++;
    }

    if (parentOp != nullptr) {
        assert(parentOp->postOp == nullptr);
        parentOp->postOp = stageHandle;
        stage->parentOp = parentOp;
    }

    // FIXME: Set orderYXZ as default for input and output
    stage->requiredInputOrder[0] = orderYXZ;
    stage->requiredOutputOrder[0] = orderYXZ;

    setter(stage.get());

    if (pos) {
        _stages.insert(*pos, stage);
    } else {
        _stages.push_back(stage);
    }

    return stageHandle;
}

class DefaultWeightsWriter : public DataWriter {
public:
    DefaultWeightsWriter(const VpuDims& dims, const Blob::Ptr& blob) : _dims(dims), _blob(blob) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override;

    void write(void* dst) const override;

private:
    VpuDims _dims;
    Blob::Ptr _blob;
};

class DefaultBiasesWriter : public DataWriter {
public:
    explicit DefaultBiasesWriter(const Blob::Ptr& blob) : _blob(blob) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override;

    void write(void* dst) const override;

private:
    Blob::Ptr _blob;
};

class ScaleWeightsWriter : public DataWriter {
public:
    ScaleWeightsWriter(float scale, uint32_t count)
        : _scale(scale), _count(count) {
    }

    size_t byteSize() const override;

    void write(void* dst) const override;

private:
    float _scale;
    uint32_t _count;
};
