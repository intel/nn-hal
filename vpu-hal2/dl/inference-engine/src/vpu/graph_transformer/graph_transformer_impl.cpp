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

#include "graph_transformer_impl.hpp"
#include <cassert>
#include <string>
#include <memory>
#include <list>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <algorithm>
#include <precision_utils.h>
#include <caseless.hpp>

#ifdef NNLOG
#include <android/log.h>
#include <cutils/log.h>
#endif

std::string mvTensorOpTypeToStr(t_MvTensorOpType type) {
    switch (type) {
    case kConv:
        return "Convolution";
    case kMaxPool:
        return "Pooling: Max";
    case kAvgPool:
        return "Pooling: Average";
    case kSoftMax:
        return "SoftMax";
    case kFC:
        return "FullyConnected";
    case kNone0:
        return "None";
    case kRelu:
        return "ReLU";
    case kReluX:
        return "ReLUX";
    case kDepthConv:
        return "DepthConvolution";
    case kBias:
        return "Bias";
    case kPRelu:
        return "PReLU";
    case kLRN:
        return "LRN";
    case kSum:
        return "Eltwise: Sum";
    case kProd:
        return "Eltwise: Prod";
    case kMax:
        return "Eltwise: Max";
    case kScale:
        return "Scale";
    case kRelayout:
        return "Relayout";
    case kSquare:
        return "Square";
    case kInnerLRN:
        return "InnerLRN";
    case kCopy:
        return "Copy";
    case kSigmoid:
        return "Sigmoid";
    case kTanh:
        return "Tanh";
    case kDeconvolution:
        return "Deconvolution";
    case kElu:
        return "Elu";
    case kReshape:
        return "Reshape";
    case kToPlaneMajor:
        return "ToPlaneMajor";
    case kPower:
        return "Power";
    case kCrop:
        return "Crop";
    case kTile:
        return "Tile";
    case kPermute:
        return "Permute";
    case kNormalize:
        return "Normalize";
    case kPriorBox:
        return "PriorBox";
    case kDetectionOutput:
        return "DetectionOutput";
    case kRegionYolo:
        return "RegionYolo";
    case kReorgYolo:
        return "ReorgYolo";
    case kConvert_u8f16:
        return "ConvertU8FP16";
    case kConvert_f32f16:
        return "ConvertF32FP16";
    case kConvert_f16f32:
        return "ConvertFP16FP32";
    case kConvertOrder:
        return "ConvertOrder";
    case kMyriadXHwConvolution:
        return "MyriadXHwConvolution";
    case kMyriadXHwPooling:
        return "MyriadXHwPooling";
    case kMyriadXHwFCL:
        return "MyriadXHwFCL";
    case kMyriadXHwPostOps:
        return "MyriadXHwPostOps";
    case kCTCDecoder:
        return "CTCDecoder";
    case kLeakyRelu:
        return "LeakyRelu";
    case kBiasRelu:
        return "BiasRelu";
    case kBiasLeakyRelu:
        return "BiasLeakyRelu";
    case kScaleShift:
        return "ScaleShift";
    case kCopyMakeBorderCHW:
        return "CopyMakeBorderCHW";
    case kIm2ColConvolution:
        return "Im2ColConvolution";
    case kCHWBiasRelu:
        return "CHWBiasRelu";
    case kCHWBiasLeakyRelu:
        return "CHWBiasLeakyRelu";
    case kCHWBias:
        return "CHWBias";
    case kCHWScale:
        return "CHWScale";
    case kCHWScaleShift:
        return "CHWScaleShift";
    case kCHWPower:
        return "CHWPower";
    case kHwFcRelayout:
        return "HwFcRelayout";
    default:
        return "Unrecognized";
    }
}

std::string mvTensorStorageOrderToStr(t_MvTensorStorageOrder order) {
    switch (order) {
    case orderYXZ:
        return "YXZ";
    case orderZYX:
        return "ZYX";
    case orderYZX:
        return "YZX";
    case orderXYZ:
        return "XYZ";
    case orderXZY:
        return "XZY";
    default:
        return "Unrecognized";
    }
}

std::string mvDataIndexToStr(IndexCodes index) {
    switch (index) {
    case IndexNone:
        return "None";
    case IndexInput:
        return "Input";
    case IndexOutput:
        return "Output";
    case IndexBlob:
        return "Blob";
    case IndexBSS:
        return "BSS";
    case IndexCMX:
        return "CMX";
    default:
        return "Unrecognized";
    }
}

std::string dataTypeToStr(VpuDataType type) {
    switch (type) {
    case VpuDataType::U8:
        return "U8";
    case VpuDataType::FP16:
        return "FP16";
    case VpuDataType::FP32:
        return "FP32";
    default:
        return "Unrecognized";
    }
}

uint32_t getDataTypeSize(VpuDataType type) {
    switch (type) {
    case VpuDataType::U8:
        return sizeof(uint8_t);
    case VpuDataType::FP16:
        return sizeof(ie_fp16);
    case VpuDataType::FP32:
        return sizeof(float);
    default:
        THROW_IE_EXCEPTION << "[VPU] Unknown data type";
    }
}

VpuDataType iePrecisionToVpu(const Precision& precision) {
    switch (precision) {
    case Precision::U8:
        return VpuDataType::U8;
    case Precision::FP16:
        return VpuDataType::FP16;
    case Precision::FP32:
        return VpuDataType::FP32;
    default:
        THROW_IE_EXCEPTION << "[VPU] Unsupported precision " << precision.name();
    }
}

VpuDims ieDimsToVpu(const SizeVector& ieDims) {
    VpuDims vpuDims(4);

    if (ieDims.size() == 2) {
        vpuDims[Dim::X] = 1;
        vpuDims[Dim::Y] = ieDims[0];
        vpuDims[Dim::Z] = ieDims[1];
        vpuDims[Dim::N] = 1;
    } else if (ieDims.size() == 3) {
        vpuDims[Dim::X] = ieDims[2];
        vpuDims[Dim::Y] = ieDims[1];
        vpuDims[Dim::Z] = ieDims[0];
        vpuDims[Dim::N] = 1;
    } else if ((ieDims.size() == 4)) {
        vpuDims[Dim::X] = ieDims[3];
        vpuDims[Dim::Y] = ieDims[2];
        vpuDims[Dim::Z] = ieDims[1];
        vpuDims[Dim::N] = ieDims[0];
    } else {
        THROW_IE_EXCEPTION << "[VPU] Unsupported dimensions";
    }

    return vpuDims;
}

VpuStrides calcStrides(const VpuDims& dims, VpuDataType type, t_MvTensorStorageOrder order, uint32_t align2nd) {
    assert(dims.count() >= 3 && dims.count() < 5);
    assert(align2nd > 0);

    VpuStrides strides(3);

    auto elemSize = getDataTypeSize(type);

    if (order == orderYXZ) {
        strides[Dim::Z] = elemSize;
        strides[Dim::X] = alignVal(strides[Dim::Z] * dims[Dim::Z], align2nd);
        strides[Dim::Y] = strides[Dim::X] * dims[Dim::X];
    } else if (order == orderXYZ) {
        strides[Dim::Z] = elemSize;
        strides[Dim::Y] = alignVal(strides[Dim::Z] * dims[Dim::Z], align2nd);
        strides[Dim::X] = strides[Dim::Y] * dims[Dim::Y];
    } else if (order == orderZYX) {
        strides[Dim::X] = elemSize;
        strides[Dim::Y] = alignVal(strides[Dim::X] * dims[Dim::X], align2nd);
        strides[Dim::Z] = strides[Dim::Y] * dims[Dim::Y];
    } else {
        THROW_IE_EXCEPTION << "[VPU] Unsupported storage order " << mvTensorStorageOrderToStr(order);
    }

    return strides;
}

VpuDims calcDataOffset(int axis, const VpuDims& dims) {
    VpuDims res(3);
    switch (axis) {
        case 0:
            res[Dim::X] = dims[Dim::X];
            break;
        case 1:
            res[Dim::Y] = dims[Dim::Y];
            break;
        case 2:
            res[Dim::Z] = dims[Dim::Z];
            break;
        default:
            THROW_IE_EXCEPTION << "[VPU] Unsupported axis " << axis;
    }
    return res;
}

DataWriter::~DataWriter() {
}

void VpuData::dumpToDot(std::ostream& os) {
    const char* color = "white";
    if (index == IndexInput)
        color = "green";
    else if (index == IndexOutput)
        color = "deepskyblue";
    else if (index == IndexBlob)
        color = "aquamarine";
    else if (index == IndexBSS)
        color = "cyan";
    else if (index == IndexCMX)
        color = "magenta";

    os << "    "
       << "data_" << static_cast<const void*>(this)
       << " [shape=box"
       << " style=filled"
       << " fillcolor=" << color
       << " label=\""
           << name << "\\n"
           << "index=" << mvDataIndexToStr(index) << "\\n"
           << "type=" << dataTypeToStr(type) << "\\n"
           << "order=" << mvTensorStorageOrderToStr(order) << "\\n"
           << "dims=" << dims << "\\n"
           << "strides=" << strides << "\\n"
           << "offset=" << offset << "\\n"
           << "offsetFromParent=" << offsetFromParent << "\\n"
           << "writer=" << (writer == nullptr ? "<none>" : typeid(*writer).name())
       << "\"];" << std::endl;
}

void VpuData::dumpToBlob(BlobWriter& writer) {
    enum {
        t_fp16,                  ///< half precision floating point
        t_u8f,                   ///< Unsigned byte
        t_int,                   ///< Integer
        t_fp32                   ///< single precision floating point
    };

    uint32_t data_type = 0;
    switch (type) {
    case VpuDataType::U8:
        data_type = t_u8f;
        break;
    case VpuDataType::FP16:
        data_type = t_fp16;
        break;
    case VpuDataType::FP32:
        data_type = t_fp32;
        break;
    }

    uint32_t reloc = (index == IndexBlob || index == IndexBSS || index == IndexCMX ? writer.dataMap[this] : offset);

    writer.write(static_cast<uint32_t>(dims[Dim::X]));
    writer.write(static_cast<uint32_t>(dims[Dim::Y]));
    writer.write(static_cast<uint32_t>(dims[Dim::Z]));
    writer.write(static_cast<uint32_t>(strides[Dim::X]));
    writer.write(static_cast<uint32_t>(strides[Dim::Y]));
    writer.write(static_cast<uint32_t>(strides[Dim::Z]));
    writer.write(static_cast<uint32_t>(reloc));
    writer.write(static_cast<uint32_t>(index));
    writer.write(static_cast<uint32_t>(data_type));
    writer.write(static_cast<uint32_t>(order));
}

VpuDataHandle getDataTopParent(const VpuDataHandle &data) {
    return data->parent != nullptr ? getDataTopParent(data->parent) : data;
}

VpuStage::~VpuStage() {
}

void VpuStage::dumpToDot(std::ostream& /*os*/) {
}

void VpuStage::dumpToBlob(BlobWriter& writer) {
    if (!inputs.empty())
        inputs[0]->dumpToBlob(writer);

    if (!outputs.empty())
        outputs[0]->dumpToBlob(writer);
}

bool isHwStage(const VpuStageHandle& stage) {
    return stage->type == kMyriadXHwConvolution ||
           stage->type == kMyriadXHwFCL ||
           stage->type == kMyriadXHwPooling;
}

GraphTransformerImpl::GraphTransformerImpl(const BlobConfig &blobConfig,
                                           const Common::LoggerPtr &log)
    : _blobConfig(blobConfig), _log(log) {
}

void GraphTransformerImpl::checkBatchDefault(const CNNLayerPtr& layer,
                                               const std::vector<VpuDataHandle>& inputs,
                                               const std::vector<VpuDataHandle>& outputs) {
    assert(layer != nullptr);

    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto layerInput = layer->insData[i].lock();
        assert(layerInput != nullptr);

        Layout layout = layerInput->getTensorDesc().getLayout();
        if ((layout == InferenceEngine::CN) || (layout == InferenceEngine::NCHW) || (layout == InferenceEngine::NHWC) || (layout == InferenceEngine::NC)) {
            auto dims = layerInput->getDims();

            if ((dims[1] != 1) && (layout == InferenceEngine::CN)) {
                THROW_IE_EXCEPTION << "[VPU] input " << layer->name << " has invalid batch";
            } else if ((dims[0] != 1) && (layout != InferenceEngine::CN)) {
                THROW_IE_EXCEPTION << "[VPU] input " << layer->name << " has invalid batch";
            }
        }
    }

    for (size_t i = 0; i < layer->outData.size(); ++i) {
        auto layerOutput = layer->outData[i];
        assert(layerOutput != nullptr);

        Layout layout = layerOutput->getTensorDesc().getLayout();
        if ((layout == InferenceEngine::CN) || (layout == InferenceEngine::NCHW) || (layout == InferenceEngine::NHWC) || (layout == InferenceEngine::NC)) {
            auto dims = layerOutput->getDims();

            if ((dims[1] != 1) && (layout == InferenceEngine::CN)) {
                THROW_IE_EXCEPTION << "[VPU] output " << layer->name << " has invalid batch";
            } else if ((dims[0] != 1) && (layout != InferenceEngine::CN)) {
                THROW_IE_EXCEPTION << "[VPU] output " << layer->name << " has invalid batch";
            }
        }
    }
}

void GraphTransformerImpl::checkBatchVPUDimensions(const CNNLayerPtr& layer,
                                               const std::vector<VpuDataHandle>& inputs,
                                               const std::vector<VpuDataHandle>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    if ((input->dims[Dim::N] != 1) || (output->dims[Dim::N] != 1)) {
         THROW_IE_EXCEPTION << "[VPU] Reshape input or output " << layer->name << " has invalid batch";
    }
}


namespace {

typedef void (GraphTransformerImpl::*parser_t)(const CNNLayerPtr& layer,
                                               const std::vector<VpuDataHandle>& inputs,
                                               const std::vector<VpuDataHandle>& outputs);

typedef void (GraphTransformerImpl::*checkBatch_t)(const CNNLayerPtr& layer,
                                               const std::vector<VpuDataHandle>& inputs,
                                               const std::vector<VpuDataHandle>& outputs);

struct parsers {
    parser_t parser;
    checkBatch_t checkBatch;
};

caseless_map<std::string, parsers> g_parsers = {
    {"Convolution",        {&GraphTransformerImpl::parseConvolution, &GraphTransformerImpl::checkBatchDefault}},
    {"Pooling",            {&GraphTransformerImpl::parsePooling, &GraphTransformerImpl::checkBatchDefault}},
    {"ReLU",               {&GraphTransformerImpl::parseReLU, &GraphTransformerImpl::checkBatchDefault}},
    {"FullyConnected",     {&GraphTransformerImpl::parseFullyConnected, &GraphTransformerImpl::checkBatchFC}},
    {"SoftMax",            {&GraphTransformerImpl::parseSoftMax, &GraphTransformerImpl::checkBatchVPUDimensions}},
    {"Norm",               {&GraphTransformerImpl::parseNorm, &GraphTransformerImpl::checkBatchDefault}},
    {"Concat",             {&GraphTransformerImpl::parseConcat, &GraphTransformerImpl::checkBatchVPUDimensions}},
    {"Eltwise",            {&GraphTransformerImpl::parseEltwise, &GraphTransformerImpl::checkBatchDefault}},
    {"Split",              {&GraphTransformerImpl::parseSplit, &GraphTransformerImpl::checkBatchVPUDimensions}},
    {"Sigmoid",            {&GraphTransformerImpl::parseSigmoid, &GraphTransformerImpl::checkBatchDefault}},
    {"TanH",               {&GraphTransformerImpl::parseTanH, &GraphTransformerImpl::checkBatchDefault}},
    {"PReLU",              {&GraphTransformerImpl::parsePReLU, &GraphTransformerImpl::checkBatchDefault}},
    {"Bias",               {&GraphTransformerImpl::parseBias, &GraphTransformerImpl::checkBatchDefault}},
    // Slice is transformed to Split by IE
    {"Slice",              {&GraphTransformerImpl::parseSplit, &GraphTransformerImpl::checkBatchVPUDimensions}},
    {"BatchNormalization", {&GraphTransformerImpl::parseBatchNorm, &GraphTransformerImpl::checkBatchDefault}},
    {"ScaleShift",         {&GraphTransformerImpl::parseScale, &GraphTransformerImpl::checkBatchDefault}},
    {"Deconvolution",      {&GraphTransformerImpl::parseDeconvolution, &GraphTransformerImpl::checkBatchDefault}},
    {"Power",              {&GraphTransformerImpl::parsePower, &GraphTransformerImpl::checkBatchDefault}},
    {"Copy",               {&GraphTransformerImpl::parseCopy, &GraphTransformerImpl::checkBatchDefault}},
    {"Reshape",            {&GraphTransformerImpl::parseReshape, &GraphTransformerImpl::checkBatchVPUDimensions}},
    {"ELU",                {&GraphTransformerImpl::parseELU, &GraphTransformerImpl::checkBatchDefault}},
    {"Flatten",            {&GraphTransformerImpl::parseFlatten, &GraphTransformerImpl::checkBatchVPUDimensions}},
    {"Crop",               {&GraphTransformerImpl::parseCrop, &GraphTransformerImpl::checkBatchDefault}},
    {"Tile",               {&GraphTransformerImpl::parseTile, &GraphTransformerImpl::checkBatchVPUDimensions}},
    {"Normalize",          {&GraphTransformerImpl::parseNormalize, &GraphTransformerImpl::checkBatchDefault}},
    {"PriorBox",           {&GraphTransformerImpl::parsePriorBox, &GraphTransformerImpl::checkBatchDefault}},
    {"PriorBoxClustered",  {&GraphTransformerImpl::parsePriorBoxClustered, &GraphTransformerImpl::checkBatchDefault}},
    {"Permute",            {&GraphTransformerImpl::parsePermute, &GraphTransformerImpl::checkBatchPermute}},
    {"DetectionOutput",    {&GraphTransformerImpl::parseDetectionOutput, &GraphTransformerImpl::checkBatchDefault}},
    {"RegionYolo",         {&GraphTransformerImpl::parseRegionYolo, &GraphTransformerImpl::checkBatchDefault}},
    {"ReorgYolo",          {&GraphTransformerImpl::parseReorgYolo, &GraphTransformerImpl::checkBatchDefault}},
    {"CTCGreedyDecoder",   {&GraphTransformerImpl::parseCTCDecoder, &GraphTransformerImpl::checkBatchCTCDecoder}},
};

#ifndef NDEBUG
class InternalGraphToDotAuto {
public:
    explicit InternalGraphToDotAuto(const GraphTransformerImpl* transformer)
        : _transformer(transformer) {
    }

    ~InternalGraphToDotAuto() {
        if (auto dumpFileName = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_FILE_NAME")) {
            _transformer->dumpInternalGraphToDot(dumpFileName);
        }
    }

private:
    const GraphTransformerImpl* _transformer;
};
#endif

}  // namespace

void GraphTransformerImpl::generate(ICNNNetwork& network,
                                    std::vector<char>& blob,
                                    std::vector<BlobMetaData>& metaData,
                                    size_t& numStages) {
#ifndef NDEBUG
    InternalGraphToDotAuto autoDumper(this);
    (void)autoDumper;
#endif

    parseNetwork(network);

    parseInputAndOutputData();
    addInputConvertStages();
    addPreProcessStages();

    generateStages();

    addOutputConvertStages();

    packPostOps();
    // this optimization must be before addConvertOrderStages();
    // because it can wrap reshape with additional convert order stages
    if (_blobConfig.reshapeOptimization) {
        eliminateReshapeStages();
    }
    if (_blobConfig.hwOptimization) {
        addHWStages();
        if (_blobConfig.copyOptimization) {
            packHWConcat();
        }
    }
    addConvertOrderStages();
    if (_blobConfig.copyOptimization) {
        eliminateCopyStages();
    }
    if (_blobConfig.hwOptimization) {
        fillHWDescriptors();
    }
    packMemory();

    finalize(blob);

#ifndef NDEBUG
    if (auto dumpFileName = std::getenv("IE_VPU_DUMP_BLOB_FILE_NAME")) {
        std::ofstream file(dumpFileName, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open()) {
            THROW_IE_EXCEPTION << "[VPU] Cannot open file " << dumpFileName << " for writing";
        }
        file.write(blob.data(), blob.size());
    }
#endif


    getMetaData(metaData);
    LOG_INFO("[VPU] GraphTransformer : getMetaData done");
    numStages = 0;
    for (const auto& stage : _stages) {
        if (!stage->optimized)
            ++numStages;
    }
}

void GraphTransformerImpl::generateStages() {
    for (const auto& layer : _orderedLayers) {
        assert(layer != nullptr);

        LOG_DEBUG("[VPU] GraphTransformer : try to parse layer %s", layer->name.c_str());
        #ifdef NNLOG
        ALOGI("[VPU] GraphTransformer : try to parse layer %s", layer->name.c_str());
        #endif


        std::vector<VpuDataHandle> stageInputs, stageOutputs;
        getInputAndOutputData(layer, stageInputs, stageOutputs);

        auto it = g_parsers.find(layer->type);
        if (it == g_parsers.end()) {
            if (_blobConfig.ignoreUnknownLayers) {
                addNoneStage(layer->name, layer, stageInputs, stageOutputs);
                continue;
            } else {
                THROW_IE_EXCEPTION << "Cannot convert layer \""
                                   << layer->name
                                   << "\" due to unsupported layer type \""
                                   << layer->type
                                   << "\"";
            }
        }

        auto checkBatchLayer = it->second.checkBatch;
        assert(checkBatchLayer != nullptr);

        (this->*checkBatchLayer)(layer, stageInputs, stageOutputs);

        auto parser = it->second.parser;
        assert(parser != nullptr);

        if (auto layerWithWeights = std::dynamic_pointer_cast<WeightableLayer>(layer)) {
            if (layerWithWeights->_weights == nullptr) {
                THROW_IE_EXCEPTION << "[VPU] Missing weights for layer " << layerWithWeights->name;
            }
        }

        (this->*parser)(layer, stageInputs, stageOutputs);
    }
}

#ifndef NDEBUG

namespace {

void addAllConnectedData(const VpuDataHandle& data, std::unordered_set<VpuDataHandle, VpuDataHandleHash>& set) {
    auto topParent = getDataTopParent(data);

    set.insert(topParent);

    loopOverSubData(topParent, [&set](VpuDataHandle subData) {
        set.insert(subData);
    });
}

}  // namespace

void GraphTransformerImpl::dumpInternalGraphToDot(const std::string& fileName) const {
    std::ofstream file(fileName);
    if (!file.is_open()) {
        THROW_IE_EXCEPTION << "[VPU] Failed to open file " << fileName;
    }

    file << "digraph " << "ie_vpu_graph_" << static_cast<const void*>(this) << " {" << std::endl;
    file << "    labelloc=top;" << std::endl;
    file << "    labeljust=left;" << std::endl;
    file << "    label=\"" << _networkName << "\";" << std::endl;

    std::unordered_set<VpuDataHandle, VpuDataHandleHash> usedDatas;
    for (const auto& stage : _stages) {
        assert(stage != nullptr);

        if (stage->optimized)
            continue;

        for (const auto& in : stage->inputs)
            addAllConnectedData(in, usedDatas);

        for (const auto& out : stage->outputs)
            addAllConnectedData(out, usedDatas);
    }

    for (const auto& data : usedDatas) {
        assert(data != nullptr);

        data->dumpToDot(file);
    }

    for (const auto& data : usedDatas) {
        if (data->parent != nullptr) {
            bool hasProducer = data->producer != nullptr && !data->producer->optimized;
            if (!hasProducer) {
                // check all subData
                loopOverSubData(data, [&hasProducer](VpuDataHandle subData) {
                    if (subData->producer != nullptr && !subData->producer->optimized)
                        hasProducer = true;
                });
            }

            if (hasProducer) {
                file << "    "
                     << "data_" << static_cast<const void*>(data.get()) << "->"
                     << "data_" << static_cast<const void*>(data->parent.get())
                     << ";" << std::endl;
            } else {
                file << "    "
                     << "data_" << static_cast<const void*>(data->parent.get()) << "->"
                     << "data_" << static_cast<const void*>(data.get())
                     << ";" << std::endl;
            }
        }
    }

    int stageIdx = 0;
    for (const auto& stage : _stages) {
        assert(stage != nullptr);

        if (stage->optimized)
            continue;

        file << "    "
             << "stage_" << static_cast<const void*>(stage.get())
             << " [shape=ellipse"
             << " style=filled"
             << " fillcolor=gold"
             << " label=\""
                 << "[" << stageIdx << "] " << stage->name << "\\n"
                 << "type=" << mvTensorOpTypeToStr(stage->type) << "\\n"
                 << "layer=" << (stage->layer == nullptr ? "<none>" : stage->layer->name.c_str()) << "\\n"
                 << "optMask=" << stage->optMask << "\\n"
                 << "optimized=" << stage->optimized << "\\n";
        stage->dumpToDot(file);
        file << "\"];" << std::endl;

        int inIndex = 0;
        for (const auto& input : stage->inputs) {
            file << "    "
                 << "data_" << static_cast<const void*>(input.get()) << "->"
                 << "stage_" << static_cast<const void*>(stage.get())
                 << " [label=\""
                    << "index=" << inIndex
                 << "\"];" << std::endl;

            ++inIndex;
        }

        int outIndex = 0;
        for (const auto& output : stage->outputs) {
            file << "    "
                 << "stage_" << static_cast<const void*>(stage.get()) << "->"
                 << "data_" << static_cast<const void*>(output.get())
                 << " [label=\""
                    << "index=" << outIndex
                 << "\"];" << std::endl;

            ++outIndex;
        }

        if (stage->buffer != nullptr) {
            stage->buffer->dumpToDot(file);

            file << "    "
                 << "data_" << static_cast<const void*>(stage->buffer.get()) << "->"
                 << "stage_" << static_cast<const void*>(stage.get())
                 << " [style=dotted];" << std::endl;
        }

        ++stageIdx;
    }

    for (const auto& stage : _stages) {
        assert(stage != nullptr);

        if (stage->optimized)
            continue;

        if (stage->postOp != nullptr && !stage->postOp->optimized) {
            file << "    "
                 << "stage_" << static_cast<const void*>(stage.get()) << "->"
                 << "stage_" << static_cast<const void*>(stage->postOp.get())
                 << " [style=dashed];" << std::endl;
        }
    }

    file << "}" << std::endl;
}
#endif

void GraphTransformerImpl::finalize(std::vector<char>& blob) {
    ElfN_Ehdr elfHdr = {};
    // TODO : what do this numbers mean?
    elfHdr.e_type = 1;
    elfHdr.e_machine = 2;
    elfHdr.e_version = 2;
    elfHdr.e_ehsize = 8 * sizeof(elfHdr);

    mv_blob_header blobHdr = {};
    blobHdr.magic_number = 8708;
    blobHdr.blob_ver_major = 2;
    blobHdr.blob_ver_minor = 1;
    // TODO : can only choose number of SHAVEs
    blobHdr.num_shaves = _blobConfig.lastShave - _blobConfig.firstShave + 1;
    blobHdr.bss_mem_size = _bssMemSize;

    auto dataSecOffset = alignVal(sizeof(elfHdr) + sizeof(blobHdr), size_t(16));
    assert(dataSecOffset % 16 == 0);

    auto dataSecPreFill = dataSecOffset - (sizeof(elfHdr) + sizeof(blobHdr));

    mv_buffer_section_header bufSecHdr = {};
    bufSecHdr.buffer_section_size = sizeof(bufSecHdr) + _blobTotalDataSize;

    std::vector<mv_reloc_info> blobBufRelocInfo;
    std::vector<mv_reloc_info> blobWorkRelocInfo;

    BlobWriter stagesWriter;

    uint32_t inputSize = 0;
    uint32_t outputSize = 0;
    for (const auto& data : _datas) {
        assert(data != nullptr);

        if (data->index == IndexBSS || data->index == IndexCMX) {
            stagesWriter.dataMap[data.get()] = blobWorkRelocInfo.size();
            mv_reloc_info info{data->offset, static_cast<uint32_t>(data->index)};
            blobWorkRelocInfo.push_back(info);
        } else if (data->index == IndexBlob) {
            stagesWriter.dataMap[data.get()] = blobBufRelocInfo.size();
            mv_reloc_info info{data->offset, static_cast<uint32_t>(data->index)};
            blobBufRelocInfo.push_back(info);
        } else if (data->index == IndexInput) {
            if (data->parent == nullptr) {
                inputSize += data->dims.totalSize() * getDataTypeSize(data->type);
            }
        } else if (data->index == IndexOutput) {
            if (data->parent == nullptr) {
                outputSize += data->dims.totalSize() * getDataTypeSize(data->type);
            }
        }
    }

    auto relocSecOffset = dataSecOffset + bufSecHdr.buffer_section_size;

    mv_relocation_section_header mvRelocSecHdr = {};
    mvRelocSecHdr.blob_buffer_reloc_size = blobBufRelocInfo.size() * sizeof(blobBufRelocInfo[0]);
    mvRelocSecHdr.work_buffer_reloc_size = blobWorkRelocInfo.size() * sizeof(blobWorkRelocInfo[0]);
    mvRelocSecHdr.relocation_buffer_size = sizeof(mvRelocSecHdr) + mvRelocSecHdr.blob_buffer_reloc_size + mvRelocSecHdr.work_buffer_reloc_size;
    mvRelocSecHdr.blob_buffer_reloc_offset = relocSecOffset + sizeof(mvRelocSecHdr);
    mvRelocSecHdr.work_buffer_reloc_offset = mvRelocSecHdr.blob_buffer_reloc_offset + mvRelocSecHdr.blob_buffer_reloc_size;

    auto stageSecOffset = relocSecOffset + mvRelocSecHdr.relocation_buffer_size;

    size_t numStages = 0;
    for (const auto& stage : _stages) {
        assert(stage != nullptr);

        if (stage->optimized)
            continue;

        ++numStages;
    }

    mv_stage_section_header stageSecHdr = {};
    stageSecHdr.stage_count = numStages;
    stageSecHdr.input_size = inputSize;
    stageSecHdr.output_size = outputSize;

    int stageIdx = 0;
    for (const auto& stage : _stages) {
        assert(stage != nullptr);

        if (stage->optimized)
            continue;

        auto isSkip = std::find(_blobConfig.NoneLayers.begin(),
                                _blobConfig.NoneLayers.end(),
                                mvTensorOpTypeToStr(stage->type))
                      != _blobConfig.NoneLayers.end();

        auto curSize = stagesWriter.stagesData.size();

        mv_stage_header stageHdr = {};
        if (isSkip) {
            stageHdr.stage_type = kNone0;
        } else {
            stageHdr.stage_type = stage->type;
        }
        stageHdr.implementation_flag = stage->optMask;
        stagesWriter.write(stageHdr);

        stage->dumpToBlob(stagesWriter);

        uint32_t pre_op_type = kNone0;
        stagesWriter.write(pre_op_type);

        uint32_t post_op_type = kNone0;
        stagesWriter.write(post_op_type);

        auto stageHdrPtr = reinterpret_cast<mv_stage_header*>(&stagesWriter.stagesData[curSize]);
        stageHdrPtr->next_stage = stageIdx == numStages - 1 ? 0 : stagesWriter.stagesData.size() + sizeof(stageSecHdr);

        ++stageIdx;
    }

    stageSecHdr.stage_section_size = sizeof(stageSecHdr) + stagesWriter.stagesData.size();

    blobHdr.stage_section_offset = stageSecOffset;
    blobHdr.buffer_section_offset = dataSecOffset;
    blobHdr.relocation_section_offset = relocSecOffset;
    blobHdr.file_size = stageSecOffset + stageSecHdr.stage_section_size;

    blob.clear();
    blob.resize(blobHdr.file_size, 0);

    size_t curBlobOffset = 0;

    std::copy_n(&elfHdr, 1, reinterpret_cast<ElfN_Ehdr*>(&blob[curBlobOffset]));
    curBlobOffset += sizeof(elfHdr);

    std::copy_n(&blobHdr, 1, reinterpret_cast<mv_blob_header*>(&blob[curBlobOffset]));
    curBlobOffset += sizeof(blobHdr);

    curBlobOffset += dataSecPreFill;

    std::copy_n(&bufSecHdr, 1, reinterpret_cast<mv_buffer_section_header*>(&blob[curBlobOffset]));
    curBlobOffset += sizeof(bufSecHdr);

    for (const auto& data : _datas) {
        assert(data != nullptr);

        if (data->index == IndexBlob) {
            if (data->writer != nullptr) {
                data->writer->write(&blob[curBlobOffset] + data->offset);
            }
        }
    }
    curBlobOffset += _blobTotalDataSize;

    std::copy_n(&mvRelocSecHdr, 1, reinterpret_cast<mv_relocation_section_header*>(&blob[curBlobOffset]));
    curBlobOffset += sizeof(mvRelocSecHdr);

    std::copy(blobBufRelocInfo.begin(), blobBufRelocInfo.end(), reinterpret_cast<mv_reloc_info*>(&blob[curBlobOffset]));
    curBlobOffset += blobBufRelocInfo.size() * sizeof(mv_reloc_info);

    std::copy(blobWorkRelocInfo.begin(), blobWorkRelocInfo.end(), reinterpret_cast<mv_reloc_info*>(&blob[curBlobOffset]));
    curBlobOffset += blobWorkRelocInfo.size() * sizeof(mv_reloc_info);

    std::copy_n(&stageSecHdr, 1, reinterpret_cast<mv_stage_section_header*>(&blob[curBlobOffset]));
    curBlobOffset += sizeof(stageSecHdr);

    std::copy(stagesWriter.stagesData.begin(), stagesWriter.stagesData.end(), &blob[curBlobOffset]);
    curBlobOffset += stagesWriter.stagesData.size();

    LOG_INFO("[VPU] GraphTransformer : blobSize=%u", static_cast<uint32_t>(sizeof(char) * blob.size()));
    #ifdef NNLOG
    ALOGI("[VPU] GraphTransformer : blobSize=%u", static_cast<uint32_t>(sizeof(char) * blob.size()));
    #endif
}

void GraphTransformerImpl::getMetaData(std::vector<BlobMetaData>& metaData) {
    metaData.clear();


    for (const auto& stage : _stages) {
        assert(stage != nullptr);

        BlobMetaData meta;
        meta.name = stage->name;
        meta.exec_type = mvTensorOpTypeToStr(stage->type);
        meta.layer_type = meta.exec_type;
        meta.status = stage->optimized ? InferenceEngineProfileInfo::OPTIMIZED_OUT
                                       : InferenceEngineProfileInfo::EXECUTED;
        metaData.push_back(meta);
    }

    // TODO : support config to disable timings
    // and not to add this meta if it is not required by user
    BlobMetaData receiveFathomTensorMeta;
    receiveFathomTensorMeta.exec_type = "Receive-Tensor";
    receiveFathomTensorMeta.layer_type = "Receive-Tensor";
    receiveFathomTensorMeta.name = "Receive-Tensor";
    receiveFathomTensorMeta.status = InferenceEngineProfileInfo::EXECUTED;
    metaData.push_back(receiveFathomTensorMeta);

    BlobMetaData loadUsbTransferMeta;
    loadUsbTransferMeta.exec_type = "Set-USB-Transfer";
    loadUsbTransferMeta.layer_type = "Set-USB-Transfer";
    loadUsbTransferMeta.name = "LoadInput";
    loadUsbTransferMeta.status = InferenceEngineProfileInfo::EXECUTED;
    metaData.push_back(loadUsbTransferMeta);

    BlobMetaData saveUsbTransferMeta;
    saveUsbTransferMeta.exec_type = "Get-USB-Transfer";
    saveUsbTransferMeta.layer_type = "Get-USB-Transfer";
    saveUsbTransferMeta.name = "GetOutput";
    saveUsbTransferMeta.status = InferenceEngineProfileInfo::EXECUTED;
    metaData.push_back(saveUsbTransferMeta);
}

VpuDataHandle GraphTransformerImpl::getVpuData(const DataPtr& ieData) {
    auto it = _vpuDatasById.find(dataId(ieData));

#ifdef NNLOG
    //ALOGI("GraphTransformerImpl::getVpuData");
#endif
    if (it == _vpuDatasById.end()) {
  #ifdef NNLOG
      ALOGI("[VPU] Can't find data ");
  #endif
        THROW_IE_EXCEPTION << "[VPU] Can't find data " << ieData->getName();
    }

    return it->second;
}

VpuDataHandle GraphTransformerImpl::getVpuDataFP16(const DataPtr& ieData) {
    auto dataIt = _vpuDatasById.find(dataId(ieData));
    if (dataIt == _vpuDatasById.end()) {
        THROW_IE_EXCEPTION << "[VPU] Can't find FP16 data " << ieData->getName();
    }

    auto vpuData = dataIt->second;
    assert(vpuData != nullptr);

    if (vpuData->type == VpuDataType::FP16)
        return vpuData;

    auto idIt = _fp16Ids.find(ieData);
    if (idIt == _fp16Ids.end()) {
        THROW_IE_EXCEPTION << "[VPU] Can't find FP16 data " << ieData->getName();
    }

    dataIt = _vpuDatasById.find(idIt->second);
    if (dataIt == _vpuDatasById.end()) {
        THROW_IE_EXCEPTION << "[VPU] Can't find FP16 data " << ieData->getName();
    }

    vpuData = dataIt->second;
    assert(vpuData != nullptr);

    if (vpuData->type != VpuDataType::FP16) {
        THROW_IE_EXCEPTION << "[VPU] data " << vpuData->name << " has invalid type";
    }

    return vpuData;
}

void GraphTransformerImpl::getInputAndOutputData(const CNNLayerPtr& layer,
                                               std::vector<VpuDataHandle>& inputs,
                                               std::vector<VpuDataHandle>& outputs) {
    assert(layer != nullptr);

    inputs.resize(layer->insData.size());
    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto layerInput = layer->insData[i].lock();
        assert(layerInput != nullptr);

        inputs[i] = getVpuDataFP16(layerInput);
    }

    outputs.resize(layer->outData.size());
    for (size_t i = 0; i < layer->outData.size(); ++i) {
        auto layerOutput = layer->outData[i];
        assert(layerOutput != nullptr);

        auto dataIt = _vpuDatasById.find(dataId(layerOutput));
        if (dataIt != _vpuDatasById.end()) {
            auto origData = dataIt->second;
            assert(origData != nullptr);

            if (origData->type == VpuDataType::FP16) {
                outputs[i] = origData;
            } else {
                outputs[i] = addNewData(
                    dataId_FP16(layerOutput),
                    [origData](VpuData* data) {
                        data->name = origData->name + "@FP16";
                        data->index = IndexBSS;
                        data->type = VpuDataType::FP16;
                        data->dims = origData->dims;
                        data->order = origData->order;
                        data->strides = calcStrides(data->dims, data->type, data->order);
                    });
            }
        } else {
            outputs[i] = addNewData(
                dataId(layerOutput),
                [layerOutput](VpuData* data) {
                    data->name = layerOutput->getName();
                    data->index = IndexBSS;
                    data->type = VpuDataType::FP16;
                    data->dims = ieDimsToVpu(layerOutput->getDims());
                    data->strides = calcStrides(data->dims, data->type, data->order);
                });
        }
    }
}

GraphTransformerImpl::DataId GraphTransformerImpl::dataId(const DataPtr& data) {
    return data.get();
}

GraphTransformerImpl::DataId GraphTransformerImpl::dataId_FP16(const DataPtr& data) {
    auto id = newDataId();
    auto res = _fp16Ids.insert({data, id});
    assert(res.second);
    return id;
}

GraphTransformerImpl::DataId GraphTransformerImpl::newDataId() {
    std::unique_ptr<char> idPtr(new char);
    auto id = idPtr.get();
    _dataIds.push_back(std::move(idPtr));
    return id;
}

size_t DefaultWeightsWriter::byteSize() const {
    return _blob->byteSize();
}

void DefaultWeightsWriter::write(void* dst) const {
    kchw_to_hwck(_blob->cbuffer().as<const ie_fp16*>(), static_cast<ie_fp16*>(dst), _dims);
}

size_t DefaultBiasesWriter::byteSize() const {
    return _blob->byteSize();
}

void DefaultBiasesWriter::write(void* dst) const {
    std::copy_n(_blob->cbuffer().as<const ie_fp16*>(), _blob->size(), static_cast<ie_fp16*>(dst));
}

size_t ScaleWeightsWriter::byteSize() const {
    return _count * sizeof(ie_fp16);
}

void ScaleWeightsWriter::write(void* dst) const {
    auto dstPtr = static_cast<ie_fp16*>(dst);
    for (uint32_t i = 0; i < _count; i++) {
        dstPtr[i] = PrecisionUtils::f32tof16(_scale);
    }
}

std::shared_ptr<IGraphTransformer> VPU::createGraphTransformer(const BlobConfig& blobConfig,
                                                               const Common::LoggerPtr& log) {
    return std::make_shared<GraphTransformerImpl>(blobConfig, log);
}

#ifdef AKS
    VpuData::~VpuData(){}
    VpuConvertStage::~VpuConvertStage(){}
    VpuEltwiseStage::~VpuEltwiseStage(){}
    VpuConvStage::~VpuConvStage(){}
    VpuPowerStage::~VpuPowerStage(){}
    VpuPoolStage::~VpuPoolStage(){}
    VpuReluStage::~VpuReluStage(){}
    VpuLRNStage::~VpuLRNStage(){}
    VpuSoftMaxStage::~VpuSoftMaxStage(){}
    VpuScaleStage::~VpuScaleStage(){}
    VpuScaleShiftStage::~VpuScaleShiftStage(){}
    VpuPermuteStage::~VpuPermuteStage(){}
    VpuDetectionOutputStage::~VpuDetectionOutputStage(){}
    VpuPReluStage::~VpuPReluStage(){}
    VpuEluStage::~VpuEluStage(){}
    VpuCropStage::~VpuCropStage(){}
    VpuTileStage::~VpuTileStage(){}
    VpuNormalizeStage::~VpuNormalizeStage(){}
    VpuRegionYoloStage::~VpuRegionYoloStage(){}
    VpuReorgYoloStage::~VpuReorgYoloStage(){}
    VpuCTCDecoderStage::~VpuCTCDecoderStage(){}
    VpuMyriadXHwConvolutionStage::~VpuMyriadXHwConvolutionStage(){}
    VpuMyriadXHwFullyConnectedStage::~VpuMyriadXHwFullyConnectedStage(){}
    VpuMyriadXHwPoolingStage::~VpuMyriadXHwPoolingStage(){}
#endif
