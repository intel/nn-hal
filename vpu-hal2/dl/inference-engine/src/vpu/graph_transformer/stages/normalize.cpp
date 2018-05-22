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
#include <vector>
#include <memory>

void VpuNormalizeStage::dumpToDot(std::ostream& os) {
    os << "acrossSpatial=" << acrossSpatial << "\\n"
       << "channelShared=" << channelShared;
}

void VpuNormalizeStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<int32_t>(acrossSpatial));
    writer.write(static_cast<int32_t>(channelShared));

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
    inputs[1]->dumpToBlob(writer);
}

namespace {

class NormalizeScalesWriter : public DataWriter {
public:
    NormalizeScalesWriter(const Blob::Ptr& weights, uint32_t C, bool channelShared)
        : _weights(weights), _C(C), _channelShared(channelShared) {
        assert(_weights != nullptr);
    }

    size_t byteSize() const override {
        return _C * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto dstPtr = static_cast<ie_fp16*>(dst);
        auto srcPtr = _weights->cbuffer().as<const ie_fp16*>();

        if (_channelShared) {
            for (int i = 0; i < _C; ++i) {
                dstPtr[i] = srcPtr[0];
            }
        } else {
            std::copy_n(srcPtr, _C, dstPtr);
        }
    }

private:
    Blob::Ptr _weights;
    uint32_t _C;
    bool _channelShared;
};

}  // namespace

void GraphTransformerImpl::parseNormalize(const CNNLayerPtr& layer,
                                          const std::vector<VpuDataHandle>& inputs,
                                          const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    // Current mvTensor Normalize function uses only the following parameters:
    //   - across_spatial (and only 0 value is supported)
    //   - channel_shared (and only 0 value is supported)
    auto acrossSpatial = layer->GetParamAsInt("across_spatial", 0);
    auto channelShared = layer->GetParamAsInt("channel_shared", 0);
    if (acrossSpatial != 0) {
        THROW_IE_EXCEPTION << "[VPU] Normalize layer supports across_spatial=0 only. Layer name is: " << layer->name;
    }

    auto weightsIt = layer->blobs.find("weights");
    if (weightsIt == layer->blobs.end()) {
        THROW_IE_EXCEPTION << "[VPU] Missing weights for " << layer->name << " layer";
    }

    auto weights = weightsIt->second;
    if (weights->precision() != Precision::FP16) {
        THROW_IE_EXCEPTION << "[VPU] Invalid precision for weights in " << layer->name << " layer";
    }

    auto output = outputs[0];

    auto scales = addNewData(
        newDataId(),
        [layer, output, weights, channelShared, this](VpuData* data) {
            data->name = layer->name + "@scales";
            data->index = IndexBlob;
            data->type = VpuDataType::FP16;
            data->dims = VpuDims({1, 1, output->dims[Dim::Z]});
            data->strides = calcStrides(data->dims, data->type, data->order);
            data->writer = std::make_shared<NormalizeScalesWriter>(weights, data->dims[Dim::Z], channelShared);
        });

    addNewStage<VpuNormalizeStage>(
        layer->name,
        kNormalize,
        layer,
        [acrossSpatial](VpuNormalizeStage* stage) {
            stage->acrossSpatial = acrossSpatial;
            // we emulate channel_shared=1 by duplicating weights in NormalizeWeightsWriter
            stage->channelShared = 0;
        },
        {inputs[0], scales},
        {outputs[0]});
}
