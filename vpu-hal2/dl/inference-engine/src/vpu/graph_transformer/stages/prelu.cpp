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

void VpuPReluStage::dumpToBlob(BlobWriter& writer) {
    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
    inputs[1]->dumpToBlob(writer);
}

namespace {

class PReLUWeightsWriter : public DataWriter {
public:
    PReLUWeightsWriter(const Blob::Ptr& blob, uint32_t count, bool channelShared)
        : _blob(blob), _count(count), _channelShared(channelShared) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _count * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto dstPtr = static_cast<ie_fp16*>(dst);
        auto srcPtr = _blob->cbuffer().as<const ie_fp16*>();

        if (_channelShared) {
            for (uint32_t i = 0; i < _count; ++i) {
                dstPtr[i] = srcPtr[0];
            }
        } else {
            std::copy_n(srcPtr, _count, dstPtr);
        }
    }

private:
    Blob::Ptr _blob;
    uint32_t _count;
    bool _channelShared;
};

}  // namespace

void GraphTransformerImpl::parsePReLU(const CNNLayerPtr& layer,
                                      const std::vector<VpuDataHandle>& inputs,
                                      const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    auto outDims = outputs[0]->dims;

    auto it = layer->blobs.find("weights");
    if (it == layer->blobs.end()) {
        THROW_IE_EXCEPTION << "[VPU] PReLU doesn't have weights";
    }
    auto weightsBlob = it->second;

    auto channelShared = layer->GetParamAsInt("channel_shared", 0);
    auto numValues = channelShared ? 1u : outDims[Dim::Z];

    if ((weightsBlob == nullptr) || (weightsBlob->size() != numValues)) {
        THROW_IE_EXCEPTION << "[VPU] PReLU weights size error";
    }

    auto weights = addNewData(
        newDataId(),
        [layer, outDims, weightsBlob, channelShared, this](VpuData* data) {
            data->name = layer->name + "@weights";
            data->index = IndexBlob;
            data->type = VpuDataType::FP16;
            data->order = orderXYZ;
            data->dims = VpuDims({1, 1, outDims[Dim::Z]});
            data->strides = calcStrides(data->dims, data->type, data->order);
            data->writer = std::make_shared<PReLUWeightsWriter>(weightsBlob, data->dims.totalSize(), channelShared);
        });

    addNewStage<VpuPReluStage>(
        layer->name,
        kPRelu,
        layer,
        [](VpuPReluStage* /*stage*/) {
        },
        {inputs[0], weights},
        {outputs[0]});
}
