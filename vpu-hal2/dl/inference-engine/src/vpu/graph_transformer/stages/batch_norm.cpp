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

namespace {

// The conversion of weights and biases for inference stage
// is required according to the arxiv.org/pdf/1502.03167.pdf.
// For details see Sub-clause 3.1, Stage 11 in Algorithm 2.

class BatchNormalizationWeightsWriter : public DataWriter {
public:
    BatchNormalizationWeightsWriter(const Blob::Ptr& blob, float epsilon) : _blob(blob), _epsilon(epsilon) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _blob->byteSize();
    }

    void write(void* dst) const override {
        auto dstPtr = static_cast<ie_fp16*>(dst);
        auto srcPtr = _blob->cbuffer().as<const ie_fp16*>();

        for (int i = 0; i < _blob->size(); ++i) {
            float val = PrecisionUtils::f16tof32(srcPtr[i]) + _epsilon;
            val = 1.0f / std::sqrt(val);
            dstPtr[i] = PrecisionUtils::f32tof16(val);
        }
    }

private:
    Blob::Ptr _blob;
    float _epsilon;
};

class BatchNormalizationBiasesWriter : public DataWriter {
public:
    BatchNormalizationBiasesWriter(const Blob::Ptr& biases, const Blob::Ptr& weights, float epsilon)
        : _biases(biases), _weights(weights), _epsilon(epsilon) {
        assert(biases != nullptr);
        assert(weights != nullptr);
    }

    size_t byteSize() const override {
        return _biases->byteSize();
    }

    void write(void* dst) const override {
        auto dstPtr = static_cast<ie_fp16*>(dst);
        auto bPtr = _biases->cbuffer().as<const ie_fp16*>();
        auto wPtr = _weights->cbuffer().as<const ie_fp16*>();

        // TODO : need to be extracted from IE layer.
        float beta = 0.0f;

        for (int i = 0; i < _biases->size(); ++i) {
            float val = PrecisionUtils::f16tof32(wPtr[i]) + _epsilon;
            val = 1.0f / std::sqrt(val);
            dstPtr[i] = PrecisionUtils::f32tof16(beta - val * PrecisionUtils::f16tof32(bPtr[i]));
        }
    }

private:
    Blob::Ptr _biases;
    Blob::Ptr _weights;
    float _epsilon;
};

}  // namespace

void GraphTransformerImpl::parseBatchNorm(const CNNLayerPtr& _layer,
                                          const std::vector<VpuDataHandle>& inputs,
                                          const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<BatchNormalizationLayer>(_layer);
    assert(layer != nullptr);

    VpuDims wDims({1, 1, inputs[0]->dims[Dim::Z]});

    assert(layer->_weights != nullptr);
    auto weights = addNewData(
        newDataId(),
        [layer, wDims, this](VpuData* data) {
            data->name = layer->name + "@weights";
            data->index = IndexBlob;
            data->type = VpuDataType::FP16;
            data->order = orderXYZ;
            data->dims = wDims;
            data->strides = calcStrides(data->dims, data->type, data->order);
            data->writer = std::make_shared<BatchNormalizationWeightsWriter>(layer->_weights, layer->epsilon);
        });

    auto scaleStage = addNewStage<VpuScaleStage>(
        layer->name,
        kScale,
        layer,
        [layer](VpuScaleStage* /*stage*/) {
        },
        {inputs[0], weights},
        {outputs[0]});

    if (layer->_biases != nullptr) {
        auto biases = addNewData(
            newDataId(),
            [layer, this](VpuData* data) {
                data->name = layer->name + "@biases";
                data->index = IndexBlob;
                data->type = VpuDataType::FP16;
                data->dims = VpuDims({static_cast<uint32_t>(layer->_biases->size()), 1, 1});
                data->strides = calcStrides(data->dims, data->type, data->order);
                data->writer = std::make_shared<BatchNormalizationBiasesWriter>(layer->_biases, layer->_weights, layer->epsilon);
            });

        addNewStage<VpuBiasStage>(
            biases->name,
            kBias,
            layer,
            [](VpuBiasStage* /*stage*/) {
            },
            {outputs[0], biases},
            {outputs[0]},
            scaleStage);
    }
}
