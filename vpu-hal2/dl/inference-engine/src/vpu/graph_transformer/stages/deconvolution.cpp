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

void deconvolutionRelayout(const ie_fp16 *src, size_t src_size,
                           ie_fp16 *dst, size_t dst_size,
                           size_t KX, size_t KY,
                           size_t IC, size_t OC,
                           size_t g, size_t GR) {
    for (size_t goc = 0; goc < OC / GR; ++goc) {
        for (size_t gic = 0; gic < IC / GR; ++gic) {
            for (size_t ky = 0; ky < KY; ++ky) {
                for (size_t kx = 0; kx < KX; ++kx) {
                    size_t iidx = gic * OC * KY * KX
                                  + (g * OC / GR + goc) * KY * KX
                                  + ky * KX
                                  + kx;
                    assert(iidx < src_size);

                    size_t inv_kx = KX - kx - 1;
                    size_t inv_ky = KY - ky - 1;
                    size_t oidx = goc * (IC / GR) * KY * KX
                                  + gic * KY * KX
                                  + inv_ky * KX
                                  + inv_kx;
                    assert(oidx < dst_size);

                    dst[oidx] = src[iidx];
                }
            }
        }
    }
}

class DeconvolutionWeightsWriter : public DataWriter {
public:
    DeconvolutionWeightsWriter(const VpuDims &wDims,
                               const Blob::Ptr &weights,
                               size_t KX, size_t KY,
                               size_t g, size_t GR,
                               size_t IC, size_t OC)
            : _wDims(wDims), _weights(weights), _KX(KX), _KY(KY), _g(g), _GR(GR), _IC(IC), _OC(OC) {
        assert(weights != nullptr);
    }

    size_t byteSize() const override {
        return _wDims.totalSize() * sizeof(ie_fp16);
    }

    void write(void *dst) const override {
        auto srcPtr = _weights->cbuffer().as<const ie_fp16 *>();
        auto dstPtr = static_cast<ie_fp16 *>(dst);

        std::vector<ie_fp16> wsrcConverted(_wDims.totalSize());
        deconvolutionRelayout(srcPtr, _weights->size(), wsrcConverted.data(), wsrcConverted.size(),
                              _KX, _KY,
                              _IC, _OC,
                              _g, _GR);

        kchw_to_hwkc(wsrcConverted.data(), dstPtr, _wDims);
    }

private:
    VpuDims _wDims;
    Blob::Ptr _weights;
    size_t _KX;
    size_t _KY;
    size_t _g;
    size_t _GR;
    size_t _IC;
    size_t _OC;
};

class DeconvolutionBiasesWriter : public DataWriter {
public:
    DeconvolutionBiasesWriter(const std::shared_ptr<DeconvolutionLayer>& layer,
                              uint32_t group)
        : _layer(layer), _group(group) {
        assert(_layer != nullptr);
    }

    size_t byteSize() const override {
        return _layer->_biases->byteSize() / _layer->_group;
    }

    void write(void* dst) const override {
        auto grouping = _layer->_group;
        auto biasByteSize = _layer->_biases->byteSize() / grouping;
        auto biasOffset = _group * biasByteSize;

        auto bsrc = _layer->_biases->cbuffer().as<const ie_fp16*>() + biasOffset / sizeof(ie_fp16);
        std::copy_n(bsrc, biasByteSize / sizeof(ie_fp16), static_cast<ie_fp16*>(dst));
    }

private:
    std::shared_ptr<DeconvolutionLayer> _layer;
    uint32_t _group;
};

}  // namespace

void GraphTransformerImpl::parseDeconvolution(const CNNLayerPtr& _layer,
                                              const std::vector<VpuDataHandle>& inputs,
                                              const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<DeconvolutionLayer>(_layer);
    assert(layer != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    if (   (layer->_group == 0)
        || (layer->_group > input->dims[Dim::Z])
        || (input->dims[Dim::Z] % layer->_group != 0)
        || (layer->_group > output->dims[Dim::Z])
        || (output->dims[Dim::Z] % layer->_group != 0)) {
        THROW_IE_EXCEPTION << "[VPU] DeconvolutionLayer has invalid group value";
    }

    if (layer->_group == 1) {
        VpuDims wDims(3);
        wDims[Dim::X] = layer->_kernel_x * layer->_kernel_y;
        wDims[Dim::Y] = input->dims[Dim::Z];
        wDims[Dim::Z] = output->dims[Dim::Z];

        assert(layer->_weights != nullptr);
        auto weights = addNewData(
            newDataId(),
            [layer, wDims, input, output](VpuData* data) {
                data->name = layer->name + "@weights";
                data->index = IndexBlob;
                data->type = VpuDataType::FP16;
                data->order = orderXYZ;
                data->dims = wDims;
                data->strides = calcStrides(data->dims, data->type, data->order);
                data->writer = std::make_shared<DeconvolutionWeightsWriter>(wDims, layer->_weights,
                                                                            layer->_kernel_x, layer->_kernel_y,
                                                                            0, 1,
                                                                            input->dims[Dim::Z], output->dims[Dim::Z]);
            });

        auto convStage = addNewStage<VpuConvStage>(
            layer->name,
            kDeconvolution,
            layer,
            [layer](VpuConvStage* stage) {
                if (layer->_kernel_x == 3 && layer->_kernel_y == 3 &&
                    layer->_stride_x == 1 && layer->_stride_y == 1) {
                    stage->optMask = 1u << opt_deconv_3_3_1_1_same_specific;
                }
                stage->radixX = layer->_kernel_x;
                stage->radixY = layer->_kernel_y;
                stage->strideX = layer->_stride_x;
                stage->strideY = layer->_stride_y;
                stage->padX = layer->_padding_x;
                stage->padY = layer->_padding_y;
                stage->dilationX = stage->dilationY = 1;
            },
            {input, weights},
            {output});

        if (layer->_biases != nullptr) {
            auto biases = addNewData(
                newDataId(),
                [layer, this](VpuData* data) {
                    data->name = layer->name + "@biases";
                    data->index = IndexBlob;
                    data->type = VpuDataType::FP16;
                    data->dims = VpuDims({static_cast<uint32_t>(layer->_biases->size()), 1, 1});
                    data->strides = calcStrides(data->dims, data->type, data->order);
                    data->writer = std::make_shared<DeconvolutionBiasesWriter>(layer, 0);
                });

            addNewStage<VpuBiasStage>(
                biases->name,
                kBias,
                layer,
                [](VpuBiasStage* /*stage*/) {
                },
                {output, biases},
                {output},
                convStage);
        }
    } else {
        auto inGroupDimZ = input->dims[Dim::Z] / layer->_group;
        auto outGroupDimZ = layer->_out_depth / layer->_group;

        for (int i = 0; i < layer->_group; ++i) {
            // Copy input to subInput

            auto subInput = addNewData(
                newDataId(),
                [layer, input, inGroupDimZ, i](VpuData* data) {
                    data->name = layer->name + "@subInput" + std::to_string(i);
                    data->index = input->index;
                    data->type = VpuDataType::FP16;
                    data->dims = VpuDims({input->dims[Dim::X], input->dims[Dim::Y], inGroupDimZ});
                    data->strides = input->strides;
                    data->offsetFromParent = VpuDims({0u, 0u, i * inGroupDimZ});
                },
                input);

            auto subInputCopy = addNewData(
                newDataId(),
                [layer, subInput, i](VpuData* data) {
                    data->name = layer->name + "@subInputCopy" + std::to_string(i);
                    data->index = IndexBSS;
                    data->type = VpuDataType::FP16;
                    data->dims = subInput->dims;
                    data->strides = calcStrides(data->dims, data->type, data->order);
                });

            addCopyStage(subInputCopy->name, layer, subInput, subInputCopy);

            // Deconvolve the subInput

            auto subDeconv = addNewData(
                newDataId(),
                [layer, output, outGroupDimZ, i](VpuData* data) {
                    data->name = layer->name + "@subDeconv" + std::to_string(i);
                    data->index = IndexBSS;
                    data->type = VpuDataType::FP16;
                    data->dims = VpuDims({output->dims[Dim::X], output->dims[Dim::Y], outGroupDimZ});
                    data->strides = calcStrides(data->dims, data->type, data->order);
                });

            VpuDims wDims(3);
            wDims[Dim::X] = layer->_kernel_x * layer->_kernel_y;
            wDims[Dim::Y] = inGroupDimZ;
            wDims[Dim::Z] = outGroupDimZ;

            assert(layer->_weights != nullptr);
            auto weights = addNewData(
                newDataId(),
                [layer, wDims, i, input, output](VpuData* data) {
                    data->name = layer->name + "@weights";
                    data->index = IndexBlob;
                    data->type = VpuDataType::FP16;
                    data->order = orderXYZ;
                    data->dims = wDims;
                    data->strides = calcStrides(data->dims, data->type, data->order);
                    data->writer = std::make_shared<DeconvolutionWeightsWriter>(wDims, layer->_weights,
                                                                                layer->_kernel_x, layer->_kernel_y,
                                                                                i, layer->_group,
                                                                                input->dims[Dim::Z], output->dims[Dim::Z]);
                });

            auto convStage = addNewStage<VpuConvStage>(
                subDeconv->name,
                kDeconvolution,
                layer,
                [layer](VpuConvStage* stage) {
                    if (layer->_kernel_x == 3 && layer->_kernel_y == 3 &&
                        layer->_stride_x == 1 && layer->_stride_y == 1) {
                        stage->optMask = 1u << opt_deconv_3_3_1_1_same_specific;
                    }
                    stage->radixX = layer->_kernel_x;
                    stage->radixY = layer->_kernel_y;
                    stage->strideX = layer->_stride_x;
                    stage->strideY = layer->_stride_y;
                    stage->padX = layer->_padding_x;
                    stage->padY = layer->_padding_y;
                    stage->dilationX = stage->dilationY = 1;
                },
                {subInputCopy, weights},
                {subDeconv});

            if (layer->_biases != nullptr) {
                auto biases = addNewData(
                    newDataId(),
                    [layer, i, this](VpuData* data) {
                        data->name = layer->name + "@biases";
                        data->index = IndexBlob;
                        data->type = VpuDataType::FP16;
                        data->dims = VpuDims({static_cast<uint32_t>(layer->_biases->size()), 1, 1});
                        data->strides = calcStrides(data->dims, data->type, data->order);
                        data->writer = std::make_shared<DeconvolutionBiasesWriter>(layer, i);
                    });

                addNewStage<VpuBiasStage>(
                    biases->name,
                    kBias,
                    layer,
                    [](VpuBiasStage* /*stage*/) {
                    },
                    {subDeconv, biases},
                    {subDeconv},
                    convStage);
            }

            // copy subConvData to output

            auto subOutput = addNewData(
                newDataId(),
                [layer, output, outGroupDimZ, i](VpuData* data) {
                    data->name = layer->name + "@subOutput" + std::to_string(i);
                    data->index = output->index;
                    data->type = VpuDataType::FP16;
                    data->dims = VpuDims({output->dims[Dim::X], output->dims[Dim::Y], outGroupDimZ});
                    data->strides = output->strides;
                    data->offsetFromParent = VpuDims({0u, 0u, i * outGroupDimZ});
                },
                output);

            addCopyStage(subOutput->name, layer, subDeconv, subOutput);
        }
    }
}
