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
#include <string>
#include <memory>

void VpuConvStage::dumpToDot(std::ostream& os) {
    os << "radixX=" << radixX << "\\n"
       << "radixY=" << radixY << "\\n"
       << "strideX=" << strideX << "\\n"
       << "strideY=" << strideY << "\\n"
       << "padX=" << padX << "\\n"
       << "padY=" << padY << "\\n"
       << "dilationX=" << dilationX << "\\n"
       << "dilationY=" << dilationY;
}

void VpuConvStage::dumpToBlob(BlobWriter& writer) {
    if (dilationX != dilationY) {
        THROW_IE_EXCEPTION << "[VPU] Convolution layer " << name << " has different dilations per x and y";
    }

    writer.write(static_cast<uint32_t>(radixX));
    writer.write(static_cast<uint32_t>(radixY));
    writer.write(static_cast<uint32_t>(strideX));
    writer.write(static_cast<uint32_t>(strideY));
    writer.write(static_cast<uint32_t>(padX));
    writer.write(static_cast<uint32_t>(padY));
    writer.write(static_cast<uint32_t>(paddStyleCaffe));
    writer.write(static_cast<uint32_t>(dilationX));

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
    inputs[1]->dumpToBlob(writer);

    if (buffer != nullptr) {
        buffer->dumpToBlob(writer);
    }

    VpuData fakeBiases;
    fakeBiases.dumpToBlob(writer);
}

namespace {

class GroupedConvolutionWeightsWriter : public DataWriter {
public:
    GroupedConvolutionWeightsWriter(int group, const VpuDims& dims, const Blob::Ptr& blob,
                                    const std::shared_ptr<ConvolutionLayer>& layer)
        : _group(group), _dims(dims), _blob(blob), _layer(layer) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _dims.totalSize() * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto grouping = _layer->_group;

        auto weightsByteSize = _blob->byteSize() / grouping;
        auto weightsOffset = _group * weightsByteSize;

        if (weightsByteSize != _dims.totalSize() * sizeof(ie_fp16)) {
            THROW_IE_EXCEPTION << "[VPU] Invalid weights size for Convolution layer " << _layer->name;
        }

        auto wsrc = _blob->cbuffer().as<const ie_fp16*>() + weightsOffset / sizeof(ie_fp16);

        kchw_to_hwck(wsrc, static_cast<ie_fp16*>(dst), _dims);
    }

private:
    int _group;
    VpuDims _dims;
    Blob::Ptr _blob;
    std::shared_ptr<ConvolutionLayer> _layer;
};

class Conv1x1s1WeightsWriter : public DataWriter {
public:
    Conv1x1s1WeightsWriter(const Blob::Ptr& blob) : _blob(blob) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _blob->byteSize();
    }

    void write(void* dst) const override {
        std::copy_n(_blob->cbuffer().as<const ie_fp16*>(), _blob->size(), static_cast<ie_fp16*>(dst));
    }

private:
    Blob::Ptr _blob;
};

class GroupedConvolution1x1s1WeightsWriter : public DataWriter {
public:
    GroupedConvolution1x1s1WeightsWriter(int group, const VpuDims& dims, const Blob::Ptr& blob,
                                    const std::shared_ptr<ConvolutionLayer>& layer)
        : _group(group), _dims(dims), _blob(blob), _layer(layer) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _dims.totalSize() * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto grouping = _layer->_group;

        auto weightsByteSize = _blob->byteSize() / grouping;
        auto weightsOffset = _group * weightsByteSize;

        if (weightsByteSize != _dims.totalSize() * sizeof(ie_fp16)) {
            THROW_IE_EXCEPTION << "[VPU] Invalid weights size for Convolution layer " << _layer->name;
        }

        auto wsrc = _blob->cbuffer().as<const ie_fp16*>() + weightsOffset / sizeof(ie_fp16);

        std::copy_n(wsrc, _dims[Dim::Y] * _dims[Dim::Z], static_cast<ie_fp16*>(dst));
    }

private:
    int _group;
    VpuDims _dims;
    Blob::Ptr _blob;
    std::shared_ptr<ConvolutionLayer> _layer;
};

class OldDepthConvolutionWeightsWriter : public DataWriter {
public:
    explicit OldDepthConvolutionWeightsWriter(const Blob::Ptr& blob) : _blob(blob) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _blob->byteSize();
    }

    void write(void* dst) const override {
        std::copy_n(_blob->cbuffer().as<const ie_fp16*>(), _blob->size(), static_cast<ie_fp16*>(dst));
    }

private:
    Blob::Ptr _blob;
};

class GroupedConvolutionBiasesWriter : public DataWriter {
public:
    explicit GroupedConvolutionBiasesWriter(int group, const Blob::Ptr& blob,
                                            const std::shared_ptr<ConvolutionLayer>& layer)
        : _group(group), _blob(blob), _layer(layer) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _blob->byteSize() / _layer->_group;
    }

    void write(void* dst) const override {
        auto grouping = _layer->_group;

        auto biasByteSize = _blob->byteSize() / grouping;
        auto biasOffset = _group * biasByteSize;

        auto bsrc = _blob->cbuffer().as<const ie_fp16*>() + biasOffset / sizeof(ie_fp16);
        std::copy_n(bsrc, biasByteSize / sizeof(ie_fp16), static_cast<ie_fp16*>(dst));
    }

private:
    int _group;
    Blob::Ptr _blob;
    std::shared_ptr<ConvolutionLayer> _layer;
};

class ConvIm2ColWeightsWriter : public DataWriter {
public:
    ConvIm2ColWeightsWriter(const VpuDims& dims, const Blob::Ptr& blob) : _dims(dims), _blob(blob) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _blob->byteSize();
    }

    void write(void* dst) const override {
        kchw_to_khwc(_blob->cbuffer().as<const ie_fp16*>(), static_cast<ie_fp16*>(dst), _dims);
    }

private:
    VpuDims _dims;
    Blob::Ptr _blob;
};

class ConvIm2ColGroupWeightsWriter : public DataWriter {
public:
    ConvIm2ColGroupWeightsWriter(int group, const VpuDims& dims, const Blob::Ptr& blob,
                                    const std::shared_ptr<ConvolutionLayer>& layer)
        : _group(group), _dims(dims), _blob(blob), _layer(layer) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _dims.totalSize() * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto grouping = _layer->_group;

        auto weightsByteSize = _blob->byteSize() / grouping;
        auto weightsOffset = _group * weightsByteSize;

        if (weightsByteSize != _dims.totalSize() * sizeof(ie_fp16)) {
            THROW_IE_EXCEPTION << "[VPU] Invalid weights size for Convolution layer " << _layer->name;
        }

        auto wsrc = _blob->cbuffer().as<const ie_fp16*>() + weightsOffset / sizeof(ie_fp16);

        kchw_to_khwc(wsrc, static_cast<ie_fp16*>(dst), _dims);
    }

private:
    int _group;
    VpuDims _dims;
    Blob::Ptr _blob;
    std::shared_ptr<ConvolutionLayer> _layer;
};

class Conv3x3WeightsWriter : public DataWriter {
public:
    Conv3x3WeightsWriter(const VpuDims& dims, const Blob::Ptr& blob) : _dims(dims), _blob(blob) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _blob->byteSize();
    }

    void write(void* dst) const override {
        kchw_to_hwkc(_blob->cbuffer().as<const ie_fp16*>(), static_cast<ie_fp16*>(dst), _dims);
    }

private:
    VpuDims _dims;
    Blob::Ptr _blob;
};

class GroupedConvolution3x3WeightsWriter : public DataWriter {
public:
    GroupedConvolution3x3WeightsWriter(int group, const VpuDims& dims, const Blob::Ptr& blob,
                                    const std::shared_ptr<ConvolutionLayer>& layer)
        : _group(group), _dims(dims), _blob(blob), _layer(layer) {
        assert(blob != nullptr);
    }

    size_t byteSize() const override {
        return _dims.totalSize() * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto grouping = _layer->_group;

        auto weightsByteSize = _blob->byteSize() / grouping;
        auto weightsOffset = _group * weightsByteSize;

        if (weightsByteSize != _dims.totalSize() * sizeof(ie_fp16)) {
            THROW_IE_EXCEPTION << "[VPU] Invalid weights size for Convolution layer " << _layer->name;
        }

        auto wsrc = _blob->cbuffer().as<const ie_fp16*>() + weightsOffset / sizeof(ie_fp16);

        kchw_to_hwkc(wsrc, static_cast<ie_fp16*>(dst), _dims);
    }

private:
    int _group;
    VpuDims _dims;
    Blob::Ptr _blob;
    std::shared_ptr<ConvolutionLayer> _layer;
};

}  // namespace

void GraphTransformerImpl::parseConvolution(const CNNLayerPtr& _layer,
                                            const std::vector<VpuDataHandle>& inputs,
                                            const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ConvolutionLayer>(_layer);
    assert(layer != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    bool isFC = (layer->_kernel_x == 1 && layer->_kernel_y == 1
            && layer->_stride_x == 1 && layer->_stride_y == 1
            && layer->_padding_x == 0 && layer->_padding_y == 0
            && input->dims[Dim::X] == 1 && input->dims[Dim::Y] == 1
            && output->dims[Dim::X] == 1 && output->dims[Dim::Y] == 1
            && (layer->_dilation_x == 1) && (layer->_dilation_y == 1));

    bool isConv1x1 = ((layer->_kernel_x == 1) && (layer->_kernel_y == 1) && (!isFC) && (layer->_dilation_x == 1) && (layer->_dilation_y == 1));

    bool isConv3x3 = ((layer->_kernel_x == 3) && (layer->_kernel_y == 3)
                       && (input->dims[Dim::Z] / layer->_group > 3));

    bool isKernelSizeOdd = (layer->_kernel_x == layer->_kernel_y) && (layer->_kernel_x % 2 == 1);
    bool iskernelSizeLessThan10 = (layer->_kernel_x == layer->_kernel_y) && (layer->_kernel_x < 10);

    // TODO : FIX SpatialConv!!!!
    // Spatial Convolution has bug when stride > 1
#define pad_right(ISize, OSize, stride, radix, dilation, pad) \
    ((pad) + ((OSize) - (2 * (pad) + (ISize) - (dilation) * ((radix)-1) + (stride)-1) / (stride)) * (stride))

    int pad_left = layer->_padding_x;
    int pad_right = pad_right(input->dims[Dim::X], output->dims[Dim::X], layer->_stride_x, layer->_kernel_x, layer->_dilation_x, layer->_padding_x);
    int pad_top = layer->_padding_y;
    int pad_bottom = pad_right(input->dims[Dim::Y], output->dims[Dim::Y], layer->_stride_y, layer->_kernel_y, layer->_dilation_y, layer->_padding_y);
    bool symmetrical_paddings = (pad_right == pad_left) && (pad_top == pad_bottom);
#undef pad_right

    bool isSpatialConv = ((isKernelSizeOdd && iskernelSizeLessThan10 && (layer->_stride_x == 1 && layer->_stride_y == 1))
                          || ((layer->_kernel_x == 7 && layer->_kernel_y == 7) && (layer->_stride_x == 2 && layer->_stride_y == 2) && symmetrical_paddings))
                         && (layer->_group == 1) && (input->dims[Dim::Z] < 4)
                         && (layer->_padding_x > 0) && (layer->_padding_y > 0);

    auto createWeights = [layer, this, input](const VpuDims& weightsDims, int grouping, const std::string& extraSuffix) {
        assert(layer->_weights != nullptr);
        return addNewData(
            newDataId(),
            [layer, extraSuffix, weightsDims, grouping, this, input](VpuData* data) {
                data->name = layer->name + "@weights" + extraSuffix;
                data->index = IndexBlob;
                data->type = VpuDataType::FP16;
                data->order = orderXYZ;
                data->dims = weightsDims;
                data->strides = calcStrides(data->dims, data->type, data->order);
                data->writer = std::make_shared<ConvIm2ColWeightsWriter>(weightsDims, layer->_weights);
            });
    };

    auto createBiases = [layer, this](int grouping, const std::string& extraSuffix) {
        if (layer->_biases == nullptr)
            return VpuDataHandle();

        return addNewData(
            newDataId(),
            [layer, extraSuffix, grouping, this](VpuData* data) {
                data->name = layer->name + "@biases" + extraSuffix;
                data->index = IndexBlob;
                data->type = VpuDataType::FP16;
                data->dims = VpuDims({static_cast<uint32_t>(layer->_biases->size() / grouping), 1, 1});
                data->strides = calcStrides(data->dims, data->type, data->order);
                data->writer = std::make_shared<DefaultBiasesWriter>(layer->_biases);
            });
    };

    auto createConvStage = [layer, this, isConv1x1, isFC, isSpatialConv, isConv3x3](const std::string& name,
                                         const VpuDataHandle& input,
                                         const VpuDataHandle& output,
                                         const VpuDataHandle& weights,
                                         const VpuDataHandle& biases) {
        int im2ColBufSize = layer->_kernel_x * layer->_kernel_y * output->dims[Dim::X] *
                output->dims[Dim::Y] * input->dims[Dim::Z] + 32;

        VpuStageHandle convStage;
        uint32_t optMask = MV_TENSOR_DEFAULT_OPT;
        if (isFC) {
            convStage = addNewStage<VpuFullyConnectedStage>(
                name,
                kFC,
                layer,
                [](VpuFullyConnectedStage* /*stage*/) {
                },
                {input, weights},
                {output});
        } else if ((isConv1x1) || (isSpatialConv) || (isConv3x3)) {
            convStage = addNewStage<VpuConvStage>(
                name,
                kConv,
                layer,
                [layer, optMask](VpuConvStage* stage) {
                    stage->optMask = optMask;
                    stage->radixX = layer->_kernel_x;
                    stage->radixY = layer->_kernel_y;
                    stage->strideX = layer->_stride_x;
                    stage->strideY = layer->_stride_y;
                    stage->padX = layer->_padding_x;
                    stage->padY = layer->_padding_y;
                    stage->dilationX = layer->_dilation_x;
                    stage->dilationY = layer->_dilation_y;
                },
                {input, weights, },
                {output});
        } else {
            convStage = addNewStage<VpuConvStage>(
                name,
                kIm2ColConvolution,
                layer,
                [layer, optMask](VpuConvStage* stage) {
                    stage->optMask = optMask;
                    stage->radixX = layer->_kernel_x;
                    stage->radixY = layer->_kernel_y;
                    stage->strideX = layer->_stride_x;
                    stage->strideY = layer->_stride_y;
                    stage->padX = layer->_padding_x;
                    stage->padY = layer->_padding_y;
                    stage->dilationX = layer->_dilation_x;
                    stage->dilationY = layer->_dilation_y;
                },
                {input, weights },
                {output});

            convStage->buffer = addNewData(
                    newDataId(),
                    [layer, im2ColBufSize](VpuData* data) {
                        data->name = layer->name + "@im2ColBufData";
                        data->index = IndexBSS;
                        data->type = VpuDataType::FP16;
                        data->dims = VpuDims({static_cast<uint32_t>(im2ColBufSize), 1, 1});
                        data->strides = calcStrides(data->dims, data->type, data->order);
                    });
        }

        if (biases != nullptr) {
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
    };

    VpuDims weightsDims(3);
    weightsDims[Dim::X] = layer->_kernel_x * layer->_kernel_y;

    if (layer->_group == 1) {
        weightsDims[Dim::Y] = input->dims[Dim::Z];
        weightsDims[Dim::Z] = output->dims[Dim::Z];

        auto weights = createWeights(weightsDims, 1, "");
        assert(weights != nullptr);

        if ((isSpatialConv) || (isFC)) {
            weights->writer = std::make_shared<DefaultWeightsWriter>(weightsDims, layer->_weights);
        } else if (isConv1x1) {
            weights->writer = std::make_shared<Conv1x1s1WeightsWriter>(layer->_weights);
        } else if (isConv3x3) {
            weights->writer = std::make_shared<Conv3x3WeightsWriter>(weightsDims, layer->_weights);
        }

        auto biases = createBiases(1, "");

        createConvStage(layer->name, input, output, weights, biases);

    } else if (layer->_group == inputs[0]->dims[Dim::Z] && layer->_group == outputs[0]->dims[Dim::Z]) {
        weightsDims[Dim::Y] = 1;
        weightsDims[Dim::Z] = outputs[0]->dims[Dim::Z];

        auto weights = createWeights(weightsDims, 1, "");
        weights->writer = std::make_shared<DefaultWeightsWriter>(weightsDims, layer->_weights);
        auto biases = createBiases(1, "");

        const int availableMemory = 84 * 1024;  // MVTENSOR_HEAP_DATA_SIZE

        const int nullPixel = 1;
        const int outputPixel = 1;
        const int local_weights = layer->_kernel_y * layer->_kernel_x;
        const int memoryForPixel = inputs[0]->dims[Dim::Z] * (layer->_kernel_y * layer->_kernel_x + nullPixel + outputPixel) * sizeof(ie_fp16);
        const int memoryForFullLineWeights = inputs[0]->dims[Dim::Z] *
                (inputs[0]->dims[Dim::X] * layer->_kernel_y + nullPixel + local_weights +
                 outputs[0]->dims[Dim::X]) * sizeof(ie_fp16);
        const int memoryForFullLine = inputs[0]->dims[Dim::Z] *
                (inputs[0]->dims[Dim::X] * layer->_kernel_y + nullPixel +
                 outputs[0]->dims[Dim::X]) * sizeof(ie_fp16);

        bool useOldDepthConvolution = false;
        if (memoryForFullLineWeights <= availableMemory)
            useOldDepthConvolution = false;
        else if (memoryForFullLine <= availableMemory)
            useOldDepthConvolution = false;
        else if (memoryForPixel <= availableMemory)
            useOldDepthConvolution = true;  // TODO : the mvDepthConvByPixel is not implemented yet
        else
            useOldDepthConvolution = true;  // TODO : this case is not implemented yet

        // If we don't have enough CMX memory, we will use old implementation of DepthConvolution.
        // It requires another order of weights coefficients, so we use another weights writer.
        if (useOldDepthConvolution) {
            weights->writer = std::make_shared<OldDepthConvolutionWeightsWriter>(layer->_weights);
        }

        auto convStage = addNewStage<VpuConvStage>(
            layer->name,
            kDepthConv,
            layer,
            [layer](VpuConvStage* stage) {
                stage->radixX = layer->_kernel_x;
                stage->radixY = layer->_kernel_y;
                stage->strideX = layer->_stride_x;
                stage->strideY = layer->_stride_y;
                stage->padX = layer->_padding_x;
                stage->padY = layer->_padding_y;
                stage->dilationX = layer->_dilation_x;
                stage->dilationY = layer->_dilation_y;
            },
            {input, weights},
            {output});

        if (biases != nullptr) {
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

        weightsDims[Dim::Y] = inGroupDimZ;
        weightsDims[Dim::Z] = outGroupDimZ;

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

            // Convolve the subInput

            auto subConv = addNewData(
                newDataId(),
                [layer, output, outGroupDimZ, i](VpuData* data) {
                    data->name = layer->name + "@subConv" + std::to_string(i);
                    data->index = IndexBSS;
                    data->type = VpuDataType::FP16;
                    data->dims = VpuDims({output->dims[Dim::X], output->dims[Dim::Y], outGroupDimZ});
                    data->strides = calcStrides(data->dims, data->type, data->order);
                });

            auto weights = createWeights(weightsDims, layer->_group, std::to_string(i));
            assert(weights != nullptr);

            // conv1x1, stride1
            if (isConv1x1) {
                weights->writer = std::make_shared<GroupedConvolution1x1s1WeightsWriter>(i, weightsDims, layer->_weights, layer);
            } else if (isConv3x3) {
                weights->writer = std::make_shared<GroupedConvolution3x3WeightsWriter>(i, weightsDims, layer->_weights, layer);
            } else {
                weights->writer = std::make_shared<ConvIm2ColGroupWeightsWriter>(i, weightsDims, layer->_weights, layer);
            }

            auto biases = createBiases(layer->_group, std::to_string(i));
            if (biases != nullptr) {
                biases->writer = std::make_shared<GroupedConvolutionBiasesWriter>(i, layer->_biases, layer);
            }

            createConvStage(subConv->name, subInputCopy, subConv, weights, biases);

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

            addCopyStage(subOutput->name, layer, subConv, subOutput);
        }
    }
}
