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
#include <algorithm>
#include <vector>
#include <memory>

namespace {

class PriorBoxClusteredWriter : public DataWriter {
public:
    PriorBoxClusteredWriter(const VpuDims& inDims0, const VpuDims& inDims1, const VpuDims& outDims, const CNNLayerPtr& layer)
        : _inDims0(inDims0), _inDims1(inDims1), _outDims(outDims), _layer(layer) {
        assert(layer != nullptr);
    }

    size_t byteSize() const override {
        return _outDims.totalSize() * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto widths_ = _layer->GetParamAsFloats("width");
        auto heights_ = _layer->GetParamAsFloats("height");
        auto clip_ = _layer->GetParamAsInt("clip");
        auto variance_ = _layer->GetParamAsFloats("variance");
        auto img_h_ = _layer->GetParamAsInt("img_h", 0);
        auto img_w_ = _layer->GetParamAsInt("img_w", 0);
        auto step_ = _layer->GetParamAsFloat("step", 0);
        auto step_h_ = _layer->GetParamAsFloat("step_h", 0);
        auto step_w_ = _layer->GetParamAsFloat("step_w", 0);
        auto offset_ = _layer->GetParamAsFloat("offset", 0);

        auto num_priors_ = widths_.size();

        if (variance_.empty())
            variance_.push_back(0.1);

        auto layer_width  = _inDims0[Dim::X];
        auto layer_height = _inDims0[Dim::Y];

        auto img_width  = img_w_ == 0 ? _inDims1[Dim::X] : img_w_;
        auto img_height = img_h_ == 0 ? _inDims1[Dim::Y] : img_h_;

        auto step_w = step_w_ == 0 ? step_ : step_w_;
        auto step_h = step_h_ == 0 ? step_ : step_h_;
        if (step_w == 0 || step_h == 0) {
            step_w = static_cast<float>(img_width) / layer_width;
            step_h = static_cast<float>(img_height) / layer_height;
        }

        auto expetected_output_dimx = layer_height * layer_width * num_priors_ * 4;
        if (_outDims[Dim::X] != expetected_output_dimx || _outDims[Dim::Y] != 2) {
            THROW_IE_EXCEPTION << "PriorBoxClustered output have invalid dimension, exptected " << expetected_output_dimx << "x2"
                               << ", got " << _outDims[Dim::X] << "x" << _outDims[Dim::Y] << ", layer name is: " << _layer->name;
        }

        auto offset = _outDims[Dim::X];
        auto var_size = variance_.size();

        auto top_data_0 = static_cast<ie_fp16*>(dst);
        auto top_data_1 = top_data_0 + offset;

        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width;  ++w) {
                auto center_x = (w + offset_) * step_w;
                auto center_y = (h + offset_) * step_h;

                for (int s = 0; s < num_priors_; ++s) {
                    auto box_width  = widths_[s];
                    auto box_height = heights_[s];

                    auto xmin = (center_x - box_width  / 2.0f) / img_width;
                    auto ymin = (center_y - box_height / 2.0f) / img_height;
                    auto xmax = (center_x + box_width  / 2.0f) / img_width;
                    auto ymax = (center_y + box_height / 2.0f) / img_height;

                    if (clip_) {
                        xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                        ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                        xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                        ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                    }

                    top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 0] = PrecisionUtils::f32tof16(xmin);
                    top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 1] = PrecisionUtils::f32tof16(ymin);
                    top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 2] = PrecisionUtils::f32tof16(xmax);
                    top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 3] = PrecisionUtils::f32tof16(ymax);

                    for (int j = 0; j < var_size; j++) {
                        auto index = h * layer_width * num_priors_ * var_size + w * num_priors_ * var_size + s * var_size + j;
                        top_data_1[index] = PrecisionUtils::f32tof16(variance_[j]);
                    }
                }
            }
        }
    }

private:
    VpuDims _inDims0;
    VpuDims _inDims1;
    VpuDims _outDims;
    CNNLayerPtr _layer;
};

}  // namespace

void GraphTransformerImpl::parsePriorBoxClustered(const CNNLayerPtr& layer,
                                                  const std::vector<VpuDataHandle>& inputs,
                                                  const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);

    auto inDims0 = inputs[0]->dims;
    auto inDims1 = inputs[1]->dims;
    auto output = outputs[0];

    if (output->index != IndexBSS ||
        output->type != VpuDataType::FP16) {
        THROW_IE_EXCEPTION << "[VPU] Unsupported usage of PriorBoxClustered layer " << layer->name;
    }

    output->index = IndexBlob;
    output->writer = std::make_shared<PriorBoxClusteredWriter>(inDims0, inDims1, output->dims, layer);

    addNewStage<VpuStage>(
        layer->name,
        kPriorBox,
        layer,
        [](VpuStage* stage) {
            stage->optimized = true;
        },
        inputs,
        outputs);
}
