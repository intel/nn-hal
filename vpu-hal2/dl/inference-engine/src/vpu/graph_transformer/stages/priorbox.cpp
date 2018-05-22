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

class PriorBoxWriter : public DataWriter {
public:
    PriorBoxWriter(const VpuDims& inDims0, const VpuDims& inDims1, const VpuDims& outDims, const CNNLayerPtr& layer)
        : _inDims0(inDims0), _inDims1(inDims1), _outDims(outDims), _layer(layer) {
        assert(layer != nullptr);
    }

    size_t byteSize() const override {
        return _outDims.totalSize() * sizeof(ie_fp16);
    }

    void write(void* dst) const override {
        auto min_sizes_ = _layer->GetParamAsFloats("min_size");
        auto max_sizes_ = _layer->GetParamAsFloats("max_size");
        auto p_aspect_ratio = _layer->GetParamAsFloats("aspect_ratio");
        auto flip_ = _layer->GetParamAsInt("flip");
        auto clip_ = _layer->GetParamAsInt("clip");
        auto variance_ = _layer->GetParamAsFloats("variance");
        auto img_h_ = _layer->GetParamAsInt("img_h", 0);
        auto img_w_ = _layer->GetParamAsInt("img_w", 0);
        auto step_ = _layer->GetParamAsFloat("step", 0);
        auto step_h_ = _layer->GetParamAsFloat("step_h", 0);
        auto step_w_ = _layer->GetParamAsFloat("step_w", 0);
        auto offset_ = _layer->GetParamAsFloat("offset", 0);

        std::vector<float> aspect_ratios_;
        aspect_ratios_.push_back(1.0);
        for (int i = 0; i < p_aspect_ratio.size(); ++i) {
            float ar = p_aspect_ratio[i];
            bool already_exist = false;
            for (int j = 0; j < aspect_ratios_.size(); ++j) {
                if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
                    already_exist = true;
                    break;
                }
            }
            if (!already_exist) {
                aspect_ratios_.push_back(ar);
                if (flip_) {
                    aspect_ratios_.push_back(1.0 / ar);
                }
            }
        }

        auto num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();

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

        int dim = layer_height * layer_width * num_priors_ * 4;
        if (_outDims[Dim::X] != dim || _outDims[Dim::Y] != 2) {
            THROW_IE_EXCEPTION << "[VPU] PriorBox output have invalid dimension, exptected " << dim << "x2"
                               << ", got " << _outDims[Dim::X] << "x" << _outDims[Dim::Y] << ", layer name is: " << _layer->name;
        }

        auto top_data = static_cast<ie_fp16*>(dst);

        int idx = 0;
        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width;  ++w) {
                float center_x = (w + offset_) * step_w;
                float center_y = (h + offset_) * step_h;

                float box_width, box_height;
                float xmin, ymin, xmax, ymax;
                for (int s = 0; s < min_sizes_.size(); ++s) {
                    float min_size_ = min_sizes_[s];

                    // first prior: aspect_ratio = 1, size = min_size
                    box_width = box_height = min_size_;
                    xmin = (center_x - box_width / 2.) / img_width;
                    ymin = (center_y - box_height / 2.) / img_height;
                    xmax = (center_x + box_width / 2.) / img_width;
                    ymax = (center_y + box_height / 2.) / img_height;

                    if (clip_) {
                        xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                        ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                        xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                        ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                    }

                    top_data[idx++] = PrecisionUtils::f32tof16(xmin);
                    top_data[idx++] = PrecisionUtils::f32tof16(ymin);
                    top_data[idx++] = PrecisionUtils::f32tof16(xmax);
                    top_data[idx++] = PrecisionUtils::f32tof16(ymax);

                    if (max_sizes_.size() > 0) {
                        float max_size_ = max_sizes_[s];

                        // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        box_width = box_height = sqrt(min_size_ * max_size_);

                        xmin = (center_x - box_width / 2.) / img_width;
                        ymin = (center_y - box_height / 2.) / img_height;
                        xmax = (center_x + box_width / 2.) / img_width;
                        ymax = (center_y + box_height / 2.) / img_height;

                        if (clip_) {
                            xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                            ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                            xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                            ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                        }

                        top_data[idx++] = PrecisionUtils::f32tof16(xmin);
                        top_data[idx++] = PrecisionUtils::f32tof16(ymin);
                        top_data[idx++] = PrecisionUtils::f32tof16(xmax);
                        top_data[idx++] = PrecisionUtils::f32tof16(ymax);
                    }

                    // rest of priors
                    for (int r = 0; r < aspect_ratios_.size(); ++r) {
                        float ar = aspect_ratios_[r];
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }

                        box_width = min_size_ * sqrt(ar);
                        box_height = min_size_ / sqrt(ar);

                        xmin = (center_x - box_width / 2.) / img_width;
                        ymin = (center_y - box_height / 2.) / img_height;
                        xmax = (center_x + box_width / 2.) / img_width;
                        ymax = (center_y + box_height / 2.) / img_height;

                        if (clip_) {
                            xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                            ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                            xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                            ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                        }

                        top_data[idx++] = PrecisionUtils::f32tof16(xmin);
                        top_data[idx++] = PrecisionUtils::f32tof16(ymin);
                        top_data[idx++] = PrecisionUtils::f32tof16(xmax);
                        top_data[idx++] = PrecisionUtils::f32tof16(ymax);
                    }
                }
            }
        }

        top_data += dim;

        // set the variance.
        if (variance_.size() == 0) {
            // Set default to 0.1.
            for (int d = 0; d < dim; ++d) {
                top_data[d] = PrecisionUtils::f32tof16(0.1f);
            }
        } else if (variance_.size() == 1) {
            for (int d = 0; d < dim; ++d) {
                top_data[d] = PrecisionUtils::f32tof16(variance_[0]);
            }
        } else {
            if (variance_.size() != 4) {
                THROW_IE_EXCEPTION << "PriorBox layer must have only 4 variance";
            }

            int idx = 0;
            for (int h = 0; h < layer_height; ++h) {
                for (int w = 0; w < layer_width; ++w) {
                    for (int i = 0; i < num_priors_; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            top_data[idx++] = PrecisionUtils::f32tof16(variance_[j]);
                        }
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

void GraphTransformerImpl::parsePriorBox(const CNNLayerPtr& layer,
                                         const std::vector<VpuDataHandle>& inputs,
                                         const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);

    auto output = outputs[0];

    if (output->index != IndexBSS ||
        output->type != VpuDataType::FP16) {
        THROW_IE_EXCEPTION << "[VPU] Unsupported usage of PriorBox layer " << layer->name;
    }

    output->index = IndexBlob;
    output->writer = std::make_shared<PriorBoxWriter>(inputs[0]->dims, inputs[1]->dims, output->dims, layer);

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
