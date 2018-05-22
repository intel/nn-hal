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

namespace {

enum PriorBox_CodeType {
    CORNER = 1,
    CENTER_SIZE,
    CORNER_SIZE
};

}  // namespace

void VpuDetectionOutputStage::dumpToDot(std::ostream& os) {
    os << "num_classes=" << params.num_classes << "\\n"
       << "share_location=" << params.share_location << "\\n"
       << "background_label_id=" << params.background_label_id << "\\n"
       << "nms_threshold=" << params.nms_threshold << "\\n"
       << "top_k=" << params.top_k << "\\n"
       << "code_type=" << params.code_type << "\\n"
       << "keep_top_k=" << params.keep_top_k << "\\n"
       << "confidence_threshold=" << params.confidence_threshold << "\\n"
       << "variance_encoded_in_target=" << params.variance_encoded_in_target << "\\n"
       << "eta=" << params.eta << "\\n"
       << "num_priors=" << params.num_priors;
}

void VpuDetectionOutputStage::dumpToBlob(BlobWriter& writer) {
    writer.write(params);

    inputs[0]->dumpToBlob(writer);
    inputs[1]->dumpToBlob(writer);
    inputs[2]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
}

void GraphTransformerImpl::parseDetectionOutput(const CNNLayerPtr& layer,
                                                const std::vector<VpuDataHandle>& inputs,
                                                const std::vector<VpuDataHandle>& outputs) {
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);

    DetectionOutputParams detParams;
    detParams.num_classes = layer->GetParamAsInt("num_classes", 0);
    detParams.share_location = layer->GetParamAsInt("share_location", 1);
    detParams.background_label_id = layer->GetParamAsInt("background_label_id", 0);
    detParams.nms_threshold = layer->GetParamAsFloat("nms_threshold", 0);
    detParams.top_k = layer->GetParamAsInt("top_k", 0);
    {
        auto code_type_str = layer->GetParamAsString("code_type", "caffe.PriorBoxParameter.CENTER_SIZE");
        if (code_type_str.find("CORNER") != std::string::npos) {
            detParams.code_type = CORNER;
        } else if (code_type_str.find("CENTER_SIZE") != std::string::npos) {
            detParams.code_type = CENTER_SIZE;
        } else if (code_type_str.find("CORNER_SIZE") != std::string::npos) {
            detParams.code_type = CORNER_SIZE;
        } else {
            THROW_IE_EXCEPTION << "[VPU] Unknown code_type " << code_type_str << " for DetectionOutput layer " << layer->name;
        }
    }
    detParams.keep_top_k = layer->GetParamAsInt("keep_top_k", 0);
    detParams.confidence_threshold = layer->GetParamAsFloat("confidence_threshold", 0);
    detParams.variance_encoded_in_target = layer->GetParamAsInt("variance_encoded_in_target", 0);
    detParams.eta = layer->GetParamAsFloat("eta", 0);
    {
        auto prior = layer->insData[2].lock();
        assert(prior != nullptr);

        // Each prior consists of 4 values.
        detParams.num_priors = prior->dims[0] / 4;
    }

    addNewStage<VpuDetectionOutputStage>(
        layer->name,
        kDetectionOutput,
        layer,
        [detParams](VpuDetectionOutputStage* stage) {
            stage->params = detParams;

            stage->requiredOutputOrder[0] = stage->outputs[0]->order;
        },
        inputs,
        outputs);
}
