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

void VpuCropStage::dumpToDot(std::ostream& os) {
    os << "offset=(" << offset[0] << ", " << offset[1] << ", " << offset[2] << ")";
}

void VpuCropStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<int32_t>(offset[0]));
    writer.write(static_cast<int32_t>(offset[1]));
    writer.write(static_cast<int32_t>(offset[2]));

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
}

void GraphTransformerImpl::parseCrop(const CNNLayerPtr& _layer,
                                     const std::vector<VpuDataHandle>& inputs,
                                     const std::vector<VpuDataHandle>& outputs) {
    // TODO : Crop layer in IR might have 1 or 2 inputs
    assert(inputs.size() >= 1);
    assert(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<CropLayer>(_layer);
    assert(layer != nullptr);

    auto cropAxis = layer->axis[0];
    if (cropAxis < 0)
        cropAxis += 4;

    if (cropAxis < 0 || cropAxis > 3) {
        THROW_IE_EXCEPTION << "[VPU] Invalid Crop axis value. Layer name is: " << layer->name;
    }
    if (cropAxis == 0) {
        THROW_IE_EXCEPTION << "[VPU] Invalid Crop axis value. Cannot crop batch channel. Layer name is: " << layer->name;
    }

    addNewStage<VpuCropStage>(
        layer->name,
        kCrop,
        layer,
        [layer, cropAxis](VpuCropStage* stage) {
            // mvTensor expects that the order of offsets is XYZ, so reverse the offsets
            for (size_t i = 0; i < layer->offset.size(); i++) {
                stage->offset[3 - cropAxis - i] = layer->offset[i];
            }
        },
        inputs,
        outputs);
}
