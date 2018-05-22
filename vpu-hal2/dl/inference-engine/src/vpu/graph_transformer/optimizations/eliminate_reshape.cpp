//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation.
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

void GraphTransformerImpl::eliminateReshapeStages() {
    for (auto &stage : _stages) {
        if (stage->optimized) {
            continue;
        }

        if (stage->type == kReshape) {
            assert(stage->layer != nullptr);

            assert(stage->layer->insData.size() == 1);
            auto layerInput = stage->layer->insData[0].lock();
            assert(layerInput != nullptr);

            assert(stage->layer->outData.size() == 1);
            auto layerOutput = stage->layer->outData[0];
            assert(layerOutput != nullptr);


            if (stage->outputs[0]->consumers.size() == 1) {
                auto consumerOfReshape = *stage->outputs[0]->consumers.begin();
                if (consumerOfReshape->type == kFC) {
                    stage->outputs[0]->consumers.erase(consumerOfReshape);

                    auto &inputReshape = stage->inputs[0];
                    inputReshape->consumers.erase(stage);
                    inputReshape->consumers.insert(consumerOfReshape);

                    consumerOfReshape->inputs.clear();
                    consumerOfReshape->inputs.push_back(inputReshape);
                    stage->optimized = true;
                }
            }

            if (!stage->optimized) {
                // 3D to other dim
                if (layerInput->dims.size() == 4 && layerInput->dims.size() != layerOutput->dims.size()) {
                    continue;
                }
                // 3D->3D with channel number change
                if (layerInput->dims.size() == 4 && layerOutput->dims.size() == 4) {
                    int IN_C = layerInput->dims[2];
                    int OUT_C = layerOutput->dims[2];
                    if (IN_C != OUT_C)
                        continue;
                }

                auto input = stage->inputs[0];
                auto output = stage->outputs[0];

                if (input->index != IndexBSS)
                    continue;
                if (output->index != IndexBSS)
                    continue;

                output->producer = nullptr;
                output->producerOutInd = -1;
                output->parent = input;
                input->subData.insert(output);

                stage->optimized = true;
            }
        }
    }
}
