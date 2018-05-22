// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include "graph_transformer.h"

namespace InferenceEngine {

void replaceLayerWithNewLayer(ICNNNetwork &network, const CNNLayerPtr &layer, const CNNLayerPtr &newLayer) {
    assert(layer->name == newLayer->name);

    // Redirect srd data
    for (auto& src : layer->insData) {
        src.lock()->getInputTo()[layer->name] = newLayer;
    }
    newLayer->insData = layer->insData;

    // Redirect dst data
    for (auto& dst : layer->outData) {
        dst->creatorLayer = newLayer;
    }
    newLayer->outData = layer->outData;

    network.addLayer(newLayer);
}

}  // namespace InferenceEngine
