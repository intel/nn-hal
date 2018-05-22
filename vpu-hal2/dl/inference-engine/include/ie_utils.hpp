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


#ifndef INFERENCEENGINE_IE_UTILS_HPP_HPP
#define INFERENCEENGINE_IE_UTILS_HPP_HPP

#include <unordered_map>
#include <string>
#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {

struct LayerComplexity {
    // number of multiply and addition
    // operations for reference CPU implementation
    size_t flops;
    size_t params;
};

/**
 * @brief Computes per layer theoretical computational and memory
 * complexity.
 *
 * @param network input graph
 *
 * @return map from layer name to layer complexity
 */
std::unordered_map<std::string, LayerComplexity> getNetworkComplexity(const InferenceEngine::ICNNNetwork &network);

}  // namespace InferenceEngine

#endif  // INFERENCEENGINE_IE_UTILS_HPP_HPP
