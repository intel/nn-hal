////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#pragma once

#include <ie_blob.h>
#include <ie_layers.h>

#include <string>
#include <functional>
#include <unordered_set>
#include <vector>
#include <utility>

namespace InferenceEngine {
class ICNNNetwork;

using LayersSet = std::unordered_set<CNNLayerPtr>;

/// Split network on subgraphs based on layer affinity
///
/// @param network - source network
/// @param checkers - list of supported plugins
///
/// @return list of subgraphs
INFERENCE_ENGINE_API_CPP(std::vector<LayersSet>)
splitGraph(ICNNNetwork& network,
           const std::vector<std::string>& plugins);

/// Sort sugraphs topologically, behaviour is undefined if there are circular
/// refences between subgraps
///
/// @param subgraphs - list of subgraphs
INFERENCE_ENGINE_API_CPP(void)
sortSubgraphs(std::vector<LayersSet>& subgraphs);

}  // namespace InferenceEngine

