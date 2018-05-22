////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#include "ade_util.hpp"

#include <unordered_map>
#include <utility>

#include <ie_icnn_network.hpp>
#include <ie_layers.h>

#include <util/algorithm.hpp>
#include <graph.hpp>
#include <typed_graph.hpp>

namespace InferenceEngine {
namespace {
using VisitedLayersMap = std::unordered_map<CNNLayer::Ptr, ade::NodeHandle>;
using TGraph = ade::TypedGraph<CNNLayerMetadata>;

void translateVisitLayer(VisitedLayersMap& visited,
                TGraph& gr,
                const ade::NodeHandle& prevNode,
                const CNNLayer::Ptr& layer) {
    assert(nullptr != layer);;
    assert(!util::contains(visited, layer));
    auto node = gr.createNode();
    gr.metadata(node).set(CNNLayerMetadata{layer});
    if (nullptr != prevNode) {
        gr.link(prevNode, node);
    }
    visited.insert({layer, node});
    for (auto&& data : layer->outData) {
        for (auto&& layerIt : data->inputTo) {
            auto nextLayer = layerIt.second;
            auto it = visited.find(nextLayer);
            if (visited.end() == it) {
                translateVisitLayer(visited, gr, node, nextLayer);
            } else {
                gr.link(node, it->second);
            }
        }
    }
}
}  // namespace

void translateNetworkToAde(ade::Graph& gr, ICNNNetwork& netrwork) {
    TGraph tgr(gr);
    VisitedLayersMap visited;
    InputsDataMap inputs;
    netrwork.getInputsInfo(inputs);
    for (auto&& it : inputs) {
        auto data = it.second->getInputData();
        assert(nullptr != data);
        for (auto&& layerIt : data->inputTo) {
            auto layer = layerIt.second;
            assert(nullptr != layer);
            if (!util::contains(visited, layer)) {
                translateVisitLayer(visited, tgr, nullptr, layer);
            }
        }
    }
}

const char* CNNLayerMetadata::name() {
    return "CNNLayerMetadata";
}

}  // namespace InferenceEngine
