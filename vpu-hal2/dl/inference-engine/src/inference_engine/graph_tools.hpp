//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
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

#pragma once

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <list>
#include <memory>
#include <functional>
#include "layer_transform.hpp"


#include "ie_icnn_network.hpp"
#include "cnn_network_impl.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief implementation of DFS with visiting checking to avoid multientry
 * @param visited - set to store visited layers
 * @param layer - current layer to start DFS from
 * @param visit - user callback on visited node
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 * @return false if cycle detected
 */
template<class T>
inline bool DFS(std::unordered_map<CNNLayer *, bool> &visited,
                const InferenceEngine::CNNLayerPtr &layer,
                const T &visit,
                bool visitBefore) {
    if (layer == nullptr) {
        return true;
    }

    if (visitBefore) visit(layer);
    visited[layer.get()] = false;
    for (auto &od : layer->outData) {
        for (auto nl : od->getInputTo()) {
            auto i = visited.find(nl.second.get());
            if (i != visited.end()) {
                /**
                 * cycle detected we entered still not completed node
                 */
                if (!i->second) {
                    return false;
                }
                continue;
            }
            if (!DFS(visited, nl.second, visit, visitBefore)) {
                return false;
            }
        }
    }
    if (!visitBefore) visit(layer);
    visited[layer.get()] = true;
    return true;
}


/**
 * @brief implementation of DFS in unordered graph, mean next layers not just child but also parents
 * @param visited - set to store visited layers
 * @param layer - current layer to start UnorderedDFS from
 * @param visit - user callback on visited node
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template<class T>
inline void UnorderedDFS(std::unordered_set<CNNLayer *> &visited,
                const InferenceEngine::CNNLayerPtr &layer,
                const T &visit,
                bool visitBefore) {
    if (layer == nullptr) {
        return;
    }
    if (visited.end() != visited.find(layer.get())) {
        return;
    }

    if (visitBefore) visit(layer);
    visited.insert(layer.get());

    // visit childs
    for (auto &od : layer->outData) {
        for (auto nl : od->getInputTo()) {
            UnorderedDFS(visited, nl.second, visit, visitBefore);
        }
    }

    // visit parents
    for (auto && input  : layer->insData) {
        UnorderedDFS(visited, input.lock()->getCreatorLayer().lock(), visit, visitBefore);
    }

    if (!visitBefore) visit(layer);
}

/**
 * @brief implementation of DFS with visiting checking to avoid multyentry
 * @param visited - set to store visited layers
 * @param layer - current layer to start DFS from
 * @param visit - user callback on visited node
 */
template<class T>
inline void BFS(InferenceEngine::CNNLayerPtr layer, const T &visit, int maxDepth) {
    std::set<InferenceEngine::CNNLayer*> visited;
    std::list<InferenceEngine::CNNLayerPtr> nextLayers;
    nextLayers.push_back(layer);

    int layersOnLevel = 1;
    for (; !nextLayers.empty() && maxDepth != 0;) {
        visit(*nextLayers.begin());
        for (auto &od : (*nextLayers.begin())->outData) {
            for (auto nl : od->getInputTo()) {
                if (visited.find(nl.second.get()) == visited.end()) {
                    nextLayers.push_back(nl.second);
                    visited.insert(nl.second.get());
                }
            }
        }
        nextLayers.pop_front();
        // move to nextLayer
        if (!--layersOnLevel) {
            layersOnLevel = nextLayers.size();
            maxDepth--;
        }
    }
}

}  // namespace details

/**
 * Generic DFS algorithm traverser
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template<class T>
inline bool CNNNetDFS(const InferenceEngine::CNNLayerPtr &layer, const T &visit, bool visitBefore = true) {
    if (layer == nullptr) {
        return true;
    }

    std::unordered_map < CNNLayer *, bool> visited;
    return details::DFS(visited, layer, visit, visitBefore);
}

/**
 * DFS algorithm with multiple starting nodes
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template<class Forest, class T>
inline bool CNNNetForestDFS(const Forest &heads, const T &visit, bool bVisitBefore) {
    if (heads.empty()) {
        return true;
    }

    std::unordered_map< CNNLayer *, bool> visited;
    for (auto & layer : heads) {
        if (!details::DFS(visited, layer, visit, bVisitBefore)) {
            return false;
        }
    }
    return true;
}

/**
 * Generic BFS algorithm traverser - with limiting depth
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 */
template<class T>
inline void CNNNetNBFS(const InferenceEngine::CNNLayerPtr &layer, int maxDept, const T &visit) {
    if (!layer) {
        return;
    }
    details::BFS(layer, visit, maxDept + 1);
}

/**
 * Generic BFS algorithm traverser
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 */
template<class T>
inline void CNNNetBFS(const InferenceEngine::CNNLayerPtr &layer, const T &visit) {
    if (!layer) {
        return;
    }

    details::BFS(layer, visit, -1);
}

/**
 * @brief name of the previous layer
 * @param layer
 */
inline std::string  CNNNetPrevLayerName(const InferenceEngine::CNNLayerPtr & layer) {
    InferenceEngine::CNNLayerPtr prevLayer = layer->input()->getCreatorLayer().lock();
    if (prevLayer == nullptr) {
        return layer->input()->getName();
    }
    return prevLayer->name;
}

/**
 * @brief name of the previous layer for given data
 * @param layer
 */
inline std::string  CNNNetPrevLayerName(const InferenceEngine::DataWeakPtr & dataWeak) {
    DataPtr dataStrong;

    IE_ASSERT(dataStrong = dataWeak.lock());

    CNNLayerPtr layerStrong;
    if (!(layerStrong = dataStrong->getCreatorLayer().lock())) {
        return dataStrong->getName();
    }

    return layerStrong->name;
}


/**
 * @brief pointer of previous layers
 * @param idx - index in previous layer collection
 * @param layer
 */
inline InferenceEngine::CNNLayerPtr  CNNNetPrevLayer(const InferenceEngine::CNNLayerPtr & layer, int idx = 0) {
    auto prevData = layer->insData[idx].lock();
    return prevData->getCreatorLayer().lock();
}

/**
 * @brief pointer of previous layers
 * @param idx - index in previous layer collection
 * @param layer
 */
inline InferenceEngine::CNNLayerPtr  CNNNetPrevLayer(const InferenceEngine::CNNLayer* layer, int idx = 0) {
    IE_ASSERT(layer != nullptr);
    auto prevData = layer->insData[idx].lock();
    return prevData->getCreatorLayer().lock();
}

/**
 * @brief pointer of previous layers
 * @param idx - index in previous layer collection
 * @param layer
 */
inline bool  CNNNetHasPrevLayer(const InferenceEngine::CNNLayer* layer, int idx = 0) {
    IE_ASSERT(layer != nullptr);
    if (layer->insData.empty() || layer->insData.size() <= idx) {
        return false;
    }
    auto prevData = layer->insData[idx].lock();
    return !!prevData->getCreatorLayer().lock();
}


/**
 * @brief to allow storing of LayersSP in collections ordered by  names
*/

class LayerNameLess {
 public:
     bool operator()(const CNNLayerPtr& lhs, const CNNLayerPtr& rhs) const {
         return std::less<std::string>()(lhs->name, rhs->name);
     }
};

using CNNLayerSet = std::set<CNNLayerPtr, LayerNameLess>;

/**
 * @brief returns all layers that are input or memory
 * @param network
 * @return set of input layers
 */
inline CNNLayerSet CNNNetGetAllInputLayers(ICNNNetwork &network) {
    InputsDataMap inputs;
    network.getInputsInfo(inputs);

    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer *> allLayers;

    if (inputs.empty())
        return inputLayers;

    auto & secondLayers = inputs.begin()->second->getInputData()->getInputTo();
    if (secondLayers.empty())
        return inputLayers;

    details::UnorderedDFS(allLayers, secondLayers.begin()->second, [&](CNNLayerPtr layer){
       if (layer->insData.empty()) {
           inputLayers.insert(layer);
       }
    }, false);
    return inputLayers;
}


/**
 * Sort network layers topologically
 * @param network
 * @return sorted vector
 * @throws if sorting not possible - for example if loop detected
 */
inline std::vector<CNNLayerPtr> CNNNetSortTopologically(ICNNNetwork & network) {
    std::vector<CNNLayerPtr> stackOfVisited;
    auto res = CNNNetForestDFS(CNNNetGetAllInputLayers(network), [&](CNNLayerPtr  current){
        stackOfVisited.push_back(current);
    }, false);

    if (!res) {
        THROW_IE_EXCEPTION << "Sorting not possible, due to existed loop.";
    }

    std::reverse(std::begin(stackOfVisited), std::end(stackOfVisited));

    return stackOfVisited;
}
/**
 * @brief copy Data from original graph, and insert into new graph, using layers remap information
 * @param input
 * @return
 */
inline DataPtr copyData(DataPtr input, std::unordered_map<CNNLayer*, CNNLayerPtr> &layersRemap) {
    auto newData = std::make_shared<Data>(*(input.get()));
    newData->getCreatorLayer() = layersRemap[input->getCreatorLayer().lock().get()];
    for (auto && input : newData->getInputTo()) {
        input.second = layersRemap[input.second.get()];
    }
    return newData;
}

using CNNNetPtr = std::shared_ptr<ICNNNetwork>;

/**
 * @brief deep copy of the entire network, structure using custom copier for layers
 * @return copied network
 */
template <class Copier>
inline CNNNetPtr CNNNetCopy(ICNNNetwork &input, const Copier &cp) {
    auto net = std::make_shared<details::CNNNetworkImpl>();

    // setting base args
    net->setTargetDevice(input.getTargetDevice());
    net->setPrecision(input.getPrecision());

    char name[1024];
    input.getName(name, sizeof(name));
    net->setName(name);

    // rest info is layer dependent so have to create graph clone
    std::unordered_map<CNNLayer*, CNNLayerPtr> oldToNewLayers;

    auto starters = CNNNetGetAllInputLayers(input);

    // 1st pass node creation
    auto res = CNNNetForestDFS(starters, [&](CNNLayerPtr  current){
        auto newLayer = cp(current);
        oldToNewLayers[current.get()] = newLayer;
        net->addLayer(newLayer);
    }, true);

    if (!res) {
        THROW_IE_EXCEPTION << "Copying of network not possible, due to existed loop.";
    }

    // internal utility to locate out data idx in layer
    auto findOutDataIdx = [&](DataPtr sourceData) {
        int dataIdx = -1;
        auto sourceLayer = sourceData->getCreatorLayer().lock();
        for (int j = 0; j < sourceLayer->outData.size(); j++) {
            if (sourceData.get() == sourceLayer->outData[j].get()) {
                dataIdx = j;
                break;
            }
        }
        IE_ASSERT(dataIdx != -1);
        return dataIdx;
    };

    // compares data, for copied network and in old network
    auto areEqualDatas = [&](DataPtr source, DataPtr target) {
        if (source.get() == target.get()) {
            return true;
        }

        // dims comparison -
        // actual dims value might be incorrect dueto syntetic case
        // , when getbatch() size returns value not reflect in actual data

        if (source->dims.size() != target->dims.size()) {
            return false;
        }

        // name comparison
        if (source->name != target->name) {
            return false;
        }

        // inputTO layers are identical by design
        return true;
    };
    // internal utility to locate input data idx in layer
    auto findInsDataIdx = [&](DataPtr sourceData, CNNLayerPtr layer) {
        int dataIdx = -1;
        auto sourceLayerMap = sourceData->inputTo;
        for (auto & layersMapping : sourceLayerMap) {
            if (layersMapping.second.get() != layer.get()) {
                continue;
            }
            for (int j = 0; j < layer->insData.size(); j++) {
                if (areEqualDatas(layer->insData[j].lock(), sourceData)) {
                    dataIdx = j;
                }
            }
            if (dataIdx != -1) {
                break;
            }
        }
        IE_ASSERT(dataIdx != -1);
        return dataIdx;
    };

    // 2nd pass edges creation
    CNNNetForestDFS(starters, [&](CNNLayerPtr  current){
        auto newLayer = oldToNewLayers[current.get()];
        // remap output data
        for (int i = 0; i != current->outData.size(); i++) {
            newLayer->outData[i]->getCreatorLayer() = CNNLayerWeakPtr(newLayer);

            // transfer data info for getData routine
            net->getData(newLayer->outData[i]->name) = newLayer->outData[i];

            for (auto inputTo = std::begin(newLayer->outData[i]->getInputTo());
                 inputTo != std::end(newLayer->outData[i]->getInputTo());
                 inputTo++) {
                inputTo->second = oldToNewLayers[inputTo->second.get()];
            }
        }
        // remap input data
        for (int i = 0; i != current->insData.size(); i++) {
            // found that data IDX
            auto sourceData = current->insData[i].lock();
            auto sourceLayer = sourceData->getCreatorLayer().lock();

            // find insData Entry in outData of sourceLayer
            newLayer->insData[i] = oldToNewLayers[sourceLayer.get()]->outData[findOutDataIdx(sourceData)];
        }
    }, true);

    // transfer input info
    InputsDataMap inputsInfo;
    input.getInputsInfo(inputsInfo);
    std::set<DataPtr> insDatas;
    for (auto &&info : inputsInfo) {
        for (auto secondLayer : info.second->getInputData()->inputTo) {
            auto secondLayerNew = oldToNewLayers[secondLayer.second.get()];
            InputInfo::Ptr infoNew = std::make_shared<InputInfo>();
            infoNew->setInputData(secondLayerNew->insData[findInsDataIdx(info.second->getInputData(), secondLayer.second)].lock());
            net->setInputInfo(infoNew);
        }
    }

    // transfer output info
    OutputsDataMap outmap;
    input.getOutputsInfo(outmap);
    for (auto && data : outmap) {
        ResponseDesc dsc;
        if (OK != net->addOutput(data.second->getCreatorLayer().lock()->name, findOutDataIdx(data.second), &dsc)) {
            THROW_IE_EXCEPTION << dsc.msg;
        }
    }

    // transfer batch size
    net->setBatchSize(input.getBatchSize());


    return net;
}

}  // namespace InferenceEngine