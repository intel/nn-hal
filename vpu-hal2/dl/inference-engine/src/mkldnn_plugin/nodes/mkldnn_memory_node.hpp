//
// INTEL CONFIDENTIAL
// Copyright 2017 Intel Corporation.
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

#include <ie_common.h>
#include "ie_algorithm.hpp"
#include "mkldnn_input_node.h"
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <map>

namespace MKLDNNPlugin {

class MKLDNNMemoryNode {
    std::string _id;
 public:
    explicit MKLDNNMemoryNode(std::string id) : _id(id) {}
    explicit MKLDNNMemoryNode(InferenceEngine::CNNLayerPtr lp) {
        if (lp->params.find("id") != lp->params.end()) {
            _id = lp->GetParamAsString("id");
        }
    }
    std::string getId() {
        return _id;
    }
    virtual void setInputNode(MKLDNNNode *) = 0;
};
class MKLDNNMemoryOutputNode;
class MKLDNNMemoryInputNode;

/**
 * @brief
 * TODO: ATTENTION: this is a temporary solution, this connection should be keep in graph
 */
class MKLDNNMemoryNodeVirtualEdge {
    using Holder = std::map<std::string, MKLDNNMemoryNode*>;
    static Holder & getExisted() {
        static Holder existed;
        return existed;
    }

    static MKLDNNMemoryNode * getByName(std::string name) {
        auto result = getExisted().find(name);
        if (result != getExisted().end()) {
            return result->second;
        }
        return nullptr;
    }

 public:
    static void registerOutput(MKLDNNMemoryOutputNode * node);
    static void registerInput(MKLDNNMemoryInputNode * node);
    static void remove(MKLDNNMemoryNode * node) {
        InferenceEngine::details::erase_if(getExisted(), [&](const Holder::value_type & it){
            return it.second == node;
        });
        // std::cout <<"[remove]   " << node << ", size="<< getExisted().size() <<"\n" << std::flush;
    }
};

class MKLDNNMemoryOutputNode : public MKLDNNNode, public MKLDNNMemoryNode {
 public:
    MKLDNNMemoryOutputNode(Type type, const std::string &name);
    explicit MKLDNNMemoryOutputNode(InferenceEngine::CNNLayerPtr layer);
    ~MKLDNNMemoryOutputNode() override;
    void createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) override;
    void initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) override;
    const MKLDNNEdgePtr getChildEdgeAt(size_t idx) const override;
    void createPrimitive() override {}
    void execute(mkldnn::stream strm) override;
    bool created() override {
        return getType() == MemoryOutput;
    }

    void setInputNode(MKLDNNNode* node) override {
        inputNode = node;
    }
 private:
    /**
     * @brief keeps reference to input sibling node
     */
    MKLDNNNode* inputNode = nullptr;
    static Register<MKLDNNMemoryOutputNode> reg;
};


class MKLDNNMemoryInputNode : public MKLDNNInputNode, public MKLDNNMemoryNode {
 protected:
    static std::string nameFromCombinedName(std::string name);
    static std::string idFromCombinedName(std::string name);
 public:
    MKLDNNMemoryInputNode(Type type, const std::string &name);
    explicit MKLDNNMemoryInputNode(InferenceEngine::CNNLayerPtr layer);
    ~MKLDNNMemoryInputNode() override;

    bool created() override {
        return getType() == MemoryInput;
    }

    void setInputNode(MKLDNNNode* node) override {}
 private:
    static Register<MKLDNNMemoryInputNode> reg;
};



}  // namespace MKLDNNPlugin

