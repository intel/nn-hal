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

#include <string>
#include <mkldnn_types.h>
#include "mkldnn_memory_node.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNMemoryOutputNode::MKLDNNMemoryOutputNode(Type type, const std::string &name) : MKLDNNNode(type, name), MKLDNNMemoryNode("unknown") {
}

MKLDNNMemoryOutputNode::MKLDNNMemoryOutputNode(InferenceEngine::CNNLayerPtr layer)
: MKLDNNNode(layer)
, MKLDNNMemoryNode(layer) {
    if (created()) {
        MKLDNNMemoryNodeVirtualEdge::registerOutput(this);
    }
}

MKLDNNMemoryOutputNode::~MKLDNNMemoryOutputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this);
}

void MKLDNNMemoryOutputNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;
}

void MKLDNNMemoryOutputNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    supportedPrimitiveDescriptors.push_back({engine,
                                             {{getParentEdgeAt(0)->getDims(), getInputDataType(), memory::format::any}},
                                             {},
                                             impl_desc_type::unknown});
}

const MKLDNNEdgePtr MKLDNNMemoryOutputNode::getChildEdgeAt(size_t idx) const {
    if (inputNode != nullptr) {
        return inputNode->getChildEdgeAt(idx);
    }
    return MKLDNNNode::getChildEdgeAt(idx);
}

void MKLDNNMemoryOutputNode::execute(mkldnn::stream strm)  {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();
    const size_t data_size = srcMemory.GetSize() / sizeof(float);

    const float *src_ptr = reinterpret_cast<const float*>(srcMemory.GetData()) +
            srcMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_ptr = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    // TODO: this can be eliminated by completely removing MKLDNN memory output NODE, to fuse it with output of prev layer
    memcpy(dst_ptr, src_ptr, srcMemory.GetSize());
}

std::string MKLDNNMemoryInputNode::nameFromCombinedName(std::string name) {
    auto idSplitter = name.find("/id=");
    return name.substr(0, idSplitter);
}

std::string MKLDNNMemoryInputNode::idFromCombinedName(std::string name) {
    auto idSplitter = name.find("/id=");
    return name.substr(idSplitter == std::string::npos ? 0 : idSplitter + 4);
}

MKLDNNMemoryInputNode::MKLDNNMemoryInputNode(Type type, const std::string &name)
    : MKLDNNInputNode(type, nameFromCombinedName(name))
    , MKLDNNMemoryNode(idFromCombinedName(name)) {
    if (created()) {
        MKLDNNMemoryNodeVirtualEdge::registerInput(this);
    }
}
MKLDNNMemoryInputNode::MKLDNNMemoryInputNode(InferenceEngine::CNNLayerPtr layer)
    : MKLDNNInputNode(layer)
    , MKLDNNMemoryNode(layer) {
    if (created()) {
        MKLDNNMemoryNodeVirtualEdge::registerInput(this);
    }
}

MKLDNNMemoryInputNode::~MKLDNNMemoryInputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this);
}

void MKLDNNMemoryNodeVirtualEdge::registerInput(MKLDNNMemoryInputNode * node) {
    // in case of output already registered
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(node->getId());
    if (sibling != nullptr) {
        auto outputNode = dynamic_cast<MKLDNNMemoryOutputNode*>(sibling);
        IE_ASSERT(outputNode != nullptr);
        outputNode->setInputNode(node);
    } else {
        getExisted()[node->getId()] = node;
    }
    // std::cout <<"[register] " << node << ", size="<< getExisted().size() <<"\n" << std::flush;
}

void MKLDNNMemoryNodeVirtualEdge::registerOutput(MKLDNNMemoryOutputNode * node) {
    // in case of output layer
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(node->getId());
    if (sibling != nullptr) {
        auto inputNode = dynamic_cast<MKLDNNMemoryInputNode*>(sibling);
        IE_ASSERT(inputNode != nullptr);
        node->setInputNode(inputNode);
    } else {
        getExisted()[node->getId()] = node;
    }
    // std::cout <<"[register] " << node << ", size="<< getExisted().size() <<"\n" << std::flush;
}
