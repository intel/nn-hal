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
#include "mkldnn_activation_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <algorithm>
#include <string>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

// TODO: (ichuraev) I don't fully sure that names of types and parameters are correct for square, abs, sqrt, linear, bounded_relu and soft_relu
caseless_map<std::string, std::function<void(GenericLayer*, mkldnn::algorithm&, float&, float&)>> MKLDNNActivationNode::initializers = {
        {"relu", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("negative_slope", 0.0f);
            beta = 0.0f;
            algorithm = eltwise_relu;
        }},
        {"elu", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 1.0f);
            beta = 0.0f;
            algorithm = eltwise_elu;
        }},
        {"tanh", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_tanh;
        }},
        {"logistic", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_logistic;
        }},
        {"square", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_square;
        }},
        {"abs", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_abs;
        }},
        {"sqrt", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_sqrt;
        }},
        {"linear", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 1.0f);
            beta = activationLayer->GetParamAsFloat("beta", 0.0f);
            algorithm = eltwise_linear;
        }},
        {"bounded_relu", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 0.0f);
            beta = 0.0f;
            algorithm = eltwise_bounded_relu;
        }},
        {"soft_relu", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_soft_relu;
        }}
};

MKLDNNActivationNode::MKLDNNActivationNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNActivationNode::MKLDNNActivationNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {
}

void MKLDNNActivationNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (!descs.empty())
        return;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    auto parentOutDims = getParentEdgeAt(0)->getDims();

    for (auto format : getAvailableFormatsForDims(parentOutDims)) {
        memory::desc in_candidate{autoBlockingDims(parentOutDims, format), inputDataType, format};
        MKLDNNDescriptor desc(std::shared_ptr<eltwise_forward::desc>(
                new eltwise_forward::desc(prop_kind::forward_scoring, getAlgorithm(), in_candidate, getAlpha(), getBeta())));
        descs.push_back(desc);
    }
}

void MKLDNNActivationNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<relu_forward::primitive_desc, relu_forward::desc>();

    prim.reset(new relu_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNActivationNode::created() {
    return getType() == Activation;
}

void MKLDNNActivationNode::initValues() {
    GenericLayer* activationLayer = getCnnLayer().get();
    if (activationLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get CNNLayer.";

    std::string type = activationLayer->type;
    CaselessEq<std::string> comparator;
    if (comparator(type, "activation"))
        type = activationLayer->GetParamAsString("type");
    if (comparator(type, "sigmoid"))
        type = "logistic";

    if (initializers.find(type) == initializers.end())
        THROW_IE_EXCEPTION << "Node " << getName() << "has unsupported activation primitive: "
                           << activationLayer->type << " : " << type;
    initializers[type](activationLayer, algorithm, alpha, beta);
    initialized = true;
}
