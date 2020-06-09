/*
 * INTEL CONFIDENTIAL
 * Copyright 2020 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#include "../IRLayers.h"
#include "ie_network.hpp"
#include "ie_builders.hpp"

namespace IEBuilder = InferenceEngine::Builder;
using IRBlob = android::hardware::neuralnetworks::nnhal::IRBlob;
using idx_t = InferenceEngine::idx_t;

namespace LN
{

class BuilderNode
{
public:
    idx_t weights;
    idx_t bias;


public:
    std::string nodeType;
    BuilderNode(idx_t weights_id, idx_t bias_id, std::string nodeType_id) : weights(weights_id), bias(bias_id), nodeType(nodeType_id) {}
};

class LayerNorm
{
    using idx_t = InferenceEngine::idx_t;

private:
    idx_t inputLayerId;
    idx_t inputLayerId2;
    idx_t outputLayerId;
    IEBuilder::Network* builderNetwork;
    unsigned long N;
    int layer_name_count = 0;
    std::string output_node;

public:
    LayerNorm(idx_t inLayer, idx_t inLayer2, unsigned long Num, IEBuilder::Network* bNetwork, std::string output)
    {
        inputLayerId = inLayer;
        inputLayerId2 = inLayer2;
        N = Num;
        builderNetwork = bNetwork;
        output_node = output;
    }

    IRBlob::Ptr generateBlobwithData(InferenceEngine::SizeVector dims, InferenceEngine::Layout layout, std::vector<std::vector<float>> data_to_set)
    {

        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, dims, layout);

        InferenceEngine::TBlob<float>::Ptr blob =
            std::make_shared<InferenceEngine::TBlob<float>>(td);
        blob->allocate();

        int cnt = 0;
        float* blbData = blob->buffer().as<float*>();
        size_t m = data_to_set.size();
        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < data_to_set[i].size(); j++)
            {
                blbData[cnt++] = data_to_set[i][j];
            }
        }
        return blob;
    }

    IRBlob::Ptr generateBlob(InferenceEngine::SizeVector dims, InferenceEngine::Layout layout, std::vector<std::vector<float>> data_to_set)
    {

        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, dims, layout);

        InferenceEngine::TBlob<float>::Ptr blob =
            std::make_shared<InferenceEngine::TBlob<float>>(td);
        blob->allocate();
        return blob;
    }

    idx_t addLayerNorm(IRBlob::Ptr norm_weights, IRBlob::Ptr norm_biases);
    idx_t add_Node(idx_t prevLayerID, BuilderNode *nodeToAdd, bool bias, int outputNum);

};
}
