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

#include "IRLayerNorm.h"
#include <fstream>
#include "IRBuilder.h"

using android::hardware::neuralnetworks::nnhal::IRBuilder::ModelBuilder;
using Port = InferenceEngine::Port;
using PortData = InferenceEngine::PortData;

using FCLayer = InferenceEngine::Builder::FullyConnectedLayer;
using INLayer = InferenceEngine::Builder::InputLayer;
using CONSTLayer = InferenceEngine::Builder::ConstLayer;
using ELTWISELayer = InferenceEngine::Builder::EltwiseLayer;
using SIGMOIDLayer = InferenceEngine::Builder::SigmoidLayer;
using LOGLayer = InferenceEngine::Builder::LogLayer;
using EXPLayer = InferenceEngine::Builder::ExpLayer;
using DIVBYNLayer = InferenceEngine::Builder::DivByNLayer;
using TANHLayer = InferenceEngine::Builder::TanHLayer;
using CLAMPLayer = InferenceEngine::Builder::ClampLayer;
using SCALESHIFTLayer = InferenceEngine::Builder::ScaleShiftLayer;
using FCLayer = InferenceEngine::Builder::FullyConnectedLayer;

using namespace LN;
idx_t LayerNorm::add_Node(idx_t prevLayerID, BuilderNode* nodeToAdd, bool bias, int outputNum = 2048)
{
    idx_t newNode;
    auto getLayerName = [&](std::string layerName) -> std::string
    {
        std::string strName(layerName);
        strName = "LN" + strName + "_" + std::to_string(layer_name_count++);
        return strName;
    };

    if(nodeToAdd->nodeType == "fc")
    {
        if (bias)
        {
            newNode =  builderNetwork->addLayer \
            ({{prevLayerID}, {nodeToAdd->weights}},\
            FCLayer(getLayerName("affine")).setOutputNum(outputNum));
        }
        else
        {
            newNode =  builderNetwork->addLayer \
            ({{prevLayerID}, {nodeToAdd->weights}},\
            FCLayer(getLayerName("affine")).setOutputNum(outputNum));
        }
    }
    else
    {
        ELTWISELayer eltmul_layer = ELTWISELayer(getLayerName("mul"));
        eltmul_layer.setEltwiseType(ELTWISELayer::EltwiseType::MUL);

        newNode = builderNetwork->addLayer(eltmul_layer);

        builderNetwork->connect({prevLayerID}, {newNode, 0});
        builderNetwork->connect({nodeToAdd->weights}, {newNode, 1});
    }
    return newNode;
}


idx_t LayerNorm::addLayerNorm(IRBlob::Ptr norm_weights, IRBlob::Ptr norm_biases)
{

    InferenceEngine::Layout layout = InferenceEngine::Layout::NC;
    InferenceEngine::SizeVector dims_weights = {8, N};
    InferenceEngine::SizeVector dims_weights_inverse = {N, 8};
    int cellSize = static_cast<int> (N);

    float init_value = -1.0/N;
    std::vector<std::vector<float>> mean_weights (8, std::vector<float> (N, (init_value)));
    std::vector<std::vector<float>> transpose_weights (N, std::vector<float> (8, (0)));
    for(int j= 0; j < N; j++)
    {
        transpose_weights[j][0] = 1.0;
    }

    IRBlob::Ptr mean_weights_blob = generateBlobwithData(dims_weights, layout, mean_weights);
    IRBlob::Ptr transpose_blob = generateBlobwithData(dims_weights_inverse, layout, transpose_weights);

    InferenceEngine::SizeVector dims_bias = {1, N};
    std::vector<std::vector<float>> zero_bias (1, std::vector<float> (N, 0));
    IRBlob::Ptr zero_bias_blob = generateBlobwithData(dims_bias, layout, zero_bias);


    idx_t weight_in_mean_id = builderNetwork->addLayer(CONSTLayer("weights").setData(mean_weights_blob));
    BuilderNode *mean_node = new BuilderNode(weight_in_mean_id, 0, "fc");
    idx_t mean_node_id = add_Node(inputLayerId, mean_node, false, 8);
    idx_t finalNode;
    idx_t transpose_id = builderNetwork->addLayer(CONSTLayer("weights_transpose").setData(transpose_blob));
    BuilderNode *transpose_node = new BuilderNode(transpose_id, 0, "fc");
    idx_t transpose_mul_id = add_Node(mean_node_id, transpose_node, false, cellSize);

    if (output_node == "mean")
    {
        finalNode = transpose_mul_id;
        return finalNode;
    }

    ELTWISELayer eltadd_layer = ELTWISELayer("add");
    eltadd_layer.setEltwiseType(ELTWISELayer::EltwiseType::SUM);
    auto eltadd_layer_id = builderNetwork->addLayer(eltadd_layer);
    builderNetwork->connect({transpose_mul_id}, {eltadd_layer_id, 0});
    builderNetwork->connect({inputLayerId}, {eltadd_layer_id, 1});

    std::cout << "output_node = " << output_node << std::endl;
    if (output_node == "xmean")
    {
        finalNode = eltadd_layer_id;
        return finalNode;
    }

    idx_t bias_square;
    BuilderNode *square = new BuilderNode(eltadd_layer_id, 0, "mul");
    idx_t squareNode = add_Node(eltadd_layer_id, square, false);

    if (output_node == "square")
    {
        finalNode = squareNode;
        return finalNode;
    }

    idx_t DIVBYNNodeActivationFn = builderNetwork->addLayer(DIVBYNLayer("DIVBYN") \
                                   .setPort(Port({1,N}, InferenceEngine::Precision::FP32)));
    builderNetwork->connect({squareNode}, {DIVBYNNodeActivationFn});

    if (output_node == "divbyn")
    {
        finalNode = DIVBYNNodeActivationFn;
        return finalNode;
    }

    init_value = (1.0);
    for(auto& row:mean_weights)
    {
        for(auto& col:row)
        {
            col = init_value;
        }
    }

    auto weights_in_ln = generateBlobwithData(dims_weights, layout, mean_weights);

    idx_t weight_in_ln_id = builderNetwork->addLayer(CONSTLayer("weights").setData(weights_in_ln));
    //idx_t bias_in_ln_t = builderNetwork->addLayer(CONSTLayer("bias").setData(zero_bias_blob));
    BuilderNode *log_square = new BuilderNode(weight_in_ln_id, 0, "fc");

    idx_t log_squareNode_id = add_Node(DIVBYNNodeActivationFn, log_square, false, 8);
    if (output_node == "sqsum")
    {
        finalNode = log_squareNode_id;
        return finalNode;
    }

    auto LogActivationFn = builderNetwork->addLayer(LOGLayer("LOG") \
                           .setPort(Port({1,8}, InferenceEngine::Precision::FP32)));
    builderNetwork->connect({log_squareNode_id}, {LogActivationFn});
    if (output_node == "log")
    {
        finalNode = LogActivationFn;
        return finalNode;
    }


    float *src = norm_weights->buffer().as<float*>();
    int k = 0;

    BuilderNode *exp_mul_aff = new BuilderNode(inputLayerId2, 0, "mul");
    idx_t exp_LogNode = add_Node(LogActivationFn, exp_mul_aff,false);
    if (output_node == "logsum")
    {
        finalNode = exp_LogNode;
        return finalNode;
    }

    auto expNodeActivationFn = builderNetwork->addLayer(EXPLayer("EXP") \
                               .setPort(Port({1,8}, InferenceEngine::Precision::FP32)));
    builderNetwork->connect({exp_LogNode}, {expNodeActivationFn});

    if (output_node == "exp")
    {
        finalNode = expNodeActivationFn;
        return finalNode;
    }
    idx_t transpose_exp_id = add_Node(expNodeActivationFn, transpose_node, false, cellSize);

    BuilderNode *exp_mul = new BuilderNode(eltadd_layer_id, 0, "mul");
    idx_t exp_sqrtNode = add_Node(transpose_exp_id, exp_mul, false);

    if (output_node == "norm")
    {
        finalNode = exp_sqrtNode;
        return finalNode;
    }
    idx_t norm_weights_layer = builderNetwork->addLayer(CONSTLayer("norm_weights").setData(norm_weights));
    idx_t scaleShiftLayerId = builderNetwork->addLayer({{exp_sqrtNode}},SCALESHIFTLayer("SSL"));
    builderNetwork->connect({norm_weights_layer}, {scaleShiftLayerId, 1});

    float *src2 = norm_biases->buffer().as<float*>();
    for (int i = 0; i < N; i++)
    {
        zero_bias[0][i] = *(src2 + i);
        std::cout << "zero_bias[0][i] = " << zero_bias[0][i] << "\n";
    }
    auto bias_blob = generateBlobwithData(dims_bias, layout, zero_bias);
    idx_t norm_bias_layer = builderNetwork->addLayer(CONSTLayer("norm_bias").setData(bias_blob));
    builderNetwork->connect({norm_bias_layer}, {scaleShiftLayerId, 2});
    finalNode = scaleShiftLayerId;


    return finalNode;
}
