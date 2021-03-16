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
using NEGHALFLOGLayer = InferenceEngine::Builder::NegHalfLogLayer;
using EXPLayer = InferenceEngine::Builder::ExpLayer;
using DIVBYNLayer = InferenceEngine::Builder::DivByNLayer;
using TANHLayer = InferenceEngine::Builder::TanHLayer;
using CLAMPLayer = InferenceEngine::Builder::ClampLayer;
using SCALESHIFTLayer = InferenceEngine::Builder::ScaleShiftLayer;
using FCLayer = InferenceEngine::Builder::FullyConnectedLayer;
using CONCATLayer = InferenceEngine::Builder::ConcatLayer;
using SPLITLayer = InferenceEngine::Builder::SplitLayer;
using RESHAPELayer = InferenceEngine::Builder::ReshapeLayer;
using PERMUTELayer = InferenceEngine::Builder::PermuteLayer;
using IDENTITYLayer = InferenceEngine::Builder::IdentityLayer;

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
            newNode =  getBuiltNetwork()->addLayer \
            ({{prevLayerID}, {nodeToAdd->weights}},\
            FCLayer(getLayerName("affine")).setOutputNum(outputNum));
        }
        else
        {
            newNode =  getBuiltNetwork()->addLayer \
            ({{prevLayerID}, {nodeToAdd->weights}},\
            FCLayer(getLayerName("affine")).setOutputNum(outputNum));
        }
    }
    else
    {
        ELTWISELayer eltmul_layer = ELTWISELayer(getLayerName("mul"));
        eltmul_layer.setEltwiseType(ELTWISELayer::EltwiseType::MUL);

        newNode = getBuiltNetwork()->addLayer(eltmul_layer);

        getBuiltNetwork()->connect({prevLayerID}, {newNode, 0});
        getBuiltNetwork()->connect({nodeToAdd->weights}, {newNode, 1});
    }
    return newNode;
}


idx_t LayerNorm::addLayerNorm(IRBlob::Ptr norm_weights, IRBlob::Ptr norm_biases)
{

    InferenceEngine::Layout layout = InferenceEngine::Layout::NC;
    InferenceEngine::SizeVector dims_weights = {8, getCellSize()};
    InferenceEngine::SizeVector dims_weights_inverse = {getCellSize(), 8};
    int cellSize = static_cast<int> (getCellSize());

    float init_value = -1.0/cellSize;
    std::vector<std::vector<float>> mean_weights (8, std::vector<float> (cellSize, (init_value)));
    std::vector<std::vector<float>> transpose_weights (cellSize, std::vector<float> (8, (0)));
    for(int j= 0; j < cellSize; j++)
    {
        transpose_weights[j][0] = 1.0;
    }

    IRBlob::Ptr mean_weights_blob = generateBlobwithData(dims_weights, layout, mean_weights);
    IRBlob::Ptr transpose_blob = generateBlobwithData(dims_weights_inverse, layout, transpose_weights);

    InferenceEngine::SizeVector dims_bias = {1, getCellSize()};
    std::vector<std::vector<float>> zero_bias (1, std::vector<float> (cellSize, 0));
    IRBlob::Ptr zero_bias_blob = generateBlobwithData(dims_bias, layout, zero_bias);


    idx_t weight_in_mean_id = getBuiltNetwork()->addLayer(CONSTLayer("weights").setData(mean_weights_blob));
    BuilderNode *mean_node = new BuilderNode(weight_in_mean_id, 0, "fc");
    idx_t mean_node_id = add_Node(inputLayerId, mean_node, false, 8);
    idx_t finalNode;
    idx_t transpose_id = getBuiltNetwork()->addLayer(CONSTLayer("weights_transpose").setData(transpose_blob));
    BuilderNode *transpose_node = new BuilderNode(transpose_id, 0, "fc");
    idx_t transpose_mul_id = add_Node(mean_node_id, transpose_node, false, cellSize);

    if (output_node == "mean")
    {
        finalNode = transpose_mul_id;
        return finalNode;
    }

    ELTWISELayer eltadd_layer = ELTWISELayer("add");
    eltadd_layer.setEltwiseType(ELTWISELayer::EltwiseType::SUM);
    auto eltadd_layer_id = getBuiltNetwork()->addLayer(eltadd_layer);
    getBuiltNetwork()->connect({transpose_mul_id}, {eltadd_layer_id, 0});
    getBuiltNetwork()->connect({inputLayerId}, {eltadd_layer_id, 1});

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

    idx_t DIVBYNNodeActivationFn = getBuiltNetwork()->addLayer(DIVBYNLayer("DIVBYN") \
                                   .setPort(Port({1,getCellSize()}, InferenceEngine::Precision::FP32)));
    getBuiltNetwork()->connect({squareNode}, {DIVBYNNodeActivationFn});

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

    idx_t weight_in_ln_id = getBuiltNetwork()->addLayer(CONSTLayer("weights").setData(weights_in_ln));
    //idx_t bias_in_ln_t = getBuiltNetwork()->addLayer(CONSTLayer("bias").setData(zero_bias_blob));
    BuilderNode *log_square = new BuilderNode(weight_in_ln_id, 0, "fc");

    idx_t log_squareNode_id = add_Node(DIVBYNNodeActivationFn, log_square, false, 8);
    if (output_node == "sqsum")
    {
        finalNode = log_squareNode_id;
        return finalNode;
    }

    auto LogActivationFn = getBuiltNetwork()->addLayer(NEGHALFLOGLayer("NEGHALFLOG") \
                           .setPort(Port({1, 8}, InferenceEngine::Precision::FP32)));
    getBuiltNetwork()->connect({log_squareNode_id}, {LogActivationFn});
    if (output_node == "neglog")
    {
        finalNode = LogActivationFn;
        return finalNode;
    }

    float *src = norm_weights->buffer().as<float*>();
    auto expNodeActivationFn = getBuiltNetwork()->addLayer(EXPLayer("EXP") \
                               .setPort(Port({1, 8}, InferenceEngine::Precision::FP32)));
    getBuiltNetwork()->connect({LogActivationFn}, {expNodeActivationFn});

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
    idx_t norm_weights_layer = getBuiltNetwork()->addLayer(CONSTLayer("norm_weights").setData(norm_weights));
    idx_t scaleShiftLayerId = getBuiltNetwork()->addLayer({{exp_sqrtNode}},SCALESHIFTLayer("SSL"));
    getBuiltNetwork()->connect({norm_weights_layer}, {scaleShiftLayerId, 1});

    float *src2 = norm_biases->buffer().as<float*>();
    for (int i = 0; i < cellSize; i++)
    {
        zero_bias[0][i] = *(src2 + i);
    }
    auto bias_blob = generateBlobwithData(dims_bias, layout, zero_bias);
    idx_t norm_bias_layer = getBuiltNetwork()->addLayer(CONSTLayer("norm_bias").setData(bias_blob));
    getBuiltNetwork()->connect({norm_bias_layer}, {scaleShiftLayerId, 2});
    finalNode = scaleShiftLayerId;


    return finalNode;
}

idx_t BatchedLayerNorm::addBatchedLayerNorm(LstmParams& params)
{
    int cellSize = static_cast<int> (getCellSize());
    InferenceEngine::Layout layout = InferenceEngine::Layout::NC;
    InferenceEngine::SizeVector dims_weights = {8, getCellSize()};
    InferenceEngine::SizeVector dims_weights_LN = {getCellSize(), getCellSize()};
    InferenceEngine::SizeVector dims_weights_inverse = {getCellSize(), 8};

    int numGates = 4;
    if (inputGateLayerId == -1) {
        //VLOG(L1, "NUmber of gatestt are 3");
        numGates = 3; // CIFG layer
    }

    float init_value = -1.0/cellSize;
    std::vector<std::vector<float>> mean_weights (8, std::vector<float> (cellSize, (init_value)));
    IRBlob::Ptr mean_weights_blob = generateBlobwithData(dims_weights, layout, mean_weights);

    std::vector<std::vector<float>> transpose_weights (cellSize, std::vector<float> (8, (0)));
    for(int j= 0; j < cellSize; j++)
    {
        transpose_weights[j][0] = 1.0;
    }

    IRBlob::Ptr transpose_blob = generateBlobwithData(dims_weights_inverse, layout, transpose_weights);
    InferenceEngine::SizeVector dims_bias = {1, getCellSize()};
    std::vector<std::vector<float>> zero_bias (1, std::vector<float> (cellSize, 0));
    std::vector<std::vector<float>> norm_bias (1, std::vector<float> (cellSize, 0));
    IRBlob::Ptr zero_bias_blob = generateBlobwithData(dims_bias, layout, zero_bias);

    idx_t concatLayerId = getBuiltNetwork()->addLayer(CONCATLayer("concat") \
                                .setAxis(0) \
                                .setInputPorts({Port({1, getCellSize()}), Port({1, getCellSize()})}) \
                                .setOutputPort(Port({2, getCellSize()})));
    idx_t concatLayer_2_Id = getBuiltNetwork()->addLayer(CONCATLayer("concat_2") \
                                .setAxis(0) \
                                .setInputPorts({Port({1, getCellSize()}), Port({2, getCellSize()})}) \
                                .setOutputPort(Port({3, getCellSize()})));

    // Concatinate accumulated InputGate Forget Gate , Cell Gate and output gate
    // For Batched Input to Layer Normalization

    idx_t permuteLayerId, reshapeLayer_idx, reshapeXLayer_idx;
    if (inputGateLayerId != -1) {
        idx_t concatLayer_final_Id = getBuiltNetwork()->addLayer(CONCATLayer("concat") \
                                .setAxis(0) \
                                .setInputPorts({Port({1, getCellSize()}), Port({3, getCellSize()})}) \
                                .setOutputPort(Port({4, getCellSize()})));

        getBuiltNetwork()->connect({inputGateLayerId}, {concatLayerId, 0});
        getBuiltNetwork()->connect({forgetGateLayerId}, {concatLayerId, 1});

        getBuiltNetwork()->connect({cellGateLayerId}, {concatLayer_2_Id, 0});
        getBuiltNetwork()->connect({concatLayerId}, {concatLayer_2_Id, 1});

        getBuiltNetwork()->connect({outputGateLayerId}, {concatLayer_final_Id, 0});
        getBuiltNetwork()->connect({concatLayer_2_Id}, {concatLayer_final_Id, 1});

        permuteLayerId = getBuiltNetwork()->addLayer({concatLayer_final_Id}, PERMUTELayer("permute") \
                               .setOrder({1, 0}) \
                               .setOutputPort(Port({getCellSize(), static_cast<unsigned long>(numGates)})));
        reshapeLayer_idx = getBuiltNetwork()->addLayer({permuteLayerId}, RESHAPELayer("reshape_1").setDims({numGates, cellSize}) \
	                                                       .setOutputPort(Port({static_cast<unsigned long>(numGates), getCellSize()})));
    } else {
        getBuiltNetwork()->connect({cellGateLayerId}, {concatLayerId, 0});
        getBuiltNetwork()->connect({forgetGateLayerId}, {concatLayerId, 1});

        getBuiltNetwork()->connect({outputGateLayerId}, {concatLayer_2_Id, 0});
        getBuiltNetwork()->connect({concatLayerId}, {concatLayer_2_Id, 1});

        permuteLayerId = getBuiltNetwork()->addLayer({concatLayer_2_Id}, PERMUTELayer("permute") \
                               .setOrder({1, 0}) \
                               .setOutputPort(Port({getCellSize(), static_cast<unsigned long>(numGates)})));
        reshapeLayer_idx = getBuiltNetwork()->addLayer({permuteLayerId}, RESHAPELayer("reshape_1").setDims({numGates, cellSize}) \
	                                                       .setOutputPort(Port({static_cast<unsigned long>(numGates), getCellSize()})));
    }

    // Calculate Mean
    idx_t weight_in_mean_id = getBuiltNetwork()->addLayer(CONSTLayer("weights").setData(mean_weights_blob));
    BuilderNode *mean_node = new BuilderNode(weight_in_mean_id, 0, "fc");
    idx_t mean_node_id = add_Node(reshapeLayer_idx, mean_node, false, 8);

    idx_t transpose_id = getBuiltNetwork()->addLayer(CONSTLayer("weights_transpose").setData(transpose_blob));
    BuilderNode *transpose_node = new BuilderNode(transpose_id, 0, "fc");
    idx_t transpose_mul_id = add_Node(mean_node_id, transpose_node, false, cellSize);

    // Calculate X - mean
    // Reshape for Eltwise operations  (X - MEAN)
    idx_t ssl_weights, ssl_bias;
    reshapeLayer_idx = getBuiltNetwork()->addLayer({transpose_mul_id}, RESHAPELayer("reshape_mean_1X4N").setDims({1, (numGates * cellSize)}) \
                                                        .setOutputPort(Port({1, (numGates * getCellSize())})));

    reshapeXLayer_idx = getBuiltNetwork()->addLayer({permuteLayerId}, RESHAPELayer("reshape_x_1X4N").setDims({1, numGates * cellSize}) \
                                                        .setOutputPort(Port({1, numGates * getCellSize()})));
    // Scale Shift Layer Needed for before Eltwise SUM
    std::vector<std::vector<float>> identity_vec (1, std::vector<float> (numGates * cellSize, 1.0));
    IRBlob::Ptr identity_vec_blob = generateBlobwithData({1, numGates * getCellSize()}, layout, identity_vec);
    ssl_weights = getBuiltNetwork()->addLayer(CONSTLayer("SSL_x-mean").setData(identity_vec_blob));
    std::vector<std::vector<float>> zero_bias_4N (1, std::vector<float> (numGates * cellSize, 0));
    IRBlob::Ptr zero_bias_blob_4N = generateBlobwithData({1, numGates * getCellSize()}, layout, zero_bias_4N);
    ssl_bias = getBuiltNetwork()->addLayer(CONSTLayer("zero_bias").setData(zero_bias_blob_4N));

    idx_t scaleShiftLayerId = getBuiltNetwork()->addLayer({{reshapeXLayer_idx}}, SCALESHIFTLayer("SSL"));
    getBuiltNetwork()->connect({ssl_weights}, {scaleShiftLayerId, 1});
    getBuiltNetwork()->connect({ssl_bias}, {scaleShiftLayerId, 2});

    ELTWISELayer eltadd_layer = ELTWISELayer("add");
    eltadd_layer.setEltwiseType(ELTWISELayer::EltwiseType::SUM);
    auto x_mean_id = getBuiltNetwork()->addLayer(eltadd_layer);
    getBuiltNetwork()->connect({reshapeLayer_idx}, {x_mean_id, 1});
    getBuiltNetwork()->connect({scaleShiftLayerId}, {x_mean_id, 0});

    // Calculate (x-mean)^2
    BuilderNode *square = new BuilderNode(x_mean_id, 0, "mul");
    idx_t squareNode = add_Node(x_mean_id, square, false);

    // Calculate ( x - mean ^2) / 4N
    idx_t DIVBYNNodeActivationFn = getBuiltNetwork()->addLayer(DIVBYNLayer("DIVBYN") \
                                .setPort(Port({1, getCellSize() * numGates}, InferenceEngine::Precision::FP32)));
    getBuiltNetwork()->connect({squareNode}, {DIVBYNNodeActivationFn});
    reshapeLayer_idx = getBuiltNetwork()->addLayer({DIVBYNNodeActivationFn}, RESHAPELayer("reshape_1").setDims({numGates, cellSize}) \
                                                        .setOutputPort(Port({static_cast<unsigned long>(numGates), getCellSize()})));

    // Calculate SUM( (x - mean)^ 2 / 4 cellSize)
    init_value = 1.0f;
    std::vector<std::vector<float>> sum_weights (8, std::vector<float> (cellSize, (init_value)));
    auto weights_sum = generateBlobwithData(dims_weights, layout, sum_weights);
    idx_t weight_sum_id = getBuiltNetwork()->addLayer(CONSTLayer("weights").setData(weights_sum));

    BuilderNode *sum_node = new BuilderNode(weight_sum_id, 0, "fc");
    idx_t sum_Node_id = add_Node(reshapeLayer_idx, sum_node, false, 8);

    // Calculate log (SUM( (x - mean)^ 2 / 4 cellSize))
    auto NegHalfLogActivationFn = getBuiltNetwork()->addLayer(NEGHALFLOGLayer("NEGHALFLOG") \
                        .setPort(Port({static_cast<unsigned long>(numGates), 8}, InferenceEngine::Precision::FP32)));
    getBuiltNetwork()->connect({sum_Node_id}, {NegHalfLogActivationFn});

    NegHalfLogActivationFn = getBuiltNetwork()->addLayer({NegHalfLogActivationFn}, RESHAPELayer("Reshape_Log") \
                                                .setDims({1, numGates * 8}) \
                                                .setOutputPort(Port({1, static_cast<unsigned long>(numGates * 8)})));
    NegHalfLogActivationFn = getBuiltNetwork()->addLayer({NegHalfLogActivationFn}, IDENTITYLayer("iden_infg"));

    auto expNodeActivationFn = getBuiltNetwork()->addLayer(EXPLayer("EXP") \
                            .setPort(Port({1, static_cast<unsigned long>(numGates * 8)}, InferenceEngine::Precision::FP32)));
    getBuiltNetwork()->connect({NegHalfLogActivationFn}, {expNodeActivationFn});

    reshapeLayer_idx = getBuiltNetwork()->addLayer({expNodeActivationFn}, RESHAPELayer("reshape_exp").setDims({numGates, 8}) \
                                                        .setOutputPort(Port({static_cast<unsigned long>(numGates), 8})));

    BuilderNode *transpose_node_exp = new BuilderNode(transpose_id, 0, "fc");
    idx_t transpose_exp = add_Node(reshapeLayer_idx, transpose_node_exp, false, cellSize);

    reshapeLayer_idx = getBuiltNetwork()->addLayer({transpose_exp}, RESHAPELayer("reshape").setDims({1, (numGates * cellSize)}) \
                                                        .setOutputPort(Port({1, numGates * getCellSize()})));

    // Calculate x - mean * exp (log (SUM( (x - mean)^ 2 / 4 cellSize))) ...
    BuilderNode *exp_mul = new BuilderNode(x_mean_id, 0, "mul");
    idx_t exp_sqrtNode = add_Node(reshapeLayer_idx, exp_mul, false);

    // Permute to  cellSize X 4 before Spliting
    reshapeLayer_idx = getBuiltNetwork()->addLayer({exp_sqrtNode}, RESHAPELayer("reshape_4XN").setDims({numGates,  cellSize}) \
                                                        .setOutputPort(Port({static_cast<unsigned long>(numGates), getCellSize()})));

    permuteLayerId = getBuiltNetwork()->addLayer({reshapeLayer_idx}, PERMUTELayer("permute_NX4") \
                            .setOrder({1, 0}) \
                            .setInputPort(Port({static_cast<unsigned long>(numGates), getCellSize()})) \
                            .setOutputPort(Port({getCellSize(), static_cast<unsigned long>(numGates)})));

    reshapeLayer_idx = getBuiltNetwork()->addLayer({permuteLayerId}, RESHAPELayer("reshape_1X4N").setDims({1, numGates * cellSize}) \
                                                        .setOutputPort(Port({1, numGates * getCellSize()})));

    idx_t splitLayer_3N, splitLayer_2N, splitLayer_1N;
    if (inputGateLayerId != -1) {
        std::vector<std::vector<float>> identity_vec_3N (1, std::vector<float> (cellSize * 3, 1.0));
        auto id_vec_3N_blob = generateBlobwithData({1, getCellSize() * 3 }, layout, identity_vec_3N);
        idx_t id_vec_3N_blob_layer = getBuiltNetwork()->addLayer(CONSTLayer("norm_weights").setData(id_vec_3N_blob));
        std::vector<std::vector<float>> identity_vec_2N (1, std::vector<float> (cellSize * 2, 1.0));
        auto id_vec_2N_blob = generateBlobwithData({1, getCellSize() * 2 }, layout, identity_vec_2N);
        idx_t id_vec_2N_blob_layer = getBuiltNetwork()->addLayer(CONSTLayer("norm_weights").setData(id_vec_2N_blob));

        // Split 4 * cellSize -> 1 * cellSize + 3 * cellSize
        splitLayer_3N = getBuiltNetwork()->addLayer({reshapeLayer_idx}, SPLITLayer("split_3N").setAxis(1) \
                                        .setInputPort({Port({1, 4 * getCellSize()})}) \
                                        .setOutputPorts({Port({1, 1 * getCellSize()}), Port({1, 3 * getCellSize()})}));

        // Add SSL Layer Needed before splitting again
        scaleShiftLayerId = getBuiltNetwork()->addLayer({{splitLayer_3N, 1}},SCALESHIFTLayer("SSL_3N"));
        getBuiltNetwork()->connect({id_vec_3N_blob_layer}, {scaleShiftLayerId, 1});

        // Split 3 * cellSize -> 1 * cellSize + 2 * cellSize
        splitLayer_2N = getBuiltNetwork()->addLayer({scaleShiftLayerId}, SPLITLayer("split_2N").setAxis(1) \
                                        .setInputPort({Port({1, (3 * getCellSize())})}) \
                                        .setOutputPorts({Port({1, getCellSize()}), Port({1, 2 * getCellSize()})}));
        scaleShiftLayerId = getBuiltNetwork()->addLayer({{splitLayer_2N, 1}},SCALESHIFTLayer("SSL_2N"));
        getBuiltNetwork()->connect({id_vec_2N_blob_layer}, {scaleShiftLayerId, 1});

        // // Split 2 * cellSize -> 1 * cellSize + 1 * cellSize
        splitLayer_1N = getBuiltNetwork()->addLayer({scaleShiftLayerId}, SPLITLayer("split_N").setAxis(1) \
                                        .setInputPort({Port({1, (2 * getCellSize())})}) \
                                        .setOutputPorts({Port({1, getCellSize()}), Port({1, getCellSize()})}));
    } else {
        std::vector<std::vector<float>> identity_vec_2N (1, std::vector<float> (cellSize * 2, 1.0));
        auto id_vec_2N_blob = generateBlobwithData({1, getCellSize() * 2 }, layout, identity_vec_2N);
        idx_t id_vec_2N_blob_layer = getBuiltNetwork()->addLayer(CONSTLayer("norm_weights").setData(id_vec_2N_blob));

        // Split 3 * cellSize -> 1 * cellSize + 2 * cellSize
        splitLayer_2N = getBuiltNetwork()->addLayer({reshapeLayer_idx}, SPLITLayer("split_2N").setAxis(1) \
                                        .setInputPort({Port({1, 3 * getCellSize()})}) \
                                        .setOutputPorts({Port({1, 1 * getCellSize()}), Port({1, 2 * getCellSize()})}));

        // Add SSL Layer Needed before splitting again
        scaleShiftLayerId = getBuiltNetwork()->addLayer({{splitLayer_2N, 1}},SCALESHIFTLayer("SSL_2N"));
        getBuiltNetwork()->connect({id_vec_2N_blob_layer}, {scaleShiftLayerId, 1});

        // Split 2 * cellSize -> 1 * cellSize + 1 * cellSize
        splitLayer_1N = getBuiltNetwork()->addLayer({scaleShiftLayerId}, SPLITLayer("split_N").setAxis(1) \
                                        .setInputPort({Port({1, (2 * getCellSize())})}) \
                                        .setOutputPorts({Port({1, getCellSize()}), Port({1, getCellSize()})}));
    }

    // Normalize Output Gate
    idx_t norm_weights_layer = getBuiltNetwork()->addLayer(CONSTLayer("OutputG_norm_weights").setData(params.outputLayerNormWeights.data));
    scaleShiftLayerId = getBuiltNetwork()->addLayer({{(numGates > 3)?splitLayer_3N:splitLayer_2N, 0}},SCALESHIFTLayer("OutputG_SSL"));
    getBuiltNetwork()->connect({norm_weights_layer}, {scaleShiftLayerId, 1});

    float *src2 = params.outputGateBias.data->buffer().as<float*>();
    for (int i = 0; i < cellSize; i++)
    {
        norm_bias[0][i] = *(src2 + i);
    }

    auto bias_blob = generateBlobwithData(dims_bias, layout, norm_bias);
    idx_t norm_bias_layer = getBuiltNetwork()->addLayer(CONSTLayer("OutputG_norm_bias").setData(bias_blob));
    getBuiltNetwork()->connect({norm_bias_layer}, {scaleShiftLayerId, 2});
    outputGateLayerId = scaleShiftLayerId;

    // Normalize Cell Gate
    norm_weights_layer = getBuiltNetwork()->addLayer(CONSTLayer("CellG_norm_weights").setData(params.cellLayerNormWeights.data));
    scaleShiftLayerId = getBuiltNetwork()->addLayer({{(numGates > 3)?splitLayer_2N:splitLayer_1N, 0}},SCALESHIFTLayer("CellG_SSL"));
    getBuiltNetwork()->connect({norm_weights_layer}, {scaleShiftLayerId, 1});

    src2 = params.cellBias.data->buffer().as<float*>();
    for (int i = 0; i < cellSize; i++)
    {
        norm_bias[0][i] = *(src2 + i);
    }
    bias_blob = generateBlobwithData(dims_bias, layout, norm_bias);
    norm_bias_layer = getBuiltNetwork()->addLayer(CONSTLayer("CellG_norm_bias").setData(bias_blob));
    getBuiltNetwork()->connect({norm_bias_layer}, {scaleShiftLayerId, 2});
    cellGateLayerId = scaleShiftLayerId;

    if (numGates >  3) {
        // Normalize Input Gate
        norm_weights_layer = getBuiltNetwork()->addLayer(CONSTLayer("InputG_norm_weights").setData(params.inputLayerNormWeights.data));
        scaleShiftLayerId = getBuiltNetwork()->addLayer({{splitLayer_1N, 0}},SCALESHIFTLayer("InputG_SSL"));
        getBuiltNetwork()->connect({norm_weights_layer}, {scaleShiftLayerId, 1});

        src2 = params.inputGateBias.data->buffer().as<float*>();
        for (int i = 0; i < cellSize; i++)
        {
            norm_bias[0][i] = *(src2 + i);
        }
        bias_blob = generateBlobwithData(dims_bias, layout, norm_bias);
        norm_bias_layer = getBuiltNetwork()->addLayer(CONSTLayer("InputG_norm_bias").setData(bias_blob));
        getBuiltNetwork()->connect({norm_bias_layer}, {scaleShiftLayerId, 2});
        inputGateLayerId = scaleShiftLayerId;
    }

    // Normalize Forget Gate
    norm_weights_layer = getBuiltNetwork()->addLayer(CONSTLayer("ForgetG_norm_weights").setData(params.forgetLayerNormWeights.data));
    scaleShiftLayerId = getBuiltNetwork()->addLayer({{splitLayer_1N, 1}},SCALESHIFTLayer("ForgetG_SSL"));
    getBuiltNetwork()->connect({norm_weights_layer}, {scaleShiftLayerId, 1});

    src2 = params.forgetGateBias.data->buffer().as<float*>();
    for (int i = 0; i < cellSize; i++)
    {
        norm_bias[0][i] = *(src2 + i);
    }

    bias_blob = generateBlobwithData(dims_bias, layout, norm_bias);
    norm_bias_layer = getBuiltNetwork()->addLayer(CONSTLayer("ForgetG_norm_bias").setData(bias_blob));
    getBuiltNetwork()->connect({norm_bias_layer}, {scaleShiftLayerId, 2});
    forgetGateLayerId = scaleShiftLayerId;

    return forgetGateLayerId;
}
