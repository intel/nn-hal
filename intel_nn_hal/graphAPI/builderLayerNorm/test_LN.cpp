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

#include "IRLayer.h"
#include "IRBuilder.h"
#include "IRLayers.h"
#include "ie_builders.hpp"
#include "ie_network.hpp"
#include "IRLayerNorm.h"
#include "tflite_Ln.h"
#include "BuilderNetwork.h"
#include "gna_config.hpp"
#include "ie_plugin_cpp.hpp"
#include <inference_engine.hpp>
#include <fstream>
#include <cmath>
#include <cstdlib>

namespace IEBuilder = InferenceEngine::Builder;

using IRBlob = android::hardware::neuralnetworks::nnhal::IRBlob;
using android::hardware::neuralnetworks::nnhal::IRBuilder::ModelBuilder;
using idx_t = InferenceEngine::idx_t;
using Port = InferenceEngine::Port;
using PortData = InferenceEngine::PortData;

using FCLayer = InferenceEngine::Builder::FullyConnectedLayer;
using INLayer = InferenceEngine::Builder::InputLayer;
using CONSTLayer = InferenceEngine::Builder::ConstLayer;
using ELTWISELayer = InferenceEngine::Builder::EltwiseLayer;
using SIGMOIDLayer = InferenceEngine::Builder::SigmoidLayer;
using LOGLayer = InferenceEngine::Builder::LogLayer;
//using NEGHALFLOGLayer = InferenceEngine::Builder::NegHalfLogLayer;
using EXPLayer = InferenceEngine::Builder::ExpLayer;
using DIVBYNLayer = InferenceEngine::Builder::DivByNLayer;
using TANHLayer = InferenceEngine::Builder::TanHLayer;
using CLAMPLayer = InferenceEngine::Builder::ClampLayer;
using SCALESHIFTLayer = InferenceEngine::Builder::ScaleShiftLayer;
using FCLayer = InferenceEngine::Builder::FullyConnectedLayer;
using CONCATLayer = InferenceEngine::Builder::ConcatLayer;

IRBlob::Ptr generateBlobwithData(InferenceEngine::SizeVector dims, InferenceEngine::Layout layout, std::vector<std::vector<float>> data_to_set) {

        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, dims, layout);

        InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
        blob->allocate();

        int cnt = 0;
        float* blbData = blob->buffer().as<float*>();
        size_t m = data_to_set.size();
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < data_to_set[i].size(); j++) {
            blbData[cnt++] = data_to_set[i][j];
            }
        }
        return blob;
    }

void log_term(std::vector<float> &input_vector) {
    for (int i = 0; i < input_vector.size(); i++)
        input_vector[i] = log(input_vector[i]);
    return;
}

void log_sum_term(std::vector<float> &input_vector) {
    for (int i = 0; i < input_vector.size(); i++)
        input_vector[i] = -0.5 * log(input_vector[i]);
    return;
}

void exp_term(std::vector<float> &input_vector) {
    for (int i = 0; i < input_vector.size(); i++)
        input_vector[i] = exp(-0.5 * log(input_vector[i]));
    return;
}

void final_term(std::vector<float> &input_vector, std::vector<float> &x_mean) {
    for (int i = 0; i < input_vector.size(); i++)
        input_vector[i] = input_vector[i] * x_mean[i];
    return;
}
void init_sum_term(std::vector<float> &input_vector) {
    float sum = accumulate(input_vector.begin(), input_vector.end(), 0.0);
    for (auto& element : input_vector) {
        element = sum;
    }
    return;
}

void mean_term(std::vector<float> &input_vector) {
    float mean = accumulate(input_vector.begin(), input_vector.end(), 0.0)/input_vector.size();

    for (auto& element : input_vector) {
        element = -mean;
    }
    return;
}
void x_mean_term(std::vector<float> &input_vector) {
    float mean = accumulate(input_vector.begin(), input_vector.end(), 0.0)/input_vector.size();

    for (auto& element : input_vector) {
        element -= mean;
    }
    return;
}

void square_term(std::vector<float> &input_vector) {
    float mean = accumulate(input_vector.begin(), input_vector.end(), 0.0)/input_vector.size();

    for (auto& element : input_vector) {
        element -= mean;
     //   element /= ((input_vector.size() > 161) ? 161 : input_vector.size());
        element *= element;
    }
    return;
}

void divbyn_term(std::vector<float> &input_vector) {
    float mean = accumulate(input_vector.begin(), input_vector.end(), 0.0)/input_vector.size();

    for (auto& element : input_vector) {
        element -= mean;
     //   element /= ((input_vector.size() > 161) ? 161 : input_vector.size());
        element *= element;
        element /= input_vector.size();
    }
    return;
}

void sqsum_term(std::vector<float> &input_vector) {
    float sum = 0;
    sum = accumulate(input_vector.begin(), input_vector.end(), sum);
    for (auto& element : input_vector)
        element = sum;
    return;
}

float rand_no(float min, float max)
{
    return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;
}

int main(int argc, char* argv[]) {

    if(argc != 5) {
        std::cout << "Usage : ./test_LN value_of_N output_node input_SF range" << std::endl;
        return EXIT_FAILURE;
    }

    std::string cellSize_str{argv[1]};
    std::string output_node{argv[2]};
    std::string scale_factor{argv[3]};
    std::string range{argv[4]};

    unsigned long cellSize = std::stoi(cellSize_str);
    float range_no = std::stof(range);

    ModelBuilder* LNBuilderModel = new ModelBuilder();
    LNBuilderModel->initializeBuilder();

    InferenceEngine::Layout layout = InferenceEngine::Layout::NC;

    std::vector<std::vector<float>> norm_weights (1, std::vector<float> (cellSize, 0));
    InferenceEngine::SizeVector dims_bias = {1, cellSize};
    IRBlob::Ptr norm_weights_blob = generateBlobwithData(dims_bias, layout, norm_weights);
    std::vector<std::vector<float>> bias_to_set (1, std::vector<float> (cellSize, (0)));
    IRBlob::Ptr bias_in = generateBlobwithData(dims_bias, layout, bias_to_set);

    idx_t inputLayerId;
    idx_t inputLayerId2;

    inputLayerId = LNBuilderModel->getBuilderNetwork()->getBuilder()->addLayer(INLayer("input") \
                                                    .setPort(Port({1,cellSize})));

    LN::LayerNorm *LN = new LN::LayerNorm(inputLayerId, inputLayerId2, cellSize, LNBuilderModel->getBuilderNetwork()->getBuilder(), output_node);

    auto final_layer = LN->addLayerNorm(norm_weights_blob, bias_in);
    LNBuilderModel->getBuilderNetwork()->mConnections.push_back(final_layer);
    LNBuilderModel->getBuilderNetwork()->finalMemLayerId = final_layer;
    LNBuilderModel->getBuilderNetwork()->getBuilder()->addLayer({final_layer},
                                                    InferenceEngine::Builder::OutputLayer("output_layer"));
    auto network = LNBuilderModel->convertBuilder();

    std::map<std::string, std::string> config;
	//config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
    std::map<std::string, std::string> gnaPluginConfig;
    gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE] = "GNA_HW";
    gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION] = "I16";

    std::string scaleFactorConfigKey_1 = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(0);
    std::string scaleFactorConfigKey_2 = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(1);
    gnaPluginConfig[scaleFactorConfigKey_1] = scale_factor;
    gnaPluginConfig[scaleFactorConfigKey_2] = "32766";
    gnaPluginConfig[GNA_CONFIG_KEY(COMPACT_MODE)] = CONFIG_VALUE(NO);
    config.insert(std::begin(gnaPluginConfig), std::end(gnaPluginConfig));

    // InferenceEngine::CNNNetwork passed_network({network});
    std::string networkName = network->getName();

	InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork passed_network = ie.ReadNetwork("/data/local/tmp/network.xml");
    //CNNNetwork network = ie.ReadNetwork(input_model, input_model.substr(0, input_model.size() - 4) + WEIGHTS_EXT);
    passed_network.setBatchSize(1);

    InferenceEngine::InputInfo::Ptr input_info = passed_network.getInputsInfo().begin()->second;
    std::string input_name = passed_network.getInputsInfo().begin()->first;
    std::cout << "input name = " << input_name << "\n";
    input_info->setPrecision(InferenceEngine::Precision::FP32);


    auto executable_network = ie.LoadNetwork(passed_network, "GNA", config);
    std::cout << "Loaded Network\n";
    auto inferRequest = executable_network.CreateInferRequest();

    std::vector<InferenceEngine::Blob::Ptr> ptrInputBlobs;
    for (auto& input : passed_network.getInputsInfo()) {
        //VLOG(L1,"%p", (void*)enginePtr->inferRequest.GetBlob(input.first));
        ptrInputBlobs.push_back(inferRequest.GetBlob(input.first));
        std::cout << " ip name = " << input.first << "\n";
    }
    std::cout << "ptrInBlob = " << ptrInputBlobs[0] << " size = " << ptrInputBlobs.size() << " " << ptrInputBlobs[0]->byteSize()/4 << std::endl;
    std::vector<std::vector<float>> image_data;
    float* dest = ptrInputBlobs[0]->buffer().as<float*>();
    int i = 0;
    int j =0;
    std::ifstream input;
    std::vector<float> input_vals(cellSize);
    input.open("/data/local/tmp/ip_to_ln.csv");
    if(input.is_open())
    {
    for (int i = 0; i < ptrInputBlobs[0]->byteSize()/4; i++) {

	    input >> input_vals[i];
	    input.get();
	//        //image_data[j][k];
    }
    }

    for (j = 0; j < ptrInputBlobs[0]->byteSize()/4 ; j++) {
        *(dest + j) = input_vals[j];
        //y_values << input_vals[j] << ",";
    }
    if (ptrInputBlobs.size() > 1) {
    float* dest2 = ptrInputBlobs[1]->buffer().as<float*>();
    for (j = 0; j < ptrInputBlobs[1]->byteSize()/4 ; j++) {
		*(dest2 + j) = -0.5;
    }
    }
    for (int i = 0;i < 1; i++) {
        inferRequest.StartAsync();  //for async infer
        inferRequest.Wait(100); //check right value to infer
    }

    InferenceEngine::OutputsDataMap outputInfo = passed_network.getOutputsInfo();
    auto outputInfoItem = outputInfo.begin()->second;

    auto outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);
    float *op = outputBlob->buffer().as<float*>();
    std::cout << "Intel LN  values = " <<  outputBlob->byteSize()/4 << "\n";
    for (i = 0; i < outputBlob->byteSize()/4 ; i++) {
        std::cout << *(op + i) << " ";
    }
    return 1;
   /* float* dest = ptrInputBlobs[0]->buffer().as<float*>();
    //float* dest2 = ptrInputBlobs[1]->buffer().as<float*>();
    int i = 0;

    std::vector<float> input_vals(cellSize);
    std::vector<float> input_vals2(cellSize);
    std::vector<float> back_mean(cellSize);
    std::ofstream y_values;
    std::ifstream input("/data/local/tmp/ip_ln_vals.csv");
    std::ofstream ip_values;
    std::string line;
    std::vector<std::vector<float>> v;
    y_values.open("/data/local/tmp/y_" + std::to_string(cellSize) + ".csv");
    if(input.is_open())
    {
    for (int i = 0; i < ptrInputBlobs[0]->byteSize()/4; i++) {

	    input >> input_vals[i];
	    input.get();
	//        //image_data[j][k];
    }
    }
    //std::sort(input_vals.begin(), input_vals.end());
    for (int i = 0; i < ptrInputBlobs[0]->byteSize()/4; i++) {
	    input_vals2[i] = input_vals[i];
        back_mean[i] = input_vals[i];
    }
    int j =0;
    for (j = 0; j < ptrInputBlobs[0]->byteSize()/4 - 1; j++) {
        *(dest + j) = input_vals[j];
	y_values << input_vals[j] << ",";
    }

    *(dest + j) = input_vals[j];
    y_values << input_vals[j] << "\n";
    for (int i = 0; i < ptrInputBlobs[1]->byteSize()/4; i++) {
        *(dest2 + i) = -0.5;
    }

    int size = static_cast<int>(cellSize);
    float output[size];
    float *output_vector = &output[0];
    PortableMeanStddevNormalization(dest, output_vector, static_cast<int> (cellSize), 1, 1e-8);
    std::cout << "Tflite values = \n";
    std::ofstream golden_values;
    golden_values.open("/data/local/tmp/cpu_tflite_" + std::to_string(cellSize) + ".csv");
    int k = 0;
    for ( k = 0; k < size - 1; k++) {
        std::cout << output[k] << " ";
	golden_values << output[k] << ",";
    }
    std::cout << "\n";
    golden_values << output[k] << "\n";

    for (int i = 0;i < 300; i++) {
        inferRequest.StartAsync();  //for async infer
        inferRequest.Wait(100); //check right value to infer
    }

    InferenceEngine::OutputsDataMap outputInfo = passed_network.getOutputsInfo();
    auto outputInfoItem = outputInfo.begin()->second;

    auto outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);
    float *op = outputBlob->buffer().as<float*>();
    std::cout << "Intel LN  values = \n";
    float off_by;
    std::ofstream gna_values;
    std::ofstream output_values_gna("/data/local/tmp/output_values_gna.csv", std::ofstream::app);
    gna_values.open("/data/local/tmp/gna_" + std::to_string(cellSize) + ".csv");
    if(output_node == "mean")
	    mean_term(input_vals);
    else if(output_node == "xmean")
        x_mean_term(input_vals);
    else if(output_node =="square")
        square_term(input_vals);
    else if(output_node == "divbyn")
        divbyn_term(input_vals);
    else if(output_node == "sqsum") {
        divbyn_term(input_vals);
        sqsum_term(input_vals);
    }
    else if(output_node == "log") {
        divbyn_term(input_vals);
        sqsum_term(input_vals);
        log_term(input_vals);
    }
    else if(output_node == "logsum") {
        divbyn_term(input_vals);
        sqsum_term(input_vals);
        log_sum_term(input_vals);
    }
    else if(output_node == "exp") {
        divbyn_term(input_vals);
        sqsum_term(input_vals);
        exp_term(input_vals);
    }
    else {
        divbyn_term(input_vals);
        sqsum_term(input_vals);
        exp_term(input_vals);
        x_mean_term(back_mean);
        final_term(input_vals, back_mean);
    }
    //sum = sum/input_vals.size();
    //float log_val = log_term(sum);
    //std::cout << "log value is " << log_val << "\n";
    for (i = 0; i < outputBlob->byteSize()/4 ; i++) {
        std::cout << *(op + i) << " ";
        off_by = *(op + i) - output[i];
        std::cout << " off_by = " << off_by << " ";
	gna_values << input_vals2[i] << ",";
	gna_values << input_vals[i]  << ",";
	gna_values << *(op + i ) << "\n";
    }
    i = 0;
    for (i = 0; i < outputBlob->byteSize()/4 - 1 ; i++) {
        output_values_gna << *( op + i ) << ",";
    }
    output_values_gna << *( op + i ) << "\n";
    output_values_gna.close();

    // oid PortableMeanStddevNormalization(const float* input_vector,
    //                                 float* output_vector, int v_size,
    //                                 int n_batch, float normalization_epsilon);

    for(auto element : norm_weights[0])
  //     std::cout << element << " ";
    return 1;*/
}
