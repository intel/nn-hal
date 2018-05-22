//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation.
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

#include "graph_transformer_impl.hpp"
#include <memory>

#ifdef NNLOG
#include <android/log.h>
#include <cutils/log.h>
#endif

void VpuConvertStage::dumpToDot(std::ostream& os) {
    os << "scale=" << scale << "\\n"
       << "bias=" << bias;
}

void VpuConvertStage::dumpToBlob(BlobWriter& writer) {
    writer.write(static_cast<float>(scale));
    writer.write(static_cast<float>(bias));

    inputs[0]->dumpToBlob(writer);
    outputs[0]->dumpToBlob(writer);
}

void GraphTransformerImpl::addInputConvertStages() {
    for (const auto& inputInfo : _networkInputs) {
        auto netInput = inputInfo.second;
        assert(netInput != nullptr);

        auto input = getVpuData(netInput->getInputData());
        assert(input != nullptr);

        if (input->type != VpuDataType::FP16) {
            LOG_DEBUG("[VPU] GraphTransformer : convert input %s to FP16", input->name.c_str());
            #ifdef NNLOG
            ALOGI("[VPU] GraphTransformer : convert input %s to FP16", input->name.c_str());
            #endif

            auto inputFP16 = addNewData(
                dataId_FP16(netInput->getInputData()),
                [input](VpuData* data) {
                    data->name = input->name + "@FP16";
                    data->index = IndexBSS;
                    data->type = VpuDataType::FP16;
                    data->dims = input->dims;
                    data->order = input->order;
                    data->strides = calcStrides(data->dims, data->type, data->order);
                });

            if (input->type == VpuDataType::FP32) {
            #ifdef NNLOG
                  ALOGI("[VPU] GraphTransformer input->type == VpuDataType::FP32");
            #endif
          }

            t_MvTensorOpType stageType = kNone0;
            switch (input->type) {
            case VpuDataType::U8:
                stageType = kConvert_u8f16;
                break;
            case VpuDataType::FP32:
                stageType = kConvert_f32f16;
            }

            addNewStage<VpuConvertStage>(
                inputFP16->name,
                stageType,
                nullptr,
                [this, input, inputFP16](VpuConvertStage* stage) {
                    stage->scale = _blobConfig.inputScale;
                    stage->bias = _blobConfig.inputBias;
                    stage->requiredInputOrder[0] = input->order;
                    stage->requiredOutputOrder[0] = inputFP16->order;
                },
                {input},
                {inputFP16});
        } else if (_blobConfig.inputScale != 1.0f) {
            auto weights = addNewData(
                newDataId(),
                [input, this](VpuData* data) {
                    data->name = input->name + "@scale";
                    data->index = IndexBlob;
                    data->type = VpuDataType::FP16;
                    data->order = orderXYZ;
                    data->dims = VpuDims({1, 1, input->dims[Dim::Z]});
                    data->strides = calcStrides(data->dims, data->type, data->order);
                    data->writer = std::make_shared<ScaleWeightsWriter>(_blobConfig.inputScale, data->dims[Dim::Z]);
                });

            addNewStage<VpuScaleStage>(
                weights->name,
                kScale,
                nullptr,
                [](VpuScaleStage* /*stage*/) {
                },
                {input, weights},
                {input});
        }
    }
}

void GraphTransformerImpl::addOutputConvertStages() {

  LOG_INFO("[VPU] GraphTransformer : addOutputConvertStages");
#ifdef NNLOG
  ALOGI("[VPU] GraphTransformer : addOutputConvertStages");
#endif
    for (const auto& outputInfo : _networkOutputs) {
        auto netOutput = outputInfo.second;
        assert(netOutput != nullptr);

        auto output = getVpuData(netOutput);
        assert(output != nullptr);

        if (output->type == VpuDataType::FP16){
          #ifdef NNLOG
            ALOGI("[VPU] GraphTransformer : output %s is FP16", output->name.c_str());
          #endif
        }


        if (output->type != VpuDataType::FP16) {
            LOG_DEBUG("[VPU] GraphTransformer : convert output %s from FP16", output->name.c_str());
#ifdef NNLOG
            ALOGI("[VPU] GraphTransformer : convert output %s from FP16", output->name.c_str());
#endif

            assert(output->type == VpuDataType::FP32);

            auto outputFP16 = getVpuDataFP16(netOutput);
            assert(outputFP16 != nullptr);


            if (output->type == VpuDataType::FP32){
            #ifdef NNLOG
                  ALOGI("[VPU] GraphTransformer output->type == VpuDataType::FP32");
            #endif
          }
            addNewStage<VpuConvertStage>(
                outputFP16->name,
                kConvert_f16f32,
                nullptr,
                [output, outputFP16](VpuConvertStage* stage) {
                    stage->requiredInputOrder[0] = output->order;
                    stage->requiredOutputOrder[0] = outputFP16->order;
                },
                {outputFP16},
                {output});
        }
    }
}
