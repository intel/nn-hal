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

#include <ie_blob.h>
#include <ie_plugin.hpp>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>

#include "precision_utils.h"
#include "myriad_executable_network.h"
#include "myriad_infer_request.h"
#include "common.h"

#ifdef NNLOG
#include <android/log.h>
#include <cutils/log.h>
#endif

using namespace VPU::Common;
using namespace VPU::MyriadPlugin;
using namespace InferenceEngine;

MyriadInferRequest::MyriadInferRequest(GraphDesc &graphDesc,
                                        InferenceEngine::InputsDataMap networkInputs,
                                        InferenceEngine::OutputsDataMap networkOutputs,
                                        const EnvironmentPtr &env, const LoggerPtr &log,
                                        const MyriadExecutorPtr &executor) :
        InferRequestInternal(networkInputs, networkOutputs), _graphDesc(
                graphDesc), _env(env), _log(log), _executor(executor) {


          //LOG_DEBUG("myriad InferRequest allocate network input blob");

    _deviceLayout = _env->parsedConfig.blobConfig.hwOptimization ? NCHW : NHWC;

    // allocate inputs
    for (auto &networkInput : _networkInputs) {
      #ifdef NNLOG
      ALOGI("allocate inputs");
      #endif
        // TODO: use TensorDesc instead of deprecated methods
        SizeVector dims = networkInput.second->getDims();
        Precision precision = networkInput.second->getInputPrecision();
        Layout layout = networkInput.second->getTensorDesc().getLayout();

        Blob::Ptr inputBlob;
        switch (precision) {
            case Precision::FP32:
                inputBlob = InferenceEngine::make_shared_blob<float, const SizeVector>(Precision::FP32, layout, dims);
                break;
            case Precision::FP16:
                inputBlob = InferenceEngine::make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, layout, dims);
                break;
            case Precision::U8:
                inputBlob = InferenceEngine::make_shared_blob<uint8_t, const SizeVector>(Precision::U8, layout, dims);
                break;
            default:
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: "
                                   << precision << "! Supported precisions: FP32, FP16 and U8";
        }
        // allocate the input blob
        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
    }

    // allocate outputs
    for (auto &networkOutput : _networkOutputs) {
        #ifdef NNLOG
        ALOGI("allocate outputs");
        #endif
        SizeVector dims = networkOutput.second->dims;
        Precision precision = networkOutput.second->precision;
        Layout layout = networkOutput.second->layout;

        Blob::Ptr outputBlob;
        switch (precision) {
            case Precision::FP32:
                outputBlob = InferenceEngine::make_shared_blob<float, const SizeVector>(Precision::FP32, layout, dims);
                break;
            case Precision::FP16:
                outputBlob = InferenceEngine::make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, layout, dims);
                break;
            default:
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: "
                                   << precision << "! Supported precisions: FP32, FP16";
        }
        // allocate the output blob
        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
    }
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }
}

void MyriadInferRequest::Infer() {
    InferAsync();
    GetResult();
}

void MyriadInferRequest::InferAsync() {
#ifdef NNLOG
  ALOGI("myriad InferAsync");
  printf("myriad InferAsync\n");
#endif
  //LOG_DEBUG(" myriad InferAsync");


    for (auto input : _inputs) {
        auto const inputBlobPtr = input.second;
        if (inputBlobPtr->precision() != Precision::FP16
            && inputBlobPtr->precision() != Precision::FP32
            && inputBlobPtr->precision() != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input blob precision";
    }
    for (auto output : _outputs) {
        auto const outputBlobPtr = output.second;
        if (outputBlobPtr->precision() != Precision::FP16
            && outputBlobPtr->precision() != Precision::FP32)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output blob precision";
    }

    std::vector<uint8_t> tmpMemory;
    Blob::Ptr tmpBlob;

    void* inputPtr = nullptr;
    size_t inputSize = 0;

    if (_inputs.size() > 1) {
        inputSize = 0;
        for (auto input : _inputs) {
            auto const inputBlobPtr = input.second;
            size_t byteSize = inputBlobPtr->size() * inputBlobPtr->element_size();
            inputSize += byteSize;
        }

        tmpMemory.resize(inputSize);
        auto dst = tmpMemory.data();

        for (auto input : _inputs) {
            auto inputBlobPtr = input.second;
            size_t byteSize = inputBlobPtr->size() * inputBlobPtr->element_size();
            Layout layout = inputBlobPtr->getTensorDesc().getLayout();
            if (layout != _deviceLayout && (layout == NCHW || layout == NHWC)) {
                switch (inputBlobPtr->precision()) {
                    case Precision::U8:
                        ConvertBlobToLayout<uint8_t>(_deviceLayout, inputBlobPtr);
                        break;
                    case Precision::FP16:
                        ConvertBlobToLayout<ie_fp16>(_deviceLayout, inputBlobPtr);
                        break;
                    case Precision::FP32:
                        ConvertBlobToLayout<float>(_deviceLayout, inputBlobPtr);
                        break;
                    default:
                        THROW_IE_EXCEPTION << "unsupported blob precision for converting layout";
                        break;
                }
            }
            memcpy(dst, inputBlobPtr->buffer(), byteSize);
            dst += byteSize;
        }

        inputPtr = tmpMemory.data();
    } else {
        auto dataName = _networkInputs.begin()->first;
        auto foundInputBlob = _inputs.find(dataName);
        if (foundInputBlob == _inputs.end())
            THROW_IE_EXCEPTION << "Error: input [" << dataName << "] is not provided.";

        tmpBlob = foundInputBlob->second;
        size_t byteSize = tmpBlob->size() * tmpBlob->element_size();
        Layout layout = tmpBlob->getTensorDesc().getLayout();
        if (layout != _deviceLayout && (layout == NCHW || layout == NHWC)) {
            switch (tmpBlob->precision()) {
                case Precision::U8:
                    ConvertBlobToLayout<uint8_t>(_deviceLayout, tmpBlob);
                    break;
                case Precision::FP16:
                    ConvertBlobToLayout<ie_fp16>(_deviceLayout, tmpBlob);
                    break;
                case Precision::FP32:
                    ConvertBlobToLayout<float>(_deviceLayout, tmpBlob);
                    break;
                default:
                    THROW_IE_EXCEPTION << "unsupported blob precision for converting layout";
                    break;
            }
        }

        inputPtr = tmpBlob->buffer();
        inputSize = byteSize;
    }

    _executor->queueInference(_graphDesc, inputPtr, inputSize, nullptr, nullptr);
}

void MyriadInferRequest::GetResult() {
    void *resultPtr = NULL;
    size_t resultSize = 0;

    _executor->getResult(_graphDesc, &resultPtr, &resultSize);

    size_t resultOffset = 0;
    for (auto pp : _outputs) {
        if (resultOffset > resultSize) {
            THROW_IE_EXCEPTION << "unexpected result data size";
        }
        auto const outputBlobPtr = pp.second;
        Layout layout = outputBlobPtr->getTensorDesc().getLayout();
        SizeVector dims = outputBlobPtr->getTensorDesc().getDims();
        if (layout != _deviceLayout && (layout == NCHW || layout == NHWC)
            && (dims[0] != 1 || dims[1] != 1) && (dims[2] != 1 || dims[3] != 1)) {
            Blob::Ptr tmpBlob = nullptr;
            switch (outputBlobPtr->precision()) {
            case Precision::FP32:
                tmpBlob = InferenceEngine::make_shared_blob<float>(Precision::FP32, _deviceLayout, outputBlobPtr->dims(),
                                                                   reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(resultPtr) + resultOffset));
                ConvertBlobToLayout<float>(layout, tmpBlob);
                break;
            case Precision::FP16:
                tmpBlob = InferenceEngine::make_shared_blob<ie_fp16>(Precision::FP16, _deviceLayout, outputBlobPtr->dims(),
                                                                     reinterpret_cast<short *>(reinterpret_cast<uint8_t *>(resultPtr) + resultOffset));
                ConvertBlobToLayout<ie_fp16>(layout, tmpBlob);
                break;
            default:
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: "
                                   << outputBlobPtr->precision() << "! Supported precisions: FP32, FP16";
            }
            memcpy(outputBlobPtr->buffer(), tmpBlob->cbuffer(), outputBlobPtr->byteSize());
        } else {
            memcpy(outputBlobPtr->buffer(), reinterpret_cast<uint8_t *>(resultPtr) + resultOffset, outputBlobPtr->byteSize());
        }

        resultOffset += outputBlobPtr->byteSize();
    }

#if 0
    _executor->printThrottlingStatus();
#endif
}

void MyriadInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    std::shared_ptr<GraphInfo<float>> graphInfo = _executor->getPerfTimeInfo(_graphDesc._graphHandle);
    if (_log->getLogLevel() >= LogLevel::eLOGINFO) {
        if (graphInfo != nullptr && graphInfo->numElements()) {
            LOG_INFO("** Device execution time %.3lf **"
                    , graphInfo->info()[graphInfo->numElements()- 1]);
        }
    }
    Common::GetPerformanceCounts(_env->blobMetaData, graphInfo, perfMap,
            _env->parsedConfig.printReceiveTensorTime);
}
