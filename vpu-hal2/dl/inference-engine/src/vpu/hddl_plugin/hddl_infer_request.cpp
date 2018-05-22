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

#include <functional>

#include <ie_blob.h>
#include <ie_plugin.hpp>
#include <ie_layouts.h>
#include <description_buffer.hpp>
#include <debug.h>
#include <precision_utils.h>

#include "hddl_infer_request.h"
#include "common.h"

#include "hddl_allocator.h"

using namespace VPU::Common;
using namespace VPU::HDDLPlugin;
using namespace InferenceEngine;

HDDLInferRequest::HDDLInferRequest(InputsDataMap networkInputs,
                                   OutputsDataMap networkOutputs,
                                   const Common::EnvironmentPtr &env,
                                   const Common::LoggerPtr &log,
                                   const ExecutorPtr &executor,
                                   HDDLAllocatorPtr &hddlAllocatorPtr)
        : AsyncInferRequestInternal(networkInputs, networkOutputs),
          _env(env), _log(log), _executor(executor), _taskHandle(0),
          _hddlAllocatorPtr(hddlAllocatorPtr) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "invalid inputs/outputs";
    }

    for (auto &networkInput : _networkInputs) {
        if (networkInput.second->getInputPrecision() != Precision::FP16
            && networkInput.second->getInputPrecision() != Precision::U8
            && networkInput.second->getInputPrecision() != Precision::FP32) {
            THROW_IE_EXCEPTION << "Unsupported input blob format. Supported FP16, U8, FP32.";
        }
    }

    for (auto &networkOutput : _networkOutputs) {
        if (networkOutput.second->precision != Precision::FP16
            && networkOutput.second->precision != Precision::FP32) {
            THROW_IE_EXCEPTION << "Unsupported output blob format. Supported FP16 and FP32.";
        }
    }

    _deviceLayout = _env->parsedConfig.blobConfig.hwOptimization ? NCHW : NHWC;

    size_t output_buffer_size = 0;  // bytes

    for (auto &networkOutput : _networkOutputs) {
        DataPtr outputData = networkOutput.second;
        SizeVector dims = outputData->dims;
        Precision precision = outputData->precision;
        size_t elemSize = 0;

        switch (precision) {
            case Precision::FP16:
                elemSize = 2;
                break;
            case Precision::FP32:
                elemSize = 4;
                break;
            default:
                THROW_IE_EXCEPTION << "Unsupported precision for allocating output";
        }

        output_buffer_size += (std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()) * elemSize);
    }


    size_t input_buffer_size = 0;  // bytes

    for (auto &networkInput : _networkInputs) {
        SizeVector dims = networkInput.second->getDims();
        Precision precision = networkInput.second->getInputPrecision();
        size_t elemSize = 0;

        switch (precision) {
            case Precision::U8:
                elemSize = 1;
                break;
            case Precision::FP16:
                elemSize = 2;
                break;
            case Precision::FP32:
                elemSize = 4;
                break;
            default:
                THROW_IE_EXCEPTION << "Unsupported precision for allocating input";
        }
        input_buffer_size += (std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()) * elemSize);
    }

    // the following code guarantees that inputs are created in one hddl buffer and placed sequentially
    hddlAllocatorPtr->reserveHddlBuffer(input_buffer_size);
    for (auto &networkInput : _networkInputs) {
        // TODO: use getTensorDesc
        SizeVector dims = networkInput.second->getDims();
        Precision precision = networkInput.second->getInputPrecision();
        Layout layout = networkInput.second->getTensorDesc().getLayout();
        Layout ionBlobLayout = layout != _deviceLayout && (layout == NCHW || layout == NHWC) ? _deviceLayout : layout;

        Blob::Ptr inputBlob = nullptr;
        switch (precision) {
            case Precision::U8:
                inputBlob = std::make_shared<TBlob<uint8_t>>(precision, ionBlobLayout, dims, hddlAllocatorPtr);
                break;
            case Precision::FP16:
                inputBlob = std::make_shared<TBlob<ie_fp16>>(precision, ionBlobLayout, dims, hddlAllocatorPtr);
                break;
            case Precision::FP32:
                inputBlob = std::make_shared<TBlob<float>>(precision, ionBlobLayout, dims, hddlAllocatorPtr);
                break;
            default:
                THROW_IE_EXCEPTION << "Unsupported precision for allocating";
        }
        inputBlob->allocate();
        _hddlInputs[networkInput.first] = inputBlob;

        if (layout != _deviceLayout && (layout == NCHW || layout == NHWC)) {
            switch (precision) {
                case Precision::U8:
                    inputBlob = std::make_shared<TBlob<uint8_t>>(precision, layout, dims);
                    break;
                case Precision::FP16:
                    inputBlob = std::make_shared<TBlob<ie_fp16>>(precision, layout, dims);
                    break;
                case Precision::FP32:
                    inputBlob = std::make_shared<TBlob<float>>(precision, layout, dims);
                    break;
                default:
                    THROW_IE_EXCEPTION << "Unsupported precision for allocating";
            }
            inputBlob->allocate();
            _inputs[networkInput.first] = inputBlob;
        }
    }

    hddlAllocatorPtr->reserveHddlBuffer(output_buffer_size);
    for (auto &networkOutput : _networkOutputs) {
        SizeVector dims = networkOutput.second->dims;
        Precision precision = networkOutput.second->precision;
        Layout layout = networkOutput.second->layout;
        Layout ionBlobLayout = layout != _deviceLayout && (layout == NCHW || layout == NHWC) ? _deviceLayout : layout;

        Blob::Ptr outputBlob = nullptr;
        switch (precision) {
            case Precision::FP16:
                outputBlob = std::make_shared<TBlob<ie_fp16>>(precision, ionBlobLayout, dims, hddlAllocatorPtr);
                break;
            case Precision::FP32:
                outputBlob = std::make_shared<TBlob<float>>(precision, ionBlobLayout, dims, hddlAllocatorPtr);
                break;
            default:
                THROW_IE_EXCEPTION << "Unsupported precision for allocating output";
        }
        outputBlob->allocate();
        _hddlOutputs[networkOutput.first] = outputBlob;

        if (layout != _deviceLayout && (layout == NCHW || layout == NHWC)) {
            switch (precision) {
                case Precision::FP16:
                    outputBlob = std::make_shared<TBlob<ie_fp16>>(precision, layout, dims);
                    break;
                case Precision::FP32:
                    outputBlob = std::make_shared<TBlob<float>>(precision, layout, dims);
                    break;
                default:
                    THROW_IE_EXCEPTION << "Unsupported precision for allocating output";
            }
            outputBlob->allocate();
            _outputs[networkOutput.first] = outputBlob;
        }
    }
}

HDDLInferRequest::~HDDLInferRequest() {
    try {
        if (isRequestBusy()) {
            Wait(IInferRequest::WaitMode::RESULT_READY);
        }
    } catch (...) {}
}

void HDDLInferRequest::StartAsync_ThreadUnsafe() {
    if (_hddlInputs.empty() || _hddlOutputs.empty()) {
        THROW_IE_EXCEPTION << NOT_FOUND_str << "Inputs or outputs aren't provided";
    }

    if (!_inputs.empty()) {
        for (auto &input : _inputs) {
            auto name = input.first;
            auto inputBlobPtr = input.second;
            auto hddlBlobPtr = _hddlInputs[name];
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
            std::copy_n(inputBlobPtr->cbuffer().as<uint8_t *>(), inputBlobPtr->byteSize(),
                        hddlBlobPtr->buffer().as<uint8_t *>());
        }
    }
    auto firstInBlobPtr = _hddlInputs.begin()->second;
    HddlBuffer *inputBuffer = _hddlAllocatorPtr->getHddlBufferByPointer(firstInBlobPtr->cbuffer().as<void *>());

    auto firstOutBlobPtr = _hddlOutputs.begin()->second;
    HddlBuffer *outputBuffer = _hddlAllocatorPtr->getHddlBufferByPointer(firstOutBlobPtr->cbuffer().as<void *>());

    _executor->InferAsync(inputBuffer, outputBuffer, &_taskHandle, this);
}

StatusCode HDDLInferRequest::Wait(int64_t millis_timeout) {
    if (millis_timeout < -1) {
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Timiout must be >= -1";
    }

    int hddlStatus = _executor->Wait(_taskHandle, millis_timeout);
    std::map<int, StatusCode> statusMap = {
            {HDDL_ERROR_NONE,               OK},
            {HDDL_TASK_STATUS_DONE,         OK},
            {HDDL_STATUS_TASK_NOT_FINISHED, RESULT_NOT_READY},
            {HDDL_TASK_STATUS_UNKNOWN,      INFER_NOT_STARTED},
            {HDDL_ERROR_DEVICE_BUSY,        REQUEST_BUSY},
            {HDDL_ERROR_DEVICE_NOT_FOUND,   NOT_FOUND},
            {HDDL_ERROR_INVAL_TASK_HANDLE,  INFER_NOT_STARTED},
    };
    auto found = statusMap.find(hddlStatus);

    if (found != statusMap.end()) {
        auto status = found->second;
        // TODO: which error means that request is still ongoing and which that request done or there is error?
        if (status != REQUEST_BUSY && status != RESULT_NOT_READY) {
            setIsRequestBusy(false);
        }
        return status;
    }
    setIsRequestBusy(false);
    return GENERAL_ERROR;
}

void HDDLInferRequest::Infer_ThreadUnsafe() {
    if (_hddlInputs.empty() || _hddlOutputs.empty()) {
        THROW_IE_EXCEPTION << NOT_FOUND_str << "Inputs or outputs aren't provided";
    }

    if (!_inputs.empty()) {
        for (auto &input : _inputs) {
            auto name = input.first;
            auto inputBlobPtr = input.second;
            auto hddlBlobPtr = _hddlInputs[name];
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
            std::copy_n(inputBlobPtr->cbuffer().as<uint8_t *>(), inputBlobPtr->byteSize(),
                        hddlBlobPtr->buffer().as<uint8_t *>());
        }
    }
    auto firstInBlobPtr = _hddlInputs.begin()->second;
    HddlBuffer *inputBuffer = _hddlAllocatorPtr->getHddlBufferByPointer(firstInBlobPtr->cbuffer().as<void *>());

    auto firstOutBlobPtr = _hddlOutputs.begin()->second;
    HddlBuffer *outputBuffer = _hddlAllocatorPtr->getHddlBufferByPointer(firstOutBlobPtr->cbuffer().as<void *>());

    _executor->InferSync(inputBuffer, outputBuffer, &_taskHandle);

    if (!_outputs.empty()) {
        CopyToExternalOutputs();
    }
}

void HDDLInferRequest::GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    Common::GetPerformanceCounts(_env->blobMetaData, _executor->getPerfTimeInfo(), perfMap);
}

void HDDLInferRequest::GetBlob_ThreadUnsafe(const char *name, Blob::Ptr &data) {
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        auto input = _inputs.find(name);
        if (input != _inputs.end()) {
            data = input->second;
        } else {
            data = _hddlInputs[name];
        }
    } else {
        auto output = _outputs.find(name);
        if (output != _outputs.end()) {
            data = output->second;
        } else {
            data = _hddlOutputs[name];
        }
    }
}

void HDDLInferRequest::SetBlob_ThreadUnsafe(const char *name, const Blob::Ptr &data) {
    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    if (data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        size_t inputSize = details::product(foundInput->getDims());
        if (dataSize != inputSize) {
            THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                               << dataSize << "!=" << inputSize << ").";
        }
        if (foundInput->getInputPrecision() != data->precision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding user input precision";
        }

        auto *inputData = data->cbuffer().as<void *>();
        if (_networkInputs.size() == 1 && _hddlAllocatorPtr->isIonBuffer(inputData)
            && _hddlAllocatorPtr->isBlobPlacedWholeBuffer(inputData)) {
            _hddlInputs[name] = data;
        } else {
            _inputs[name] = data;
        }
    } else {
        size_t outputSize = details::product(foundOutput->getDims());
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != data->precision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding user output precision";
        }

        auto *outputData = data->cbuffer().as<void *>();
        if (_networkOutputs.size() == 1 && _hddlAllocatorPtr->isIonBuffer(outputData)
            && _hddlAllocatorPtr->isBlobPlacedWholeBuffer(outputData)) {
            _hddlOutputs[name] = data;
        } else {
            _outputs[name] = data;
        }
    }
}

void HDDLInferRequest::SetCompletionCallback_ThreadUnsafe(IInferRequest::CompletionCallback callback) {
    AsyncInferRequestInternal::SetCompletionCallback(callback);
}

void HDDLInferRequest::GetUserData_ThreadUnsafe(void **data) {
    AsyncInferRequestInternal::GetUserData(data);
}

void HDDLInferRequest::SetUserData_ThreadUnsafe(void *data) {
    AsyncInferRequestInternal::SetUserData(data);
}

void *HDDLInferRequest::HDDLCallback() {
    if (!_outputs.empty()) {
        CopyToExternalOutputs();
    }

    if (_callback && _publicInterface.lock()) {
        setIsRequestBusy(false);
        auto status = Wait(IInferRequest::WaitMode::STATUS_ONLY);
        _callback(_publicInterface.lock(), status);
    }
    return nullptr;
}

void HDDLInferRequest::CopyToExternalOutputs() {
    for (auto &output : _outputs) {
        auto name = output.first;
        auto outputBlobPtr = output.second;
        auto hddlBlobPtr = _hddlOutputs[name];
        Layout layout = hddlBlobPtr->getTensorDesc().getLayout();
        SizeVector dims = hddlBlobPtr->getTensorDesc().getDims();
        if (layout != _deviceLayout && (layout == NCHW || layout == NHWC)
            && (dims[0] != 1 || dims[1] != 1) && (dims[2] != 1 || dims[3] != 1)) {
            switch (outputBlobPtr->precision()) {
                case Precision::FP32:
                    ConvertBlobToLayout<float>(layout, hddlBlobPtr);
                    break;
                case Precision::FP16:
                    ConvertBlobToLayout<ie_fp16>(layout, hddlBlobPtr);
                    break;
                default:
                    THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: "
                                       << outputBlobPtr->precision() << "! Supported precisions: FP32, FP16";
            }
        }
        std::copy_n(hddlBlobPtr->cbuffer().as<uint8_t *>(), hddlBlobPtr->byteSize(),
                    outputBlobPtr->buffer().as<uint8_t *>());
    }
}
