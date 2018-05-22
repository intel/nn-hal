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

#include "mkldnn_infer_request.h"
#include "mkldnn_extension_utils.h"
#include <vector>
#include <string>
#include <map>
#include <blob_factory.hpp>
#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_split_node.h>

MKLDNNPlugin::MKLDNNInferRequest::MKLDNNInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                                     InferenceEngine::OutputsDataMap networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs) {}


template <typename T> void MKLDNNPlugin::MKLDNNInferRequest::pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob) {
    InferenceEngine::TBlob<T> *in_f = dynamic_cast<InferenceEngine::TBlob<T> *>(inputBlob.get());

    if (in_f == nullptr) {
        THROW_IE_EXCEPTION << "Input data precision not supported. Expected float.";
    }

    if (in_f->readOnly() == nullptr) {
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }

    graph->PushInputData(inputName, inputBlob);
}

void MKLDNNPlugin::MKLDNNInferRequest::Infer() {
    IE_PROFILING_AUTO_SCOPE(MKLDNN_INFER)

    if (!graph || !graph->IsReady()) {
        THROW_IE_EXCEPTION << "Network not loaded.";
    }
    changeDefaultPtr();
    // need to retain converted blobs until infer finish
    std::vector<InferenceEngine::Blob::Ptr> convertedInputs;
    for (auto input : _inputs) {
        if (!_networkInputs[input.first]) {
            THROW_IE_EXCEPTION <<
                               "input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name "
                               << input.first;
        }
        /*if (_networkInputs[input.first]->getInputPrecision() != input.second->precision()) {
            THROW_IE_EXCEPTION << "Different input precision for input " << input.first
                               << " registered in IInferencePlugin::LoadNetwork network and IInferencePlugin::Infer. "
                               << _networkInputs[input.first]->getInputPrecision() << " vs "
                               << input.second->precision();
        }*/



        InferenceEngine::Blob::Ptr iconv;
        InferenceEngine::TBlob<float> *in_f = nullptr;
        switch (input.second->precision()) {
            case InferenceEngine::Precision::FP32:
                pushInput<float>(input.first, input.second);
                break;
            case InferenceEngine::Precision::U16:
                // U16 is unsupported by mkldnn, so here we convert the blob and send FP32
                iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32,
                                                                                                    input.second->dims());
                convertedInputs.push_back(iconv);
                iconv->allocate();
                in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                InferenceEngine::copyToFloat<uint16_t>(in_f->data(), input.second.get());
                pushInput<float>(input.first, iconv);
                break;
            case InferenceEngine::Precision::I16:
                if (graph->hasMeanImageFor(input.first)) {
                    // If a mean image exists, we convert the blob and send FP32
                    iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32,
                                                                                                        input.second->dims());
                    convertedInputs.push_back(iconv);
                    iconv->allocate();
                    in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                    InferenceEngine::copyToFloat<int16_t>(in_f->data(), input.second.get());
                    pushInput<float>(input.first, iconv);
                } else {
                    // Instead we can send I16 directly
                    pushInput<int16_t>(input.first, input.second);
                }
                break;
            case InferenceEngine::Precision::U8:
                if (graph->hasMeanImageFor(input.first)) {
                    // If a mean image exists, we convert the blob and send FP32
                    iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32,
                                                                                                        input.second->dims());
                    convertedInputs.push_back(iconv);
                    iconv->allocate();
                    in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                    InferenceEngine::copyToFloat<uint8_t>(in_f->data(), input.second.get());
                    pushInput<float>(input.first, iconv);
                } else {
                    // Instead we can send I8 directly
                    pushInput<uint8_t>(input.first, input.second);
                }
                break;
            default:
                THROW_IE_EXCEPTION << "Unsupported input precision " << input.second->precision();
        }
    }
    graph->Infer();
    graph->PullOutputData(_outputs);
    resetDefaultPtr();
}

void MKLDNNPlugin::MKLDNNInferRequest::GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";
    graph->GetPerfData(perfMap);
}

void MKLDNNPlugin::MKLDNNInferRequest::GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) {
    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";

    InferenceEngine::BlobMap blobs;
    graph->getInputBlobs(blobs);

    if (blobs.find(name) != blobs.end()) {
        if (_inputs.find(name) != _inputs.end()) {
            data = _inputs[name];
            return;
        }

        InferenceEngine::TensorDesc desc = blobs[name]->getTensorDesc();
        if (_networkInputs.find(name) != _networkInputs.end()) {
            desc = _networkInputs[name]->getTensorDesc();
            desc.setPrecision(_networkInputs[name]->getInputPrecision());
        }

        _inputs[name] = make_blob_with_precision(desc);
        _inputs[name]->allocate();
        if (desc.getPrecision() == InferenceEngine::Precision::FP32 &&
                graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {
            externalPtr[name] = _inputs[name]->buffer();
        }
        data = _inputs[name];
        return;
    }
    blobs.clear();
    graph->getOutputBlobs(blobs);

    if (blobs.find(name) != blobs.end()) {
        if (_outputs.find(name) != _outputs.end()) {
            data = _outputs[name];
            return;
        }

        _outputs[name] = make_blob_with_precision(blobs[name]->getTensorDesc());
        _outputs[name]->allocate();
        if (blobs[name]->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                !graph->getProperty().batchLimit) {
            externalPtr[name] = _outputs[name]->buffer();
        }
        data = _outputs[name];
        return;
    }
    THROW_IE_EXCEPTION << "Cannot find blob with name: " << name;
}

void MKLDNNPlugin::MKLDNNInferRequest::SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) {
    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    if (data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        size_t inputSize = InferenceEngine::details::product(foundInput->getDims());
        if (dataSize != inputSize) {
            THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                               << dataSize << "!=" << inputSize << ").";
        }

        // Checking for the input precision
        switch (data->precision()) {
        case InferenceEngine::Precision::FP32:
        case InferenceEngine::Precision::FP16:
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::U16:
        case InferenceEngine::Precision::U8:
            // These Precisions are supported
            break;
        default:
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set Blob with precision " << data->precision();
        }

        if (data->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {
            externalPtr[name] = data->buffer();
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _inputs[name] = data;
    } else {
        size_t outputSize = InferenceEngine::details::product(foundOutput->getDims());
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != data->precision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding user output precision";
        }
        if (data->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                !graph->getProperty().batchLimit) {
            externalPtr[name] = data->buffer();
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _outputs[name] = data;
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::changeDefaultPtr() {
    for (auto& it : externalPtr) {
        auto input = graph->inputNodes.find(it.first);
        if (input != graph->inputNodes.end()) {
            // Input cannot be in-place with other primitives
            bool canBeInPlace = true;
            for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {
                auto& child = input->second->getChildEdgeAt(i)->getChild();
                if (child->isConstant(true))
                    canBeInPlace = false;
                auto* concat = dynamic_cast<MKLDNNConcatNode *>(child.get());
                if (canBeInPlace && concat && concat->isOptimized())
                    canBeInPlace = false;
                // Cannot be in-place before split because split is using different ptrs without offsets
                auto* split = dynamic_cast<MKLDNNSplitNode *>(child.get());
                if (canBeInPlace && split)
                    canBeInPlace = false;
                for (size_t j = 0; canBeInPlace && j < child->getChildEdges().size(); j++) {
                    if (child->getChildEdgeAt(j)->getMemory().GetPrimitive().get_data_handle() ==
                            input->second->getChildEdgeAt(i)->getMemory().GetPrimitive().get_data_handle())
                        canBeInPlace = false;
                }
            }
            for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {
                defaultPtr[it.first] = input->second->getChildEdgeAt(i)->getMemory().GetPrimitivePtr()->get_data_handle();
                changeAllPtrs(defaultPtr[it.first], it.second);
            }
            continue;
        }
        MKLDNNNodePtr output;
        for (auto& out : graph->outputNodes) {
            if (out->getName() == "out_" + it.first) {
                output = out;
                break;
            }
        }
        if (output) {
            bool canBeInPlace = true;
            defaultPtr[it.first] = output->getParentEdgeAt(0)->getMemory().GetPrimitivePtr()->get_data_handle();
            // Cannot be in-place after concat because concat is using different ptrs without offsets
            auto parent = output->getParentEdgeAt(0)->getParent();
            MKLDNNNodePtr previousParent;
            do {
                previousParent = parent;
                auto *concat = dynamic_cast<MKLDNNConcatNode *>(parent.get());
                if (output->getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1 ||
                    output->getParentEdgeAt(0)->getParent()->isConstant(true) ||
                    (concat && concat->isOptimized())) {
                    canBeInPlace = false;
                    break;
                }

                for (size_t i = 0; i < parent->getParentEdges().size(); i++) {
                    if (parent->getParentEdgeAt(i)->getMemory().GetPrimitivePtr()->get_data_handle() == defaultPtr[it.first]) {
                        parent = parent->getParentEdgeAt(i)->getParent();
                        break;
                    }
                }
            } while (previousParent != parent);
            if (canBeInPlace)
                changeAllPtrs(defaultPtr[it.first], it.second);
            continue;
        }
        THROW_IE_EXCEPTION << "Cannot find input/output blob: " << it.first;
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::resetDefaultPtr() {
    for (auto& it : externalPtr) {
        auto input = graph->inputNodes.find(it.first);
        if (input != graph->inputNodes.end()) {
            for (size_t i = 0; i < input->second->getChildEdges().size(); i++) {
                changeAllPtrs(it.second, defaultPtr[it.first]);
            }
            continue;
        }
        MKLDNNNodePtr output;
        for (auto& out : graph->outputNodes) {
            if (out->getName() == "out_" + it.first) {
                output = out;
                break;
            }
        }
        if (output) {
            for (size_t i = 0; i < output->getParentEdges().size(); i++) {
                changeAllPtrs(it.second, defaultPtr[it.first]);
            }
            continue;
        }
        THROW_IE_EXCEPTION << "Cannot find input/output blob: " << it.first;
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::changeAllPtrs(void *oldPtr, void *newPtr) {
    for (auto &edge : graph->graphEdges) {
        if (edge->getMemory().GetPrimitivePtr()->get_data_handle() == oldPtr) {
            edge->getMemory().GetPrimitivePtr()->set_data_handle(newPtr);
        }
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::SetGraph(const MKLDNNPlugin::MKLDNNGraph::Ptr &graph) {
    this->graph = graph;
}
