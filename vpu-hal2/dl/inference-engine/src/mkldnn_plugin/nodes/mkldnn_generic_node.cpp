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
#include <mkldnn_extension_mngr.h>
#include <mkldnn_extension_utils.h>
#include "mkldnn_generic_node.h"
#include <vector>
#include <string>
#include <blob_factory.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNGenericNode::MKLDNNGenericNode(Type type, const std::string &name) : MKLDNNNode(type, name) {}
MKLDNNGenericNode::MKLDNNGenericNode(InferenceEngine::CNNLayerPtr layer) : MKLDNNNode(layer) {}

void MKLDNNGenericNode::createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) {
    this->inputDataType = inputDataType;
    this->outputDataType = outputDataType;

    if (!genericPrimitive && !extFactory) {
        std::string type = getCnnLayer() ? getCnnLayer()->type : "Generic";
        THROW_IE_EXCEPTION << "Cannot get generic primitive for layer: " << getName() << " with type: " << type;
    }
    if (genericPrimitive && extFactory) {
        extFactory.reset();
    }
}

void MKLDNNGenericNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    if (genericPrimitive) {
        std::vector<InferenceEngine::MKLDNNPlugin::MKLDNNGenericFormats> formats = genericPrimitive->GetSupportedFormats();
        if (formats.empty())
            THROW_IE_EXCEPTION << "External primitive doesn't have supported formats";
        for (auto &format : formats) {
            bool isAny = false;
            bool isNotAny = false;
            std::vector<MKLDNNMemoryDesc> inputs;
            bool isCompatible = true;
            for (size_t i = 0; i < format.GetInputs().size() && i < getParentEdges().size(); i++) {
                if (!MKLDNNMemory::isConsistant(getParentEdgeAt(i)->getDims(),
                                                MKLDNNExtensionUtils::MemoryFormatToMKLFormat(format.GetInputs()[i]))) {
                    isCompatible = false;
                    break;
                }
                mkldnn::memory::format mkldnnFormat = MKLDNNExtensionUtils::MemoryFormatToMKLFormat(format.GetInputs()[i]);
                inputs.push_back({autoBlockingDims(getParentEdgeAt(i)->getDims(), mkldnnFormat),
                                  getInputDataType(), mkldnnFormat});
                if (inputs.at(inputs.size() - 1) == memory::any) {
                    isAny = true;
                } else {
                    isNotAny = true;
                }
            }
            if (isAny && isNotAny) {
                THROW_IE_EXCEPTION << "Layer " << getName() << " has incorrect input formats "
                                   << " (any and not any formats don't supported in the same time).";
            }
            isAny = false;
            isNotAny = false;
            std::vector<MKLDNNMemoryDesc> outputs;
            for (size_t i = 0; i < format.GetOutputs().size() && i < getChildEdges().size(); i++) {
                if (!MKLDNNMemory::isConsistant(getChildEdgeAt(i)->getDims(),
                                                MKLDNNExtensionUtils::MemoryFormatToMKLFormat(
                                                        format.GetOutputs()[i]))) {
                    isCompatible = false;
                    break;
                }
                mkldnn::memory::format mkldnnFormat = MKLDNNExtensionUtils::MemoryFormatToMKLFormat(format.GetOutputs()[i]);
                outputs.push_back({autoBlockingDims(getChildEdgeAt(i)->getDims(), mkldnnFormat),
                                   getOutputDataType(), mkldnnFormat});
                if (outputs.at(outputs.size() - 1) == memory::any) {
                    isAny = true;
                } else {
                    isNotAny = true;
                }
            }
            if (isAny && isNotAny) {
                THROW_IE_EXCEPTION << "Layer " << getName() << " has incorrect output formats "
                                   << " (any and not any formats don't supported in the same time).";
            }
            if (isCompatible) {
                supportedPrimitiveDescriptors.push_back({engine, inputs, outputs, impl_desc_type::unknown});
            }
        }
    } else if (extFactory) {
        InferenceEngine::ResponseDesc resp;
        InferenceEngine::StatusCode rc = extFactory->getImplementations(impls, &resp);
        if (rc != InferenceEngine::OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }
        for (auto &impl : impls) {
            std::vector<InferenceEngine::LayerConfig> configs;
            rc = impl->getSupportedConfigurations(configs, &resp);
            implsConfigs.push_back(configs);

            if (rc != InferenceEngine::OK) {
                THROW_IE_EXCEPTION << resp.msg;
            }

            for (auto& config : configs) {
                std::vector<MKLDNNMemoryDesc> inputs;
                for (auto& input : config.inConfs) {
                    inputs.push_back(MKLDNNMemoryDesc(input.desc));
                }
                std::vector<MKLDNNMemoryDesc> outputs;
                for (auto& output : config.outConfs) {
                    outputs.push_back(MKLDNNMemoryDesc(output.desc));
                }
                supportedPrimitiveDescriptors.push_back({engine, inputs, outputs, impl_desc_type::unknown});
            }
        }
        if (impls.empty()) {
            THROW_IE_EXCEPTION << "Layer " << getName() << " hasn't available configurations!";
        }
    } else {
        THROW_IE_EXCEPTION << "Descriptor for generic primitive doesn't exist";
    }
}

void MKLDNNGenericNode::createPrimitive() {
    if (extFactory) {
        return;
    }
    if (!genericPrimitive)
        THROW_IE_EXCEPTION << "Descriptor for generic primitive doesn't exist";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNGenericNode::execute(mkldnn::stream strm) {
    if (genericPrimitive) {
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto& mklMemory = getParentEdgeAt(i)->getMemory();
            inputs.push_back(MKLDNNExtensionUtils::MKLMemoryToGenericMemory(mklMemory));
        }

        for (size_t i = 0; i < getChildEdges().size(); i++) {
            auto& mklMemory = getChildEdgeAt(i)->getMemory();
            outputs.push_back(MKLDNNExtensionUtils::MKLMemoryToGenericMemory(mklMemory));
        }

        genericPrimitive->SetMemory(inputs, outputs);
        genericPrimitive->Execute();
    } else if (extFactory) {
        execLayer();
    } else {
        THROW_IE_EXCEPTION << "Descriptor for generic primitive doesn't exist";
    }
}

bool MKLDNNGenericNode::created() {
    return Generic == getType();
}



bool MKLDNNGenericNode::created(const MKLDNNExtensionManager::Ptr &extMgr) {
    if (getCnnLayer() && extMgr) {
        // We should save extension manager in otder to avoid situation when
        // it will destroyed before extensibility primitives
        extensionManager = extMgr;
        genericPrimitive.reset(extensionManager->CreateExtensionPrimitive(getCnnLayer()));
        extFactory.reset(extensionManager->CreateExtensionFactory(getCnnLayer()));

        if (genericPrimitive || extFactory)
            setType(Generic);
    }
    return created();
}

void MKLDNNGenericNode::selectOptimalPrimitiveDescriptor() {
    if (genericPrimitive) {
        MKLDNNNode::selectOptimalPrimitiveDescriptor();
        return;
    }

    InferenceEngine::StatusCode rc;
    InferenceEngine::ResponseDesc resp;
    for (size_t k = 0; k < impls.size(); k++) {
        std::vector<InferenceEngine::LayerConfig> configs = implsConfigs[k];

        int selectedPrimitive = -1;
        int equalsFormatCount = -1;
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
            int equalsLocalFormatCount = 0;
            bool isSupportedType = true;
            if (getSupportedPrimitiveDescriptors()[i].getInputDescs().size() > getParentEdges().size())
                continue;
            for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[i].getInputDescs().size(); j++) {
                auto parentEdge = getParentEdgeAt(j);
                auto parentPtr = parentEdge->getParent();
                if (j < getParentEdges().size() &&
                        autoBlockingDims(parentEdge->getDims(),
                                         getSupportedPrimitiveDescriptors()[i].getInputDescs()[j].getFormat()) !=
                                getSupportedPrimitiveDescriptors()[i].getInputDescs()[j].getDims()) {
                    isSupportedType = false;
                    break;
                }
                auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();
                if (parent_spd != nullptr && parent_spd->getOutputDescs().size()) {
                    int inNum = parentEdge->getInputNum();
                    if (inNum < 0 || inNum >= parent_spd->getOutputDescs().size()) {
                        inNum = 0;
                    }
                    if (getSupportedPrimitiveDescriptors()[i].getInputDescs()[j] ==
                        parent_spd->getOutputDescs()[inNum]) {
                        equalsLocalFormatCount++;
                    }
                }
            }
            std::vector<memory::format> outputs;
            for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[i].getOutputDescs().size(); j++) {
                if (autoBlockingDims(getChildEdgeAt(j)->getDims(),
                                     getSupportedPrimitiveDescriptors()[i].getOutputDescs()[j].getFormat()) !=
                        getSupportedPrimitiveDescriptors()[i].getOutputDescs()[j].getDims()) {
                    isSupportedType = false;
                    break;
                }
            }
            if (!isSupportedType)
                continue;
            if (equalsLocalFormatCount > equalsFormatCount) {
                equalsFormatCount = equalsLocalFormatCount;
                selectedPrimitive = static_cast<int>(i);
            }
        }
        if (selectedPrimitive >= 0) {
            InferenceEngine::LayerConfig config;

            config.dynBatchSupport = false;

            for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[selectedPrimitive].getInputDescs().size(); j++) {
                InferenceEngine::DataConfig cfg;
                cfg.desc = getSupportedPrimitiveDescriptors()[selectedPrimitive].getInputDescs()[j];
                cfg.constant = configs[selectedPrimitive].inConfs[j].constant;
                if (configs[selectedPrimitive].inConfs[j].inPlace >= 0) {
                     if (getParentEdgeAt(j)->getParent()->getChildEdges().size() > 1) {
                         configs[selectedPrimitive].inConfs[j].inPlace = -1;
                     } else {
                         cfg.inPlace = configs[selectedPrimitive].inConfs[j].inPlace;
                     }
                } else {
                    cfg.inPlace = configs[selectedPrimitive].inConfs[j].inPlace;
                }
                config.inConfs.push_back(cfg);
            }
            for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[selectedPrimitive].getOutputDescs().size(); j++) {
                InferenceEngine::DataConfig cfg;
                cfg.desc = getSupportedPrimitiveDescriptors()[selectedPrimitive].getOutputDescs()[j];
                cfg.constant = configs[selectedPrimitive].outConfs[j].constant;
                if (configs[selectedPrimitive].outConfs[j].inPlace >= 0) {
                    if (configs[selectedPrimitive].outConfs[j].inPlace < getParentEdges().size() &&
                            getParentEdgeAt(configs[selectedPrimitive].outConfs[j].inPlace)->getParent()->getChildEdges().size() > 1) {
                        configs[selectedPrimitive].inConfs[j].inPlace = -1;
                    } else {
                        cfg.inPlace = configs[selectedPrimitive].outConfs[j].inPlace;
                    }
                } else {
                    cfg.inPlace = configs[selectedPrimitive].outConfs[j].inPlace;
                }
                config.outConfs.push_back(cfg);
            }
            if (configs[selectedPrimitive].dynBatchSupport) {
                config.dynBatchSupport = dynBatchLim > 0;
            }

            rc = impls[k]->init(config, &resp);
            if (rc != InferenceEngine::OK) {
                continue;
            }
            while (impls.size() > 1) {
                impls.erase(impls.begin() + 1);
            }
            implsConfigs.clear();
            implsConfigs.push_back({config});
            while (supportedPrimitiveDescriptors.size() > configs.size()) {
                supportedPrimitiveDescriptors.erase(supportedPrimitiveDescriptors.begin() + configs.size());
            }
            selectPrimitiveDescriptorByIndex(selectedPrimitive);
            return;
        }
        impls.erase(impls.begin());
        implsConfigs.erase(implsConfigs.begin());
        for (size_t i = 0; i < configs.size(); i++) {
            supportedPrimitiveDescriptors.erase(supportedPrimitiveDescriptors.begin());
        }
        k--;
    }
    if (impls.empty()) {
        THROW_IE_EXCEPTION << "Cannot find available configurations!.";
    }
    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNGenericNode::isConstant(bool fromCache) {
    if (!fromCache) {
        if (!MKLDNNNode::isConstant(fromCache)) {
            constant = true;
            if (!extFactory) {
                constant = false;
            } else {
                if (implsConfigs.empty() || implsConfigs[0].empty())
                    THROW_IE_EXCEPTION << "Cannot find selected primitive descriptor for layer " << getName();
                InferenceEngine::LayerConfig conf = implsConfigs[0][0];
                for (auto &input : conf.inConfs) {
                    if (!input.constant)
                        constant = false;
                }
                for (auto &output : conf.outConfs) {
                    if (!output.constant)
                        constant = false;
                }
            }
        }
    }
    return constant;
}

void MKLDNNGenericNode::execLayer() {
    bool isDynBatch = dynBatchLim > 0;
    std::vector<InferenceEngine::Blob::Ptr> inputs;
    std::vector<InferenceEngine::TensorDesc> inputDescs;
    std::vector<InferenceEngine::TensorDesc> outputDescs;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &mkldnnMemory = getParentEdgeAt(i)->getMemory();
        inputs.push_back(mkldnnMemory.GetBlob());
        if (isDynBatch && dynBatchLim >= inputs[inputs.size() - 1]->getTensorDesc().getDims()[0]) {
            isDynBatch = false;
        } else {
            // TODO: Ask the right dims using getShape() from previous node
            inputDescs.push_back(inputs[inputs.size() - 1]->getTensorDesc());
            inputDescs[inputDescs.size() - 1].getDims()[0] = static_cast<size_t>(dynBatchLim);
        }
    }

    if (isDynBatch) {
        auto sts = extFactory->getShapes(inputDescs, outputDescs, nullptr);
        if (sts != InferenceEngine::StatusCode::OK)
            isDynBatch = false;
    }

    if (isDynBatch) {
        for (size_t i = 0; i < inputs.size(); i++) {
            auto td = inputs[i]->getTensorDesc();
            td.setDims(inputDescs[i].getDims());
            inputs[i] = make_blob_with_precision(td, getParentEdgeAt(i)->getMemory().GetData());
        }
    }
    std::vector<InferenceEngine::Blob::Ptr> outputs;
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto& mkldnnMemory = getChildEdgeAt(i)->getMemory();
        if (isDynBatch) {
            size_t idx = i >= outputDescs.size() ? 0 : i;
            auto td = mkldnnMemory.GetBlob()->getTensorDesc();
            td.setDims(outputDescs[idx].getDims());
            outputs.push_back(make_blob_with_precision(td, mkldnnMemory.GetData()));
        } else {
            outputs.push_back(mkldnnMemory.GetBlob());
        }
    }
    auto * execImpl = dynamic_cast<InferenceEngine::ILayerExecImpl *>(impls[0].get());
    if (execImpl != nullptr) {
        InferenceEngine::ResponseDesc resp;
        InferenceEngine::StatusCode rc = execImpl->execute(inputs, outputs, &resp);
        if (rc != InferenceEngine::OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }
    }
}

void MKLDNNGenericNode::initEdges() {
    if (genericPrimitive) {
        MKLDNNNode::initEdges();
        return;
    }

    if (implsConfigs.empty() || implsConfigs[0].empty())
        THROW_IE_EXCEPTION << "Cannot find selected primitive descriptor for layer " << getName();

    InferenceEngine::LayerConfig conf = implsConfigs[0][0];
    // TODO: Fix issue with two and more outputs from one real edge -> move to ports
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        auto childEdge = getChildEdgeAt(cIdx);

        if (cIdx >= conf.outConfs.size()) {
            childEdge->sharedMemFrom(getChildEdgeAt(0));
        } else if (conf.outConfs[cIdx].inPlace >= 0 && conf.outConfs[cIdx].inPlace < getParentEdges().size() &&
                getParentEdgeAt(conf.outConfs[cIdx].inPlace)->getParent()->getChildEdges().size() == 1) {
            childEdge->sharedMemFrom(getParentEdgeAt(conf.outConfs[cIdx].inPlace));
        } else {
            childEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        }
    }
    for (size_t pIdx = 0 ; pIdx < getParentEdges().size(); pIdx++) {
        auto parentEdge = getParentEdgeAt(pIdx);
        if (pIdx >= conf.inConfs.size()) {
            parentEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        } else if (conf.inConfs[pIdx].inPlace >= 0 && conf.inConfs[pIdx].inPlace < getChildEdges().size() &&
                parentEdge->getParent()->getChildEdges().size() == 1) {
            if (getChildEdgeAt(conf.inConfs[pIdx].inPlace)->getStatus() == MKLDNNEdge::Status::NotAllocated) {
                parentEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
            } else {
                parentEdge->sharedMemFrom(getChildEdgeAt(conf.inConfs[pIdx].inPlace));
            }
        } else {
            parentEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        }
    }
}

void MKLDNNGenericNode::resolveNotAllocatedEdges() {
    if (genericPrimitive) {
        MKLDNNNode::resolveNotAllocatedEdges();
        return;
    }
    if (implsConfigs.empty() || implsConfigs[0].empty())
        THROW_IE_EXCEPTION << " Cannot find selected primitive descriptor for layer " << getName();

    InferenceEngine::LayerConfig conf = implsConfigs[0][0];
    for (size_t i = 0; i < getChildEdges().size() && i < conf.outConfs.size(); i++) {
        auto childEdge = getChildEdgeAt(i);

        if (childEdge->getStatus() == MKLDNNEdge::Status::NotAllocated &&
                conf.outConfs[i].inPlace >= 0 &&
                conf.outConfs[i].inPlace < getParentEdges().size() &&
                childEdge->getSharedEdge().get() == getParentEdgeAt(conf.outConfs[i].inPlace).get() &&
                getParentEdgeAt(conf.outConfs[i].inPlace)->getStatus() != MKLDNNEdge::Status::NotAllocated) {
            auto * memPtr = reinterpret_cast<char*>(childEdge->getMemory().GetData());
            childEdge->getMemoryPtr().reset(new MKLDNNMemory(getSelectedPrimitiveDescriptor()->getEngine()));
            childEdge->getMemoryPtr()->Create(MKLDNNMemoryDesc(conf.outConfs[i].desc), memPtr);
        }
    }
    for (size_t i = 0; i < getParentEdges().size() && i < conf.inConfs.size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge->getStatus() == MKLDNNEdge::Status::NotAllocated &&
                conf.inConfs[i].inPlace >= 0 &&
                conf.inConfs[i].inPlace < getChildEdges().size() &&
                parentEdge->getSharedEdge().get() == getChildEdgeAt(conf.inConfs[i].inPlace).get() &&
                getChildEdgeAt(conf.inConfs[i].inPlace)->getStatus() != MKLDNNEdge::Status::NotAllocated) {
            auto *memPtr = reinterpret_cast<char *>(parentEdge->getMemory().GetData());
            parentEdge->getMemoryPtr().reset(new MKLDNNMemory(getSelectedPrimitiveDescriptor()->getEngine()));
            parentEdge->getMemoryPtr()->Create(MKLDNNMemoryDesc(conf.inConfs[i].desc), memPtr);
        }
    }
}

MKLDNNGenericNode::~MKLDNNGenericNode() {
    extFactory.reset();
    genericPrimitive.reset();
    extensionManager.reset();
}
