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
#include "mkldnn_node.h"
#include "mkldnn_extension_mngr.h"

#include "caseless.hpp"
#include <vector>
#include <string>

#include <nodes/mkldnn_batchnorm_node.h>
#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_conv_node.h>
#include <nodes/mkldnn_crop_node.h>
#include <nodes/mkldnn_deconv_node.h>
#include <nodes/mkldnn_eltwise_node.h>
#include <nodes/mkldnn_fullyconnected_node.h>
#include <nodes/mkldnn_generic_node.h>
#include <nodes/mkldnn_input_node.h>
#include <nodes/mkldnn_lrn_node.h>
#include <nodes/mkldnn_pooling_node.h>
#include <nodes/mkldnn_power_node.h>
#include <nodes/mkldnn_activation_node.h>
#include <nodes/mkldnn_clamp_node.h>
#include <nodes/mkldnn_reorder_node.h>
#include <nodes/mkldnn_reshape_node.h>
#include <nodes/mkldnn_roi_pooling_node.h>
#include <nodes/mkldnn_scaleshift_node.h>
#include <nodes/mkldnn_softmax_node.h>
#include <nodes/mkldnn_tile_node.h>
#include <nodes/mkldnn_split_node.h>
#include <nodes/mkldnn_permute_node.h>
#include <nodes/mkldnn_copy_node.h>
#include <nodes/mkldnn_memory_node.hpp>

#include "mkldnn_extension_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;

std::vector<MKLDNNNode::Registry::CreatorByNameFunction> MKLDNNNode::Registry::_dataByName;
std::vector<MKLDNNNode::Registry::CreatorByLayerFunction> MKLDNNNode::Registry::_dataByLayer;

MKLDNNNode::Register<MKLDNNGenericNode> MKLDNNGenericNode::reg;
MKLDNNNode::Register<MKLDNNBatchNormalizationNode> MKLDNNBatchNormalizationNode::reg;
MKLDNNNode::Register<MKLDNNConcatNode> MKLDNNConcatNode::reg;
MKLDNNNode::Register<MKLDNNConvolutionNode> MKLDNNConvolutionNode::reg;
MKLDNNNode::Register<MKLDNNCropNode> MKLDNNCropNode::reg;
MKLDNNNode::Register<MKLDNNDeconvolutionNode> MKLDNNDeconvolutionNode::reg;
MKLDNNNode::Register<MKLDNNEltwiseNode> MKLDNNEltwiseNode::reg;
MKLDNNNode::Register<MKLDNNFullyConnectedNode> MKLDNNFullyConnectedNode::reg;
MKLDNNNode::Register<MKLDNNInputNode> MKLDNNInputNode::reg;
MKLDNNNode::Register<MKLDNNLrnNode> MKLDNNLrnNode::reg;
MKLDNNNode::Register<MKLDNNPoolingNode> MKLDNNPoolingNode::reg;
MKLDNNNode::Register<MKLDNNPowerNode> MKLDNNPowerNode::reg;
MKLDNNNode::Register<MKLDNNActivationNode> MKLDNNActivationNode::reg;
MKLDNNNode::Register<MKLDNNClampNode> MKLDNNClampNode::reg;
MKLDNNNode::Register<MKLDNNReorderNode> MKLDNNReorderNode::reg;
MKLDNNNode::Register<MKLDNNReshapeNode> MKLDNNReshapeNode::reg;
MKLDNNNode::Register<MKLDNNROIPoolingNode> MKLDNNROIPoolingNode::reg;
MKLDNNNode::Register<MKLDNNScaleShiftNode> MKLDNNScaleShiftNode::reg;
MKLDNNNode::Register<MKLDNNSoftMaxNode> MKLDNNSoftMaxNode::reg;
MKLDNNNode::Register<MKLDNNSplitNode> MKLDNNSplitNode::reg;
MKLDNNNode::Register<MKLDNNTileNode> MKLDNNTileNode::reg;
MKLDNNNode::Register<MKLDNNPermuteNode> MKLDNNPermuteNode::reg;
MKLDNNNode::Register<MKLDNNCopyNode> MKLDNNCopyNode::reg;
MKLDNNNode::Register<MKLDNNMemoryInputNode> MKLDNNMemoryInputNode::reg;
MKLDNNNode::Register<MKLDNNMemoryOutputNode> MKLDNNMemoryOutputNode::reg;

const std::vector<impl_desc_type> MKLDNNNode::primitivesPriority = {
        impl_desc_type::unknown,
        impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,
        impl_desc_type::jit_uni,
        impl_desc_type::jit_avx512_dw,
        impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,
        impl_desc_type::jit_avx2_dw,
        impl_desc_type::jit_avx2_1x1,
        impl_desc_type::jit_avx2,
        impl_desc_type::jit_sse42_dw,
        impl_desc_type::jit_sse42_1x1,
        impl_desc_type::jit_sse42,
        impl_desc_type::gemm_any,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_sse42,
        impl_desc_type::ref_any,
        impl_desc_type::ref,
};

MKLDNNNode::MKLDNNNode(Type type, const std::string &name): name(name), type(type), typeStr(typeToStr(type)),
                                                            selectedPrimitiveDescriptorIndex(-1),
                                                            permanent(false), temporary(false), constant(false),
                                                            inputDataType(memory::data_undef), outputDataType(memory::data_undef) {
    srcMemDesc = [](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.src_primitive_desc(idx).desc());
    };
    dstMemDesc = [](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.dst_primitive_desc(idx).desc());
    };
}

MKLDNNNode::MKLDNNNode(InferenceEngine::CNNLayerPtr layer): cnnLayer(layer), name(layer->name), typeStr(layer->type),
                                                            type(TypeFromName(layer->type)),
                                                            selectedPrimitiveDescriptorIndex(-1),
                                                            permanent(false), temporary(false), constant(false),
                                                            inputDataType(memory::data_undef), outputDataType(memory::data_undef) {
    srcMemDesc = [](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.src_primitive_desc(idx).desc());
    };
    dstMemDesc = [](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.dst_primitive_desc(idx).desc());
    };
    if (!layer->outData.empty()) {
        for (auto outData : layer->outData) {
            outDims.push_back(MKLDNNDims(outData->getDims()));
        }
    } else {
        // the only know case is when layer type is memory: lets check for it
        IE_ASSERT(CaselessEq<std::string>()(layer->type, "memory"));
    }

    parentEdges.resize(layer->insData.size());
    for (auto inData : layer->insData) {
        inDims.push_back(MKLDNNDims(inData.lock()->getDims()));
    }
}

void MKLDNNNode::addEdge(const MKLDNNEdgeWeakPtr& edge, size_t pIndex, size_t cIndex) {
    auto edgePtr = edge.lock();
    if (!edgePtr)
        return;
    auto parentPtr = edgePtr->getParent();
    auto childPtr = edgePtr->getChild();
    if (!parentPtr || !childPtr)
        return;
    if (cIndex < parentPtr->childEdges.size()) {
        removeEdge(parentPtr->childEdges[cIndex]);
        parentPtr->childEdges[cIndex] = edge;
    } else {
        parentPtr->childEdges.push_back(edge);
    }
    if (pIndex < childPtr->parentEdges.size()) {
        removeEdge(childPtr->parentEdges[pIndex]);
        childPtr->parentEdges[pIndex] = edge;
    } else {
        childPtr->parentEdges.push_back(edge);
    }
}

void MKLDNNNode::removeEdge(const MKLDNNEdgeWeakPtr& edge) {
    auto edgePtr = edge.lock();
    if (!edgePtr)
        return;
    auto parentPtr = edgePtr->getParent();
    auto childPtr = edgePtr->getChild();
    if (!parentPtr || !childPtr)
        return;
    for (auto it = childPtr->parentEdges.begin(); it != childPtr->parentEdges.end(); it++) {
        auto parentEdge = (*it).lock();
        if (parentEdge && parentEdge->getChild() == childPtr && parentEdge->getParent() == parentPtr) {
            (*it).reset();
            break;
        }
    }
    for (auto it = parentPtr->childEdges.begin(); it != parentPtr->childEdges.end(); it++) {
        auto childEdge = (*it).lock();
        if (childEdge && childEdge->getChild() == childPtr && childEdge->getParent() == parentPtr) {
            (*it).reset();
            break;
        }
    }
}

void MKLDNNNode::remove() {
    for (const auto &parentEdge : parentEdges) {
        removeEdge(parentEdge);
    }
    for (const auto &childEdge : childEdges) {
        removeEdge(childEdge);
    }
}


MKLDNNNode* MKLDNNNode::CreateNode(const Type type, const std::string &name,
                                   const MKLDNNExtensionManager::Ptr& extMgr) {
    MKLDNNNode* newNode = Registry::CreateNode(type, name, extMgr);
    if (!newNode)
        THROW_IE_EXCEPTION << "Unsupported primitive of type: " << type << " name: " << name;
    return newNode;
}

MKLDNNNode* MKLDNNNode::CreateNode(const InferenceEngine::CNNLayerPtr& layer,
                                   const MKLDNNExtensionManager::Ptr& extMgr) {
    MKLDNNNode* newNode = Registry::CreateNode(layer, extMgr);
    if (!newNode)
        THROW_IE_EXCEPTION << "Unsupported primitive of type: " << layer->type << " name: " << layer->name;

    return newNode;
}

bool MKLDNNNode::isEdgesEmpty(const std::vector<MKLDNNEdgeWeakPtr>& edges) const {
    for (auto &edge : edges) {
        if (edge.lock())
            return false;
    }
    return true;
}

void MKLDNNNode::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(primitivesPriority);
}

void MKLDNNNode::selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority) {
    for (auto& type : priority) {
        int selectedPrimitive = -1;
        int equalsFormatCount = -1;
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
            impl_desc_type supportedType = getSupportedPrimitiveDescriptors()[i].getImplementationType();
            if (type == supportedType) {
                int equalsLocalFormatCount = 0;
                if (getSupportedPrimitiveDescriptors()[i].getInputDescs().size() > getParentEdges().size())
                    continue;
                for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[i].getInputDescs().size(); j++) {
                    auto parentEdge = getParentEdgeAt(j);
                    auto parentPtr = parentEdge->getParent();
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
                if (equalsLocalFormatCount > equalsFormatCount) {
                    equalsFormatCount = equalsLocalFormatCount;
                    selectedPrimitive = static_cast<int>(i);
                }
            }
        }
        if (selectedPrimitive >= 0) {
            selectPrimitiveDescriptorByIndex(selectedPrimitive);
            return;
        }
    }

    if (!getSupportedPrimitiveDescriptors().size())
        THROW_IE_EXCEPTION << "Supported primitive descriptors list is empty.";
    // fallback. If there are no primitives from priority list just select a first
    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNNode::initAsInPlace() {
    if (getParentEdges().size() != 1 || getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1)
        return false;

    MKLDNNDims dims = getParentEdgeAt(0)->getDims();
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        if (getChildEdgeAt(cIdx)->getDims() != dims) {
            return false;
        }
    }
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        auto childEdge = getChildEdgeAt(cIdx);
        if (childEdge->getStatus() != MKLDNNEdge::Status::NeedAllocation &&
                childEdge->getStatus() != MKLDNNEdge::Status::Uninitialized)
            return false;
    }
    for (size_t pIdx = 0; pIdx < getParentEdges().size(); pIdx++) {
        auto parentEdge = getParentEdgeAt(pIdx);
        parentEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
    }
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        auto childEdge = getChildEdgeAt(cIdx);
        childEdge->sharedMemFrom(getParentEdgeAt(0));
    }
    return true;
}

void MKLDNNNode::initEdges() {
    if (!initAsInPlace()) {
        for (size_t pIdx = 0; pIdx < getParentEdges().size(); pIdx++) {
            auto parentEdge = getParentEdgeAt(pIdx);
            parentEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
        }
        for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
            auto childEdge = getChildEdgeAt(cIdx);
            if (!cIdx) {
                childEdge->changeStatus(MKLDNNEdge::Status::NeedAllocation);
            } else {
                childEdge->sharedMemFrom(getChildEdgeAt(0));
            }
        }
    }
}

void MKLDNNNode::resolveNotAllocatedEdges() {}

std::string MKLDNNNode::getPrimitiveDescriptorType() {
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    if (selectedPrimitiveDesc) {
        return primitiveDescriptorTypeToString(selectedPrimitiveDesc->getImplementationType());
    }

    return "undef";
}

std::string MKLDNNNode::primitiveDescriptorTypeToString(impl_desc_type type) {
    switch (type) {
        case impl_desc_type::unknown:
            return "unknown";
        case impl_desc_type::undef:
            return "undef";
        case impl_desc_type::jit_avx512:
            return "jit_avx512";
        case impl_desc_type::jit_avx2:
            return "jit_avx2";
        case impl_desc_type::jit_sse42:
            return "jit_sse42";
        case impl_desc_type::jit_uni:
            return "jit_uni";
        case impl_desc_type::jit_avx512_1x1:
            return "jit_avx512_1x1";
        case impl_desc_type::jit_avx2_1x1:
            return "jit_avx2_1x1";
        case impl_desc_type::jit_sse42_1x1:
            return "jit_sse42_1x1";
        case impl_desc_type::jit_uni_1x1:
            return "jit_uni_1x1";
        case impl_desc_type::jit_avx512_dw:
            return "jit_avx512_dw";
        case impl_desc_type::jit_avx2_dw:
            return "jit_avx2_dw";
        case impl_desc_type::jit_sse42_dw:
            return "jit_sse42_dw";
        case impl_desc_type::jit_uni_dw:
            return "jit_uni_dw";
        case impl_desc_type::gemm_any:
            return "gemm_any";
        case impl_desc_type::gemm_blas:
            return "gemm_blas";
        case impl_desc_type::gemm_avx512:
            return "gemm_avx512";
        case impl_desc_type::gemm_avx2:
            return "gemm_avx2";
        case impl_desc_type::gemm_sse42:
            return "gemm_sse42";
        case impl_desc_type::ref:
        case impl_desc_type::ref_any:
            return "ref";
        case impl_desc_type::reorder:
            return "reorder";
        default:
            return "unknown";
    }
    return "undef";
}

const MKLDNNEdgePtr MKLDNNNode::getParentEdgeAt(size_t idx) const {
    if (idx >= parentEdges.size())
        THROW_IE_EXCEPTION << "Node " << getName() << " contains less parent edges than " << idx;
    auto parentEdgePtr = parentEdges[idx].lock();
    if (!parentEdgePtr)
        THROW_IE_EXCEPTION << "Node " << getName() << " contains empty parent edge for index " << idx;
    return parentEdgePtr;
}

const MKLDNNEdgePtr MKLDNNNode::getChildEdgeAt(size_t idx) const {
    if (idx >= childEdges.size())
        THROW_IE_EXCEPTION << "Node " << getName() << " contains less child edges than " << idx;
    auto childEdgePtr = childEdges[idx].lock();
    if (!childEdgePtr)
        THROW_IE_EXCEPTION << "Node " << getName() << " contains empty child edge for index " << idx;
    return childEdgePtr;
}

std::vector<memory::format> MKLDNNNode::getAvailableFormatsForDims(const MKLDNNDims &dims) const {
    if (dims.ndims() == 1)
        return {memory::format::x};
    else if (dims.ndims() == 2)
        return {memory::format::nc};
    else if (dims.ndims() == 4)
        return {memory::format::nchw, memory::format::nChw8c, memory::format::nChw16c};
    return {memory::format::any};
}

MKLDNNDims MKLDNNPlugin::autoBlockingDims(const MKLDNNDims &dims, mkldnn::memory::format fmt) {
    MKLDNNDims res = dims;

    switch (fmt) {
        case memory::format::nChw8c:
            res[1] = rnd_up(res[1], 8);
            break;
        case memory::format::gOIhw8o8i:
            res[1] = rnd_up(res[1], 8);
            res[2] = rnd_up(res[2], 8);
            break;
        case memory::format::OIhw8o8i:
            res[0] = rnd_up(res[0], 8);
            res[1] = rnd_up(res[1], 8);
            break;
        case memory::format::nChw16c:
            res[1] = rnd_up(res[1], 16);
            break;
        case memory::format::gOIhw16o16i:
            res[1] = rnd_up(res[1], 16);
            res[2] = rnd_up(res[2], 16);
            break;
        case memory::format::OIhw16o16i:
            res[0] = rnd_up(res[0], 16);
            res[1] = rnd_up(res[1], 16);
            break;
        default:
            // keep dims
            break;
    }
    return res;
}

void MKLDNNNode::execute(mkldnn::stream strm) {
    if (prim) {
        strm.submit({*prim});
    }
}

void MKLDNNNode::initSupportedPrimitiveDescriptors(const mkldnn::engine &engine) {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    for (auto& desc : descs) {
        primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine);
        do {
            std::vector<MKLDNNMemoryDesc> srcDescs;
            for (size_t i = 0; i < desc.inputNumbers() && srcMemDesc; i++)
                srcDescs.push_back(srcMemDesc(itpd, i));

            std::vector<MKLDNNMemoryDesc> intDescs;
            for (size_t i = 0; i < internalBlobDesc.size(); i++)
                intDescs.push_back(internalBlobDesc[i](itpd, 0));

            std::vector<MKLDNNMemoryDesc> dstDescs;
            for (size_t i = 0; i < desc.outputNumbers() && dstMemDesc; i++)
                dstDescs.push_back(dstMemDesc(itpd, i));
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str().c_str());

            supportedPrimitiveDescriptors.push_back({engine, srcDescs, dstDescs, intDescs, impl_type});
        } while (itpd.next());
    }
}

InferenceEngine::Blob::Ptr MKLDNNNode::createInternalBlob(InferenceEngine::SizeVector dims, bool weights) {
    auto checkSize = [](size_t dst_size, size_t src_size) {
        if (dst_size < src_size) {
            THROW_IE_EXCEPTION << "Cannot create internal buffer. Buffer can be overrun.";
        }
    };
    auto * wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(getCnnLayer().get());
    if (wLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get weightable layer for node " << getName() << ".";

    InferenceEngine::Blob::Ptr blb = weights ? wLayer->_weights : wLayer->_biases;

    if (blb == nullptr)
        THROW_IE_EXCEPTION << "Cannot get internal blob layer for node " << getName() << ".";

    InferenceEngine::TensorDesc desc(blb->precision(), dims, InferenceEngine::TensorDesc::getLayoutByDims(dims));
    InferenceEngine::TBlob<float>::Ptr internalBlob = InferenceEngine::make_shared_blob<float>(desc);
    internalBlob->allocate();
    char *data = internalBlob->buffer();
    size_t intBuffSize = internalBlob->byteSize();

    size_t offset = blb->byteSize();
    checkSize(intBuffSize, offset);
    memcpy(data, blb->buffer(), blb->byteSize());
    data += blb->byteSize();
    for (const auto &merged : getMergeWith()) {
        wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(merged->getCnnLayer().get());
        if (wLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot convert merged weightable layer for node "
                               << getName() << ".";
        blb = weights ? wLayer->_weights : wLayer->_biases;

        if (blb == nullptr)
            THROW_IE_EXCEPTION << "Cannot get internal blob layer for node " << getName() << ".";
        offset += blb->byteSize();
        checkSize(intBuffSize, offset);
        memcpy(data, blb->buffer(), blb->byteSize());
        data += blb->byteSize();
    }

    return internalBlob;
}

void MKLDNNNode::prepareMemory(const MKLDNNPrimitiveDescInfo *selected_pd) {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    internalBlobMemory.clear();
    for (size_t i = 0; i < internalBlobs.size(); i++) {
        auto& internalBlob = internalBlobs[i];
        internalBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getSelectedPrimitiveDescriptor()->getEngine())));
        MKLDNNDims blobDims = MKLDNNDims(internalBlob->getTensorDesc().getDims());
        memory::format format = memory::oihw;

        if (blobDims.ndims() == 1) {
            format = memory::x;
        } else if (blobDims.ndims() == 2) {
            format = memory::oi;
        } else if (blobDims.ndims() == 5) {
            format = memory::goihw;
        }

        MKLDNNDims real_dims = selected_pd->getInternalDescs()[i].getDims();
        if (blobDims == real_dims) {  // No auto blocking
            // TODO: Cannot create memory from selected_pd->getInternalDescs()[i] because ScaleShift changes dims
            internalBlobMemory[i]->Create(blobDims, getInputDataType(), selected_pd->getInternalDescs()[i].getFormat());
            internalBlobMemory[i]->SetData(getInputDataType(), format, internalBlob->buffer(),
                                           blobDims.size() * MKLDNNExtensionUtils::sizeOfDataType(getInputDataType()));
        } else {  // Auto blocking, logic and real dims are different
            if (blobDims.ndims() != real_dims.ndims() || blobDims.ndims() > 5)
                THROW_IE_EXCEPTION << getName() << " Error: CPU plugin supports auto blocking only "
                                   << "for blobs with a number of dimensions less than 6!";
            InferenceEngine::Blob::Ptr tmp_wght =
                    InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32, real_dims.ToSizeVector());

            tmp_wght->allocate();

            int with_group = 0;
            if (blobDims.ndims() == 5)
                with_group = 1;

            // Logic dims
            int L_G = blobDims.ndims() > 0 && with_group ? blobDims[0] : 1;
            int L_N = blobDims.ndims() > 0 ? blobDims[0 + with_group] : 1;
            int L_C = blobDims.ndims() > 1 ? blobDims[1 + with_group] : 1;
            int L_H = blobDims.ndims() > 2 ? blobDims[2 + with_group] : 1;
            int L_W = blobDims.ndims() > 3 ? blobDims[3 + with_group] : 1;

            // Ref
            int R_G = real_dims.ndims() > 0 && with_group ? real_dims[0] : 1;
            int R_N = real_dims.ndims() > 0 ? real_dims[0 + with_group] : 1;
            int R_C = real_dims.ndims() > 1 ? real_dims[1 + with_group] : 1;
            int R_H = real_dims.ndims() > 2 ? real_dims[2 + with_group] : 1;
            int R_W = real_dims.ndims() > 3 ? real_dims[3 + with_group] : 1;

            if (L_H != R_H || L_W != R_W)
                THROW_IE_EXCEPTION << "Unsuported mode of auto blocking tensors";

            auto * tmp_data = tmp_wght->buffer().as<float*>();
            auto * in_data = internalBlob->buffer().as<float*>();
            memset(tmp_data, 0,  real_dims.size()* sizeof(float));

            for (int g = 0; g < L_G; g++)
            for (int n = 0; n < L_N; n++)
            for (int c = 0; c < L_C; c++)
            for (int h = 0; h < L_H; h++)
            for (int w = 0; w < L_W; w++) {
                int l_indx = g * L_N * L_C * L_H * L_W +
                        n * L_C * L_H * L_W +
                        c * L_H * L_W + h * L_W + w;
                int r_indx = g * R_N * R_C * R_H * R_W +
                        n * R_C * R_H * R_W +
                        c * R_H * R_W + h * R_W + w;

                tmp_data[r_indx] = in_data[l_indx];
            }

            internalBlobMemory[i]->Create(real_dims, getInputDataType(), selected_pd->getInternalDescs()[i].getFormat());
            internalBlobMemory[i]->SetData(getInputDataType(), format, tmp_wght->buffer(), tmp_wght->byteSize());
        }
    }
}

bool MKLDNNNode::isConstant(bool fromCache) {
    if (!fromCache) {
        constant = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            if (!getParentEdgeAt(i)->getParent()->isConstant(false)) {
                constant = false;
                break;
            }
        }
        if (constant)
            constant = !getParentEdges().empty();
    }
    return constant;
}

void MKLDNNNode::cleanup() {
    internalBlobs.clear();
    cnnLayer.reset();

    for (auto it : fusedWith) {
        it->cleanup();
    }

    for (auto it : mergedWith) {
        it->cleanup();
    }
}

std::string MKLDNNNode::typeToStr(Type type) {
    switch (type) {
        case Generic:
            return "Generic";
        case Reorder:
            return "Reorder";
        case Input:
            return "Input";
        case Output:
            return "Output";
        case Convolution:
            return "Convolution";
        case Deconvolution:
            return "Deconvolution";
        case Convolution_Sum:
            return "Convolution_Sum";
        case Convolution_Activation:
            return "Convolution_Activation";
        case Convolution_Sum_Activation:
            return "Convolution_Sum_Activation";
        case Activation:
            return "Activation";
        case Clamp:
            return "Clamp";
        case Lrn:
            return "Lrn";
        case Pooling:
            return "Pooling";
        case FullyConnected:
            return "FullyConnected";
        case SoftMax:
            return "SoftMax";
        case Split:
            return "Split";
        case Concatenation:
            return "Concatenation";
        case Power:
            return "Power";
        case ScaleShift:
            return "ScaleShift";
        case Eltwise:
            return "Eltwise";
        case Crop:
            return "Crop";
        case Reshape:
            return "Reshape";
        case Tile:
            return "Tile";
        case SimplerNMS:
            return "SimplerNMS";
        case ROIPooling:
            return "ROIPooling";
        case BatchNormalization:
            return "BatchNormalization";
        case Flatten:
            return "Flatten";
        case Permute:
            return "Permute";
        case Copy:
            return "Copy";
        case MemoryOutput:
            return "MemoryOutput";
        case MemoryInput:
            return "MemoryInput";
        default:
            return "Unknown";
    }
}

MKLDNNNode *MKLDNNNode::Registry::CreateNode(const Type type, const std::string &name,
                                             const MKLDNNExtensionManager::Ptr& extMgr) {
    for (auto maker : _dataByName) {
        MKLDNNNode *ol = maker(type, name);
        if (ol != nullptr && ol->created(extMgr))
            return ol;
        delete ol;
    }
    return nullptr;
}

MKLDNNNode *MKLDNNNode::Registry::CreateNode(const InferenceEngine::CNNLayerPtr &layer,
                                             const MKLDNNExtensionManager::Ptr& extMgr) {
    for (auto maker : _dataByLayer) {
        std::unique_ptr<MKLDNNNode> ol(maker(layer));
        if (ol != nullptr && ol->created(extMgr))
            return ol.release();
    }
    return nullptr;
}

void MKLDNNNode::Registry::RegisterNode(MKLDNNNode::Registry::CreatorByNameFunction f) {
    _dataByName.push_back(f);
}

void MKLDNNNode::Registry::RegisterNode(MKLDNNNode::Registry::CreatorByLayerFunction f) {
    _dataByLayer.push_back(f);
}
