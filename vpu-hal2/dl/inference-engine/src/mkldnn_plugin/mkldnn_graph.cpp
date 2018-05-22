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

#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <caseless.hpp>

#include "mkldnn_graph.h"
#include "mkldnn_graph_optimizer.h"
#include <debug.h>
#include <nodes/mkldnn_input_node.h>
#include <nodes/mkldnn_reorder_node.h>
#include "mkldnn_extension_utils.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn/omp_manager.h"
#include <omp.h>
#include <graph_tools.hpp>
#include <cpp_interfaces/ie_executor_manager.hpp>
#include "ie_algorithm.hpp"
#include "mkldnn_infer_request.h"
#include "mkldnn_async_infer_request.h"
// #define DEBUG_DUMP_PATH "/home/user/HDD/gna-mkldnn/"
// #define DEBUG_DUMP_NEW_FOLDER_PER_INFER
#ifdef DEBUG_DUMP_PATH
#include "../../thirdparty/mkl-dnn/src/common/memory_desc_wrapper.hpp"
#include <iomanip>
// #define DEBUG_BMP_OUTPUT 1
#endif

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace MKLDNNPlugin::cpu;
using namespace InferenceEngine;
using namespace InferenceEngine::MKLDNNPlugin;

void BindThreads(mkldnn::engine eng) {
    static bool alreadyBind = false;
    if (!alreadyBind) {
        int env_cores = 0;
        if (getenv("OMP_NUM_THREADS") != nullptr) {
            try {
                env_cores = std::stoi(std::string(getenv("OMP_NUM_THREADS")));
            } catch (...) {
                env_cores = 0;
            }
        }
#if !(defined(__APPLE__) || defined(_WIN32))
        OpenMpManager::setGpuDisabled();
        OpenMpManager::bindOpenMpThreads(env_cores);
#else
        int num_cores = env_cores == 0 ? OpenMpManager::getOpenMpThreadNumber() : env_cores;
        omp_set_num_threads(num_cores);
#endif

        alreadyBind = true;
    }
}

void MKLDNNGraph::CreateGraph(ICNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr) {
    if (IsReady()) {
        ForgetGraphData();
    }

    if (config.useThreadBinding) BindThreads(eng);

    // go over the inputs and create input primitives
    InputsDataMap inputs;
    network.getInputsInfo(inputs);
    if (inputs.empty()) {
        THROW_IE_EXCEPTION << "MKLDNNGraph::CreateGraph: No inputs for the topology";
    }

    for (auto input : inputs) {
        MKLDNNNodePtr inputNode;
        auto inputLayer = input.second->getInputData()->getCreatorLayer().lock();
        if (!inputLayer) {
            // For v1 parser
            inputLayer.reset(new CNNLayer({input.second->getInputData()->getName(),
                                           "Input",
                                           input.second->getInputData()->getPrecision()}));

            inputLayer->outData.push_back(input.second->getInputData());
        }

        inputNode = MKLDNNNodePtr(MKLDNNNode::CreateNode(inputLayer, extMgr));

        graphNodes.push_back(inputNode);
        inputNodes[input.first] = inputNode;

        for (const auto &layer : input.second->getInputData()->getInputTo()) {
            ParseNode(layer.second, inputNode, extMgr, 0);
        }

        // Loading mean images
        MKLDNNDims outDims(inputNode->getChildEdgeAt(0)->getDims());
        if (inputs.find(input.first) != inputs.end()) {
            InputInfo::Ptr ii = inputs[input.first];
            if (ii && ii->getPreProcess().getNumberOfChannels()) {
                _meanImages[input.first].Load(outDims, ii);
            }
        }
    }

    auto allInputs = CNNNetGetAllInputLayers(network);
    for (auto input : allInputs) {
        auto isRealInput = std::find_if(std::begin(inputs), std::end(inputs), [&](InputsDataMap::value_type& inputInfo){
            return inputInfo.second->getInputData()->getName() == input->name;
        });
        if (isRealInput != std::end(inputs)) {
            continue;
        }

        MKLDNNNodePtr inputNode;
        CaselessEq<std::string> comparator;
        if (!comparator(input->type, "Const")) {
            auto memoryId = input->GetParamAsString("id");
            inputNode = MKLDNNNodePtr(MKLDNNNode::CreateNode(Type::MemoryInput, input->name + "/id=" + memoryId, extMgr));
        } else {
            inputNode = MKLDNNNodePtr(MKLDNNNode::CreateNode(input, extMgr));
        }
        graphNodes.push_back(inputNode);
        inputNodes[input->name] = inputNode;

        size_t count_out = 0;
        for (auto &&outData : input->outData) {
            for (auto &&layer : outData->getInputTo()) {
                ParseNode(layer.second, inputNode, extMgr, count_out);
            }
            count_out++;
        }
    }

    std::map<std::string, DataPtr> output;
    network.getOutputsInfo(output);

    for (auto it = output.begin(); it != output.end(); it++) {
        MKLDNNNodePtr node = FindNodeWithName((*it).second->getCreatorLayer().lock()->name);
        if (!node)
            THROW_IE_EXCEPTION << "Cannot find output layer " << (*it).second->getCreatorLayer().lock()->name;

        std::string name = "out_" + node->getName();
        MKLDNNNodePtr outputLayer(new MKLDNNInputNode(Output, name));

        MKLDNNEdgePtr edgePtr(new MKLDNNEdge(node, outputLayer));
        graphEdges.push_back(edgePtr);
        outputLayer->addEdge(edgePtr, 0, node->getChildEdges().size());
        graphNodes.push_back(outputLayer);
        outputNodes.push_back(outputLayer);
    }

    MKLDNNGraphOptimizer optimizer;
    optimizer.Optimize(*this);

    InitNodes();
    SelectOptimalPrimitiveDescriptors();

    InitEdges();

    SortTopologically();

    Allocate();

    CreatePrimitives();

    for (auto &graphNode : graphNodes) {
        graphNode->cleanup();
    }

    mkldnn::stream stream = mkldnn::stream(stream::kind::eager);
    for (auto &graphNode : graphNodes) {
        if (!graphNode->isConstant(false))
            continue;
        graphNode->execute(stream);
    }

    status = Ready;
}

MKLDNNNodePtr MKLDNNGraph::ParseNode(const CNNLayerPtr& cnnLayer, MKLDNNNodePtr& parent,
                                     const MKLDNNExtensionManager::Ptr& extMgr, size_t outIdx) {
    if (cnnLayer->precision != Precision::FP32) {
        THROW_IE_EXCEPTION << "The plugin does not support " << cnnLayer->precision;
    }

    MKLDNNNodePtr node = FindNodeWithName(cnnLayer->name);
    bool exists = false;
    if (node) {
        exists = true;
    } else {
        node.reset(MKLDNNNode::CreateNode(cnnLayer, extMgr));
    }

    MKLDNNEdgePtr edgePtr;
    size_t shift = 0;
    if (outIdx >= parent->getChildEdges().size() || !parent->getChildEdges()[outIdx].lock()) {
        edgePtr.reset(new MKLDNNEdge(parent, node));
        graphEdges.push_back(edgePtr);
    } else {
        edgePtr = parent->getChildEdgeAt(outIdx);
        if (edgePtr->getChild() != node) {
            edgePtr.reset(new MKLDNNEdge(parent, node));
            graphEdges.push_back(edgePtr);
            shift = parent->getChildEdges().size();
        }
    }


    size_t pIndex = node->getParentEdges().size();
    if (parent->getCnnLayer() != nullptr) {
        for (size_t idx = 0; idx < cnnLayer->insData.size(); idx++) {
            if (cnnLayer->insData[idx].lock().get() == parent->getCnnLayer()->outData[outIdx].get()) {
                pIndex = idx;
                break;
            }
        }

        node->addEdge(edgePtr, pIndex, outIdx + shift);
    } else {
        for (size_t idx = 0; idx < cnnLayer->insData.size(); idx++) {
            if (cnnLayer->insData[idx].lock()->getName() == parent->getName()) {
                pIndex = static_cast<int>(idx);
                break;
            }
        }
        node->addEdge(edgePtr, pIndex, outIdx + shift);
    }

    if (exists)
        return node;

    graphNodes.push_back(node);

    size_t count_out = 0;
    for (const auto &layer : cnnLayer->outData) {
        for (const auto &data : layer->getInputTo()) {
            ParseNode(data.second, node, extMgr, count_out);
        }
        count_out++;
    }

    return node;
}

void MKLDNNGraph::InitNodes() {
    for (auto &node : graphNodes) {
        mkldnn::memory::data_type outputDataType = mkldnn::memory::f32;
        if (node->getType() == Input && _meanImages.find(node->getName()) == _meanImages.end()) {
            // If it is an input layer, its output data type is undefined because it should be equal to the CNN layer input precision
            outputDataType = mkldnn::memory::data_undef;
        }

        node->createDescriptor(mkldnn::memory::f32, outputDataType);

        node->initSupportedPrimitiveDescriptors(getEngine());
    }
}

void MKLDNNGraph::SelectOptimalPrimitiveDescriptors() {
    for (auto& node : graphNodes) {
        node->selectOptimalPrimitiveDescriptor();
    }
}

void MKLDNNGraph::InitEdges() {
    size_t numberOfEdges = graphEdges.size();
    for (auto i = 0; i < numberOfEdges; i++) {
        if (graphEdges[i]->needReorder()) {
            std::string layerName = graphEdges[i]->getParent()->getName() + "_" +
                    MKLDNNMemory::formatToString(graphEdges[i]->getInputDesc().getFormat()) + "_" +
                    MKLDNNMemory::formatToString(graphEdges[i]->getOutputDesc().getFormat());
            MKLDNNNodePtr newReorder(new MKLDNNReorderNode(Reorder, layerName));
            auto *reorderPtr = dynamic_cast<MKLDNNReorderNode *>(newReorder.get());
            if (reorderPtr) {
                reorderPtr->setDescs(graphEdges[i]->getInputDesc(), graphEdges[i]->getOutputDesc());
            }
            MKLDNNEdgePtr beforeNode(new MKLDNNEdge(graphEdges[i]->getParent(), newReorder));
            beforeNode->setDims(graphEdges[i]->getDims());
            MKLDNNEdgePtr afterNode(new MKLDNNEdge(newReorder, graphEdges[i]->getChild()));
            afterNode->setDims(graphEdges[i]->getDims());

            int oIndex = graphEdges[i]->getOutputNum();
            int iIndex = graphEdges[i]->getInputNum();
            if (iIndex < 0 || oIndex < 0)
                THROW_IE_EXCEPTION << "Cannot create reorder for nodes: "
                                   << graphEdges[i]->getParent()->getName() << " and "
                                   << graphEdges[i]->getChild()->getName() << ".";
            graphEdges[i]->getParent()->addEdge(beforeNode, 0, static_cast<size_t>(iIndex));
            graphEdges[i]->getChild()->addEdge(afterNode, static_cast<size_t>(oIndex), 0);

            newReorder->createDescriptor(graphEdges[i]->getParent()->getOutputDataType(), graphEdges[i]->getChild()->getInputDataType());
            newReorder->initSupportedPrimitiveDescriptors(getEngine());
            newReorder->selectOptimalPrimitiveDescriptor();

            graphEdges.push_back(beforeNode);
            graphEdges.push_back(afterNode);

            graphNodes.push_back(newReorder);
            graphEdges.erase(graphEdges.begin() + i);
            i--;
            numberOfEdges--;
        }
    }
}

void MKLDNNGraph::Allocate() {
    for (auto& node : graphNodes) {
        node->initEdges();
    }
    for (auto& edge : graphEdges) {
        edge->allocate();
    }
    for (auto& node : graphNodes) {
        node->resolveNotAllocatedEdges();
    }
    for (auto& edge : graphEdges) {
        edge->validate();
    }
}

void MKLDNNGraph::CreatePrimitives() {
    for (auto& node : graphNodes) {
        node->createPrimitive();
    }
}

void MKLDNNGraph::PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in) {
    if (!IsReady()) THROW_IE_EXCEPTION<< "Wrong state. Topology not ready.";

    auto input = inputNodes.find(name);
    if (input != inputNodes.end()) {
        MKLDNNDims outDims = input->second->getChildEdgeAt(0)->getDims();

        const void *ext_data_ptr = in->cbuffer();
        void *inter_data_ptr = input->second->getChildEdgeAt(0)->getMemory().GetData();

        if (ext_data_ptr != inter_data_ptr)
        input->second->getChildEdgeAt(0)->getMemory().SetData(MKLDNNExtensionUtils::IEPrecisionToDataType(in->getTensorDesc().getPrecision()),
                MKLDNNMemory::GetPlainFormat(outDims), ext_data_ptr, in->byteSize(), false);

        // todo: make sure 'name' exists in this map...
        if (_meanImages.find(name) != _meanImages.end()) {
            if (in->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                _meanImages[name].Subtract(outDims, reinterpret_cast<float *>(inter_data_ptr));
            } else {
                THROW_IE_EXCEPTION << "Mean image of type " << in->getTensorDesc().getPrecision().name() << " is unsupported";
            }
        }
    } else {
        THROW_IE_EXCEPTION << "Input blob for infer '" << name << "' doesn't correspond to input in network";
    }
}

void MKLDNNGraph::PullOutputData(BlobMap &out) {
    if (!IsReady())
        THROW_IE_EXCEPTION << "Wrong state. Topology not ready.";

    for (MKLDNNNodePtr &node : outputNodes) {
        // remove out_ from node name
        std::string name = node->getName().substr(4);
        const MKLDNNMemory& intr_blob = node->getParentEdgeAt(0)->getMemory();
        if (out.find(name) == out.end()) {
            // TODO: Create blob from MemoryDesc
            Blob::Ptr outBlob = make_shared_blob<float>({Precision::FP32, node->getParentEdgeAt(0)->getDims().ToSizeVector(),
                                                         TensorDesc::getLayoutByDims(node->getParentEdgeAt(0)->getDims().ToSizeVector())},
                                                        reinterpret_cast<float*>(intr_blob.GetData()));
            out[name] = outBlob;
        }

        Blob::Ptr &ext_blob = out[name];

        // TODO: Why we allow allocation of output memory inside Infer call??
        // Suggestion is to disable this behaviour
        if (ext_blob->buffer() == nullptr) {
            SizeVector dims = node->getParentEdgeAt(0)->getDims().ToSizeVector();
            std::reverse(dims.begin(), dims.end());  // Blobs dims are in reverse order (legacy of OpenVX :-( )
            ext_blob->Resize(dims);
            ext_blob->allocate();
        }

        if (ext_blob->byteSize() != intr_blob.GetSize())
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << ext_blob->size() << "!=" << intr_blob.GetSize()/sizeof(float) << ").";

        void *ext_blob_ptr = ext_blob->buffer();
        void *intr_blob_ptr = intr_blob.GetData();

        // That is the same memory. No need to copy
        if (ext_blob_ptr == intr_blob_ptr) continue;

        int MB = intr_blob.GetDims()[0];
        int MB_to_process = config.batchLimit ?
                std::min<int>(config.batchLimit, MB) : MB;
        size_t size_to_copy = intr_blob.GetSize() * MB_to_process / MB;

        memcpy(ext_blob_ptr, intr_blob_ptr, size_to_copy);
    }
}

#ifdef DEBUG_BMP_OUTPUT
#include <sys/types.h>
#include <sys/stat.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../thirdparty/stb_lib/stb_image_write.h"

#if defined(_WIN32)
#define mkdir(dir, mode) _mkdir(dir)
#endif

void dump_as_bitmaps(const std::string name, const float* data,
                     const SizeVector& cdims,
                    mkldnn::impl::memory_format_t format = mkldnn::impl::memory_format::nchw) {
    std::string dir_name = name + "_bmp_dir/";
    mkdir(dir_name.c_str(), 0755);

    std::ofstream layer_bmp_log;
    layer_bmp_log.open(dir_name + "bmp_dump_log.txt");
    layer_bmp_log << "Format " << format << std::endl;

    if (cdims.size() == 1) {
        layer_bmp_log << "Only one dimension: " << cdims[0] << std::endl;
        layer_bmp_log.close();
        return;
    }

    SizeVector dims(cdims.rbegin(), cdims.rend());

    size_t x = dims[0], y = dims[1], total_images = 1;
    size_t img_sz = x*y;

    for (size_t k = 0; k < dims.size(); ++k)
        if (dims[k])
            total_images *= dims[k];

    total_images /= img_sz;

    //  sanity checks
    if (img_sz < 100) {
        layer_bmp_log << "Image size is too small" << std::endl;
        layer_bmp_log.close();
        return;
    } else if (x < 10 || y < 10 || x > 2048 || y > 2048) {
        layer_bmp_log << "Dimensions are unapropriate to dump - " << y << "x" << x << std::endl;
        layer_bmp_log.close();
        return;
    } else {
        float ratio = static_cast<float>(x) / static_cast<float>(y);
        if (ratio < 1.0) ratio = 1.0 / ratio;

        if (ratio > 8.f) {
            layer_bmp_log << "Suspicious aspect ratio - " << ratio << std::endl;
            layer_bmp_log.close();
            return;
        }
    }

    layer_bmp_log << total_images << " images to write ..." << std::endl;

    const float* dataPtr = data;
    for (size_t img = 0; img < total_images; img++) {
        std::string img_name = "img" + std::to_string(img) + ".bmp";

        //  copy image plane to separate buffer,
        //  normalize and convert to 3-channel 8-bit bmp
        std::vector<float> imgbuf(img_sz);
        int stride = 1;
        switch (format) {
        case mkldnn::impl::memory_format::nChw8c:
            stride = 8;
            break;
        case mkldnn::impl::memory_format::nChw16c:
            stride = 16;
            break;
        case mkldnn::impl::memory_format::nchw:
        default:
            break;
        }

        float maxval = -FLT_MAX, minval = FLT_MAX;
        for (size_t i = 0; i < y; i++)
            for (size_t j = 0; j < x; j++) {
                float val = dataPtr[(i*x + j) * stride];
                if (val > maxval) maxval = val;
                if (val < minval) minval = val;
                imgbuf[i*x + j] = val;
            }

        if (minval >= 0.f && maxval <= 0.f) {
            layer_bmp_log << img_name << " all zero." << std::endl;
        } else {
            const float mult = 256.f / (maxval - minval);
            std::vector<unsigned char> bmpbuf(img_sz * 3);
            unsigned char* bmp_ptr = bmpbuf.data();

            for (int i = 0; i < imgbuf.size(); i++, bmp_ptr += 3) {
                if (imgbuf[i] >= 0.f && imgbuf[i] <= 0.f) {
                    bmp_ptr[0] = 65;
                    bmp_ptr[1] = bmp_ptr[2] = 0;
                } else {
                    bmp_ptr[0] = bmp_ptr[1] = bmp_ptr[2] = (unsigned char)((imgbuf[i] - minval) * mult);
                }
            }

            //  write bmp file
            std::string full_name = dir_name + img_name;
            stbi_write_bmp(full_name.c_str(), x, y, 3, (const void *)bmpbuf.data());
        }

        switch (format) {
        case mkldnn::impl::memory_format::nChw8c:
            if ( ( img & 7 ) < 7 )   dataPtr++;
            else                dataPtr += img_sz * 8;
            break;
        case mkldnn::impl::memory_format::nChw16c:
            if ( ( img & 15 ) < 15 )    dataPtr++;
            else                    dataPtr += img_sz * 16;
            break;
        case mkldnn::impl::memory_format::nchw:
        default:
            dataPtr += img_sz;
            break;
        }
    }

    layer_bmp_log.close();
}
#endif

void MKLDNNGraph::Infer() {
    if (!IsReady()) {
        THROW_IE_EXCEPTION << "Wrong state. Topology is not ready.";
    }

    mkldnn::stream stream = mkldnn::stream(stream::kind::eager);

#ifdef DEBUG_DUMP_NEW_FOLDER_PER_INFER
        static int folderIdx = 0;
        folderIdx++;
#endif
    for (int i = 0; i < graphNodes.size(); i++) {
        PERF(graphNodes[i]);
        graphNodes[i]->setDynamicBatchLim(config.batchLimit);
        if (!graphNodes[i]->isConstant(true)) {
            IE_PROFILING_AUTO_SCOPE_STRING(graphNodes[i]->name.c_str())
            graphNodes[i]->execute(stream);
        }

#ifdef DEBUG_DUMP_PATH
        {
            auto folderName = std::string(DEBUG_DUMP_PATH) +
#ifdef DEBUG_DUMP_NEW_FOLDER_PER_INFER
            std::to_string(folderIdx - 1) +
#endif
            "/";
            std::cout << "Try to create logs for " << graphNodes[i]->getName() << std::endl;
            std::string nodeName = graphNodes[i]->name;
            std::replace(nodeName.begin(), nodeName.end(), '/', '_');
            std::ofstream layer_data_dump;
            for (size_t j = 0; j < graphNodes[i]->getChildEdges().size(); j++) {
                auto childEdge = graphNodes[i]->getChildEdgeAt(j);
                std::string childName = graphNodes[i]->getChildEdgeAt(j)->getChild()->getName();
                std::replace(childName.begin(), childName.end(), '/', '_');

                //  std::string fname = DEBUG_DUMP_PATH + nodeName + "_dst_" + childName + "_" + std::to_string(j) + ".txt";
                std::string tname = folderName + nodeName + "_dst_" + childName + "_" + std::to_string(j);
                std::string fname = tname + ".txt";
                if (graphNodes[i]->getChildEdges().size() == 1) {
                    fname = folderName + nodeName + "_dst.txt";
                }
                layer_data_dump.open(fname);
                if (layer_data_dump.is_open()) {
                    float *data = static_cast<float *>(childEdge->getMemory().GetData());
                    mkldnn::impl::memory_desc_wrapper dst_d(childEdge->getMemory().GetDescriptor().data);
    #ifdef DEBUG_BMP_OUTPUT
                    dump_as_bitmaps(tname, data, childEdge->getDims().ToSizeVector(), dst_d.format());
    #endif

                    layer_data_dump << "shape: ";
                    for (size_t d = 0; d < childEdge->getDims().ndims(); d++)
                        layer_data_dump << childEdge->getDims()[d] << " ";
                    layer_data_dump << "(" << dst_d.nelems() << ")" << std::endl;
                    for (size_t i = 0; i < dst_d.nelems(); i++) {
                        layer_data_dump << std::fixed << std::setprecision(3) << data[dst_d.off_l(i)] << std::endl;
                    }
                    layer_data_dump.close();
                } else {
                    std::cout << "Cannot create file " << fname << std::endl;
                }
            }

            for (size_t p = 0 ; p < graphNodes[i]->getParentEdges().size(); p++) {
                auto parentEdge = graphNodes[i]->getParentEdgeAt(p);
                auto parent = parentEdge->getParent();
                std::string parentName = parent->getName();
                std::replace(parentName.begin(), parentName.end(), '/', '_');
                //  std::string fname = folderName + nodeName + "_src_" + parentName + "_" + std::to_string(p) + ".txt";
                std::string tname = folderName + nodeName + "_src_" + parentName + "_" + std::to_string(p);
                std::string fname = tname + ".txt";
                layer_data_dump.open(fname);
                if (layer_data_dump.is_open()) {
                    float *data = static_cast<float *>(graphNodes[i]->getParentEdges()[p]
                            .lock()->getMemory().GetData());
                    mkldnn::impl::memory_desc_wrapper src_d(graphNodes[i]->getParentEdges()[p]
                                                                    .lock()->getMemory().GetDescriptor().data);
    #ifdef DEBUG_BMP_OUTPUT
                    dump_as_bitmaps(tname, data, parentEdge->getDims().ToSizeVector(), src_d.format());
    #endif
                    layer_data_dump << "shape: ";
                    for (size_t d = 0; d < parentEdge->getDims().ndims(); d++)
                        layer_data_dump << parentEdge->getDims()[d] << " ";
                    layer_data_dump << "(" << src_d.nelems() << ")"<< std::endl;
                    for (size_t i = 0; i < src_d.nelems(); i++) {
                        layer_data_dump << std::fixed << std::setprecision(3) << data[src_d.off_l(i)] << std::endl;
                    }
                    layer_data_dump.close();
                } else {
                    std::cout << "Cannot create file " << fname << std::endl;
                }
            }

            GenericLayer* genericLayer = dynamic_cast<GenericLayer*>(graphNodes[i]->getCnnLayer().get());
            if (genericLayer != nullptr) {
                for (auto blob : genericLayer->blobs) {
                    layer_data_dump.open(folderName + nodeName + "_" + blob.first + ".txt");
                    if (layer_data_dump.is_open()) {
                        layer_data_dump << "shape: ";
                        for (size_t d = 0; d < blob.second->dims().size(); d++)
                            layer_data_dump << blob.second->dims()[d] << " ";
                        layer_data_dump << "(" << blob.second->size() << ")"<< std::endl;
                        float *data = blob.second->buffer();
                        for (size_t bs = 0; bs < blob.second->size(); bs++) {
                            layer_data_dump << std::fixed << std::setprecision(3) << data[bs] << std::endl;
                        }
                        layer_data_dump.close();
                    } else {
                        std::cout << "Cannot create file " << folderName << nodeName
                                  << "_" << blob.first << ".txt" << std::endl;
                    }
                }
            }
        }
#endif
    }
}

MKLDNNNodePtr MKLDNNGraph::FindNodeWithName(const std::string& name) const {
    if (inputNodes.empty()) {
        return std::shared_ptr<MKLDNNNode>();
    }

    auto childs = graphNodes;

    auto node = std::find_if(childs.begin(), childs.end(),
                             [&name](MKLDNNNodePtr const& item) {
                                 return item->getName() == name;
                             });

    return (node == childs.end() ? std::shared_ptr<MKLDNNNode>() : *node);
}

void MKLDNNGraph::VisitNode(MKLDNNNodePtr node, std::vector<MKLDNNNodePtr>& sortedNodes) {
    if (node->temporary) {
        return;
    }

    if (node->permanent) {
        return;
    }

    node->temporary = true;

    for (size_t i = 0; i < node->getChildEdges().size(); i++) {
        VisitNode(node->getChildEdgeAt(i)->getChild(), sortedNodes);
    }

    node->permanent = true;
    node->temporary = false;

    sortedNodes.insert(sortedNodes.begin(), node);
}

void MKLDNNGraph::SortTopologically() {
    std::vector<MKLDNNNodePtr> unsorted;
    std::vector<MKLDNNNodePtr> sorted;

    for (int i = 0; i < graphNodes.size(); i++) {
        MKLDNNNodePtr node = graphNodes[i];

        node->permanent = false;
        node->temporary = false;

        unsorted.push_back(node);
    }

    while (!unsorted.empty()) {
        MKLDNNNodePtr node = unsorted.at(0);
        unsorted.erase(unsorted.begin());

        VisitNode(node, sorted);
    }

    graphNodes.erase(graphNodes.begin(), graphNodes.end());
    graphNodes.assign(sorted.begin(), sorted.end());
}

void MKLDNNGraph::GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    std::function<void(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &, const MKLDNNNodePtr&)>
            getPerfMapFor = [&](std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap, const MKLDNNNodePtr& node) {
        InferenceEngine::InferenceEngineProfileInfo &pc = perfMap[node->getName()];
        // TODO: Why time counter is signed?
        pc.cpu_uSec = pc.realTime_uSec = (long long) node->PerfCounter().avg();
        pc.status = pc.cpu_uSec > 0 ? InferenceEngine::InferenceEngineProfileInfo::EXECUTED
                                    : InferenceEngine::InferenceEngineProfileInfo::NOT_RUN;
        std::string pdType = node->getPrimitiveDescriptorType();
        size_t typeLen = sizeof(pc.exec_type) / sizeof(pc.exec_type[0]);
        pdType.copy(pc.exec_type, typeLen, 0);
        size_t layerTypeLen = sizeof(pc.layer_type) / sizeof(pc.layer_type[0]);
        node->typeStr.copy(pc.layer_type, layerTypeLen, 0);

        for (auto& fusedNode : node->fusedWith) {
            getPerfMapFor(perfMap, fusedNode);
        }

        for (auto& mergedWith : node->mergedWith) {
            getPerfMapFor(perfMap, mergedWith);
        }
    };

    for (int i = 1; i < graphNodes.size(); i++) {
        getPerfMapFor(perfMap, graphNodes[i]);
    }
}

void MKLDNNGraph::setConfig(const Config &cfg) {
    config = cfg;
}

void MKLDNNGraph::setProperty(const std::map<std::string, std::string>& properties) {
    config.readProperties(properties);
}

Config MKLDNNGraph::getProperty() {
    return config;
}

void MKLDNNGraph::getInputBlobs(InferenceEngine::BlobMap &resp) {
    for (auto &it : inputNodes) {
        MKLDNNNodePtr &node = it.second;
        float* in_ptr = reinterpret_cast<float*>(node->getChildEdgeAt(0)->getMemory().GetData());
        SizeVector in_dims = node->getChildEdgeAt(0)->getDims().ToSizeVector();

        resp[it.first] = make_shared_blob<float>({Precision::FP32, in_dims, TensorDesc::getLayoutByDims(in_dims)}, in_ptr);
    }
}

void MKLDNNGraph::getOutputBlobs(InferenceEngine::BlobMap &resp) {
    for (auto &it : outputNodes) {
        std::string name = it->getName().substr(4);

        float* out_ptr = reinterpret_cast<float*>(it->getParentEdgeAt(0)->getMemory().GetData());
        SizeVector out_dims = it->getParentEdgeAt(0)->getDims().ToSizeVector();

        resp[name] = make_shared_blob<float>({Precision::FP32, out_dims, TensorDesc::getLayoutByDims(out_dims)}, out_ptr);
    }
}

InferenceEngine::InferRequestInternal::Ptr
MKLDNNExecNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                          InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<MKLDNNInferRequest>(networkInputs, networkOutputs);
}

MKLDNNExecNetwork::MKLDNNExecNetwork(InferenceEngine::ICNNNetwork &network,
                                     const Config &cfg,
                                     const MKLDNNExtensionManager::Ptr& extMgr) : extensionManager(extMgr) {
    graph.reset(new MKLDNNGraph());
    graph->setConfig(cfg);

    if (graph->getProperty().exclusiveAsyncRequests) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor(TargetDeviceInfo::name(TargetDevice::eCPU));
    }

    // initialization in taskExecutor thread
    auto task = std::make_shared<InferenceEngine::Task>([&]() {
        graph->CreateGraph(network, extensionManager);
    });

    _taskExecutor->startTask(task);
    Task::Status sts = task->wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    if (sts == Task::TS_ERROR) task->checkException();
}

void MKLDNNExecNetwork::setProperty(const std::map<std::string, std::string> &properties) {
    if (graph)  // TODO: graph field cannot be empty
        graph->setProperty(properties);
}

void MKLDNNExecNetwork::CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncRequestImpl = std::make_shared<MKLDNNAsyncInferRequest>(syncRequestImpl, _taskExecutor,
                                                                      _taskSynchronizer, _callbackExecutor);
    asyncRequest.reset(new InferRequestBase<MKLDNNAsyncInferRequest>(asyncRequestImpl),
                       [](IInferRequest *p) { p->Release(); });

    asyncRequestImpl->SetPointerToPublicInterface(asyncRequest);

    auto mkldnnSyncRequest = dynamic_cast<MKLDNNInferRequest *>(syncRequestImpl.get());
    if (!mkldnnSyncRequest)
        THROW_IE_EXCEPTION << " Cannot get mkldnn sync request.";
    mkldnnSyncRequest->SetGraph(graph);
}

MKLDNNExecNetwork::~MKLDNNExecNetwork() {
    graph.reset();
    extensionManager.reset();
}
