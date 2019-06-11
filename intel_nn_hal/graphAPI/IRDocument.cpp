/*
 * INTEL CONFIDENTIAL
 * Copyright 2017 Intel Corporation.
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

//#define LOG_TAG "graphAPI"

#include "IRDocument.h"
#include <fstream>
#include <locale>
#include "IRLayers.h"
#include "cnn_network_impl.hpp"

#ifdef NNLOG
#include <android/log.h>
#include <log/log.h>
#endif

using namespace IRBuilder;
using namespace std;
using namespace InferenceEngine;

class InternalNetworkImpl : public InferenceEngine::details::CNNNetworkImpl {
   public:
    InternalNetworkImpl() {}
    InternalNetworkImpl(const std::string netName) : InternalNetworkImpl() {
        setPrecision(IRBuilder::g_layer_precision);
        setName(netName);
    }

    void remove(const string &layer_name) { _layers.erase(layer_name); }

    bool hasLayer(const string &lname) const { return _layers.find(lname) != _layers.end(); }

    void addData(const DataPtr &data) { _data[data->name] = data; }

    void addOutput(const DataPtr &data) {
        addData(data);

        auto docdatadims = data->getTensorDesc().getDims();
#ifdef NNLOG
        for (auto i = 0; i < docdatadims.size(); i++)
            ALOGI("doc addOutput data dims[%d] = %d", i, docdatadims[i]);
#endif
        std::cout << "doc addOutput datadims size " << docdatadims.size() << std::endl;
        // std::cout << "docdatadims[0] "<<docdatadims[0]<< "docdatadims[1]" <<docdatadims[1]<<
        // std::endl;

        _outputData[data->name] = data;
    }
};

IRDocument::IRDocument(const std::string &cs) : _name(cs) { network = new InternalNetworkImpl(cs); }

IRDocument::~IRDocument() {
    delete network;
    network = nullptr;
}

void IRDocument::add(const IRLayer &ir_layer) {
#ifdef NNLOG
    ALOGI("add ir_layer = %s", ir_layer->name.c_str());
#endif
    network->addLayer(ir_layer);
    _layers.push_back(ir_layer);
}

bool IRDocument::shouldRemove(const IRLayer &l) {
    if (l->type != "Reshape") return false;
    return l->insData[0].lock()->getDims() == output(l)->getDims();
}

void IRDocument::process(const IRLayer &layer) {
    if (network->hasLayer(layer->name)) return;
    add(layer);
    for (auto o : layer->outData) {
        network->addData(o);
        for (auto l : o->inputTo) process(l.second);
    }
}

void IRDocument::optimize() {
    for (auto it = _layers.begin(); it != _layers.end();) {
        auto l = *it;
        if (shouldRemove(l)) {
            // l-in -> (l,l-out) -> (b-in list) ===> a-out -> (b-in list)
            auto lin = l->input();
            auto lout = output(l);

            lin->inputTo.erase(l->name);

            auto lout_targets = lout->inputTo;
            for (auto i : lout_targets) {
                lin->inputTo[i.first] = i.second;
                // reaplce target input data from lout to lin
                for (auto &tar_inp : i.second->insData) {
                    if (tar_inp.lock() == lout) {
                        tar_inp = lin;
                        break;
                    }
                }
            }

            it = _layers.erase(it);
            network->remove(l->name);
        }
        else {
            ++it;
        }
    }
}

void IRDocument::build() {
    if (_processed) return;
    network->setPrecision(IRBuilder::g_layer_precision);
    InputsDataMap inputs;
    network->getInputsInfo(inputs);
    for (auto i : inputs) {
        for (auto l : i.second->getInputData()->inputTo) process(l.second);
    }

    for (auto l : network->allLayers()) {
        process(l.second);
    }
    optimize();
    _processed = true;
}

InferenceEngine::ICNNNetwork *IRDocument::buildNetwork() {
    build();
    return network;
}
InferenceEngine::ICNNNetwork *IRDocument::getNetwork() { return network; }
/**
 * \brief save a blob to IR
 * \param binFile
 * \param blob
 * \param layer
 * \param name
 */

void IRDocument::saveBlobToIR(std::ostream &binFile,
                              const /*IRBlob::Ptr*/ InferenceEngine::Blob::Ptr &blob,
                              pugi::xml_node &layer, const std::string &name) {
    const float *fp = blob->cbuffer().as<const float *>();

    auto fit = _segmentsMap.find(fp);
    bool newBlob = fit == _segmentsMap.end();
    size_t offset;
    if (newBlob) {
        offset = binFile.tellp();
        //_segmentsMap[(float*)fp] = offset;
        _segmentsMap[fp] = offset;
    } else {
        offset = fit->second;
    }

    auto node = layer.append_child(name.c_str());
    node.append_attribute("offset").set_value(offset);
    node.append_attribute("size").set_value(blob->byteSize());
    if (newBlob) {
        binFile.write(reinterpret_cast<const char *>(fp), blob->byteSize());
    }

    node.append_attribute("precision").set_value(blob->precision().name());
}

void IRDocument::save(std::ostream &xml_os, std::ostream &bin_os) {
    pugi::xml_document doc;

    build();
    pugi::xml_node root = doc.append_child("net");
    root.append_attribute("name").set_value(_name.c_str());
    root.append_attribute("version").set_value(2);
    root.append_attribute("batch").set_value(1);
    pugi::xml_node layers = root.append_child("layers");
    int id_cnt = 0;
    InputsDataMap netInputs;
    network->getInputsInfo(netInputs);
    int icnt = 0;
    for (auto &kvp : netInputs) {  // todo: consider adding input layer to actual network...
        InferenceEngine::LayerParams prms;
        prms.name = kvp.first;
        CNNLayer::Ptr inputLayer(new CNNLayer(prms));
        inputLayer->type = "Input";
        auto input_data = kvp.second->getInputData();
        input_data->userObject.v_int = icnt++;
        inputLayer->outData.push_back(input_data);
        saveToIR(bin_os, layers, inputLayer);
    }

    for (auto &cnn_layer : _layers) {
        cnn_layer->userValue.v_int = ++id_cnt;
        int pcnt = 0;
        for (auto output : cnn_layer->outData) output->userObject.v_int = pcnt++;
        saveToIR(bin_os, layers, cnn_layer);
    }
    pugi::xml_node edgesNode = root.append_child("edges");
    for (auto &kvp : _layers) {
        int pcnt = 0;
        for (auto inputData : kvp->insData) {
            Edge edge;

            edge.to.lid = kvp->userValue.v_int;
            edge.to.pid = pcnt++;
            bool b = inputData.lock()->creatorLayer.expired();
            edge.from.lid = b ? 0 : inputData.lock()->creatorLayer.lock()->userValue.v_int;
            edge.from.pid = inputData.lock()->userObject.v_int;

            _edges.push_back(edge);

            auto node = edgesNode.append_child("edge");
            node.append_attribute("from-layer").set_value(edge.from.lid);
            node.append_attribute("from-port").set_value(edge.from.pid);
            node.append_attribute("to-layer").set_value(edge.to.lid);
            node.append_attribute("to-port").set_value(edge.to.pid);
        }
    }
    doc.save(xml_os);
}

void IRDocument::save(const std::string &filebase) {
    std::fstream xml, bin;

    xml.open(filebase + ".xml", std::ios_base::out);
    bin.open(filebase + ".bin", std::ios_base::out | std::ios_base::binary);
    save(xml, bin);
    xml.close();
    bin.close();
}

void IRDocument::crateDotFile(std::ostream &dot) const {
    dot << "digraph g {\n\tgraph[rankdir = \"LR\"];" << std::endl;

    for (auto &kvp : network->allLayers()) {
        saveLayerToDot(dot, kvp.second);
    }
    dot << std::endl << std::endl;
    for (auto &kvp : _edges) {
        dot << "\t\"layer_" << kvp.from.lid << "\":p" << kvp.from.pid << " -> \"layer_"
            << kvp.to.lid << "\":p" << kvp.to.pid << " [];" << std::endl;
    }
    dot << "}" << std::endl;
}

InferenceEngine::InputInfo::Ptr IRDocument::createInput(const std::string &name,
                                                        const TensorDims &dims) const {
    Layout layout;
    if (dims.size() == 4) layout = NCHW;
    else if (dims.size() == 2)
        layout = InferenceEngine::Layout::NC;
    else
        layout = InferenceEngine::Layout::C;

    std::cout << "createInput input data dims[0] " << dims[0] << "dims[1]" << dims[1] << std::endl;
    TensorDesc td(IRBuilder::g_layer_precision, dims, layout);

    auto inputData = std::make_shared<InferenceEngine::Data>(name, td);
    InferenceEngine::InputInfo::Ptr info(new InferenceEngine::InputInfo());

    info->setInputData(inputData);

    Precision inputPrecision = info->getInputPrecision();
    if (inputPrecision == Precision::FP16) {
        info->setInputPrecision(Precision::FP32);
    }

    network->setInputInfo(info);

#ifdef NNLOG
    ALOGI("createInput input info dims size %d and layout %d", info->getDims().size(),
          info->getLayout());
#endif

    auto indatadims = inputData->getDims();
    std::cout << " createInput input data indatadims[0] " << indatadims[0] << "indatadims[1]"
              << indatadims[1] << std::endl;

#ifdef NNLOG
    for (auto i = 0; i < indatadims.size(); i++)
        ALOGI("createInput input data dims[%d] = %d", i, indatadims[i]);
#endif

    return info;
}

void IRDocument::addOutput(const DataPtr &src) { network->addOutput(src); }

/**
 * \brief save output port to IR
 * \param parent
 * \param port
 */
void IRDocument::saveOutputToIR(pugi::xml_node &parent, const DataPtr &port) {
    auto node = parent.append_child("port");
    node.append_attribute("id").set_value(port->userObject.v_int);
    // node.append_attribute("buffer").set_value(reinterpret_cast<size_t>(buffer) & 0x00FFFFFF);
    if (!port->inputTo.empty()) {
        std::string comment = "connected to ";
        for (auto peer : port->inputTo) comment += ", " + peer.first;
        node.append_child(pugi::xml_node_type::node_comment).set_value(comment.c_str());
    }
    auto dims = port->getDims();
    for (auto d : dims) {
        node.append_child("dim").text().set(d);
    }
}

/**
 * \brief save input port to IR
 * \param parent
 * \param index
 * \param port
 */
void IRDocument::saveInputToIR(pugi::xml_node &parent, int index, const DataPtr &port) {
    auto node = parent.append_child("port");
    node.append_attribute("id").set_value(index);
    auto peer = port->creatorLayer.lock();
    if (peer) {
        auto comment = "connected to " + peer->name;
        node.append_child(pugi::xml_node_type::node_comment).set_value(comment.c_str());
    }
    auto dims = port->getDims();
    for (auto d : dims) {
        node.append_child("dim").text().set(d);
    }
}

/**
 * \brief Save layer to IR
 * \param binFile
 * \param parent
 * \param irLayer
 */
void IRDocument::saveToIR(std::ostream &binFile, pugi::xml_node &parent, const IRLayer &irLayer) {
    auto layer = parent.append_child("layer");
    layer.append_attribute("name").set_value(irLayer->name.c_str());
    layer.append_attribute("type").set_value(irLayer->type.c_str());
    layer.append_attribute("id").set_value(irLayer->userValue.v_int);
    layer.append_attribute("precision")
        .set_value(IRBuilder::g_layer_precision == Precision::FP16 ? "FP16" : "FP32");

    if (!irLayer->params.empty()) {
        auto attr = layer.append_child("data");  // todo: need to check for type and overide it
        for (auto &kvp : irLayer->params) {
            attr.append_attribute(kvp.first.c_str()).set_value(kvp.second.c_str());
        }
    }
    if (!irLayer->insData.empty()) {
        auto inputs = layer.append_child("input");
        for (int i = 0; i < irLayer->insData.size(); ++i) {
            saveInputToIR(inputs, i, irLayer->insData[i].lock());
        }
    }
    if (!irLayer->outData.empty()) {
        auto outputs = layer.append_child("output");
        for (auto &inp : irLayer->outData) {
            saveOutputToIR(outputs, inp);
        }
    }

    for (auto blob : irLayer->blobs) {
        auto fb = blob.second;
        saveBlobToIR(binFile, fb, layer, blob.first);
    }
}

void IRDocument::saveLayerToDot(std::ostream &dot, const IRLayer &irLayer) const {
    /*
    "layer_4" [
    label = "name| type | <f2> |-1"
    shape = "record"
    ];
    */
    dot << "\t\"layer_" << irLayer->userValue.v_int << "\" [ label = \"" << _name
        << "| type: " << irLayer->type;
    int pid = 0;
    for (auto &in : irLayer->insData) {
        dot << "| ";
        auto dims = in.lock()->getDims();
        dot << "<p" << (pid++) << "> " << dims[0];
        for (int i = 1; i < dims.size(); ++i) dot << ", " << dims[i];
    }
    for (auto &p : irLayer->outData) {
        dot << "| ";
        auto dims = p->getDims();
        dot << "<p" << p->userObject.v_int << "> " << dims[0];
        for (int i = 1; i < dims.size(); ++i) dot << ", " << dims[i];
    }
    dot << "\"";
    dot << "\t\tshape = \"record\" ];" << std::endl;
}
void IRDocument::setName(const char *name) {
    _name = name;
    network->setName(_name);
}
